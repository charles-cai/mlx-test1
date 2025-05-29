#!/bin/bash
# PostgreSQL Docker Setup for MNIST Project

# Configuration variables
CONTAINER_NAME="mnist_postgres"
DATABASE_NAME="mnist_db"
POSTGRES_USER="postgres"
POSTGRES_PASSWORD="password"
POSTGRES_DB="postgres"
POSTGRES_PORT="5442"
POSTGRES_IMAGE="postgres:17"

echo "🐘 PostgreSQL Docker Setup for MNIST Project"
echo "============================================="

# Function to run command and show error
run_cmd() {
    echo "$ $*"
    if ! "$@"; then
        echo "❌ Command failed with exit code $?"
        exit 1
    fi
}

# Check for command line arguments
case "$1" in
    "--help")
        echo "Usage: ./setup.sh [OPTION]"
        echo ""
        echo "Options:"
        echo "  (no args)  Full setup - creates/starts PostgreSQL container and loads schema"
        echo "  --init     Initialize schema only (assumes container is running)"
        echo "  --check    Check database status and list table schema"
        echo "  --reset    Reset database - drop and recreate tables with confirmation"
        echo ""
        echo "Examples:"
        echo "  ./setup.sh           # Full setup"
        echo "  ./setup.sh --init    # Load schema only"
        echo "  ./setup.sh --check   # Check database status"
        echo "  ./setup.sh --reset   # Reset database tables"
        exit 0
        ;;
    "--init")
        echo "📋 Initializing database schema only..."
        run_cmd docker exec -i $CONTAINER_NAME psql -U postgres -d $DATABASE_NAME < "init.sql"
        echo "✅ Database schema initialized"
        exit 0
        ;;
    "--reset")
        echo "⚠️  Database Reset"
        echo "=================="
        echo "This will:"
        echo "  • Drop the 'predictions' table and ALL data"
        echo "  • Recreate the table with the latest schema"
        echo ""
        read -p "Are you sure you want to continue? [Y/n]: " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
            echo "🗑️  Dropping existing table..."
            docker exec $CONTAINER_NAME psql -U postgres -d $DATABASE_NAME -c "DROP TABLE IF EXISTS predictions;" 2>/dev/null || true
            echo "📋 Recreating table with latest schema..."
            run_cmd docker exec -i $CONTAINER_NAME psql -U postgres -d $DATABASE_NAME < "init.sql"
            echo "✅ Database reset complete!"
        else
            echo "❌ Reset cancelled"
        fi
        exit 0
        ;;
    "--check")
        echo "🔍 Checking PostgreSQL status..."
        echo ""
        echo "📋 Table Schema:"
        run_cmd docker exec $CONTAINER_NAME psql -U postgres -d $DATABASE_NAME -c "\\d+ predictions"
        echo ""
        echo "📊 Table Statistics:"
        run_cmd docker exec $CONTAINER_NAME psql -U postgres -d $DATABASE_NAME -c "SELECT COUNT(*) as total_predictions FROM predictions;"
        exit 0
        ;;
esac

# Start or create PostgreSQL container
echo "🐳 Starting PostgreSQL container..."
if docker ps -q -f name=$CONTAINER_NAME | grep -q .; then
    echo "✅ Container already running"
elif docker start $CONTAINER_NAME 2>/dev/null; then
    echo "✅ Started existing container"
    sleep 2  # Brief wait for startup
else
    echo "Creating new container..."
    # Remove existing container if it exists (force remove to handle running containers)
    docker rm -f $CONTAINER_NAME 2>/dev/null || true
    run_cmd docker run -d \
        --name $CONTAINER_NAME \
        -e POSTGRES_PASSWORD=$POSTGRES_PASSWORD \
        -e POSTGRES_USER=$POSTGRES_USER \
        -e POSTGRES_DB=$POSTGRES_DB \
        -p $POSTGRES_PORT:5432 \
        $POSTGRES_IMAGE
    echo "⏳ Waiting for new container to start..."
    sleep 5
fi

# Create database
echo "🗄️ Creating MNIST database..."
docker exec $CONTAINER_NAME psql -U postgres -c "CREATE DATABASE $DATABASE_NAME;" 2>/dev/null || true

# Load schema
echo "📋 Loading database schema..."
run_cmd docker exec -i $CONTAINER_NAME psql -U postgres -d $DATABASE_NAME < "init.sql"

# Generate DATABASE_URL
DATABASE_URL="postgresql://$POSTGRES_USER:$POSTGRES_PASSWORD@localhost:$POSTGRES_PORT/$DATABASE_NAME"
export DATABASE_URL

echo ""
echo "🎉 PostgreSQL setup complete!"
echo ""
echo "📊 Connection details:"
echo "   Container: $CONTAINER_NAME"
echo "   Database: $DATABASE_NAME"
echo "   Host: localhost:$POSTGRES_PORT"
echo "   User: $POSTGRES_USER"
echo "   Password: $POSTGRES_PASSWORD"
echo ""
echo "🔗 Database URL:"
echo "   DATABASE_URL=$DATABASE_URL"
echo ""
echo "📋 Configuration variables:"
echo "   CONTAINER_NAME=$CONTAINER_NAME"
echo "   DATABASE_NAME=$DATABASE_NAME"
echo "   POSTGRES_USER=$POSTGRES_USER"
echo "   POSTGRES_PASSWORD=$POSTGRES_PASSWORD"
echo "   POSTGRES_DB=$POSTGRES_DB"
echo "   POSTGRES_PORT=$POSTGRES_PORT"
echo "   POSTGRES_IMAGE=$POSTGRES_IMAGE"
echo ""
echo "🔧 Useful commands:"
echo "   Connect: docker exec -it $CONTAINER_NAME psql -U $POSTGRES_USER -d $DATABASE_NAME"
echo "   Check status: ./setup.sh --check"
echo "   Reload schema: ./setup.sh --init"
echo "   Remove and reload schema: ./setup.sh --reset"
echo ""