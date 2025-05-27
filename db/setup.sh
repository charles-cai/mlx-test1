#!/bin/bash
# PostgreSQL Docker Setup for MNIST Project

CONTAINER_NAME="mnist_postgres"
DATABASE_NAME="mnist_db"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🐘 PostgreSQL Docker Setup for MNIST Project"
echo "============================================="

# Check if --init flag is provided
if [ "$1" = "--init" ]; then
    echo "📋 Initializing database schema only..."
    if docker exec -i $CONTAINER_NAME psql -U postgres -d $DATABASE_NAME < "$SCRIPT_DIR/init.sql"; then
        echo "✅ Database schema initialized"
    else
        echo "❌ Failed to initialize schema"
        exit 1
    fi
    exit 0
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker first."
    exit 1
fi

# Function to wait for PostgreSQL to be ready
wait_for_postgres() {
    echo "⏳ Waiting for PostgreSQL to be ready..."
    for i in {1..30}; do
        if docker exec $CONTAINER_NAME pg_isready -U postgres > /dev/null 2>&1; then
            echo "✅ PostgreSQL is ready!"
            return 0
        fi
        echo "   Waiting... ($i/30)"
        sleep 2
    done
    echo "❌ PostgreSQL failed to start within 60 seconds"
    return 1
}

# Check if container is already running
if docker ps -q -f name=$CONTAINER_NAME | grep -q .; then
    echo "✅ PostgreSQL container already running"
elif docker ps -a -q -f name=$CONTAINER_NAME | grep -q .; then
    echo "🚀 Starting existing PostgreSQL container..."
    docker start $CONTAINER_NAME
    wait_for_postgres
else
    echo "🐳 Creating new PostgreSQL 17 container..."
    docker run -d \
        --name $CONTAINER_NAME \
        -e POSTGRES_PASSWORD=password \
        -e POSTGRES_USER=postgres \
        -e POSTGRES_DB=postgres \
        -p 5432:5432 \
        postgres:17
    
    wait_for_postgres || exit 1
fi

# Create database if it doesn't exist
echo "🗄️ Setting up MNIST database..."
docker exec -i $CONTAINER_NAME psql -U postgres -c "CREATE DATABASE $DATABASE_NAME;" 2>/dev/null || echo "   Database might already exist"

# Run schema setup
echo "📋 Setting up database schema..."
if [ -f "$SCRIPT_DIR/init.sql" ]; then
    docker exec -i $CONTAINER_NAME psql -U postgres -d $DATABASE_NAME < "$SCRIPT_DIR/init.sql"
    echo "✅ Database schema setup complete"
else
    echo "❌ init.sql not found in $SCRIPT_DIR"
    exit 1
fi

# Test the database
echo "🧪 Testing database connection..."
if docker exec -i $CONTAINER_NAME psql -U postgres -d $DATABASE_NAME -c "SELECT COUNT(*) FROM predictions;" > /dev/null 2>&1; then
    echo "✅ Database test successful"
else
    echo "❌ Database test failed"
    exit 1
fi

echo ""
echo "🎉 PostgreSQL setup complete!"
echo ""
echo "📊 Connection details:"
echo "   Container: $CONTAINER_NAME"
echo "   Database: $DATABASE_NAME"
echo "   Host: localhost:5432"
echo "   User: postgres"
echo "   Password: password"
echo ""
echo "🔧 Useful commands:"
echo "   Connect: docker exec -it $CONTAINER_NAME psql -U postgres -d $DATABASE_NAME"
echo "   Init schema only: ./setup.sh --init"
echo "   Stop: docker stop $CONTAINER_NAME"
echo "   Start: docker start $CONTAINER_NAME"
echo ""