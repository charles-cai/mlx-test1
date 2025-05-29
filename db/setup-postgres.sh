#!/bin/bash
# Simplified PostgreSQL Docker Setup for MNIST Project

CONTAINER_NAME="mnist_postgres"
DATABASE_NAME="mnist_db"
POSTGRES_USER="postgres"
POSTGRES_PASSWORD="password"
POSTGRES_PORT="5442"
POSTGRES_IMAGE="postgres:17"

set -e

echo "🐘 PostgreSQL Docker Setup for MNIST Project"
echo "============================================="

run_cmd() {
    echo "$ $*"
    "$@"
}

case "$1" in
    --help|"")
        echo "Usage: ./setup-postgres.sh [--init | --check | --reset | --start | --stop | --help]"
        echo "  --init     Initialize schema only (assumes container is running)"
        echo "  --check    Check database status and list table schema (assumes container is running)"
        echo "  --reset    Reset database - drop and recreate tables (assumes container is running)"
        echo "  --start    Start PostgreSQL container if not running"
        echo "  --stop     Stop PostgreSQL container if running"
        echo "  --help     Show this help message"
        exit 0
        ;;
    --init)
        echo "📋 Initializing database schema only..."
        run_cmd docker exec -i $CONTAINER_NAME psql -U $POSTGRES_USER -d $DATABASE_NAME < "init.sql"
        echo "✅ Database schema initialized"
        exit 0
        ;;
    --reset)
        echo "⚠️  Resetting database..."
        run_cmd docker exec $CONTAINER_NAME psql -U $POSTGRES_USER -d $DATABASE_NAME -c "DROP TABLE IF EXISTS predictions;"
        run_cmd docker exec -i $CONTAINER_NAME psql -U $POSTGRES_USER -d $DATABASE_NAME < "init.sql"
        echo "✅ Database reset complete!"
        exit 0
        ;;
    --check)
        echo "🔍 Checking PostgreSQL status..."
        run_cmd docker exec $CONTAINER_NAME psql -U $POSTGRES_USER -d $DATABASE_NAME -c "\\d+ predictions"
        run_cmd docker exec $CONTAINER_NAME psql -U $POSTGRES_USER -d $DATABASE_NAME -c "SELECT COUNT(*) as total_predictions FROM predictions;"
        exit 0
        ;;
    --start)
        echo "🚀 Starting PostgreSQL container..."
        run_cmd docker start $CONTAINER_NAME
        exit 0
        ;;
    --stop)
        echo "🛑 Stopping PostgreSQL container..."
        run_cmd docker stop $CONTAINER_NAME
        echo "PostgreSQL container stopped."
        exit 0
        ;;
esac