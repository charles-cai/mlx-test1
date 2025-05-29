#!/bin/bash

# MNIST Project Local Development Startup Script
# This script starts PostgreSQL in Docker and runs API/App locally with Python

set -e  # Exit on any error

echo "🚀 Starting MNIST Project (Local Development Mode)..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Cleanup function
cleanup() {
    print_status "🛑 Shutting down services..."
    if [ ! -z "$API_PID" ]; then
        kill $API_PID 2>/dev/null || true
    fi
    if [ ! -z "$APP_PID" ]; then
        kill $APP_PID 2>/dev/null || true
    fi
    print_status "✅ Services stopped"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed. Please install Python 3 first."
    exit 1
fi

# Create necessary directories
print_status "Creating necessary directories..."
mkdir -p data logs

# Check if model exists
if [ ! -f "model/saved_model.pth" ]; then
    print_warning "Model file not found. Training model first..."
    
    # Create virtual environment if it doesn't exist
    if [ ! -d ".venv" ]; then
        print_status "Creating virtual environment..."
        python3 -m venv .venv
    fi
    
    # Activate virtual environment and install dependencies
    print_status "Installing dependencies..."
    source .venv/bin/activate
    pip install -r requirements.txt
    
    # Train the model
    print_status "Training MNIST model..."
    cd model
    python MnistModel.py --train --epochs 5
    cd ..
    
    print_status "Model training completed!"
else
    print_status "Model file found: model/saved_model.pth"
fi

# Start PostgreSQL Docker container
print_status "Starting PostgreSQL Docker container..."
cd db
chmod +x setup.sh
./setup.sh
cd ..

# Wait for PostgreSQL to be ready
print_status "Waiting for PostgreSQL to be ready..."
sleep 5

# Set up virtual environment for local development
if [ ! -d ".venv" ]; then
    print_status "Creating virtual environment..."
    python3 -m venv .venv
fi

print_status "Activating virtual environment and installing dependencies..."
source .venv/bin/activate
pip install -r requirements.txt > /dev/null 2>&1

# Set environment variables for local development
export DATABASE_URL="postgresql://postgres:password@localhost:5432/mnist_db"
export API_URL="http://localhost:8000"

# Start FastAPI in background
print_status "Starting FastAPI server..."
cd model
python api.py &
API_PID=$!
cd ..

# Wait for API to start
sleep 3

# Start Gradio App in background
print_status "Starting Gradio application..."
cd app
python app.py &
APP_PID=$!
cd ..

# Wait for services to start
print_status "Waiting for services to start..."
sleep 5

# Check if services are running
print_status "Checking service status..."

# Check PostgreSQL
if docker ps | grep -q "mnist_postgres"; then
    print_status "✅ PostgreSQL is running"
else
    print_error "❌ PostgreSQL failed to start"
fi

# Check API
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    print_status "✅ FastAPI is running at http://localhost:8000"
else
    print_warning "⚠️  FastAPI might still be starting..."
fi

# Check Gradio
if curl -s http://localhost:7860 > /dev/null 2>&1; then
    print_status "✅ Gradio UI is running at http://localhost:7860"
else
    print_warning "⚠️  Gradio UI might still be starting..."
fi

echo ""
print_status "🎉 MNIST Local Development Setup Complete!"
echo ""
echo "📱 Access Points:"
echo "   • Gradio UI (Main Interface): http://localhost:7860"
echo "   • FastAPI (Backend): http://localhost:8000"
echo "   • API Documentation: http://localhost:8000/docs"
echo "   • PostgreSQL: localhost:5432"
echo ""
echo "🛠️  Management Commands:"
echo "   • Stop PostgreSQL: docker stop mnist_postgres"
echo "   • View PostgreSQL logs: docker logs mnist_postgres"
echo "   • Connect to DB: docker exec -it mnist_postgres psql -U postgres -d mnist_db"
echo ""
echo "🎯 Try drawing a digit at http://localhost:7860 to test the system!"
echo ""
print_status "Press Ctrl+C to stop all services..."

# Keep the script running and wait for interrupt
while true; do
    sleep 1
done