#!/bin/bash

# Build Docker Images for MNIST Project
# This script builds the API and App Docker containers

set -e  # Exit on any error

echo "🐳 Building Docker Images for MNIST Project..."

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

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

# Build API Docker image
print_status "Building FastAPI container..."
if [ -d "api" ] && [ -d "model" ]; then
    # Build from parent directory to access both api and model folders
    docker build -f api/Dockerfile -t mnist-api:latest .
    print_status "✅ FastAPI image built successfully"
else
    print_error "❌ api/ or model/ directory not found"
    exit 1
fi

# Build App Docker image
print_status "Building Gradio App container..."
if [ -d "app" ]; then
    cd app
    if [ -f "Dockerfile" ]; then
        docker build -t mnist-app:latest .
        print_status "✅ Gradio App image built successfully"
    else
        print_error "❌ Dockerfile not found in app/ directory"
        exit 1
    fi
    cd ..
else
    print_error "❌ app/ directory not found"
    exit 1
fi

echo ""
print_status "🎉 Docker images built successfully!"
echo ""
echo "📦 Built Images:"
echo "   • mnist-api:latest (FastAPI Backend)"
echo "   • mnist-app:latest (Gradio Frontend)"
echo ""
echo "🚀 Next Steps:"
echo "   • Run with Docker Compose: docker-compose up"
echo "   • Or run individual containers manually"
echo ""

# Show built images
print_status "📋 Docker Images:"
docker images | grep -E "(mnist-api|mnist-app|REPOSITORY)"