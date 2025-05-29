#!/bin/bash

# Build and run MNIST Project with Docker Compose
# Docker Compose handles building images and managing dependencies automatically

# Check for --compose flag
if [ "$1" = "--compose" ]; then
    echo "🐳 Using Docker Compose build..."
    docker compose down --remove-orphans
    docker compose up --build
    exit 0
fi

echo "🐳 Building Docker images with detailed output..."
docker compose down --remove-orphans

# Build API image with detailed progress
echo "📦 Building FastAPI image..."
docker build -f api/Dockerfile --progress=plain -t mnist-api:latest .

# Build App image with detailed progress  
echo "📦 Building Gradio App image..."
docker build -f app/Dockerfile --progress=plain -t mnist-app:latest ./app

echo "🚀 Starting services..."
docker compose up