#!/bin/bash

# Simple Docker Compose build and run script for MNIST project

set -e

copy_model_deps() {
  MODEL_SRC_DIR="./model"
  MODEL_DST_DIR="./api/model"
  if [ -d "$MODEL_SRC_DIR" ]; then
    echo "Copying model files to $MODEL_DST_DIR ..."
    mkdir -p "$MODEL_DST_DIR"
    rsync -av --include 'MnistModel.py' --include '*.pth' --exclude 'node_modules' --exclude '__pycache__' --exclude '*' "$MODEL_SRC_DIR/" "$MODEL_DST_DIR/"
    cp "$MODEL_SRC_DIR/requirements.txt" ./api/model_requirements.txt
  fi
}

case "$1" in
  --build)
    copy_model_deps
    echo "🔧 Building Docker images..."
    docker compose build
    ;;
  --reset)
    echo "🔥 Removing project containers and images..."
    docker compose down --remove-orphans
    docker rmi -f mnist-api:latest mnist-app:latest 2>/dev/null || true
    ;;
  ""|--help)
    echo "Usage: $0 [--build | --reset | --help]"
    echo "  (no args)   Copy model dependencies for API (default)"
    echo "  --build     Copy model dependencies and build Docker images"
    echo "  --reset     Remove containers and images (keep volumes)"
    echo "  --help      Show this help message"
    ;;
  *)
    echo "Unknown option: $1"
    echo "Use --help to see available options."
    exit 1
    ;;
esac

# Default action: copy model dependencies
if [ -z "$1" ]; then
  copy_model_deps
fi