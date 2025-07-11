#!/bin/bash

# DriaClaude2 Build Script
# This script builds and optionally runs the DriaClaude2 container

set -e

echo "🚀 DriaClaude2 Build Script"
echo "=========================="

# Parse command line arguments
BUILD_ONLY=false
RUN_TESTS=false
FORCE_REBUILD=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --build-only)
            BUILD_ONLY=true
            shift
            ;;
        --test)
            RUN_TESTS=true
            shift
            ;;
        --force)
            FORCE_REBUILD=true
            shift
            ;;
        --help)
            echo "Usage: ./build.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --build-only    Only build the container, don't run it"
            echo "  --test          Run API tests after starting"
            echo "  --force         Force rebuild (no cache)"
            echo "  --help          Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run './build.sh --help' for usage information"
            exit 1
            ;;
    esac
done

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose is not installed. Please install docker-compose first."
    exit 1
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p uploads outputs voices models

# Set permissions for unRAID compatibility
echo "🔒 Setting permissions..."
chmod 755 uploads outputs voices models

# Build the container
echo "🔨 Building container..."
if [ "$FORCE_REBUILD" = true ]; then
    docker-compose build --no-cache
else
    docker-compose build
fi

if [ "$BUILD_ONLY" = true ]; then
    echo "✅ Build complete!"
    echo "Run 'docker-compose up -d' to start the container"
    exit 0
fi

# Stop any existing container
echo "🛑 Stopping existing container..."
docker-compose down

# Start the container
echo "🚀 Starting container..."
docker-compose up -d

# Wait for container to be ready
echo "⏳ Waiting for container to be ready..."
sleep 10

# Check if container is running
if docker-compose ps | grep -q "driaclaude2.*Up"; then
    echo "✅ Container is running!"
    echo ""
    echo "📌 Access points:"
    echo "   Web UI: http://localhost:4144/ui"
    echo "   API Docs: http://localhost:4144/docs"
    echo "   Health: http://localhost:4144/health"
    echo ""
    
    # Run tests if requested
    if [ "$RUN_TESTS" = true ]; then
        echo "🧪 Running API tests..."
        sleep 5  # Extra wait for model loading
        python test_api.py
    fi
    
    echo "📋 Container logs:"
    docker-compose logs --tail 20
else
    echo "❌ Container failed to start!"
    echo "Check logs with: docker-compose logs"
    exit 1
fi

echo ""
echo "🎉 DriaClaude2 is ready to use!"
echo "Stop with: docker-compose down"