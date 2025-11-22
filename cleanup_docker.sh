#!/bin/bash

# Docker Cleanup Script
# Removes old containers and images to free up space

set -e

echo "🧹 Docker Cleanup Script"
echo ""

# Stop and remove hn-virality-predictor container if running
echo "1. Checking for hn-virality-predictor container..."
if docker ps -a --format '{{.Names}}' | grep -q "^hn-virality-predictor$"; then
    echo "   Stopping and removing container..."
    docker stop hn-virality-predictor 2>/dev/null || true
    docker rm hn-virality-predictor 2>/dev/null || true
    echo "   ✓ Container removed"
else
    echo "   No container found"
fi

# Remove old hn-virality-predictor images
echo ""
echo "2. Removing old hn-virality-predictor images..."
OLD_IMAGES=$(docker images "hn-virality-predictor" -q)
if [ -n "$OLD_IMAGES" ]; then
    echo "   Found $(echo $OLD_IMAGES | wc -w) image(s)"
    docker rmi $OLD_IMAGES 2>/dev/null || true
    echo "   ✓ Images removed"
else
    echo "   No old images found"
fi

# Remove dangling images
echo ""
echo "3. Removing dangling images..."
DANGLING=$(docker images -f "dangling=true" -q)
if [ -n "$DANGLING" ]; then
    echo "   Found $(echo $DANGLING | wc -w) dangling image(s)"
    docker rmi $DANGLING 2>/dev/null || true
    echo "   ✓ Dangling images removed"
else
    echo "   No dangling images"
fi

# Show current images
echo ""
echo "4. Current Docker images:"
docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}" | grep -E "REPOSITORY|hn-virality|artbred" || echo "   No hn-virality images found"

echo ""
echo "✅ Cleanup complete!"
echo ""
echo "Disk space recovered. You can now run:"
echo "  ./train_and_deploy.sh"
