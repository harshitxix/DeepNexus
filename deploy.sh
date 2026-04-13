#!/bin/bash
# DeepNexus Quick Deployment Script
# This script automates the Docker-based deployment

set -e  # Exit on error

echo "========================================="
echo "DeepNexus Deployment Script"
echo "========================================="

# Configuration
IMAGE_NAME="${1:-deepnexus}"
REGISTRY="${2:-}"
PORT="${3:-8000}"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Docker installation
if ! command -v docker &> /dev/null; then
    echo -e "${YELLOW}Docker is not installed. Please install Docker first.${NC}"
    exit 1
fi

# Step 1: Build Docker image
echo -e "\n${BLUE}Step 1: Building Docker image...${NC}"
docker build -t ${IMAGE_NAME}:latest .
echo -e "${GREEN}✓ Docker image built successfully${NC}"

# Step 2: Test image locally
echo -e "\n${BLUE}Step 2: Testing image locally...${NC}"
docker run --rm -p ${PORT}:8000 --name deepnexus-test ${IMAGE_NAME}:latest &
TEST_PID=$!

sleep 5

if curl -f http://localhost:${PORT}/api/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Health check passed${NC}"
    kill $TEST_PID 2>/dev/null || true
else
    echo -e "${YELLOW}✗ Health check failed${NC}"
    kill $TEST_PID 2>/dev/null || true
    exit 1
fi

# Step 3: Push to registry (optional)
if [ ! -z "$REGISTRY" ]; then
    echo -e "\n${BLUE}Step 3: Pushing to registry...${NC}"
    docker tag ${IMAGE_NAME}:latest ${REGISTRY}/${IMAGE_NAME}:latest
    docker push ${REGISTRY}/${IMAGE_NAME}:latest
    echo -e "${GREEN}✓ Image pushed to registry${NC}"
fi

# Step 4: Display next steps
echo -e "\n${GREEN}=========================================${NC}"
echo -e "${GREEN}Deployment preparation complete!${NC}"
echo -e "${GREEN}=========================================${NC}"
echo -e "\n${BLUE}Next steps:${NC}"
echo "1. Choose your deployment platform:"
echo "   - Docker: docker run -p 8000:8000 ${IMAGE_NAME}:latest"
echo "   - Docker Compose: docker-compose up"
echo "   - Vercel (Frontend) + Railway (Backend)"
echo "   - AWS EC2 with Docker"
echo "   - Render.com"
echo ""
echo "2. Configure environment variables:"
echo "   - Frontend: VITE_API_URL=<backend-url>"
echo "   - Backend: CORS_ORIGINS=<frontend-url>"
echo ""
echo "3. See DEPLOYMENT.md for detailed instructions"
echo ""
echo -e "${YELLOW}Environment Variable Examples:${NC}"
echo "VITE_API_URL=https://api.yourdomain.com"
echo "CORS_ORIGINS=https://yourdomain.com,https://www.yourdomain.com"
