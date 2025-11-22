#!/bin/bash

# HackerNews Virality Model - Training and Deployment Script
# This script automates the complete pipeline:
# 1. Fetches HackerNews data
# 2. Trains the CatBoost model
# 3. Builds and starts a Docker container for inference

set -e  # Exit on error

# ============================================================================
# Configuration
# ============================================================================
HN_PARSER_REPO="${HN_PARSER_REPO:-https://github.com/artbred/hn_parser.git}"
STORIES_COUNT="${1:-10000}"
MIN_SCORE="${2:-10}"
MIN_DESCENDANTS="${3:-10}"
OUTPUT_DIR="${4:-data}"
CONCURRENT="${5:-1000}"
RPS="${6:-500}"
DOCKER_IMAGE_NAME="${DOCKER_IMAGE_NAME:-hn-virality-predictor}"
DOCKER_PORT="${DOCKER_PORT:-5000}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;36m'
NC='\033[0m' # No Color

# ============================================================================
# Helper Functions
# ============================================================================
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "\n${BLUE}======================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}======================================${NC}\n"
}

# ============================================================================
# Step 1: Fetch HackerNews Data
# ============================================================================
fetch_hn_data() {
    log_step "STEP 1: Fetching HackerNews Data"
    
    log_info "Configuration:"
    echo "  - Stories to fetch: $STORIES_COUNT"
    echo "  - Minimum score: $MIN_SCORE"
    echo "  - Minimum descendants: $MIN_DESCENDANTS"
    echo "  - Output directory: $OUTPUT_DIR"
    echo "  - Concurrent requests: $CONCURRENT"
    echo "  - Requests per second: $RPS"
    echo ""

    # Check Go installation
    log_info "Checking Go installation..."
    if ! command -v go &> /dev/null; then
        log_error "Go is not installed. Please install Go first."
        echo "Visit: https://golang.org/doc/install"
        exit 1
    fi
    GO_VERSION=$(go version | awk '{print $3}')
    log_info "Go installed: $GO_VERSION"

    # Clone hn_parser repository
    TEMP_DIR=$(mktemp -d)
    trap "rm -rf $TEMP_DIR" EXIT

    log_info "Cloning hn_parser repository..."
    git clone --quiet "$HN_PARSER_REPO" "$TEMP_DIR/hn_parser"
    cd "$TEMP_DIR/hn_parser"

    # Build hn_parser
    log_info "Building hn_parser..."
    go mod download
    go build -o hn_parser main.go

    if [ ! -f "hn_parser" ]; then
        log_error "Build failed: hn_parser binary not created"
        exit 1
    fi
    log_info "Build successful"

    # Create output directory
    mkdir -p "$OLDPWD/$OUTPUT_DIR"
    OUTPUT_FILE="$OLDPWD/$OUTPUT_DIR/hn_posts_$(date +%Y%m%d_%H%M%S).jsonl"

    # Fetch data
    log_info "Fetching HackerNews data..."
    echo "Output file: $OUTPUT_FILE"

    # Build command arguments
    ARGS="-output $OUTPUT_FILE"
    ARGS="$ARGS -concurrent $CONCURRENT"
    ARGS="$ARGS -rps $RPS"
    ARGS="$ARGS -min-score $MIN_SCORE"
    ARGS="$ARGS -min-descendants $MIN_DESCENDANTS"

    if [ "$STORIES_COUNT" != "0" ]; then
        ARGS="$ARGS -stories $STORIES_COUNT"
        echo "Fetching $STORIES_COUNT stories..."
    else
        echo "Performing incremental update..."
        LATEST_FILE=$(find "$OLDPWD/$OUTPUT_DIR" -name "*.jsonl" -type f | sort -r | head -1)
        if [ -n "$LATEST_FILE" ]; then
            ARGS="$ARGS -update-from-file $LATEST_FILE"
            echo "Updating from: $LATEST_FILE"
        fi
    fi

    # Run hn_parser
    if ./hn_parser $ARGS; then
        log_info "Data fetching completed"
    else
        log_error "Data fetching failed"
        exit 1
    fi

    # Validate output
    cd "$OLDPWD"
    if [ ! -f "$OUTPUT_FILE" ]; then
        log_error "Error: Output file not created"
        exit 1
    fi

    # Count stories and show statistics
    STORY_COUNT=$(wc -l < "$OUTPUT_FILE")
    FILE_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)

    echo ""
    log_info "Fetching Statistics:"
    echo "  - Stories fetched: $STORY_COUNT"
    echo "  - File size: $FILE_SIZE"
    echo "  - Output file: $OUTPUT_FILE"

    # Sample first few stories for validation
    echo ""
    echo "Sample stories (first 3):"
    head -3 "$OUTPUT_FILE" | while IFS= read -r line; do
        if command -v jq &> /dev/null; then
            echo "$line" | jq -r '"\(.title) (score: \(.score), by: \(.by))"' 2>/dev/null || echo "$line"
        else
            echo "$line"
        fi
    done

    echo ""
    log_info "Data fetching complete!"
    
    # Export the data file path for next steps
    export FETCHED_DATA_FILE="$OUTPUT_FILE"
}

# ============================================================================
# Step 2: Train CatBoost Model
# ============================================================================
train_model() {
    log_step "STEP 2: Training CatBoost Model"
    
    # Find the data file
    if [ -z "$FETCHED_DATA_FILE" ]; then
        FETCHED_DATA_FILE=$(find "$OUTPUT_DIR" -name "*.jsonl" -type f | sort -r | head -1)
    fi
    
    if [ ! -f "$FETCHED_DATA_FILE" ]; then
        log_error "No data file found for training"
        exit 1
    fi
    
    log_info "Using data file: $FETCHED_DATA_FILE"
    
    # Check Python installation
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is not installed"
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 --version)
    log_info "Python version: $PYTHON_VERSION"
    
    # Install/check UV
    log_info "Checking UV package manager..."
    if ! command -v uv &> /dev/null; then
        log_warn "UV not found, installing..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
    fi
    
    # Setup Python environment
    log_info "Setting up Python environment with UV..."
    uv venv
    source .venv/bin/activate
    
    log_info "Installing dependencies..."
    uv pip install -e .
    
    # Create reports directory
    mkdir -p reports
    
    # Convert JSONL to CSV if needed
    log_info "Preparing data for training..."
    TRAINING_DATA_CSV="$OUTPUT_DIR/hn_posts.csv"
    
    python3 scripts/convert_jsonl_to_csv.py "$FETCHED_DATA_FILE" "$TRAINING_DATA_CSV"

    
    log_info "Training the model..."
    echo "Configuration:"
    echo "  - Data file: $TRAINING_DATA_CSV"
    echo "  - Output directory: reports/"
    echo ""
    
    # Train the model
    cd src
    python3 train.py \
        --data-path "../$TRAINING_DATA_CSV" \
        --reports-dir ../reports \
        --train-fraction 0.85
    cd ..
    
    # Check if model was created
    if [ ! -f "reports/catboost_model.cbm" ]; then
        log_error "Model file not created"
        exit 1
    fi
    
    MODEL_SIZE=$(du -h reports/catboost_model.cbm | cut -f1)
    log_info "Model trained successfully (size: $MODEL_SIZE)"
    
    # Show metrics if available
    if [ -f "reports/metrics.json" ]; then
        echo ""
        log_info "Training Metrics:"
        if command -v jq &> /dev/null; then
            cat reports/metrics.json | jq '.'
        else
            cat reports/metrics.json
        fi
    fi
}

# ============================================================================
# Step 3: Build Docker Image with Cog
# ============================================================================
build_docker_image() {
    log_step "STEP 3: Building Docker Image with Cog"
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    log_info "Docker version: $(docker --version)"
    
    # Check if Cog is installed
    if ! command -v cog &> /dev/null; then
        log_warn "Cog not found, installing..."
        sudo curl -o /usr/local/bin/cog -L https://github.com/replicate/cog/releases/latest/download/cog_$(uname -s)_$(uname -m)
        sudo chmod +x /usr/local/bin/cog
        log_info "Cog installed successfully"
    fi
    
    COG_VERSION=$(cog --version 2>&1)
    log_info "Cog version: $COG_VERSION"
    
    # Clean up any running containers with this name
    log_info "Cleaning up old containers and images..."
    if docker ps -a --format '{{.Names}}' | grep -q "^${DOCKER_IMAGE_NAME}$"; then
        log_warn "Stopping and removing existing container: $DOCKER_IMAGE_NAME"
        docker stop "$DOCKER_IMAGE_NAME" 2>/dev/null || true
        docker rm "$DOCKER_IMAGE_NAME" 2>/dev/null || true
    fi
    
    # Remove old images to save space
    OLD_IMAGES=$(docker images "${DOCKER_IMAGE_NAME}" -q)
    if [ -n "$OLD_IMAGES" ]; then
        log_warn "Removing old images to save space..."
        docker rmi $OLD_IMAGES 2>/dev/null || true
    fi
    
    # Clean up dangling images
    DANGLING=$(docker images -f "dangling=true" -q)
    if [ -n "$DANGLING" ]; then
        log_info "Removing dangling images..."
        docker rmi $DANGLING 2>/dev/null || true
    fi
    
    # Verify model exists
    if [ ! -f "reports/catboost_model.cbm" ]; then
        log_error "Model file not found. Please train the model first."
        exit 1
    fi
    
    # Verify feature stats exist
    if [ ! -f "reports/feature_stats.json" ]; then
        log_error "Feature stats not found. Please train the model first."
        exit 1
    fi
    
    log_info "Building Docker image with Cog: $DOCKER_IMAGE_NAME"
    echo "This may take a few minutes..."
    
    # Build the Docker image using Cog
    cog build -t "$DOCKER_IMAGE_NAME:latest"
    
    if [ $? -eq 0 ]; then
        log_info "Docker image built successfully with Cog"
        
        # Show image details
        IMAGE_SIZE=$(docker images "$DOCKER_IMAGE_NAME:latest" --format "{{.Size}}")
        echo "  - Image: $DOCKER_IMAGE_NAME:latest"
        echo "  - Size: $IMAGE_SIZE"
        
        # Show what's consuming space
        log_info "Image layers breakdown:"
        docker history "$DOCKER_IMAGE_NAME:latest" --human --format "table {{.Size}}\t{{.CreatedBy}}" | head -10
    else
        log_error "Cog build failed"
        exit 1
    fi
}

# ============================================================================
# Step 4: Start Docker Container
# ============================================================================
start_docker_container() {
    log_step "STEP 4: Starting Docker Container"
    
    # Stop existing container if running
    log_info "Checking for existing container..."
    if docker ps -a --format '{{.Names}}' | grep -q "^${DOCKER_IMAGE_NAME}$"; then
        log_warn "Stopping and removing existing container..."
        docker stop "$DOCKER_IMAGE_NAME" 2>/dev/null || true
        docker rm "$DOCKER_IMAGE_NAME" 2>/dev/null || true
    fi
    
    log_info "Starting new container with Cog..."
    # Run the Cog-built container
    # Cog containers expose port 5000 by default for predictions
    docker run -d \
        --name "$DOCKER_IMAGE_NAME" \
        -p "${DOCKER_PORT}:5000" \
        "$DOCKER_IMAGE_NAME:latest"
    
    if [ $? -eq 0 ]; then
        log_info "Container started successfully"
        echo "  - Container name: $DOCKER_IMAGE_NAME"
        echo "  - Port mapping: ${DOCKER_PORT}:5000"
        echo "  - API endpoint: http://localhost:${DOCKER_PORT}/predictions"
        echo ""
        
        # Wait a moment and check if container is running
        sleep 3
        if docker ps --format '{{.Names}}' | grep -q "^${DOCKER_IMAGE_NAME}$"; then
            log_info "Container is running"
            
            # Show logs
            echo ""
            log_info "Initial container logs:"
            docker logs "$DOCKER_IMAGE_NAME" 2>&1 | tail -20
            
            echo ""
            log_info "You can interact with the container using:"
            echo "  - View logs: docker logs -f $DOCKER_IMAGE_NAME"
            echo "  - Execute commands: docker exec -it $DOCKER_IMAGE_NAME /bin/bash"
        else
            log_error "Container stopped unexpectedly"
            echo "Logs:"
            docker logs "$DOCKER_IMAGE_NAME"
            exit 1
        fi
    else
        log_error "Failed to start container"
        exit 1
    fi
}

# ============================================================================
# Main Execution
# ============================================================================
main() {
    echo -e "${GREEN}"
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║   HackerNews Virality Model - Training & Deployment Script   ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    
    # Execute all steps
    fetch_hn_data
    train_model
    build_docker_image
    start_docker_container
    
    # Final summary
    echo ""
    log_step "✅ DEPLOYMENT COMPLETE"
    echo ""
    log_info "Summary:"
    echo "  - Data fetched: $STORY_COUNT stories"
    echo "  - Model trained: reports/catboost_model.cbm"
    echo "  - Docker image: $DOCKER_IMAGE_NAME:latest (built with Cog)"
    echo "  - Container running on: http://localhost:${DOCKER_PORT}"
    echo ""
    log_info "Test the API:"
    echo "  - Predictions: curl -X POST http://localhost:${DOCKER_PORT}/predictions -H 'Content-Type: application/json' -d '{\"input\":{\"title\":\"Test\",\"url\":\"\",\"by\":\"user\",\"time\":1763807356}}'"
    echo ""
    log_info "Container management:"
    echo "  - View logs: docker logs -f $DOCKER_IMAGE_NAME"
    echo "  - Execute commands: docker exec -it $DOCKER_IMAGE_NAME /bin/bash"
    echo "  - Stop container: docker stop $DOCKER_IMAGE_NAME"
    echo ""
}

# Run the script
main
