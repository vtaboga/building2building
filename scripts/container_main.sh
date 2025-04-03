#!/bin/bash

# Exit on error
set -e

# Get the repository root directory
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Define container paths
CONTAINER="$REPO_ROOT/container/container.sif"

# Check if container exists
if [ ! -f "$CONTAINER" ]; then
    echo "❌ Container not found at $CONTAINER"
    echo "Please run setup.sh first to build the container"
    exit 1
fi

# Create required directories if they don't exist
mkdir -p "$REPO_ROOT/data"
mkdir -p "$REPO_ROOT/logs"
mkdir -p "$REPO_ROOT/results"

# Determine which command to use
SINGULARITY_CMD="singularity"
if ! command -v singularity &> /dev/null; then
    if command -v apptainer &> /dev/null; then
        SINGULARITY_CMD="apptainer"
    else
        echo "❌ Neither Singularity nor Apptainer found"
        exit 1
    fi
fi

# Run the container with appropriate bindings
echo "🚀 Running main.py in container..."
"$SINGULARITY_CMD" exec \
    --bind "$REPO_ROOT:/opt/repository" \
    --bind "$REPO_ROOT/data:/opt/repository/data" \
    --bind "$REPO_ROOT/logs:/opt/repository/logs" \
    --bind "$REPO_ROOT/results:/opt/repository/results" \
    --pwd /opt/repository \
    "$CONTAINER" \
    /opt/repository/.venv/bin/python /opt/repository/scripts/main.py "$@"

echo "✅ Container execution completed" 