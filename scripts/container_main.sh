#!/bin/bash

# Exit on error
set -e

# Get the repository root directory
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Define container paths
CONTAINER_PATH="$REPO_ROOT/container"
CONTAINER="$CONTAINER_PATH/container.sif"

# Check if container exists
if [ ! -f "$CONTAINER" ]; then
    echo "❌ Container not found at $CONTAINER"
    echo "Please run setup.sh first to build the container"
    exit 1
fi

# Create required directories if they don't exist
mkdir -p "$REPO_ROOT/data"
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

# Check if a script was specified
if [ $# -lt 1 ] || [[ "$1" != *.py ]]; then
    echo "❌ Error: You must provide a Python script to run"
    echo "Usage: $0 <script.py> [args...]"
    echo "Example: $0 scripts/main.py --config configs/default.json"
    echo ""
    echo "For remote execution with scratch folder, use container_main_remote.sh instead:"
    echo "./scripts/container_main_remote.sh --scratch=/scratch/user123 scripts/main.py --config configs/default.json"
    exit 1
fi

# Get the script path
SCRIPT_PATH="$1"
shift

# If it's a relative path, make it absolute
if [[ "$SCRIPT_PATH" != /* ]]; then
    SCRIPT_PATH="/opt/repository/$SCRIPT_PATH"
fi

# Run the container with appropriate bindings
echo "🚀 Running $(basename "$SCRIPT_PATH") in container..."

"$SINGULARITY_CMD" exec \
    --bind "$REPO_ROOT:/opt/repository" \
    --pwd /opt/repository \
    "$CONTAINER" \
    /opt/repository/.venv/bin/python "$SCRIPT_PATH" "$@"

echo "✅ Container execution completed" 
