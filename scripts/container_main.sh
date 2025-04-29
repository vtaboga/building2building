#!/bin/bash

# Exit on error
set -e

# Get the repository root directory
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Define default container paths
CONTAINER_PATH="$REPO_ROOT/container"
CONTAINER="$CONTAINER_PATH/container.sif"
DATA_PATH="$REPO_ROOT/data"
RESULTS_PATH="$REPO_ROOT/results"

# Parse arguments for scratch folder
SCRATCH_FOLDER=""
REMAINING_ARGS=()

for arg in "$@"; do
    if [[ "$arg" == "--scratch="* ]]; then
        SCRATCH_FOLDER="${arg#*=}"
    else
        REMAINING_ARGS+=("$arg")
    fi
done

# If scratch folder is provided, update paths
if [ -n "$SCRATCH_FOLDER" ]; then
    echo "🔄 Using scratch folder: $SCRATCH_FOLDER"
    CONTAINER_PATH="$SCRATCH_FOLDER/container"
    CONTAINER="$CONTAINER_PATH/container.sif"
    DATA_PATH="$SCRATCH_FOLDER/data"
    RESULTS_PATH="$SCRATCH_FOLDER/results"
    
    # Create scratch directories if they don't exist
    mkdir -p "$CONTAINER_PATH"
    mkdir -p "$DATA_PATH"
    mkdir -p "$RESULTS_PATH"
fi

# Check if container exists
if [ ! -f "$CONTAINER" ]; then
    echo "❌ Container not found at $CONTAINER"
    echo "Please run setup.sh first to build the container"
    exit 1
fi

# Create required directories if they don't exist
mkdir -p "$DATA_PATH"
mkdir -p "$RESULTS_PATH"

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
if [ ${#REMAINING_ARGS[@]} -lt 1 ] || [[ "${REMAINING_ARGS[0]}" != *.py ]]; then
    echo "❌ Error: You must provide a Python script to run"
    echo "Usage: $0 [--scratch=PATH] <script.py> [args...]"
    echo "Example: $0 scripts/main.py --config configs/default.json"
    echo "Example with scratch: $0 --scratch=/scratch/user123 scripts/main.py --config configs/default.json"
    exit 1
fi

# Get the script path
SCRIPT_PATH="${REMAINING_ARGS[0]}"

# If it's a relative path, make it absolute
if [[ "$SCRIPT_PATH" != /* ]]; then
    SCRIPT_PATH="/opt/repository/$SCRIPT_PATH"
fi

# Remove the script argument so remaining args can be passed to the script
REMAINING_ARGS=("${REMAINING_ARGS[@]:1}")

# Run the container with appropriate bindings
echo "🚀 Running $(basename "$SCRIPT_PATH") in container..."

# Prepare bind arguments
if [ -n "$SCRATCH_FOLDER" ]; then
    # When using scratch folder, bind repository without data/results
    # and then bind scratch data/results separately
    BIND_ARGS=(
        "--bind" "$REPO_ROOT:/opt/repository"
        "--bind" "$DATA_PATH:/opt/repository/data"
        "--bind" "$RESULTS_PATH:/opt/repository/results"
    )
else
    # When not using scratch, just bind the whole repository
    BIND_ARGS=("--bind" "$REPO_ROOT:/opt/repository")
fi

# Use the Python interpreter that's built into the container
# instead of looking for a .venv directory
"$SINGULARITY_CMD" exec \
    "${BIND_ARGS[@]}" \
    --pwd /opt/repository \
    "$CONTAINER" \
    python3 "$SCRIPT_PATH" "${REMAINING_ARGS[@]}"

echo "✅ Container execution completed" 
