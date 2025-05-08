#!/bin/bash

# Exit on error
set -e

# Get the repository root directory
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Parse arguments to find the scratch directory and results directory
SCRATCH_DIR=""
RESULTS_DIR_ARG=""  # Rename to avoid conflicts
ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --scratch=*) SCRATCH_DIR="${1#*=}"; shift ;;
        --scratch) SCRATCH_DIR="$2"; shift 2 ;;
        --results-dir=*) RESULTS_DIR_ARG="${1#*=}"; shift ;;
        --results-dir) RESULTS_DIR_ARG="$2"; shift 2 ;;
        *) ARGS+=("$1"); shift ;;
    esac
done

# Check if scratch directory was provided
if [ -z "$SCRATCH_DIR" ]; then
    echo "Error: Scratch directory not specified"
    echo "Usage: $0 --scratch=/path/to/scratch/folder --results-dir=/path/to/results <script.py> [args...]"
    exit 1
fi

# Check if results directory was provided
if [ -z "$RESULTS_DIR_ARG" ]; then
    echo "Error: Results directory not specified. Use --results-dir to specify where results should be copied."
    echo "Usage: $0 --scratch=/path/to/scratch/folder --results-dir=/path/to/results <script.py> [args...]"
    exit 1
fi

# Set final results directory
FINAL_RESULTS_DIR="$RESULTS_DIR_ARG"

# Restore positional arguments
set -- "${ARGS[@]}"

# Define container and data paths
REPO_CONTAINER="$REPO_ROOT/container/container.sif"
SCRATCH_CONTAINER="$SCRATCH_DIR/container/container.sif"
SCRATCH_DATA_DIR="$SCRATCH_DIR/data"
SCRATCH_RESULTS_DIR="$SCRATCH_DIR/results"
# Add a directory for wandb
SCRATCH_WANDB_DIR="$SCRATCH_DIR/wandb"

# Check if container exists in repository directory first, then scratch directory
if [ -f "$REPO_CONTAINER" ]; then
    CONTAINER="$REPO_CONTAINER"
    echo "Using container from repository: $CONTAINER"
elif [ -f "$SCRATCH_CONTAINER" ]; then
    CONTAINER="$SCRATCH_CONTAINER"
    echo "Using container from scratch directory: $CONTAINER"
else
    echo "Container not found at $REPO_CONTAINER or $SCRATCH_CONTAINER"
    exit 1
fi

# Create required directories in scratch
mkdir -p "$SCRATCH_DATA_DIR" "$SCRATCH_RESULTS_DIR" "$SCRATCH_WANDB_DIR"

# Determine which command to use
SINGULARITY_CMD="singularity"
if ! command -v singularity &> /dev/null; then
    if command -v apptainer &> /dev/null; then
        SINGULARITY_CMD="apptainer"
    else
        echo "Neither Singularity nor Apptainer found"
        exit 1
    fi
fi

# Check if a script was specified
if [ $# -lt 1 ] || [[ "$1" != *.py ]]; then
    echo "Error: You must provide a Python script to run"
    echo "Usage: $0 --scratch=/path/to/scratch/folder --results-dir=/path/to/results <script.py> [args...]"
    exit 1
fi

# Get the script path
SCRIPT_PATH="$1"
shift


if [[ "$SCRIPT_PATH" != /* ]]; then
    SCRIPT_NAME=$(basename "$SCRIPT_PATH")
else
    SCRIPT_NAME=$(basename "$SCRIPT_PATH")
fi

echo "Running $SCRIPT_NAME in container with scratch directory: $SCRATCH_DIR"

"$SINGULARITY_CMD" exec \
    --bind "$REPO_ROOT:/opt/repository-host" \
    --bind "$SCRATCH_DATA_DIR:/opt/repository/data" \
    --bind "$SCRATCH_RESULTS_DIR:/opt/repository/results" \
    --bind "$SCRATCH_WANDB_DIR:/opt/repository/wandb" \
    --pwd /opt/repository \
    "$CONTAINER" \
    bash -c "
      # Activate the virtual environment and run the script from the host-mounted directory
      source /opt/repository/.venv/bin/activate
      # Set WANDB_DIR environment variable to ensure wandb uses the correct directory
      export WANDB_DIR=/opt/repository/wandb
      python /opt/repository-host/$SCRIPT_PATH $*
    "

# After container execution
if [ -d "$SCRATCH_DIR/results" ] && [ "$(ls -A "$SCRATCH_DIR/results" 2>/dev/null)" ]; then
    # Print variables for debugging
    echo "Debug - Source directory: $SCRATCH_DIR/results"
    echo "Debug - Destination directory: $FINAL_RESULTS_DIR"
    
    # Verify FINAL_RESULTS_DIR is set
    if [ -z "$FINAL_RESULTS_DIR" ]; then
        echo "ERROR: Results directory not provided. Please specify with --results-dir option."
        exit 1
    fi
    
    echo "Copying results from $SCRATCH_DIR/results to $FINAL_RESULTS_DIR"
    mkdir -p "$FINAL_RESULTS_DIR"
    # Copy each subdirectory separately to ensure proper copying
    for dir in "$SCRATCH_DIR"/results/*/; do
        if [ -d "$dir" ]; then
            dir_name=$(basename "$dir")
            mkdir -p "$FINAL_RESULTS_DIR/$dir_name"
            echo "Copying directory $dir_name to $FINAL_RESULTS_DIR/$dir_name"
            cp -r "$dir"/* "$FINAL_RESULTS_DIR/$dir_name/" || echo "Warning: Some files in $dir_name might not have copied"
        fi
    done
else
    echo "No results to copy from $SCRATCH_DIR/results"
fi

echo "Results copied successfully. Job complete."