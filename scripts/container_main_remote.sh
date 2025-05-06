#!/bin/bash

# Exit on error
set -e

# Get the repository root directory
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Parse arguments to find the scratch directory
SCRATCH_DIR=""
ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --scratch=*) SCRATCH_DIR="${1#*=}"; shift ;;
        --scratch) SCRATCH_DIR="$2"; shift 2 ;;
        *) ARGS+=("$1"); shift ;;
    esac
done

# Check if scratch directory was provided
if [ -z "$SCRATCH_DIR" ]; then
    echo "Error: Scratch directory not specified"
    echo "Usage: $0 --scratch=/path/to/scratch/folder <script.py> [args...]"
    exit 1
fi

# Restore positional arguments
set -- "${ARGS[@]}"

# Define container and data paths
REPO_CONTAINER="$REPO_ROOT/container/container.sif"
SCRATCH_CONTAINER="$SCRATCH_DIR/container/container.sif"
SCRATCH_DATA_DIR="$SCRATCH_DIR/data"
SCRATCH_RESULTS_DIR="$SCRATCH_DIR/results"

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
mkdir -p "$SCRATCH_DATA_DIR" "$SCRATCH_RESULTS_DIR"

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
    echo "Usage: $0 --scratch=/path/to/scratch/folder <script.py> [args...]"
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
    --pwd /opt/repository \
    "$CONTAINER" \
    bash -c "
      # Activate the virtual environment and run the script from the host-mounted directory
      source /opt/repository/.venv/bin/activate
      python /opt/repository-host/$SCRIPT_PATH $*
    "

echo "Container execution completed"