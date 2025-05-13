#!/bin/bash
#SBATCH --job-name=train_dqn
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Exit on error
set -e

# --- PATHS ---
# Get project root directory - handle both direct execution and SLURM contexts
if [[ -n "$SLURM_JOB_ID" ]]; then
    echo "Running as SLURM job ID: $SLURM_JOB_ID"
    REPO_ROOT=$(pwd)
    if [[ "$REPO_ROOT" == */jobs ]]; then
        REPO_ROOT=$(dirname "$REPO_ROOT")
    fi
    echo "Repository root determined to be: $REPO_ROOT"
else
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    echo "Running directly, repository root: $REPO_ROOT"
fi

# Load environment variables if .env exists
if [ -f "$REPO_ROOT/.env" ]; then
    echo "Loading environment variables from $REPO_ROOT/.env"
    source "$REPO_ROOT/.env"
else
    echo "Error: .env file not found at $REPO_ROOT/.env"
    exit 1
fi

# Use environment variables if set, otherwise exit
: "${CONTAINER_PATH:?CONTAINER_PATH must be set in .env}"
: "${RESULTS_DIR:?RESULTS_DIR must be set in .env}"
: "${WANDB_DIR:?WANDB_DIR must be set in .env}"
: "${DATA_ROOT:?DATA_ROOT must be set in .env}"

# --- Parse arguments for the Python script ---
PY_ARGS=()
SEEDS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --seed)
            IFS=',' read -ra SEEDS <<< "$2"
            shift 2
            ;;
        *)
            PY_ARGS+=("$1")
            if [[ "$1" =~ ^- ]]; then
                if [[ -n "$2" && ! "$2" =~ ^- ]]; then
                    PY_ARGS+=("$2")
                    shift
                fi
            fi
            shift
            ;;
    esac
done

if [ ${#SEEDS[@]} -eq 0 ]; then
    SEEDS=(1)
fi

mkdir -p logs

# Use SLURM_TMPDIR for all temp data
SCRATCH_DIR="${SLURM_TMPDIR:-/tmp}/Building2Building"
mkdir -p "$SCRATCH_DIR"/{container,data/processed_buildings,data/weather,wandb}

# Extract required arguments for file copying
STATE=""
COUNTY=""
BUILDING_ID=""
WEATHER=""
for ((i=0; i<${#PY_ARGS[@]}; i++)); do
    case "${PY_ARGS[$i]}" in
        -s|--state) STATE="${PY_ARGS[$((i+1))]}";;
        -c|--county) COUNTY="${PY_ARGS[$((i+1))]}";;
        -b|--building-id) BUILDING_ID="${PY_ARGS[$((i+1))]}";;
        --weather) WEATHER="${PY_ARGS[$((i+1))]}";;
    esac
done

if [ -z "$STATE" ] || [ -z "$COUNTY" ] || [ -z "$BUILDING_ID" ] || [ -z "$WEATHER" ]; then
    echo "Error: --state, --county, --building-id, and --weather must be specified."
    exit 1
fi

# Copy container
if [ -f "$CONTAINER_PATH" ]; then
    cp "$CONTAINER_PATH" "$SCRATCH_DIR/container/container.sif"
else
    echo "Error: Container not found at $CONTAINER_PATH"
    exit 1
fi

# Copy only the required building and weather files
mkdir -p "$SCRATCH_DIR/data/processed_buildings/$STATE/$COUNTY"
cp "$DATA_ROOT/processed_buildings/$STATE/$COUNTY/$BUILDING_ID.json" "$SCRATCH_DIR/data/processed_buildings/$STATE/$COUNTY/"
cp "$DATA_ROOT/processed_buildings/$STATE/$COUNTY/$BUILDING_ID.epJSON" "$SCRATCH_DIR/data/processed_buildings/$STATE/$COUNTY/"
mkdir -p "$SCRATCH_DIR/data/weather"
cp "$DATA_ROOT/weather/$WEATHER" "$SCRATCH_DIR/data/weather/"

mkdir -p "$RESULTS_DIR" "$WANDB_DIR"

# --- TRAINING LOOP ---
for SEED in "${SEEDS[@]}"; do
    echo "========================================================"
    echo "Starting training with seed: $SEED"
    echo "========================================================"

    # Remove any previous --seed argument and its value
    ARGS=()
    skip_next=0
    for ((i=0; i<${#PY_ARGS[@]}; i++)); do
        if (( skip_next )); then
            skip_next=0
            continue
        fi
        if [[ "${PY_ARGS[$i]}" == "--seed" ]]; then
            skip_next=1
            continue
        fi
        ARGS+=("${PY_ARGS[$i]}")
    done
    ARGS+=("--seed" "$SEED")

    # Call the container script (results will be written directly to $RESULTS_DIR)
    "$REPO_ROOT/scripts/container_main_remote.sh" \
        --scratch="$SCRATCH_DIR" \
        scripts/train_dqn.py "${ARGS[@]}"

    # Copy wandb data if exists
    if [ -d "$SCRATCH_DIR/wandb" ] && [ "$(ls -A "$SCRATCH_DIR/wandb" 2>/dev/null)" ]; then
        echo "Copying wandb data from $SCRATCH_DIR/wandb/ to $WANDB_DIR/"
        mkdir -p "$WANDB_DIR"
        cp -r "$SCRATCH_DIR"/wandb/* "$WANDB_DIR/" 2>/dev/null || echo "Warning: Some wandb files might not have copied"
    else
        echo "No wandb data to copy"
    fi

    echo "Completed training with seed: $SEED"
done

echo "All training jobs completed successfully!" 
