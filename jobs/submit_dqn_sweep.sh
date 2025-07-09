#!/bin/bash
#SBATCH --job-name=dqn_sweep
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/dqn_sweep_%j.out
#SBATCH --error=logs/dqn_sweep_%j.err

# Load required modules
module load singularity

# Set up directories
# Use environment variable if set, otherwise use default
SCRATCH_DIR="${SCRATCH_DIR:-/home/mila/v/vincent.taboga/scratch/Building2Building}"
mkdir -p logs

# Export environment variables for the sweep
export SLURM_JOB_ID=${SLURM_JOB_ID}
export SCRATCH_DIR=${SCRATCH_DIR}

# Enable W&B GPU monitoring
export WANDB_MONITOR_GPU=true

# Validate scratch directory and container exist
if [ ! -d "${SCRATCH_DIR}" ]; then
    echo "Error: Scratch directory does not exist: ${SCRATCH_DIR}"
    echo "Please ensure the scratch directory is set up with the container"
    exit 1
fi

if [ ! -f "${SCRATCH_DIR}/container/container.sif" ]; then
    echo "Error: Container not found at: ${SCRATCH_DIR}/container/container.sif"
    echo "Please ensure the container is built and available in the scratch directory"
    exit 1
fi

echo "Using scratch directory: ${SCRATCH_DIR}"
echo "Container found at: ${SCRATCH_DIR}/container/container.sif"

# Run the sweep with multiple agents
echo "Starting DQN hyperparameter sweep with job ID: ${SLURM_JOB_ID}"
echo "Scratch directory: ${SCRATCH_DIR}"

# Initialize and run the sweep inside the container
# This ensures W&B is properly configured and available
# To test setup first: bash scripts/container_main_remote.sh --scratch=${SCRATCH_DIR} tests/test_wandb_connection.py
bash scripts/container_main_remote.sh \
    --scratch=${SCRATCH_DIR} \
    scripts/run_dqn_sweep.py \
    --agents=4 \
    --count=5

echo "Sweep completed!" 