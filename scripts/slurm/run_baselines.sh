#!/bin/bash
#SBATCH --job-name=baselines
#SBATCH --output=logs/baselines_%j.out
#SBATCH --error=logs/baselines_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=main

# Load conda
source $(conda info --base)/etc/profile.d/conda.sh
conda activate building2building

# Create logs directory if it doesn't exist
mkdir -p logs

# Change to project root directory
cd $SLURM_SUBMIT_DIR

# Add project root to PYTHONPATH so algorithms module can be found
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH}"

# Run the script
python scripts/baselines.py

