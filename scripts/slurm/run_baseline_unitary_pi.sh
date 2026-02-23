#!/bin/bash
#SBATCH --job-name=baseline_unitary_pi
#SBATCH --output=logs/baseline_unitary_pi_%j.out
#SBATCH --error=logs/baseline_unitary_pi_%j.err
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

# Run rule-based PI control baseline
# Uses configs/baseline.yaml which defaults to unitary_pi policy
python scripts/baselines.py

