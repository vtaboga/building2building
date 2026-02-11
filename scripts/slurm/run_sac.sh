#!/bin/bash
#SBATCH --job-name=sac_training
#SBATCH --output=logs/sac_%j.out
#SBATCH --error=logs/sac_%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
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

# Run SAC training with the online trainer
# Override the default PPO policy with SAC
python scripts/main.py policy=sac

