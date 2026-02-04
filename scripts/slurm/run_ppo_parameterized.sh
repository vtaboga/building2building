#!/bin/bash
#SBATCH --job-name=ppo_parameterized
#SBATCH --output=logs/ppo_parameterized_%j.out
#SBATCH --error=logs/ppo_parameterized_%j.err
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

# Run parameterized PPO training
# This trains a single policy that generalizes across multiple buildings
# by augmenting observations with building parameters
python scripts/parameterized_main.py policy=ppo_parameterized


