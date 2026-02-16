#!/bin/bash
#SBATCH --job-name=ppo_baseline_adaptive
#SBATCH --output=logs/ppo_baseline_adaptive_%j.out
#SBATCH --error=logs/ppo_baseline_adaptive_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=48G
#SBATCH --partition=main-cpu
#SBATCH --cpus-per-task=8

# Baseline PPO training on adaptive dynamics benchmark
# Same as parameterized variant but WITHOUT building parameter augmentation.
# The agent has no access to building parameters — it must learn a single
# policy that works across all buildings without knowing which building it is in.
# Buildings are still resampled on each episode reset.

# Load modules
module load python/3.10

# Activate conda environment
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate building2building

# Set environment variables
export B2B_REPO_ROOT=$PWD
export PYTHONPATH=$PWD:$PYTHONPATH

# Create logs directory
mkdir -p logs

# Run training
python -m scripts.baseline_adaptive_dynamics_main \
    seed=42 \
    training.total_timesteps=1000000 \
    training.eval_freq=262144 \
    wandb.project=building2building

echo "Training complete!"

