#!/bin/bash
#SBATCH --job-name=per_bldg_ppo
#SBATCH --array=0-10
#SBATCH --output=logs/per_building_ppo_test_%a.out
#SBATCH --error=logs/per_building_ppo_test_%a.err
#SBATCH --time=12:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --partition=long-cpu

# Per-building PPO baseline for the adaptive dynamics benchmark.
#
# Trains a separate PPO agent for each of the 100 test-split buildings.
# Each SLURM array task handles a single building; the task ID is the
# split_index (0-99).
#
# No building parameter augmentation, no building resampling — each agent
# is specialised to a single building.

# Load modules
module load python/3.10

# Activate conda environment
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate building2building

# Set environment variables
export B2B_REPO_ROOT=$PWD
export PYTHONPATH=$PWD:$PYTHONPATH

# Create logs directory
mkdir -p logs

echo "=========================================="
echo "Per-Building PPO | test split | building ${SLURM_ARRAY_TASK_ID}"
echo "=========================================="

# Run training for this building
# Pass SLURM_ARRAY_JOB_ID so all runs from this batch can be grouped in wandb
python -m scripts.per_building_adaptive_dynamics_main \
    split=test \
    split_index="${SLURM_ARRAY_TASK_ID}" \
    seed=42 \
    training.total_timesteps=4000000 \
    wandb.project=building2building \
    slurm_array_job_id="${SLURM_ARRAY_JOB_ID:-none}"

echo "Training complete for building ${SLURM_ARRAY_TASK_ID}!"

