#!/bin/bash
#SBATCH --job-name=rulebased_adaptive
#SBATCH --output=logs/rulebased_adaptive_%j.out
#SBATCH --error=logs/rulebased_adaptive_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# Rule-based controller evaluation on adaptive dynamics benchmark
# Runs the unitary_g36 controller on all buildings in both train and test splits.
# Override the policy with: --export=POLICY=ashrae_air_loop

# Load modules
module load python/3.10
module load cuda/11.8

# Activate conda environment
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate building2building

# Set environment variables
export B2B_REPO_ROOT=$PWD
export PYTHONPATH=$PWD:$PYTHONPATH

# Create logs directory
mkdir -p logs

# Allow overriding the policy via environment variable (default: unitary_g36)
POLICY=${POLICY:-unitary_g36}

echo "Running rule-based controller (${POLICY}) on adaptive dynamics benchmark"

# Evaluate on train split (900 buildings)
echo "=== Evaluating on train split ==="
python -m scripts.bm_adaptive_dynamics \
    policy=${POLICY} \
    benchmark.split=train \
    benchmark.start=0 \
    benchmark.limit=0 \
    wandb.project=building2building

# Evaluate on test split (100 buildings)
echo "=== Evaluating on test split ==="
python -m scripts.bm_adaptive_dynamics \
    policy=${POLICY} \
    benchmark.split=test \
    benchmark.start=0 \
    benchmark.limit=0 \
    wandb.project=building2building

echo "Evaluation complete!"

