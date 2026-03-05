#!/bin/bash
#SBATCH --job-name=eval_ppo
#SBATCH --output=logs/eval_ppo_%j.out
#SBATCH --error=logs/eval_ppo_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

source .venv/bin/activate
export PYTHONPATH=.
mkdir -p logs

echo "=== Step 1: Gather trained models ==="
python scripts/gather_trained_models.py

echo ""
echo "=== Step 2: Evaluate PPO models ==="
python scripts/eval_ppo_test_small.py --output-dir outputs/eval_ppo
