#!/bin/bash
#SBATCH --job-name=eval_singlezone
#SBATCH --output=logs/eval_singlezone_%j.out
#SBATCH --error=logs/eval_singlezone_%j.err
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

source .venv/bin/activate
export PYTHONPATH=.

echo "=== Evaluating SingleZone test buildings (100 buildings x 4 setups) ==="
python scripts/eval_singlezone_test.py \
    --run-period full_year \
    --output-dir outputs/eval_g36
