#!/bin/bash
#SBATCH --job-name=eval_officemedium
#SBATCH --output=logs/eval_officemedium_%j.out
#SBATCH --error=logs/eval_officemedium_%j.err
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

source .venv/bin/activate
export PYTHONPATH=.

echo "=== Evaluating OfficeMedium test buildings (per-CZ configs, 4 setups) ==="
python scripts/eval_officemedium_test.py \
    --run-period full_year \
    --output-dir outputs/eval_g36
