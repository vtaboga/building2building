#!/bin/bash
#SBATCH --job-name=tune_g36_test
#SBATCH --output=logs/tune_g36_test_%j.out
#SBATCH --error=logs/tune_g36_test_%j.err
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

source .venv/bin/activate
export PYTHONPATH=.

echo "=== Tuning G36 on first 8 single-zone test buildings ==="

python scripts/tune_g36_test.py \
    --n-buildings 8 \
    --n-trials 100 \
    --run-period full_year \
    --output-dir outputs/tune_g36_test
