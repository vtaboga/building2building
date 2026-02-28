#!/bin/bash
#SBATCH --job-name=tune_g36
#SBATCH --output=logs/tune_g36_%A_%a.out
#SBATCH --error=logs/tune_g36_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-2

# Load conda
source .venv/bin/activate   
export PYTHONPATH=.

BUILDING_TYPES=("Warehouse" "RetailStandalone" "RestaurantFastFood")
BTYPE=${BUILDING_TYPES[$SLURM_ARRAY_TASK_ID]}

echo "=== Tuning G36 for ${BTYPE} (array task ${SLURM_ARRAY_TASK_ID}) ==="

python scripts/tune_g36.py \
    --building-type "$BTYPE" \
    --n-trials 50 \
    --run-period full_year \
    --n-buildings 5 \
    --output-dir output/tune_g36
