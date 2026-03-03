#!/bin/bash
#SBATCH --job-name=eval_g36
#SBATCH --output=logs/eval_g36_%A_%a.out
#SBATCH --error=logs/eval_g36_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-3

source .venv/bin/activate
export PYTHONPATH=.

BUILDING_TYPES=("OfficeSmall" "RetailStandalone" "RestaurantFastFood" "Warehouse")
BUILDING_TYPE=${BUILDING_TYPES[$SLURM_ARRAY_TASK_ID]}

echo "=== Evaluating ${BUILDING_TYPE} ==="

python scripts/eval_g36_all_test.py \
    --building-type ${BUILDING_TYPE} \
    --run-period full_year \
    --output-dir outputs/eval_g36
