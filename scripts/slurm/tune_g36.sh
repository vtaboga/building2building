#!/bin/bash
#SBATCH --job-name=tune_g36
#SBATCH --output=logs/tune_g36_%A_%a.out
#SBATCH --error=logs/tune_g36_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-31

source .venv/bin/activate
export PYTHONPATH=.

BUILDING_TYPES=("OfficeSmall" "RetailStandalone" "RestaurantFastFood" "Warehouse")
BUILDING_TYPE=${BUILDING_TYPES[$((SLURM_ARRAY_TASK_ID / 8))]}
CZ=$(( (SLURM_ARRAY_TASK_ID % 8) + 1 ))

echo "=== Tuning ${BUILDING_TYPE} CZ${CZ} ==="

python scripts/tune_g36.py \
    --building-type ${BUILDING_TYPE} \
    --climate-zone ${CZ} \
    --n-trials 300 \
    --run-period full_year \
    --output-dir outputs/tune_g36
