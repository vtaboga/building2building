#!/bin/bash
#SBATCH --job-name=tune_g36
#SBATCH --output=logs/tune_g36_%A_%a.out
#SBATCH --error=logs/tune_g36_%A_%a.err
#SBATCH --time=14:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-2

# Load conda
source .venv/bin/activate   
export PYTHONPATH=.

BUILDING_TYPES=("OfficeSmall" "RetailStandalone" "RestaurantFastFood")
BUILDING_TYPE=${BUILDING_TYPES[$SLURM_ARRAY_TASK_ID]}

echo "Running building type: ${BUILDING_TYPE}"

python scripts/tune_g36.py \
    --building-type ${BUILDING_TYPE} \
    --n-trials 200 \
    --run-period full_year \
    --n-buildings 20 \
    --output-dir outputs/tune_g36/${BUILDING_TYPE}
