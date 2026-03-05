#!/bin/bash
#SBATCH --job-name=eval_g36_occ
#SBATCH --output=logs/eval_g36_occ_%j.out
#SBATCH --error=logs/eval_g36_occ_%j.err
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

source .venv/bin/activate
export PYTHONPATH=.

for BUILDING_TYPE in OfficeSmall RetailStandalone RestaurantFastFood Warehouse; do
    echo "=== Re-evaluating ${BUILDING_TYPE} — deadband_ew001_occ (occupancy-aware metric fix) ==="
    python scripts/eval_g36_all_test.py \
        --building-type "${BUILDING_TYPE}" \
        --run-period full_year \
        --output-dir outputs/eval_g36 \
        --setup deadband_ew001_occ
done
