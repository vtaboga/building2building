#!/bin/bash
#SBATCH --job-name=tune_ashrae_al
#SBATCH --output=logs/tune_ashrae_air_loop_%A_%a.out
#SBATCH --error=logs/tune_ashrae_air_loop_%A_%a.err
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-7

source .venv/bin/activate
export PYTHONPATH=.

CZ=$(( SLURM_ARRAY_TASK_ID + 1 ))

echo "=== Tuning ASHRAE AirLoop on OfficeMedium CZ${CZ} ==="

python scripts/tune_ashrae_air_loop.py \
    --climate-zone ${CZ} \
    --n-trials 150 \
    --run-period full_year \
    --output-dir outputs/tune_ashrae_air_loop
