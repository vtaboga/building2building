#!/bin/bash
#SBATCH --job-name=ppo_offmed
#SBATCH --output=logs/ppo_offmed_%A_%a.out
#SBATCH --error=logs/ppo_offmed_%A_%a.err
#SBATCH --array=0-9
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=main

source .venv/bin/activate
export PYTHONPATH=.
mkdir -p logs

SCRATCH=/network/scratch/v/vincent.taboga/Building2Building/ppo_officemedium_test

python scripts/train_multizones.py \
    multizones.building_type=OfficeMedium \
    multizones.split=test \
    multizones.index=${SLURM_ARRAY_TASK_ID} \
    reward.energy_weight=0.01 \
    hydra.run.dir=${SCRATCH}/building_${SLURM_ARRAY_TASK_ID}
