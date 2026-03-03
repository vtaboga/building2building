#!/bin/bash
#SBATCH --job-name=ppo_house
#SBATCH --output=logs/ppo_house_%A_%a.out
#SBATCH --error=logs/ppo_house_%A_%a.err
#SBATCH --array=0-31
#SBATCH --time=14:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

source .venv/bin/activate
export PYTHONPATH=.
mkdir -p logs

SCRATCH=/network/scratch/v/vincent.taboga/Building2Building/outputs

NUM_BUILDINGS=8
EXP_IDX=$((SLURM_ARRAY_TASK_ID / NUM_BUILDINGS))
BLDG_IDX=$((SLURM_ARRAY_TASK_ID % NUM_BUILDINGS))

declare -a REWARD_TYPES=(
    "deadband"  # 0: deadband ew=0.01 constant
    "deadband"  # 1: deadband ew=0.01 occupancy
    "deadband"  # 2: deadband ew=0.1  constant
    "barrier"   # 3: barrier  dT=1 penalty=10
)

declare -a ENERGY_WEIGHTS=("0.01" "0.01" "0.1" "1.0")

declare -a TARGET_MODES=("constant" "occupancy" "constant" "constant")

declare -a TAG_LABELS=(
    "sf_db001_const"  "sf_db001_occ"  "sf_db01_const"  "sf_bar"
)

REWARD=${REWARD_TYPES[$EXP_IDX]}
EW=${ENERGY_WEIGHTS[$EXP_IDX]}
TARGET=${TARGET_MODES[$EXP_IDX]}
TAG=${TAG_LABELS[$EXP_IDX]}

echo "Experiment ${EXP_IDX} | Building ${BLDG_IDX} | SingleFamily | ${REWARD} ew=${EW} target=${TARGET}"

COMMON_ARGS=(
    bldg.split=test
    bldg.index=${BLDG_IDX}
    reward=${REWARD}
    reward.energy_weight=${EW}
    task.target_temperature_mode=${TARGET}
    "wandb.tags=[test, ${TAG}]"
    hydra.run.dir=${SCRATCH}/${TAG}/building_${BLDG_IDX}
)

if [ "${REWARD}" = "barrier" ]; then
    COMMON_ARGS+=(
        reward.dT=1.0
        reward.violation_penalty=10.0
    )
fi

python scripts/train_single_zone_houses.py "${COMMON_ARGS[@]}"
