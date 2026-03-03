#!/bin/bash
#SBATCH --job-name=ppo_sweep
#SBATCH --output=logs/ppo_sweep_%A_%a.out
#SBATCH --error=logs/ppo_sweep_%A_%a.err
#SBATCH --array=0-95
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

declare -a BUILDING_TYPES=(
    # Warehouse (4 experiments: indices 0-3)
    "Warehouse"  # 0: deadband ew=0.01 constant
    "Warehouse"  # 1: deadband ew=0.01 occupancy
    "Warehouse"  # 2: deadband ew=0.1  constant
    "Warehouse"  # 3: barrier  dT=1 penalty=10
    # Retail (4 experiments: indices 4-7)
    "RetailStandalone"  # 4: deadband ew=0.01 constant
    "RetailStandalone"  # 5: deadband ew=0.01 occupancy
    "RetailStandalone"  # 6: deadband ew=0.1  constant
    "RetailStandalone"  # 7: barrier  dT=1 penalty=10
    # Restaurant (4 experiments: indices 8-11)
    "RestaurantFastFood"  # 8:  deadband ew=0.01 constant
    "RestaurantFastFood"  # 9:  deadband ew=0.01 occupancy
    "RestaurantFastFood"  # 10: deadband ew=0.1  constant
    "RestaurantFastFood"  # 11: barrier  dT=1 penalty=10
)

declare -a REWARD_TYPES=(
    "deadband" "deadband" "deadband" "barrier"
    "deadband" "deadband" "deadband" "barrier"
    "deadband" "deadband" "deadband" "barrier"
)

declare -a ENERGY_WEIGHTS=(
    "0.01" "0.01" "0.1" "1.0"
    "0.01" "0.01" "0.1" "1.0"
    "0.01" "0.01" "0.1" "1.0"
)

declare -a TARGET_MODES=(
    "constant" "occupancy" "constant" "constant"
    "constant" "occupancy" "constant" "constant"
    "constant" "occupancy" "constant" "constant"
)

declare -a TAG_LABELS=(
    "wh_db001_const"  "wh_db001_occ"  "wh_db01_const"  "wh_bar"
    "rt_db001_const"  "rt_db001_occ"  "rt_db01_const"  "rt_bar"
    "rf_db001_const"  "rf_db001_occ"  "rf_db01_const"  "rf_bar"
)

BTYPE=${BUILDING_TYPES[$EXP_IDX]}
REWARD=${REWARD_TYPES[$EXP_IDX]}
EW=${ENERGY_WEIGHTS[$EXP_IDX]}
TARGET=${TARGET_MODES[$EXP_IDX]}
TAG=${TAG_LABELS[$EXP_IDX]}

echo "Experiment ${EXP_IDX} | Building ${BLDG_IDX} | ${BTYPE} | ${REWARD} ew=${EW} target=${TARGET}"

COMMON_ARGS=(
    bldg.building_type=${BTYPE}
    bldg.split=test_small
    bldg.index=${BLDG_IDX}
    reward=${REWARD}
    reward.energy_weight=${EW}
    task.target_temperature_mode=${TARGET}
    "wandb.tags=[test_small, ${TAG}]"
    hydra.run.dir=${SCRATCH}/${TAG}/building_${BLDG_IDX}
)

if [ "${REWARD}" = "barrier" ]; then
    COMMON_ARGS+=(
        reward.dT=1.0
        reward.violation_penalty=10.0
    )
fi

python scripts/train_multizones.py "${COMMON_ARGS[@]}"
