#!/bin/bash
#SBATCH --job-name=ppo_rc_tuning
#SBATCH --output=logs/ppo_rc_tuning_%A_%a.out
#SBATCH --error=logs/ppo_rc_tuning_%A_%a.err
#SBATCH --array=0-19
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

source .venv/bin/activate
export PYTHONPATH=.
mkdir -p logs

SCRATCH=/network/scratch/v/vincent.taboga/Building2Building/outputs

# 20 buildings, 1 per job
# building_type, building_id, split, split_index
declare -a BUILDING_TYPES=(
    "RetailStandalone"    # 0:  id=2801
    "RetailStandalone"    # 1:  id=2802
    "RetailStandalone"    # 2:  id=2803
    "RetailStandalone"    # 3:  id=2804
    "RetailStandalone"    # 4:  id=2805
    "RestaurantFastFood"  # 5:  id=3801
    "RestaurantFastFood"  # 6:  id=3802  (test split)
    "RestaurantFastFood"  # 7:  id=3803
    "RestaurantFastFood"  # 8:  id=3804
    "RestaurantFastFood"  # 9:  id=3805
    "OfficeMedium"        # 10: id=4801
    "OfficeMedium"        # 11: id=4802
    "OfficeMedium"        # 12: id=4803
    "OfficeMedium"        # 13: id=4804
    "OfficeMedium"        # 14: id=4805
    "OfficeSmall"         # 15: id=5801
    "OfficeSmall"         # 16: id=5802  (test split)
    "OfficeSmall"         # 17: id=5803
    "OfficeSmall"         # 18: id=5804
    "OfficeSmall"         # 19: id=5805
)

declare -a SPLITS=(
    "train" "train" "train" "train" "train"
    "train" "test"  "train" "train" "train"
    "train" "train" "train" "train" "train"
    "train" "test"  "train" "train" "train"
)

declare -a INDICES=(
    722 723 724 725 726
    723  77 724 725 726
    721 722 723 724 725
    719  81 720 721 722
)

declare -a BUILDING_IDS=(
    2801 2802 2803 2804 2805
    3801 3802 3803 3804 3805
    4801 4802 4803 4804 4805
    5801 5802 5803 5804 5805
)

IDX=${SLURM_ARRAY_TASK_ID}
BTYPE=${BUILDING_TYPES[$IDX]}
SPLIT=${SPLITS[$IDX]}
BLDG_IDX=${INDICES[$IDX]}
BID=${BUILDING_IDS[$IDX]}

TAG="rc_tuning"

echo "Job ${IDX} | ${BTYPE} id=${BID} | split=${SPLIT} index=${BLDG_IDX}"

python scripts/train_multizones.py \
    bldg.building_type=${BTYPE} \
    bldg.split=${SPLIT} \
    bldg.index=${BLDG_IDX} \
    reward=deadband \
    reward.dT=1.0 \
    reward.energy_weight=0.001 \
    task.target_temperature_mode=occupancy \
    task.default_zone_target_temperature.occupied_c=21.0 \
    task.default_zone_target_temperature.unoccupied_c=18.0 \
    "wandb.tags=[${TAG}, ${BTYPE}, bldg_${BID}]" \
    hydra.run.dir=${SCRATCH}/${TAG}/${BTYPE}/building_${BID}
