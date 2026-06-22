#!/bin/bash
# NOTE: This is a cluster-specific example (SLURM). The #SBATCH directives below
# reflect one HPC setup; adjust partition, memory, and module/venv paths for your
# environment, or run building2building.pipeline.generate_raw_dataset directly.
#SBATCH --job-name=b2b-gen-raw
#SBATCH --output=logs/gen_raw_%A_%a.out
#SBATCH --error=logs/gen_raw_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=long
#SBATCH --array=0-5

# Stage 1 dataset generator — IDF + LHS → multizones_reference_buildings layout.
# See building2building/pipeline/generate_raw_dataset.py.
#
# 6 array tasks, one per building type:
#   0 Warehouse           id range   1-1000
#   1 HotelSmall          id range 1001-2000
#   2 RetailStandalone    id range 2001-3000
#   3 RestaurantFastFood  id range 3001-4000
#   4 OfficeMedium        id range 4001-5000
#   5 OfficeSmall         id range 5001-6000
#
# The last task (--array-task-id 5) additionally merges the 6 partial
# metadata_*.csv files into metadata.csv.
#
# Usage:
#     sbatch building2building/pipeline/scripts/generate_raw_dataset.sh
#
# After all 6 tasks finish, validate:
#     python -m pytest tests/long/test_generate_raw_dataset_matches_existing.py -m long
#
# To regenerate only a subset of types:
#     sbatch --array=4 building2building/pipeline/scripts/generate_raw_dataset.sh
# (but --merge-metadata will not be triggered; run the merge manually if needed)

set -euo pipefail

module load python/3.10
source "$HOME/Building2Building/.venv/bin/activate"

cd "$HOME/Building2Building" || exit 1

export TMPDIR="${SLURM_TMPDIR:?}"
# Route the store to persistent scratch so it survives across jobs and
# does not fill the 100 GB home quota.
export STORE_PATH="${SCRATCH:?SCRATCH must be set}/b2b"
mkdir -p "$STORE_PATH"

# The Python entry point defines the canonical mapping
#   shard_index -> ALL_BUILDING_TYPES[shard_index]
# (see generate_raw_dataset.py::ALL_BUILDING_TYPES + the
# `type_shard_index = ALL_BUILDING_TYPES.index(bt)` slice in main()).
# Do NOT pass --building-type here: the validation in main() requires
# shard_count == len(building_types), which would collapse to 1 if we
# narrowed the type list and break sharded runs.  Instead, let
# building_types default to ALL_BUILDING_TYPES and let --shard-index
# pick the single type to process this task.
SHARD_COUNT=6
SHARD_INDEX=${SLURM_ARRAY_TASK_ID:?Must be run as a SLURM array job}

OUT_DIR="$SCRATCH/b2b_raw_dataset"
mkdir -p "$OUT_DIR"

EXTRA_FLAGS=()
# Only the last shard merges the partial metadata CSVs.
if (( SHARD_INDEX == SHARD_COUNT - 1 )); then
    EXTRA_FLAGS+=(--merge-metadata)
fi

echo "=== Shard $SHARD_INDEX / $SHARD_COUNT ==="

python -m building2building.pipeline.generate_raw_dataset \
    --output-dir "$OUT_DIR" \
    --shard-index "$SHARD_INDEX" \
    --shard-count "$SHARD_COUNT" \
    --samples-per-type 1000 \
    --seed 42 \
    "${EXTRA_FLAGS[@]}"

echo "=== Shard $SHARD_INDEX done ==="
