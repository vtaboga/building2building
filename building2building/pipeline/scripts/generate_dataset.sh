#!/bin/bash
# NOTE: This is a cluster-specific example (SLURM). The #SBATCH directives below
# reflect one HPC setup; adjust partition, memory, and module/venv paths for your
# environment, or run building2building.pipeline.generate_dataset directly.
#SBATCH --job-name=b2b-gen-dataset
#SBATCH --output=logs/gen_dataset_%A_%a.out
#SBATCH --error=logs/gen_dataset_%A_%a.err
#SBATCH --time=04:00:00
# 16G empirically required: 8G OOM-killed shards 18,19 after ~46/50 buildings
# (slow per-building growth in the control-derivation/realize step).
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --partition=long
#SBATCH --array=0-19

# Stage 2 dataset generator — controlled epJSON pipeline for one building type.
# See building2building/pipeline/generate_dataset.py.
#
# Usage (OfficeMedium, 20 shards × 50 buildings each):
#     sbatch building2building/pipeline/scripts/generate_dataset.sh \
#         --export=BUILDING_TYPE=OfficeMedium
#
# Or set BUILDING_TYPE in this file before submitting.
#
# After all 20 shards finish, push to HuggingFace per REPRODUCING.md
# § "Dataset regeneration".

set -euo pipefail

module load python/3.10
source "$HOME/Building2Building/.venv/bin/activate"

cd "$HOME/Building2Building" || exit 1

export TMPDIR="${SLURM_TMPDIR:?}"
# Route the store to persistent scratch so it survives across jobs and
# does not fill the 100 GB home quota.
export STORE_PATH="${SCRATCH:?SCRATCH must be set}/b2b"
mkdir -p "$STORE_PATH"

# Set via --export=BUILDING_TYPE=... on the sbatch command line, or override here.
BUILDING_TYPE="${BUILDING_TYPE:-OfficeMedium}"

OUT_DIR="$SCRATCH/b2b_gen_dataset_${BUILDING_TYPE}"
mkdir -p "$OUT_DIR"

SHARD_INDEX=${SLURM_ARRAY_TASK_ID:?Must be run as a SLURM array job}
SHARD_COUNT=20

EXTRA_FLAGS=()
# Only the last shard rewrites the unified metadata.parquet and copies splits.json.
if (( SHARD_INDEX == SHARD_COUNT - 1 )); then
    EXTRA_FLAGS+=(--write-metadata-parquet)
fi

echo "=== Shard $SHARD_INDEX / $SHARD_COUNT  building-type=$BUILDING_TYPE ==="

python -m building2building.pipeline.generate_dataset \
    --building-type "$BUILDING_TYPE" \
    --output-dir "$OUT_DIR" \
    --shard-index "$SHARD_INDEX" \
    --shard-count "$SHARD_COUNT" \
    "${EXTRA_FLAGS[@]}"

echo "=== Shard $SHARD_INDEX done ==="
