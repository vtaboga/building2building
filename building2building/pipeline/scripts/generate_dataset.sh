#!/bin/bash
#SBATCH --job-name=b2b-gen-dataset
#SBATCH --output=logs/gen_dataset_%A_%a.out
#SBATCH --error=logs/gen_dataset_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --partition=long
#SBATCH --array=0-19

# Stage 2 dataset generator — controlled epJSON pipeline for one building type.
# See TODO.md § Phase G (G4) and building2building/pipeline/generate_dataset.py.
#
# Usage (OfficeMedium, 20 shards × 50 buildings each):
#     sbatch building2building/pipeline/scripts/generate_dataset.sh \
#         --export=BUILDING_TYPE=OfficeMedium
#
# Or set BUILDING_TYPE in this file before submitting.
#
# After all 20 shards finish, push to HuggingFace per REPRODUCING.md
# § "Dataset regeneration".
#
# This script replaces the old regen_officemedium.sh (deleted in G4).
# M2's outstanding follow-up is now:
#     sbatch .../generate_dataset.sh (with BUILDING_TYPE=OfficeMedium)

set -euo pipefail

module load python/3.10
source "$HOME/Building2Building/.venv/bin/activate"

cd "$HOME/Building2Building" || exit 1

export TMPDIR="${SLURM_TMPDIR:?}"

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
