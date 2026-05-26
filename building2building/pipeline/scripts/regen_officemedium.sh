#!/bin/bash
#SBATCH --job-name=b2b-regen-om
#SBATCH --output=logs/regen_om_%A_%a.out
#SBATCH --error=logs/regen_om_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --partition=long
#SBATCH --array=0-19

# Regenerate the OfficeMedium slice of vtaboga/building2building_dataset
# under the post-M1 pipeline (adds the OA-mixer actuator per VAV loop).
# See TODO.md § Phase M (M2) and notes.md § "OfficeMedium OA-mixer fix".
#
# 1000 OfficeMedium buildings are sharded across 20 array tasks (50
# buildings per task).  Each task writes per-building artefacts into
# its own shard of $SCRATCH/b2b_regen_officemedium/; the last task
# (--shard-index 19) additionally rewrites metadata.parquet and
# copies splits.json.  The shards write to disjoint per-building
# directories, so no inter-task locking is required.
#
# This is the canonical reproduction path on a Slurm cluster.  For
# single-machine reproduction see:
#
#     python -m building2building.pipeline.regen_dataset \
#         --building-type OfficeMedium \
#         --output-dir <staging> \
#         --write-metadata-parquet
#
# Usage:
#     sbatch building2building/pipeline/scripts/regen_officemedium.sh
#
# After all 20 array tasks finish, push to HuggingFace:
#     cd $SCRATCH/b2b_regen_officemedium
#     huggingface-cli upload \
#         vtaboga/building2building_dataset \
#         . . \
#         --repo-type dataset \
#         --revision main

set -euo pipefail

module load python/3.10
source "$HOME/Building2Building/.venv/bin/activate"

cd "$HOME/Building2Building" || exit 1

# EnergyPlus scratch files (when --rerun-discovery is used) go to
# fast node-local storage.
export TMPDIR="$SLURM_TMPDIR"

OUT_DIR="$SCRATCH/b2b_regen_officemedium"
mkdir -p "$OUT_DIR"

SHARD_INDEX=${SLURM_ARRAY_TASK_ID:?Must be run as a SLURM array job}
SHARD_COUNT=20

EXTRA_FLAGS=()
# Only the last shard rewrites the unified metadata.parquet, so other
# shards never race on that file.  Single-machine runs (shard_count=1)
# always include this flag.
if (( SHARD_INDEX == SHARD_COUNT - 1 )); then
    EXTRA_FLAGS+=(--write-metadata-parquet)
fi

echo "=== Shard $SHARD_INDEX / $SHARD_COUNT regenerating OfficeMedium ==="

python -m building2building.pipeline.regen_dataset \
    --building-type OfficeMedium \
    --output-dir "$OUT_DIR" \
    --shard-index "$SHARD_INDEX" \
    --shard-count "$SHARD_COUNT" \
    "${EXTRA_FLAGS[@]}"

echo "=== Shard $SHARD_INDEX done ==="
