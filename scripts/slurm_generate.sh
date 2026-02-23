#!/usr/bin/env bash
#SBATCH --job-name=b2b-generate
#SBATCH --array=0-7
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=logs/generate_%a.out
#SBATCH --error=logs/generate_%a.err
#
# Generate the building dataset on a SLURM cluster.
# Submits 8 parallel array tasks, one per building type.
#
# Usage:
#   mkdir -p logs
#   sbatch scripts/slurm_generate.sh
#
# After all tasks finish, merge the partial metadata files:
#   PYTHONPATH=. python scripts/generate_dataset.py --merge-metadata

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate
export PYTHONPATH=.

python scripts/generate_dataset.py \
    --type-index "${SLURM_ARRAY_TASK_ID}" \
    --output "dataset" \
    --samples-per-type 1000 \
    --seed 42
