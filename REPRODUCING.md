## Generating the datasets

## Processing pipeline

## Reward normalization constants

"${PYTHON_BIN}" -m baselines.compute_random_policy_reward_normalizers \
    --mode rollout \
    --run-periods "${RUN_PERIOD_ARGS[@]}" \
    "${EXTRA_ARGS[@]}" \
    --shard-index "${SHARD_INDEX}" \
    --shard-count "${SHARD_COUNT}" \
    --n-workers   "${N_WORKERS}" \
    --data-dir    "${SCRATCH}/b2b/reward_normalizers/data"

python -m baselines.compute_random_policy_reward_normalizers --mode aggregate \
       --data-dir "$SCRATCH/b2b/reward_normalizers/data"

## Fine tuning RBC

Pay attention: this depends on the reward (and on the energy weight)
Fine tuning was done for task_occ_e0 (no weight on energy) to remove any arbitraty choice
of multi objective trade-off.

python -m baselines.tune_controller experiment=tune_controller \
    building_type="$BUILDING_TYPE" \
    climate_zone="$CLIMATE_ZONE" \
    +reward.task_name=task_occ_e0 \
    base_dir="$SCRATCH" \
    n_building_workers=5 \
    wandb.enabled=false

## Generating baseline results table

baselines/scripts/run_baseline_returns.sh (need updating, still the old tasks definition)

baselines/scripts/merge_baseline_returns.py