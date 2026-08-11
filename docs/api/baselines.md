# Baselines

Baseline controllers and training scripts live in the top-level `baselines/`
directory (outside the `building2building` package). They are standalone scripts
that use the package API and are not importable as `building2building.baselines`.

See the [Baselines Overview](../baselines/overview.md) for usage instructions.

## Directory Structure

| Path | Description |
|---|---|
| `baselines/controllers/` | Reactive controllers (`UnitaryHvacPolicy`, `AirLoopPolicy`) |
| `baselines/models/` | Neural-network policy architectures (`AmorpheusPolicy`) |
| `baselines/utils/` | Shared training / evaluation helpers |
| `baselines/train_*.py` | Training entry points (PPO, SAC, cross-domain, dynamics) |
| `baselines/eval_*.py` | Evaluation scripts |
| `baselines/tune_controller.py`, `baselines/tune_ppo.py` | Optuna hyperparameter tuning (reactive controllers, PPO) |
| `baselines/run_reactive_control.py` | Run a reactive controller and generate `baseline_returns.csv` |
| `baselines/compute_reactive_reward_normalizers.py` | Regenerate the packaged `reward_normalizers.yaml` |
| `baselines/plotting/` | Matplotlib figure generation scripts |
| `baselines/configs/` | Hydra configuration (experiments, policies, rewards, training) |
