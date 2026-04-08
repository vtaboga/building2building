# Baselines

Baseline controllers and training scripts live in the top-level `baselines/`
directory (outside the `building2building` package). They are standalone scripts
that use the package API and are not importable as `building2building.baselines`.

See the [Baselines Overview](../baselines/overview.md) for usage instructions.

## Directory Structure

| Path | Description |
|---|---|
| `baselines/controllers/` | Rule-based controllers (`UnitaryHvacPolicy`, `AirLoopPolicy`) |
| `baselines/models/` | Neural-network policy architectures (`AmorpheusPolicy`) |
| `baselines/utils/` | Shared training / evaluation helpers |
| `baselines/train_*.py` | Training entry points (PPO, cross-domain, dynamics) |
| `baselines/eval_*.py` | Evaluation scripts |
| `baselines/tune_controller.py` | Optuna hyperparameter tuning for rule-based controllers |
| `baselines/run_rule_based.py` | Run a rule-based controller and generate `baseline_returns.csv` |
| `baselines/plotting/` | Matplotlib figure generation scripts |
| `baselines/configs/` | Hydra configuration (experiments, policies, rewards, training) |
