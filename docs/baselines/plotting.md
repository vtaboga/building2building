# Plotting

Generate paper figures from experiment result CSVs.

## Available Scripts

| Script | Description | Input |
|---|---|---|
| `plot_ppo_specialist.py` | PPO specialist vs baseline per building | `results_ppo.csv` + `baseline_returns.csv` |
| `plot_dynamics_adaptation.py` | Specialist vs baseline vs parameterized | Per-approach result CSVs |
| `plot_cross_domain.py` | Cross-domain transfer results | `results_cross_domain.csv` |

## Usage

### PPO Specialist

```bash
python -m baselines.plotting.plot_ppo_specialist \
    --ppo-csv results_ppo.csv \
    --baseline-csv baseline_returns.csv
```

### Dynamics Adaptation

```bash
python -m baselines.plotting.plot_dynamics_adaptation \
    --specialist-csv results_dynamics_specialist.csv \
    --baseline-csv results_dynamics_baseline.csv \
    --parameterized-csv results_dynamics_parameterized.csv
```

### Cross-Domain Transfer

```bash
python -m baselines.plotting.plot_cross_domain \
    --results-csv results_cross_domain.csv
```

## Shared Style

All plotting scripts use `baselines/plotting/common.py` for consistent
matplotlib styling, color palettes, and CSV loading utilities.

!!! warning "Known issue"

    `plot_ppo_specialist.py` expects a `reward_mean` column but `eval_ppo.py`
    outputs `reward`. See [Known Issues](../about/known-issues.md).
