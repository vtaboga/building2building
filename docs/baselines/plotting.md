# Plotting

Generate paper figures from experiment result CSVs.

## Available Scripts

| Script | Description | Input |
|---|---|---|
| `plot_ppo_specialist.py` | PPO specialist vs baseline per building | `results_ppo.csv` + `baseline_returns.csv` |
| `plot_dynamics_adaptation.py` | Specialist vs baseline vs parameterized | Per-approach result CSVs |
| `plot_cross_domain.py` | Cross-domain transfer results | `results_cross_domain.csv` |
| `plot_trajectory.py` | Per-episode zone temperature / actuator time-series | `.npz` trajectory (from `run_reactive_control` with `save_trajectories=true`) |
| `plot_chs_results.py` | CHS PPO tuning re-evaluation summaries | `reeval_*.yaml` (from `tune_ppo.py` with `reeval=true`) |

## Usage

### PPO Specialist

```bash
python -m baselines.plotting.plot_ppo_specialist \
    --ppo-csv results_ppo.csv \
    --baseline-csv baseline_returns.csv \
    --task task_const_e0
```

`--task` selects which task preset's rows to plot.

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

### Trajectories

```bash
python -m baselines.plotting.plot_trajectory \
    --trajectory trajectories/OfficeSmall_OfficeSmall-0001_task_const_e0_run0.npz \
    --output figures/trajectory_officesmall
```

### CHS Tuning Results

```bash
python -m baselines.plotting.plot_chs_results \
    --reeval-dir outputs/tune_ppo/.../reeval \
    --output-dir plots/chs
```

## Shared Style

All plotting scripts use `baselines/plotting/common.py` for consistent
matplotlib styling, color palettes, and CSV loading utilities.
