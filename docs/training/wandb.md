# Weights & Biases Integration

## Overview

B2B integrates with [Weights & Biases](https://wandb.ai/) (W&B) for
experiment tracking.  All training scripts and baseline rollouts automatically
log metrics, configurations, and time-series plots when W&B is enabled.

## Configuration

W&B is configured via `configs/wandb/default.yaml`:

```yaml title="configs/wandb/default.yaml"
enabled: true
project: "building2building"
entity: "pierre-luc-bacon-mila-org"
tags: []
```

| Parameter | Description |
|---|---|
| `enabled` | Enable or disable W&B logging |
| `project` | W&B project name |
| `entity` | W&B team or user entity |
| `tags` | Additional user-defined tags |

### Disabling W&B

```bash
python scripts/baselines.py wandb.enabled=false
python scripts/train_single_zone_houses.py wandb.enabled=false
```

## Automatic Tags

The logging system (`b2b.baselines.wandb_utils`) automatically derives tags
from the run configuration for easier filtering:

| Tag format | Source |
|---|---|
| `algo:<algorithm>` | `policy.algorithm` (e.g. `algo:ppo`) |
| `policy:<type>` | `policy.type` (e.g. `policy:unitary_g36`) |
| `split:<split>` | `benchmark.split` (e.g. `split:train`) |
| `building:<type>` | `bldg.bldg.building_type` (e.g. `building:OfficeSmall`) |

These are merged with any user-defined tags from the config.

## Run Naming

W&B run names are auto-generated with a descriptive format:

```
<building_type>_id<building_id>_<algorithm>_<6-char-uuid>
```

For example: `OfficeSmall_id42_unitary_g36_a3f2c1`

## Logged Config

The following configuration fields are logged as W&B run config columns:

| Column | Source |
|---|---|
| `policy/type` | Policy type |
| `policy/algorithm` | RL algorithm name |
| `reward/reward_type` | Reward function class |
| `reward/energy_weight` | Energy weight in reward |
| `task/run_period` | Simulation period |
| `task/target_temperature_mode` | Temperature target mode |
| `building_type` | Building archetype |
| `selection/split` | Train/test split |
| `env/max_steps` | Max episode length |

## Baseline Rollout Metrics

When running baseline rollouts (`scripts/baselines.py`), the following are
logged to W&B:

### Time-Series Plots

- **Temperature** — zone air temperatures (up to 10 zones) and outdoor
  temperature, plotted against global step.
- **Actions** — actuator outputs (fan mass flow rates, temperature setpoints,
  schedule values), up to 12 series.
- **Energy** — electricity and gas consumption (Wh/m²).
- **Reward** — per-step reward signal.

All time-series are downsampled to 2,000 points for chart readability.

### Summary Statistics

| Metric | Description |
|---|---|
| `rollout/mean_reward` | Mean per-step reward |
| `rollout/episode_return_mean` | Mean total episode return |
| `building/type` | Building type string |
| `building/id` | Building ID (if available) |
| `building/area_m2` | Building floor area (m²) |
| `building/controlled_zones` | List of controlled zone names |

## Training Metrics

During SB3 training, standard SB3 logging is used alongside W&B.  The
`ResampleBuildingOnResetWrapper` (used in multi-task training) additionally
tracks:

- **Episode reward** per building
- **Episode length** per building
- **Building index** at each reset

## Best Practices

1. **Use tags for filtering** — add experiment-specific tags in the config to
   group related runs.
2. **Compare baselines and RL** — log both rule-based baselines and RL runs to
   the same project for side-by-side comparison.
3. **Disable for debugging** — set `wandb.enabled=false` during local
   debugging to avoid creating noisy runs.
4. **Custom entity** — override `wandb.entity` for personal projects:
   ```bash
   python scripts/baselines.py wandb.entity=my-username
   ```
