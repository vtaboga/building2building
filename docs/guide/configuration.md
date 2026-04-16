# Configuration (Hydra)

The `baselines/` scripts use [Hydra](https://hydra.cc/) for configuration
management. This page describes the Hydra config structure, config groups, and
CLI override patterns.

!!! note

    Hydra configuration is used by the `baselines/` experiment scripts, not by
    the core `building2building` package. The package API uses typed dataclasses
    directly (see [API Reference](../api/config.md)).

---

## Config Structure

All Hydra configs live in `baselines/configs/`:

```
baselines/configs/
├── config.yaml           # Root config with defaults
├── experiment/           # Per-script experiment settings
│   ├── eval_reactive_control.yaml
│   ├── train_ppo.yaml
│   ├── train_dynamics_specialist.yaml
│   ├── train_dynamics_baseline.yaml
│   ├── train_dynamics_parameterized.yaml
│   ├── train_cross_domain.yaml
│   └── tune_controller.yaml
├── policy/               # Algorithm / controller hyperparameters
│   ├── ppo.yaml
│   ├── ppo_parameterized.yaml
│   ├── unitary_hvac.yaml
│   └── air_loop.yaml
├── reward/               # Task reward definitions
│   ├── task1.yaml ... task4.yaml
├── training/
│   └── default.yaml      # Training loop parameters
├── tuned_controllers/    # Optuna-optimized controller configs
│   └── (45 YAML files per building_type x climate_zone)
└── wandb/
    └── default.yaml      # W&B logging settings
```

## Usage Pattern

Every baseline script follows this pattern:

```bash
python -m baselines.<script> experiment=<name> [overrides...]
```

Examples:

```bash
python -m baselines.run_reactive_control experiment=eval_reactive_control
python -m baselines.train_ppo experiment=train_ppo seed=42
python -m baselines.train_dynamics_adaptation \
    experiment=train_dynamics_parameterized difficulty=medium
```

## Key Config Groups

### Policy

PPO hyperparameters (from the paper's Table 5):

```yaml
# baselines/configs/policy/ppo.yaml
algorithm: "ppo"
policy_type: "MlpPolicy"
n_steps: 2048
batch_size: 64
gamma: 0.99
learning_rate: 0.0003
clip_range: 0.2
ent_coef: 0.0
vf_coef: 0.5
max_grad_norm: 0.5
gae_lambda: 0.95
```

### Reward

Task presets are mirrored as Hydra configs:

```yaml
# baselines/configs/reward/task1.yaml
reward_type: DeadbandRewardConfig
energy_weight: 0.01
dT: 1.0
```

### Training

```yaml
# baselines/configs/training/default.yaml
total_timesteps: 1000000
n_envs: 4
eval_freq: 262144
eval_episodes: 20
```

### W&B

```yaml
# baselines/configs/wandb/default.yaml
enabled: true
project: "building2building"
entity: "pierre-luc-bacon-mila-org"
tags: []
```

## CLI Override Examples

```bash
# Override training hyperparameters
python -m baselines.train_ppo experiment=train_ppo \
    training.total_timesteps=500000 policy.learning_rate=1e-4

# Disable W&B
python -m baselines.train_ppo experiment=train_ppo wandb.enabled=false

# Multi-run sweep
python -m baselines.train_ppo --multirun \
    experiment=train_ppo seed=1,2,3 \
    building_types=[OfficeSmall],[Warehouse]
```

## Output Directory

Hydra creates timestamped output directories:

```
outputs/
└── train_ppo/
    └── 2025-01-15/
        └── 14-30-00/
            ├── .hydra/
            │   ├── config.yaml
            │   ├── hydra.yaml
            │   └── overrides.yaml
            ├── models/
            └── results.csv
```

## Typed Configuration (Package API)

The core package uses frozen dataclasses for configuration, independent of
Hydra. See the [config API reference](../api/config.md) for:

- `DatasetSelectionConfig` -- building selection
- `EnvBuildConfig` -- complete environment specification
- `TaskConfig` -- run period, temperature targets, timestep
- `RewardConfig` -- reward function parameters
