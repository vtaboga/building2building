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
├── config.yaml           # Root config with defaults (policy/training/wandb/experiment)
├── experiment/           # Per-script experiment settings
│   ├── eval_reactive_control.yaml
│   ├── eval_reactive_all_periods.yaml
│   ├── eval_reactive_control_task_study.yaml
│   ├── eval_reactive_season_ablation.yaml
│   ├── train_ppo.yaml
│   ├── train_ppo_task_study.yaml
│   ├── train_sac.yaml
│   ├── train_sac_task_study.yaml
│   ├── train_dynamics_specialist.yaml
│   ├── train_dynamics_baseline.yaml
│   ├── train_dynamics_parameterized.yaml
│   ├── train_cross_domain.yaml
│   ├── tune_controller.yaml
│   └── tune_ppo.yaml
├── policy/               # Algorithm / controller hyperparameters
│   ├── ppo.yaml
│   ├── ppo_parameterized.yaml
│   ├── sac.yaml
│   ├── unitary_hvac.yaml
│   └── air_loop.yaml
├── training/
│   ├── default.yaml      # Training loop parameters
│   └── sac.yaml
├── tuned_controllers/    # Optuna-optimized controller configs
│   └── (41 YAML files, one per building_type x climate_zone)
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
algorithm: ppo
policy_type: MlpPolicy
device: auto

learning_rate: 5.0e-5
batch_size: 336
n_steps: 672
n_epochs: 5
target_kl: 0.02
gamma: 0.98
gae_lambda: 0.95
clip_range: 0.2
ent_coef: 0.01
vf_coef: 0.5
max_grad_norm: 0.5

policy_kwargs:
  net_arch:
    pi: [256, 256]
    vf: [256, 256]
  activation_fn: Tanh
  ortho_init: true
  log_std_init: -1.0
```

### Tasks

Task presets are referenced by name in experiment configs and passed
through to `make_env(task=...)`:

```yaml
# e.g. baselines/configs/experiment/train_ppo.yaml
tasks: [task_const_e0, task_const_e05, task_occ_e0, task_occ_e05, task_rand_e0, task_rand_e05]
run_period: full_year  # one of full_year / winter / summer
```

Override from the CLI with e.g. `tasks=[task_occ_e05]`.

### Training

```yaml
# baselines/configs/training/default.yaml
total_timesteps: 5_000_000
n_envs: 14
eval_freq: 262144
eval_episodes: 1
recycle_every: null
```

### W&B

```yaml
# baselines/configs/wandb/default.yaml
# Disabled by default; set `enabled: true` and your own `entity` to log runs.
enabled: false
project: building2building
entity: null
tags: []
```

## CLI Override Examples

```bash
# Override training hyperparameters
python -m baselines.train_ppo experiment=train_ppo \
    training.total_timesteps=500000 policy.learning_rate=1e-4

# Enable W&B (disabled by default)
python -m baselines.train_ppo experiment=train_ppo \
    wandb.enabled=true wandb.entity=<your-entity>

# Multi-run sweep
python -m baselines.train_ppo --multirun \
    experiment=train_ppo seed=1,2,3 \
    building_types=[OfficeSmall],[Warehouse]
```

## Output Directory

All results are written under `${base_dir}/b2b/`, where `base_dir` defaults
to `$SCRATCH` when that environment variable is set and `./outputs`
otherwise:

```
${base_dir}/b2b/
├── <experiment name>/          # script outputs (output_dir)
│   └── 2026-01-15/
│       └── 14-30-00/
└── hydra/                      # Hydra run dirs (.hydra/config.yaml, ...)
    └── <experiment name>/
        └── 2026-01-15/
            └── 14-30-00/
```

## Typed Configuration (Package API)

The core package uses frozen dataclasses for configuration, independent of
Hydra. See the [config API reference](../api/config.md) for:

- `DatasetSelectionConfig` -- building selection
- `EnvBuildConfig` -- complete environment specification
- `TaskConfig` -- run period, temperature targets, timestep
- `RewardConfig` -- reward function parameters
