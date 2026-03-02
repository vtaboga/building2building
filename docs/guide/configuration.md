# Configuration (Hydra)

B2B uses [Hydra](https://hydra.cc/) with typed dataclasses for configuration management. This page describes the configuration hierarchy, config groups, key config files, and how to override settings from the CLI.

---

## Config Hierarchy

The base configuration file composes default config groups:

```yaml
# configs/base.yaml
defaults:
  - _self_
  - wandb: default
  - training: default
  - policy: ppo
  - reward: deadband
  - task: default
  - bldg: single_family
  - rollout: default

env:
  normalize_obs: true
  normalize_action: false
  max_steps: null

n_episodes: 1
seed: 42

benchmark:
  split: train
  start: 0
  limit: 0
  max_steps: null
```

### Config Groups

| Group | Directory | Purpose |
|---|---|---|
| `task` | `configs/task/` | Run period, target temperature mode |
| `reward` | `configs/reward/` | Reward function type and parameters |
| `policy` | `configs/policy/` | RL algorithm or baseline controller settings |
| `training` | `configs/training/` | Training hyperparameters |
| `wandb` | `configs/wandb/` | Weights & Biases logging settings |
| `bldg` | `configs/bldg/` | Building/dataset selection |
| `rollout` | `configs/rollout/` | Rollout execution settings |
| `benchmark_interface` | `configs/benchmark_interface/` | Benchmark evaluation configs |

---

## Task Configuration

Controls the simulation period and target temperatures.

```yaml
# configs/task/default.yaml
run_period: full_year        # full_year | winter | summer
target_temperature_mode: constant  # constant | occupancy

default_zone_target_temperature:
  occupied_c: 21.0
  unoccupied_c: 21.0

zone_target_temperatures: {}
```

### Run Periods

| Period | Start | End | Steps (15 min) |
|---|---|---|---|
| `full_year` | Jan 1 | Dec 31 | 35,040 |
| `winter` | Jan 1 | Mar 31 | 8,640 |
| `summer` | Jun 1 | Aug 31 | 8,832 |

### Target Temperature Modes

- **`constant`**: Target is always `occupied_c` regardless of occupancy
- **`occupancy`**: Target switches to `unoccupied_c` when zone occupancy is zero

### CLI Override Examples

```bash
# Run for winter only
python -m b2b.train task.run_period=winter

# Occupancy-based targets with setback
python -m b2b.train \
    task.target_temperature_mode=occupancy \
    task.default_zone_target_temperature.occupied_c=22.0 \
    task.default_zone_target_temperature.unoccupied_c=16.0
```

---

## Reward Configuration

### BarrierReward

```yaml
# configs/reward/barrier.yaml
reward_type: BarrierRewardConfig
energy_weight: 0.1
deadband_c: 0.5
violation_penalty: 100.0
```

### DeadbandReward

```yaml
# configs/reward/deadband.yaml
reward_type: DeadbandRewardConfig
dT: 1.0
energy_weight: 0.001
```

### CLI Override Examples

```bash
# Switch reward function
python -m b2b.train reward=barrier

# Override reward parameters
python -m b2b.train reward=barrier reward.energy_weight=0.5 reward.violation_penalty=200.0
```

---

## Policy Configuration

B2B supports RL algorithms, baseline controllers, and custom policies.

### PPO

```yaml
# configs/policy/ppo.yaml
algorithm: "ppo"
policy_type: "MlpPolicy"
device: "cpu"
n_steps: 2048
batch_size: 64
gamma: 0.99
learning_rate: 0.0003
clip_range: 0.2
ent_coef: 0.0
vf_coef: 0.5
max_grad_norm: 0.5
gae_lambda: 0.95
verbose: 1
```

### SAC

```yaml
# configs/policy/sac.yaml
algorithm: "sac"
buffer_size: 100000
batch_size: 128
learning_starts: 1000
tau: 0.005
gamma: 0.99
learning_rate: 0.0003
policy_type: "MlpPolicy"
```

### SB3 Checkpoint

```yaml
# configs/policy/sb3.yaml
type: sb3
algorithm: ppo              # SB3 algorithm name
checkpoint_path: ???        # Path to saved .zip model
```

### Custom Policy

```yaml
# configs/policy/custom.yaml
type: custom
module: ???                 # Fully-qualified Python module
class_name: ???             # Class with predict(obs, deterministic) -> (action, state)
kwargs: {}                  # Constructor keyword arguments
```

### Baseline Controllers

```yaml
# configs/policy/unitary_g36.yaml
# configs/policy/ashrae_air_loop.yaml
# configs/policy/air_loop_sat.yaml
```

### CLI Override Examples

```bash
# Switch algorithm
python -m b2b.train policy=sac

# Override hyperparameters
python -m b2b.train policy=ppo policy.learning_rate=0.001 policy.n_steps=4096

# Load a trained checkpoint
python -m b2b.benchmark.baseline_rollout policy=sb3 policy.checkpoint_path=checkpoints/ppo.zip
```

---

## Training Configuration

```yaml
# configs/training/default.yaml
total_timesteps: 1000000
eval_freq: 262144
eval_episodes: 20
num_train_envs: 4
cb_gradient_save_freq: 500
```

| Parameter | Default | Description |
|---|---|---|
| `total_timesteps` | 1,000,000 | Total training steps |
| `eval_freq` | 262,144 | Steps between evaluations |
| `eval_episodes` | 20 | Episodes per evaluation |
| `num_train_envs` | 4 | Parallel training environments (SubprocVecEnv) |
| `cb_gradient_save_freq` | 500 | Gradient logging frequency |

### CLI Override Examples

```bash
python -m b2b.train training.total_timesteps=2000000 training.num_train_envs=8
```

---

## W&B Configuration

```yaml
# configs/wandb/default.yaml
enabled: true
project: "building2building"
entity: "your-org"
tags: []
```

| Parameter | Default | Description |
|---|---|---|
| `enabled` | `true` | Enable/disable W&B logging |
| `project` | `"building2building"` | W&B project name |
| `entity` | — | W&B team/organization |
| `tags` | `[]` | Tags for organizing runs |

### CLI Override Examples

```bash
# Disable wandb
python -m b2b.train wandb.enabled=false

# Custom project
python -m b2b.train wandb.project=my-experiment wandb.tags="[transfer,vav]"
```

---

## Building Configuration

### Single-Family Houses

```yaml
# configs/bldg/single_family.yaml
# (dataset-specific building selection)
```

### Multi-Zone Buildings

```yaml
# configs/bldg/multi_zone.yaml
# (building type, climate zone, split selection)
```

---

## Benchmark Interface Configuration

For structured train/test benchmarks:

### Single-Type Benchmark

```yaml
# configs/benchmark_interface/single_type.yaml
mode: single_type
building_type: OfficeSmall
train:
  selection:
    mode: random
    n: 50
    seed: 42
  config:
    task:
      run_period: winter
    reward:
      reward_type: BarrierRewardConfig
      energy_weight: 0.1
test:
  selection:
    mode: random
    n: 10
    seed: 123
  config:
    task:
      run_period: winter
    reward:
      reward_type: BarrierRewardConfig
      energy_weight: 0.1
```

### Multi-Type Benchmark

```yaml
# configs/benchmark_interface/multi_type.yaml
mode: multi_type
train:
  types: [OfficeSmall, RetailStandalone, Warehouse]
  selection:
    mode: random
    n: 30
  config:
    task:
      run_period: winter
test:
  types: [HotelSmall, OfficeMedium]
  selection:
    mode: random
    n: 10
  config:
    task:
      run_period: winter
```

---

## CLI Patterns

### Overriding Nested Values

```bash
python -m b2b.train task.default_zone_target_temperature.occupied_c=23.0
```

### Switching Config Groups

```bash
python -m b2b.train policy=sac reward=barrier bldg=multi_zone
```

### Multi-Run Sweeps

```bash
python -m b2b.train --multirun \
    policy=ppo,sac \
    reward=barrier,deadband \
    seed=1,2,3
```

### Output Directory

Hydra creates timestamped output directories by default:

```
outputs/
└── 2025-01-15/
    └── 14-30-00/
        ├── .hydra/
        │   ├── config.yaml
        │   ├── hydra.yaml
        │   └── overrides.yaml
        ├── checkpoints/
        └── eplus_outputs/
```

---

## Typed Configuration with Dataclasses

All configuration is backed by frozen dataclasses in `b2b.config.models` and `b2b.types`, ensuring type safety:

```python
from b2b.config.models import EnvBuildConfig

# From YAML/dict — validates all fields
cfg = EnvBuildConfig.from_dict({
    "dataset_selection": {"dataset": "single_zone_houses", "split": "train"},
    "task": {"run_period": "winter"},
    "reward": {"reward_type": "BarrierRewardConfig"},
})

# Invalid values raise immediately
EnvBuildConfig.from_dict({"task": {"run_period": "spring"}})
# ValueError: task.run_period must be one of {'full_year', 'winter', 'summer'}
```

---

## Next Steps

- Write [custom policies](custom-policies.md) and load them via configuration
- See the full [API reference](../index.md) for config dataclasses
- Follow the [Getting Started](../getting-started.md) guide for end-to-end examples
