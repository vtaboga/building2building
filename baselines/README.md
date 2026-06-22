# Building2Building Baselines

Reference implementations of the control baselines and transfer experiments
from the Building2Building paper. See the
[full documentation](../docs/baselines/overview.md) for detailed guides.

## Structure

```
baselines/
├── controllers/           # Reactive controllers
│   ├── unitary_hvac.py    # Controller for single-zone unitary systems
│   └── air_loop.py        # VAV air-loop controller
├── models/
│   └── amorpheus.py       # Type-heterogeneous transformer (Section 6.2)
├── utils/
│   ├── metadata.py        # Observation/action name helpers
│   ├── evaluation.py      # Rollout and metric functions
│   ├── training.py        # SB3 PPO builder with paper defaults
│   └── callbacks.py       # W&B logging callback
├── configs/               # Hydra configuration
│   ├── config.yaml        # Root config (defaults: policy, training, wandb, experiment)
│   ├── experiment/        # Per-experiment configs (one per run; selects tasks)
│   ├── policy/            # Algorithm hyperparameters (ppo, sac, controllers)
│   ├── training/          # Training loop parameters (timesteps, envs, eval freq)
│   ├── tuned_controllers/ # Optuna-tuned controller YAMLs
│   └── wandb/             # W&B logging settings
├── plotting/              # Matplotlib plotting scripts
│   ├── common.py          # Shared styles, palettes, data loaders
│   ├── plot_ppo_specialist.py
│   ├── plot_dynamics_adaptation.py
│   └── plot_cross_domain.py
├── run_reactive_control.py       # Evaluate reactive controllers, generate baseline CSV
├── train_ppo.py           # Per-building PPO specialist (Section 5)
├── train_sac.py           # Per-building SAC specialist (Section 5)
├── train_dynamics_adaptation.py  # Section 6.1 experiments
├── train_cross_domain.py  # Section 6.2 Amorpheus cross-domain transfer
├── eval_ppo.py            # Evaluate trained PPO models
├── eval_dynamics_adaptation.py   # Evaluate dynamics adaptation models
├── eval_cross_domain.py   # Evaluate cross-domain transfer
├── tune_controller.py     # Optuna-based reactive controller tuning
├── tune_ppo.py            # Optuna-based PPO hyperparameter tuning
└── README.md
```

## Quick Start

### Install

```bash
pip install -e "../[training]"
```

This installs `building2building` in editable mode together with all
training dependencies (PyTorch, Stable-Baselines3, Hydra, Optuna, W&B).

### 1. Reactive Controllers

Evaluate the reactive controller on all buildings and generate the
`baseline_returns.csv` used for normalized scoring:

```bash
python -m baselines.run_reactive_control experiment=eval_reactive_control

# Subset:
python -m baselines.run_reactive_control experiment=eval_reactive_control \
    building_types=[OfficeSmall] tasks=[task_const_e0] max_buildings_per_type=5
```

### 2. Single-Building Specialists

Train PPO or SAC specialists across building types and tasks:

```bash
python -m baselines.train_ppo experiment=train_ppo
python -m baselines.train_sac experiment=train_sac

# Override for quick testing:
python -m baselines.train_ppo experiment=train_ppo \
    building_types=[OfficeSmall] tasks=[task_const_e0] \
    training.total_timesteps=100_000

# Full 9-task grid on the fast (test_small) split, single seed:
python -m baselines.train_ppo experiment=train_ppo_task_study \
    --multirun seed=0 building_split=test_small
python -m baselines.train_sac experiment=train_sac_task_study \
    --multirun seed=0 building_split=test_small
```

The nine task presets are `task_{const,occ,rand}_{e0,emed,ehigh}`; pass them
as a list, e.g. `tasks=[task_const_e0,task_occ_emed]`.

### 3. Dynamics Adaptation 

Three approaches: `specialist`, `baseline` (multi-building MLP), `parameterized`
(multi-building MLP with building parameter augmentation):

```bash
python -m baselines.train_dynamics_adaptation \
    experiment=train_dynamics_parameterized

python -m baselines.train_dynamics_adaptation \
    experiment=train_dynamics_specialist difficulty=medium
```

Difficulty levels:
| Level  | Building Type      | Action Dim |
|--------|--------------------|------------|
| easy   | SingleFamilyHouse  | 2          |
| medium | OfficeSmall        | 10         |
| hard   | OfficeMedium       | 33         |

### 4. Cross-Domain Transfer

Train the Amorpheus transformer on multiple building types simultaneously:

```bash
python -m baselines.train_cross_domain experiment=train_cross_domain

python -m baselines.train_cross_domain experiment=train_cross_domain \
    total_timesteps=5_000_000 model.embed_dim=128
```

### 5. Hyperparameter Tuning (Optuna)

Tune reactive controller parameters, or PPO hyperparameters per task:

```bash
python -m baselines.tune_controller experiment=tune_controller \
    building_type=OfficeSmall climate_zone=1

python -m baselines.tune_ppo experiment=tune_ppo task=task_const_e0
```

### 6. Evaluation

```bash
# PPO specialists
python -m baselines.eval_ppo --model-dir outputs/train_ppo/models

# Dynamics adaptation
python -m baselines.eval_dynamics_adaptation \
    --model-path outputs/dynamics_parameterized/models/multi_parameterized.zip \
    --difficulty easy --approach parameterized

# Cross-domain transfer
python -m baselines.eval_cross_domain \
    --model-path outputs/cross_domain/amorpheus_policy.pt \
    --test-building-types Warehouse SingleFamilyHouse
```

### 7. Plotting

Generate paper figures from result CSVs:

```bash
python -m baselines.plotting.plot_ppo_specialist \
    --ppo-csv results_ppo.csv --baseline-csv baseline_returns.csv

python -m baselines.plotting.plot_dynamics_adaptation \
    --specialist-csv results_dynamics_specialist.csv \
    --baseline-csv results_dynamics_baseline.csv \
    --parameterized-csv results_dynamics_parameterized.csv

python -m baselines.plotting.plot_cross_domain \
    --results-csv results_cross_domain.csv
```

## Configuration

All training scripts use [Hydra](https://hydra.cc/) for configuration.
The config structure lives in `baselines/configs/`:

- **`config.yaml`** -- root config with defaults for experiment, policy, training, wandb
- **`experiment/`** -- per-script experiment settings (building types, split, `tasks` list, approach, etc.)
- **`policy/`** -- algorithm hyperparameters (`ppo`, `sac`, controller defaults)
- **`training/`** -- loop parameters (timesteps, envs, eval frequency)
- **`tuned_controllers/`** -- Optuna-optimized controller configs per building type/climate zone

Tasks are selected per experiment via a `tasks: [...]` list of preset names
(`task_{const,occ,rand}_{e0,emed,ehigh}`) — reward configs are resolved from
the preset by the `building2building` package, so there is no `reward/` group
to edit here.

Override any config value from the command line:

```bash
python -m baselines.train_ppo experiment=train_ppo \
    seed=42 training.total_timesteps=1_000_000 policy.learning_rate=1e-4
```

## API Usage

All baselines use only the public `building2building` API:

```python
import building2building as b2b

# Create environment
env = b2b.make_env("OfficeSmall", task="task_const_e0")

# Access metadata
obs_names = env.metadata["observation_names"]
act_names = env.metadata["action_names"]
morphology = env.metadata["morphology"]

# Structured representation
obs, _ = env.reset()
local_obs = morphology.split_observation(obs)

# Scoring
score = b2b.compute_normalized_score(
    cumulative_return=-5000.0,
    building_type="OfficeSmall",
    task="task_const_e0",
    run_period="full_year",
    building_id="OfficeSmall-0001",
)

# Benchmarks
bench = b2b.benchmarks.DynamicsAdaptation(difficulty="easy")
train_ids = bench.train_building_ids()
test_ids = bench.test_building_ids()
```

## Wrappers for Multi-Building Training

The main package provides wrappers essential for training across diverse buildings:

- `b2b.PadObservation(env, target_size)` -- zero-pad observations to uniform size
- `b2b.NormalizeObservation(env)` -- running-mean normalization
- `b2b.AugmentObservationWithBuildingParams(env)` -- append building parameters
- `b2b.ResampleBuildingOnResetWrapper(factory, indices)` -- resample building on reset
