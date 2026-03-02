# Baselines Overview

B2B ships a suite of rule-based baseline controllers alongside support for
Stable-Baselines3 (SB3) RL policies.  Baselines are run through a unified
rollout pipeline that records observations, actions, rewards, and optional
Weights & Biases logging.

## Policy Registry

The baseline rollout system selects a controller based on the `policy.type`
field in the Hydra config.  The dispatch lives in
`b2b.benchmark.baseline_rollout._build_controller_policy()`:

| `policy.type` | Controller class | Description |
|---|---|---|
| `unitary_g36` | `UnitaryG36Policy` | G36-inspired PI airflow + Trim-and-Respond SAT for PSZ |
| `ashrae_air_loop` | `AshraeAirLoopPolicy` | ASHRAE air-loop controller for VAV systems |
| `air_loop_sat` | `AirLoopSatPolicy` | SAT-based air-loop controller for VAV systems |

Additional policy types used through SB3 training scripts:

| Config file | Algorithm |
|---|---|
| `configs/policy/ppo.yaml` | PPO |
| `configs/policy/sac.yaml` | SAC |
| `configs/policy/sb3.yaml` | Generic SB3 checkpoint |
| `configs/policy/custom.yaml` | User-defined policy |

## Running Baselines

The entry point is `scripts/baselines.py`, which uses the Hydra config
`configs/baseline.yaml`:

```yaml title="configs/baseline.yaml"
defaults:
  - _self_
  - bldg: single_family
  - policy: unitary_g36
  - reward: deadband
  - wandb: default

hydra:
  run:
    dir: outputs/${policy.type}/${now:%d-%m-%Y}/${now:%H-%M-%S}

env:
  normalize_obs: false
  max_steps: 35040

n_episodes: 1
```

### Examples

```bash
# Default: unitary_g36 on a single-family house
python scripts/baselines.py

# G36 controller on OfficeSmall
python scripts/baselines.py policy=unitary_g36 \
    bldg.building_type=OfficeSmall

# ASHRAE air-loop controller
python scripts/baselines.py policy=ashrae_air_loop

# Override episode length
python scripts/baselines.py env.max_steps=8760

# Disable W&B logging
python scripts/baselines.py wandb.enabled=false
```

## Rollout Pipeline

`run_baseline_rollout()` in `b2b.benchmark.baseline_rollout` orchestrates:

1. **Environment creation** — builds the EnergyPlus Gymnasium environment from
   the Hydra config.
2. **Policy binding** — calls `policy.bind_env(env)` so the controller
   discovers observation/action indices.
3. **Episode rollout** — runs `n_episodes` episodes up to `max_steps` per
   episode.
4. **Data recording** — saves a CSV and compressed NPZ file with observations,
   actions, rewards, and per-step metrics.
5. **W&B logging** — optionally logs temperature, action, energy, and reward
   time-series plots plus summary statistics.

## Output Structure

Each rollout produces:

```
outputs/<policy_type>/<date>/<time>/
├── config.json          # Resolved Hydra config
├── rollout.csv          # Full rollout data (obs, actions, reward, metrics)
├── rollout.npz          # Compressed NumPy archive
└── eplus_outputs/       # Raw EnergyPlus simulation output
```

## Controller Details

- [G36 Controller (unitary_g36)](unitary-g36.md)
