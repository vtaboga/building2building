# Building2Building (B2B)

**A large-scale reinforcement learning benchmark for HVAC control.**

---

Reinforcement learning (RL) has achieved strong results in control, yet learned policies remain brittle to changes in dynamics, action spaces, observation spaces, or reward changes. **Building2Building** (B2B) is a large-scale suite of realistic HVAC control environments built on EnergyPlus. Starting from ASHRAE 90.1-2022 building prototypes, B2B parametrically generates over **7,000 buildings** spanning **7 commercial and residential building types**, **16 climate zones**, and **3 distinct HVAC system types**.

B2B is designed to accelerate research in **transfer learning**, **multi-task RL**, and **meta-learning** for building energy management.

---

## Feature Highlights

- **7,000+ parametrically generated buildings** across commercial and residential archetypes
- **7 building types**: Warehouse, HotelSmall, RetailStandalone, RestaurantFastFood, OfficeMedium, OfficeSmall, and single-zone houses
- **16 ASHRAE climate zones** with real TMY3 weather files
- **3 HVAC system types**: VAV (Variable Air Volume), Unitary, and Heating-Only (unit heaters, baseboards, radiant)
- **Gymnasium-compatible** environments registered as `EnergyPlus-v0`
- **Flexible reward functions**: BaseReward (MSE + energy), BarrierReward (deadband + violation), DeadbandReward (quadratic/linear)
- **Observation & action wrappers** for normalization, padding, and building-parameter augmentation
- **Multi-building training** with `ResampleBuildingOnResetWrapper` for zero-shot generalization
- **Hydra + typed dataclass** configuration with full CLI override support
- **SB3 integration**: PPO, SAC, TRPO, DQN out of the box
- **W&B logging** for experiment tracking and visualization
- **SLURM-ready** for large-scale cluster training

---

## Quick Start

```python
from b2b.api import make_env
from b2b.config.models import DatasetSelectionConfig, EnvBuildConfig
from b2b.types import TaskConfig, reward_config_from_dict

cfg = EnvBuildConfig(
    dataset_selection=DatasetSelectionConfig(
        dataset="single_zone_houses",
        split="train",
        mode="split_index",
        split_index=0,
    ),
    task=TaskConfig.from_dict({"run_period": "winter"}),
    reward=reward_config_from_dict({"reward_type": "BarrierRewardConfig"}),
)
env = make_env(cfg, eplus_output_dir="outputs/eplus")

obs, info = env.reset()
for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
env.close()
```

!!! tip "Convenience functions"

    For common use cases, B2B provides shorthand factory functions:

    ```python
    from b2b.api import make_single_zone_env, make_multizones_env

    # Single-zone house
    env = make_single_zone_env(split="train", split_index=0, eplus_output_dir="outputs/eplus")

    # Multi-zone office
    env = make_multizones_env(
        building_type="OfficeSmall",
        split="train",
        split_index=0,
        eplus_output_dir="outputs/eplus",
    )
    ```

---

## Documentation Overview

| Section | Description |
|---|---|
| [Getting Started](getting-started.md) | End-to-end tutorial from installation to training |
| [Installation](guide/installation.md) | System requirements, installation, and verification |
| [Environments](guide/environments.md) | Architecture, factory functions, and lifecycle |
| [Building Types & Climate Zones](guide/buildings.md) | The 7,000+ building dataset |
| [HVAC Systems](guide/hvac-systems.md) | VAV, Unitary, and Heating-Only zone control interfaces |
| [Observations](guide/observations.md) | Observation space structure and bounds |
| [Actions](guide/actions.md) | Action space per HVAC type |
| [Rewards](guide/rewards.md) | Reward function definitions and math |
| [Wrappers](guide/wrappers.md) | Normalization, padding, and augmentation |
| [Configuration](guide/configuration.md) | Hydra config system walkthrough |
| [Custom Policies](guide/custom-policies.md) | Writing and loading your own controllers |

---

## Citing B2B

If you use Building2Building in your research, please cite:

```bibtex
@inproceedings{b2b2025,
  title={Building2Building: A Large-Scale RL Benchmark for HVAC Control},
  year={2025},
}
```
