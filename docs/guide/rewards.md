# Reward Functions

## Overview

B2B provides three reward functions that trade off thermal comfort against
energy consumption:

| Reward | Key Idea | Config Class |
|---|---|---|
| **BaseReward** | MSE temperature penalty + energy cost | `BaseRewardConfig` |
| **DeadbandReward** | Quadratic penalty outside deadband + energy | `DeadbandRewardConfig` |
| **BarrierReward** | Hard violation penalty outside deadband + energy | `BarrierRewardConfig` |

## DeadbandReward

The default reward for most tasks. Penalizes temperature deviations outside a
comfort deadband of width `dT` around the target:

\[
r_t = -\frac{1}{N_z} \sum_{z=1}^{N_z} \max(0, |T_z - T_{\text{target}}| - \text{dT})^2
     - w_e \cdot E_t
\]

where \(T_z\) is the zone temperature, \(T_{\text{target}}\) is the target
temperature, `dT` is the deadband half-width, and \(E_t\) is the normalized
energy consumption.

**Configuration:**

```python
from building2building.types import DeadbandRewardConfig

reward = DeadbandRewardConfig(
    energy_weight=0.01,  # weight on energy term
    dT=1.0,              # deadband half-width (C)
)
```

## BarrierReward

Adds a hard violation penalty when temperature leaves the deadband:

\[
r_t = -\frac{1}{N_z} \sum_{z=1}^{N_z} \left[
    \max(0, |T_z - T_{\text{target}}| - \text{dT})^2
    + \lambda \cdot \mathbf{1}[|T_z - T_{\text{target}}| > \text{dT}]
\right] - w_e \cdot E_t
\]

where \(\lambda\) is the `violation_penalty`.

**Configuration:**

```python
from building2building.types import BarrierRewardConfig

reward = BarrierRewardConfig(
    energy_weight=0.01,
    dT=1.0,
    violation_penalty=10.0,
)
```

## BaseReward

Simple MSE penalty on temperature deviation plus energy cost:

\[
r_t = -\frac{1}{N_z} \sum_{z=1}^{N_z} (T_z - T_{\text{target}})^2 - w_e \cdot E_t
\]

**Configuration:**

```python
from building2building.types import BaseRewardConfig

reward = BaseRewardConfig(energy_weight=0.01)
```

## Target Temperature Modes

**Constant:** the target is always `occupied_c` regardless of occupancy.

**Occupancy-based:** the target switches to `unoccupied_c` when zone occupancy
is zero.

```python
env = b2b.new_make_env(
    "OfficeSmall",
    task="task3",  # uses occupancy mode
)
```

## Energy Weight Guidelines

| Weight | Behaviour |
|---|---|
| `0.001` | Comfort-dominated (energy nearly free) |
| `0.01` | Balanced (default for most tasks) |
| `0.1` | Energy-conscious |
| `1.0` | Aggressive energy minimization |

## Using Custom Rewards

Pass a reward config directly to `new_make_env`:

```python
import building2building as b2b
from building2building.types import BarrierRewardConfig

env = b2b.new_make_env(
    "OfficeSmall",
    task="task1",
    reward=BarrierRewardConfig(energy_weight=0.05, dT=0.5, violation_penalty=50.0),
)
```

## Task Presets

The four named task presets define specific reward configurations:

| Task | Reward | Energy Weight | dT | Mode |
|---|---|---|---|---|
| `task1` | Deadband | 0.01 | 1.0 | Constant |
| `task2` | Deadband | 0.10 | 1.0 | Constant |
| `task3` | Deadband | 0.01 | 1.0 | Occupancy |
| `task4` | Barrier | 0.01 | 1.0 | Constant |
