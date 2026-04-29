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

**Occupancy-based:** the target switches to an unoccupied setpoint when zone
occupancy is zero.  The unoccupied setpoint may be:

* `fixed` (a single year-round value via `unoccupied_c`), or
* `seasonal` — dispatched by the current simulation month via
  `seasonal_unoccupied_c = {"winter": ..., "shoulder": ..., "summer": ...}`.
  This avoids the pathology of a fixed 18 °C target driving unnecessary
  summer cooling.  The paper's `task3` uses the seasonal policy by default
  (18 / 21 / 26 °C).

!!! warning "Buildings without `People` objects"
    Occupancy-based tasks read EnergyPlus's `Zone People Occupant Count`
    variable.  If a building's epJSON does not define any `People`
    objects, that variable is always zero and `task3` degenerates to a
    constant "unoccupied" setpoint — i.e. it behaves like a
    `constant`-mode task whose target is the seasonal unoccupied value
    (18 / 21 / 26 °C) instead of 21 °C.  In the bundled dataset,
    **`SingleFamilyHouse` has no occupancy schedule**, so `task3` on
    that building type is not informative.  Prefer `task5`
    (random schedule), which drives its own occupancy signal from the
    Python side and works uniformly across every building type.

**Random schedule:** each simulated day, a fresh arrival time, departure time,
occupied setpoint, and unoccupied setpoint are sampled from a per-building-type
distribution.  This is what `task5` uses.

```python
env = b2b.new_make_env(
    "OfficeSmall",
    task="task3",  # seasonal occupancy-based targets
)

env = b2b.new_make_env(
    "OfficeSmall",
    task="task5",  # per-day random arrival/departure + setpoints
    random_schedule_seed=42,
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

The five named task presets define specific reward configurations:

| Task | Reward | Energy Weight | dT | Mode | Notes |
|---|---|---|---|---|---|
| `task1` | Deadband | 0.01 | 1.0 | Constant | Comfort-leaning baseline |
| `task2` | Deadband | 0.10 | 1.0 | Constant | Energy-leaning baseline |
| `task3` | Deadband | 0.01 | 1.0 | Occupancy | Seasonal unoccupied setpoint (18 / 21 / 26 °C) |
| `task4` | Barrier | 0.01 | 1.0 | Constant | Hard-band penalty |
| `task5` | Deadband | 0.01 | 1.0 | Random schedule | Per-day random arrival / departure / setpoints |

An additional preset `task3_legacy` reproduces the paper's original
fixed 18 °C unoccupied setpoint and is intended only for the
seasonal-ablation study in `analysis/task_study/`.
