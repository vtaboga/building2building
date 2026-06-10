# Reward Functions

## Overview

B2B uses one reward function: **NormalizedDeadbandReward**, which penalises
temperature deviations outside a comfort deadband and optionally penalises
energy consumption.  Per-bucket `(tau_T, tau_E)` normalizers make
`energy_weight` dimensionless and comparable across building types and
climate zones.

\[
r_t = -\left(
    \frac{1}{\tau_T} \cdot \frac{1}{N_z}\sum_{z=1}^{N_z}(T_z - T_\text{target})^2
    + w_E \cdot \frac{E_t}{\tau_E}
\right)
\]

where `(tau_T, tau_E)` are median comfort and energy penalties under a random
policy for the building's `(building_type, climate_zone)` bucket (stored in
`building2building/data/reward_normalizers.yaml`).

**Configuration:**

```python
from building2building.types import NormalizedDeadbandRewardConfig

reward = NormalizedDeadbandRewardConfig(
    energy_weight=1.0,  # dimensionless trade-off weight
    dT=1.0,             # comfort deadband half-width (°C)
    tau_T=None,         # None = auto-resolved at env-build time
    tau_E=None,
)
```

`tau_T = tau_E = None` is the *unfilled sentinel state*.  Pass the preset
name to `make_env` and it resolves the constants automatically:

```python
import building2building as b2b

env = b2b.make_env("OfficeSmall", task="task_const_e0")
```

## Target Temperature Modes

**Constant:** the target is always `occupied_c` regardless of occupancy.

**Occupancy-based:** the target switches to an unoccupied setpoint when zone
occupancy is zero.  The unoccupied setpoint may be:

* `fixed` (a single year-round value via `unoccupied_c`), or
* `seasonal` — dispatched by the current simulation month via
  `seasonal_unoccupied_c = {"winter": ..., "shoulder": ..., "summer": ...}`.
  This avoids the pathology of a fixed 18 °C target driving unnecessary
  summer cooling.  The `task_occ_*` family uses the seasonal policy
  (18 / 21 / 26 °C).

!!! warning "Buildings without `People` objects"
    Occupancy-based tasks read EnergyPlus's `Zone People Occupant Count`
    variable.  If a building's epJSON does not define any `People`
    objects, that variable is always zero and `task_occ_*` degenerates to a
    constant "unoccupied" setpoint — i.e. it behaves like a
    `constant`-mode task whose target is the seasonal unoccupied value
    (18 / 21 / 26 °C) instead of 21 °C.  In the bundled dataset,
    **`SingleFamilyHouse` has no occupancy schedule**.  Prefer
    `task_rand_*` (random schedule), which drives its own occupancy signal
    from the Python side and works uniformly across every building type.

**Random schedule:** each simulated day, a fresh arrival time, departure time,
occupied setpoint, and unoccupied setpoint are sampled from a per-building-type
distribution.  This is what the `task_rand_*` family uses.

```python
env = b2b.make_env(
    "OfficeSmall",
    task="task_occ_e0",  # seasonal occupancy-based targets
)

env = b2b.make_env(
    "OfficeSmall",
    task="task_rand_e0",  # per-day random arrival/departure + setpoints
    random_schedule_seed=42,
)
```

## Energy Weight Levels

The three weight levels in the normalized preset grid:

| Level suffix | `energy_weight` | Behaviour |
|---|---|---|
| `e0` | 0.0 | Comfort-only; energy consumption not penalised |
| `emed` | 1.0 | Balanced; comfort and energy roughly equally weighted |
| `ehigh` | 5.0 | Energy-emphasis; aggressive energy minimization |

## Task Presets

Nine named presets form a 3×3 grid over `(setpoint_mode, energy_weight)`:

| Task | Mode | Energy weight | dT |
|---|---|---|---|
| `task_const_e0` | Constant | 0.0 | 1.0 |
| `task_const_emed` | Constant | 1.0 | 1.0 |
| `task_const_ehigh` | Constant | 5.0 | 1.0 |
| `task_occ_e0` | Occupancy (seasonal) | 0.0 | 1.0 |
| `task_occ_emed` | Occupancy (seasonal) | 1.0 | 1.0 |
| `task_occ_ehigh` | Occupancy (seasonal) | 5.0 | 1.0 |
| `task_rand_e0` | Random schedule | 0.0 | 1.0 |
| `task_rand_emed` | Random schedule | 1.0 | 1.0 |
| `task_rand_ehigh` | Random schedule | 5.0 | 1.0 |

The default task is `task_const_e0` (constant setpoint, comfort-only).
