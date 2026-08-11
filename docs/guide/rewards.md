# Reward Functions

## Overview

B2B uses one reward function: **NormalizedDeadbandReward**, which penalises
squared deviations of each controlled zone's temperature from its target and
optionally penalises energy consumption.  A per-bucket energy normalizer
`tau_E` makes `energy_weight` dimensionless and comparable across building
types and climate zones.

\[
r_t = -\left(
    \frac{1}{\tau_T} \cdot \frac{1}{N_z}\sum_{z=1}^{N_z}(T_z - T^\text{target}_z)^2
    + w_E \cdot \frac{E_t}{\tau_E}
\right)
\]

where `E_t` is the HVAC energy use per square metre at step `t`.  The
normalization is asymmetric:

* **Comfort is unnormalized**: `tau_T ≡ 1`, so the comfort term is a raw
  mean squared deviation in °C² — the same physical unit in every building,
  zone, and climate.
* **Only energy is normalized**: `tau_E` is the median per-step energy spend
  of the reference reactive controller for the building's
  `(building_type, climate_zone)` bucket, calibrated under the occupancy
  regime (`task_occ_*`, `dT=1.0`) on the train split.  `power_penalty /
  tau_E = 1` means "spends like the reference controller for this bucket".

The constants live in `building2building/data/reward_normalizers.yaml`.
`tau_E` is **seasonal** (heating and cooling energy differ), so the YAML has
one `constants` section per run period; the packaged file ships `winter`,
`summer`, and `full_year`.  Requesting a `run_period` that has no section
raises a clear error at env-build time until that section is calibrated.
Regenerate the YAML with
`python -m baselines.compute_reactive_reward_normalizers --mode all`.

Because `tau_E` is calibrated under the occupancy regime, using the
`const`/`rand` setpoint modes or a non-default `dT` triggers a one-shot
`RuntimeWarning` at simulator construction.  This is intentional, the
constants are applied as-is and the calibration is approximate outside its
regime.

**Configuration:**

```python
from building2building.types import NormalizedDeadbandRewardConfig

reward = NormalizedDeadbandRewardConfig(
    energy_weight=0.5,  # dimensionless trade-off weight (w_E)
    dT=1.0,             # calibration deadband half-width (°C)
    tau_T=None,         # None = auto-resolved at env-build time
    tau_E=None,
)
```

`tau_T = tau_E = None` is the *unfilled sentinel state*.  Constructing a
simulator with an unfilled config raises.  Pass the preset name to
`make_env` and it resolves the constants automatically:

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

!!! warning "Zones without `People` objects"
    Occupancy-based tasks read EnergyPlus's `Zone People Occupant Count`
    variable.  If a controlled zone has no `People` object, that variable
    is always zero and `task_occ_*` degenerates to a constant
    "unoccupied" setpoint for that zone — i.e. it behaves like a
    `constant`-mode task whose target is the seasonal unoccupied value
    (18 / 21 / 26 °C) instead of 21 °C.  `task_rand_*` (random schedule)
    drives its own occupancy signal from the Python side and works
    uniformly across every building type.

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

The two weight levels in the normalized preset grid:

| Level suffix | `energy_weight` | Behaviour |
|---|---|---|
| `e0` | 0.0 | Comfort-only; energy consumption not penalised |
| `e05` | 0.5 | Comfort with a moderate energy penalty |

## Task Presets

Six named presets form a 3×2 grid over `(setpoint_mode, energy_weight)`:

| Task | Mode | Energy weight | dT |
|---|---|---|---|
| `task_const_e0` | Constant | 0.0 | 1.0 |
| `task_const_e05` | Constant | 0.5 | 1.0 |
| `task_occ_e0` | Occupancy (seasonal) | 0.0 | 1.0 |
| `task_occ_e05` | Occupancy (seasonal) | 0.5 | 1.0 |
| `task_rand_e0` | Random schedule | 0.0 | 1.0 |
| `task_rand_e05` | Random schedule | 0.5 | 1.0 |

The default task is `task_const_e0` (constant setpoint, comfort-only).
For `w_E` or `dT` values outside this grid, build a filled preset with
`building2building.config.tasks.make_normalized_deadband_task(...)`.
