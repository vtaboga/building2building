# Reactive Controllers

B2B includes two reactive controllers that serve as baselines for comparison
with learned policies.

## UnitaryHvacPolicy

Supervisory controller for single-zone PSZ (Packaged Single Zone) unitary
systems. Used for all building types except `OfficeMedium`.

```python
from baselines.controllers import UnitaryHvacConfig, UnitaryHvacPolicy

policy = UnitaryHvacPolicy(UnitaryHvacConfig())
policy.bind_env(env)
action, _ = policy.predict(obs, deterministic=True)
```

### Control Strategy

**Airflow PI loop:** converts zone temperature error into a fan mass flow rate
command. Error is the distance from the nearest deadband edge (heating or
cooling setpoint). The PI output scales from `min_fan_fraction * fan_max` to
`fan_max`.

**SAT Trim-and-Respond:** each timestep the supply air temperature setpoint is
adjusted:

- **Respond down** (too warm): if zone temp exceeds the cooling setpoint by
  more than `demand_deadband`, SAT is lowered by `sat_respond`.
- **Respond up** (too cold): if zone temp falls below the heating setpoint by
  more than `demand_deadband`, SAT is raised by `sat_respond`.
- **Trim** (satisfied): SAT drifts toward `sat_initial_c` at rate `sat_trim`.

### Parameters

```yaml
# baselines/configs/policy/unitary_hvac.yaml
type: unitary_hvac
heating_setpoint_c: 20.0
cooling_setpoint_c: 22.0
kp: 0.8
ki: 0.02
integral_max: 5.0
min_fan_fraction: 0.3
sat_min_c: 12.0
sat_max_c: 40.0
sat_initial_c: 14.0
sat_trim: 0.2
sat_respond: 0.5
demand_deadband: 0.5
availability_on: 1.0
target_schedule:
  enabled: false
```

| Parameter | Default | Description |
|---|---|---|
| `heating_setpoint_c` | 20.0 | Heating zone setpoint (C) |
| `cooling_setpoint_c` | 22.0 | Cooling zone setpoint (C) |
| `kp` | 0.8 | Proportional gain |
| `ki` | 0.02 | Integral gain |
| `integral_max` | 5.0 | Anti-windup clamp |
| `min_fan_fraction` | 0.3 | Minimum fan speed fraction |
| `sat_min_c` / `sat_max_c` | 12.0 / 40.0 | SAT limits (C) |
| `sat_initial_c` | 14.0 | Neutral SAT at reset (C) |
| `sat_trim` | 0.2 | SAT drift rate (C/step) |
| `sat_respond` | 0.5 | SAT response step (C/step) |
| `demand_deadband` | 0.5 | Overshoot threshold (C) |
| `availability_on` | 1.0 | Availability-schedule actuator value |
| `fan_error_mode` | `nearest_setpoint` | Fan PI error signal (`nearest_setpoint` or `center_of_band`) |
| `target_schedule` | disabled | Optional weekday/weekend/setback setpoint schedule |

### Multi-Zone Support

The controller auto-discovers per-zone unitary systems from `env.metadata` and
maintains independent PI and T&R states for each zone.

---

## AirLoopPolicy

Controller for VAV air-loop systems. Used for `OfficeMedium`.

```python
from baselines.controllers import AirLoopConfig, AirLoopPolicy

policy = AirLoopPolicy(AirLoopConfig())
policy.bind_env(env)
action, _ = policy.predict(obs, deterministic=True)
```

### Control Strategy

- **SAT P-control:** the central supply air temperature setpoint is a
  continuous P-controller biased toward the warmest/coldest zone, with rate
  limiting and optional cold/warm biases.
- **Directional PI flow control:** per-zone VAV airflow is driven by a
  rate-limited PI controller; when `sat_aware_flow` is enabled the sign of
  the response flips with the current SAT.
- **Reheat setpoint control:** the per-zone heating setpoint is driven by
  heating demand (P-control with deadband and rate limiting); a constant
  per-loop outdoor-air mass flow (`oa_mass_flow`) is also commanded.

### Parameters

```yaml
# baselines/configs/policy/air_loop.yaml
type: air_loop
target_temp: 21.0
deadband: 1.0
sat_aware_flow: true
sat_neutral: 20.5
sat_kp: 0.8
sat_min: 10.0
sat_max: 60.0
flow_base: 0.4
flow_kp: 0.2
flow_ki: 0.025
reheat_sp_kp: 3.0
# ... (see full config for all ~26 parameters)
```

---

## Policy Interface

Both controllers implement the same interface, compatible with SB3:

```python
class PolicyLike:
    def bind_env(self, env: gym.Env) -> None: ...
    def reset(self) -> None: ...
    def predict(self, obs, deterministic=True) -> tuple[np.ndarray, None]: ...
```

- `bind_env()` reads observation/action names from `env.metadata` and caches
  actuator indices.
- `reset()` clears internal state (PI integrators, SAT setpoint).

## Running Baselines

```bash
# Evaluate on all building types and tasks
python -m baselines.run_reactive_control experiment=eval_reactive_control

# Specific building type
python -m baselines.run_reactive_control experiment=eval_reactive_control \
    building_types=[OfficeSmall] tasks=[task_const_e0] max_buildings_per_type=5

# Save trajectories and plot temperature / actuator time-series
python -m baselines.run_reactive_control experiment=eval_reactive_control \
    save_trajectories=true plot_trajectories=true \
    building_types=[OfficeSmall] tasks=[task_const_e0] max_buildings_per_type=1
```

The script auto-selects `AirLoopPolicy` for `OfficeMedium` and
`UnitaryHvacPolicy` for all other building types. If a tuned controller
config exists in `baselines/configs/tuned_controllers/` for the building
type and climate zone, it is loaded automatically; otherwise the dataclass
defaults are used.

## Output

`run_reactive_control.py` generates `baseline_returns.csv` with one row per
`(building_type, task, run_period, building_id)`:

| Column | Description |
|---|---|
| `building_type` | Building type |
| `task` | Task preset name |
| `run_period` | Simulation period (`full_year`, `winter`, `summer`) |
| `building_id` | Building identifier |
| `reward_run1` ... `reward_runN` | Per-run episode returns (`n_runs` columns) |
| `reward_mean` | Mean return (over runs) |

Rows are appended as each building finishes, so an interrupted sweep can be
resumed by re-running the same command — completed rows are skipped.
This CSV is used by `b2b.compute_normalized_score()`.
