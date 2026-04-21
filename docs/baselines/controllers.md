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
kp: 0.25
ki: 0.02
integral_max: 200.0
min_fan_fraction: 0.15
sat_min_c: 12.0
sat_max_c: 35.0
sat_initial_c: 21.0
sat_trim: 0.5
sat_respond: 1.0
demand_deadband: 0.3
availability_on: 2.0
```

| Parameter | Default | Description |
|---|---|---|
| `heating_setpoint_c` | 20.0 | Heating zone setpoint (C) |
| `cooling_setpoint_c` | 22.0 | Cooling zone setpoint (C) |
| `kp` | 0.25 | Proportional gain |
| `ki` | 0.02 | Integral gain |
| `integral_max` | 200.0 | Anti-windup clamp |
| `min_fan_fraction` | 0.15 | Minimum fan speed fraction |
| `sat_min_c` / `sat_max_c` | 12.0 / 35.0 | SAT limits (C) |
| `sat_initial_c` | 21.0 | Neutral SAT at reset (C) |
| `sat_trim` | 0.5 | SAT drift rate (C/step) |
| `sat_respond` | 1.0 | SAT response step (C/step) |
| `demand_deadband` | 0.3 | Overshoot threshold (C) |

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

- **SAT P-control:** central supply air temperature is adjusted proportionally
  to the maximum zone cooling demand.
- **Directional PI flow control:** per-zone VAV damper position is driven by a
  PI controller with separate gains for heating and cooling regimes.
- **Reheat setpoint control:** per-zone heating and cooling setpoints are
  maintained at fixed values.

### Parameters

```yaml
# baselines/configs/policy/air_loop.yaml
type: air_loop
heating_setpoint_c: 20.0
cooling_setpoint_c: 22.0
sat_min_c: 12.0
sat_max_c: 18.0
sat_kp: 0.5
flow_kp_heat: 0.3
flow_kp_cool: 0.15
flow_ki_heat: 0.01
flow_ki_cool: 0.005
# ... (see full config for all ~30 parameters)
```

---

## Policy Interface

Both controllers implement the same interface, compatible with SB3:

```python
class PolicyLike:
    def bind_env(self, env: gym.Env) -> None: ...
    def reset(self) -> None: ...
    def predict(self, obs, deterministic=True) -> tuple[np.ndarray, None]: ...
    def step_metrics(self, obs, *, action) -> dict[str, float]: ...
```

- `bind_env()` reads observation/action names from `env.metadata` and caches
  actuator indices.
- `reset()` clears internal state (PI integrators, SAT setpoint).
- `step_metrics()` returns per-step metrics for logging (zone temps, setpoints).

## Running Baselines

```bash
# Evaluate on all building types and tasks
python -m baselines.run_reactive_control experiment=eval_reactive_control

# Specific building type
python -m baselines.run_reactive_control experiment=eval_reactive_control \
    building_types=[OfficeSmall] tasks=[task1] max_buildings_per_type=5

# Override controller parameters
python -m baselines.run_reactive_control experiment=eval_reactive_control \
    policy.heating_setpoint_c=19.0 policy.cooling_setpoint_c=24.0
```

The script auto-selects `AirLoopPolicy` for `OfficeMedium` and
`UnitaryHvacPolicy` for all other building types. If tuned controller configs
exist for the building type and climate zone, they are loaded automatically.

For full-year diagnostic rollouts of the tuned controllers (trajectories,
per-building figures, and aggregated per-type summaries), see
[Tuned-Controller Analysis](analysis.md). That page documents the
reproducible pipeline used to produce the `analysis/*_tuned_v2/` artefacts
shipped with the repo.

## Output

`run_reactive_control.py` generates `baseline_returns.csv` with columns:

| Column | Description |
|---|---|
| `building_type` | Building type |
| `task` | Task preset name |
| `building_id` | Building identifier |
| `reward_run1` | Episode return |
| `reward_mean` | Mean return (over runs) |

This CSV is used by `b2b.compute_normalized_score()`.
