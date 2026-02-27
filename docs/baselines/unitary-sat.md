# SAT Controller (`unitary_sat`)

## Overview

`UnitaryAirflowFirstSatPolicy` is a supply air temperature (SAT) reset
controller for unitary HVAC systems.  It uses a two-loop control strategy:

1. **Primary loop** — PI controller on fan mass flow rate (fast loop).
2. **Secondary loop** — SAT trim/reset that adjusts the outlet temperature
   setpoint only when fan airflow is saturated (slow loop).

This "airflow-first" approach prioritises fan modulation for responsive zone
temperature control and only resorts to SAT adjustments when the fan reaches
its limits.

```python
from b2b.baselines.controllers.unitary_sat import UnitaryAirflowFirstSatPolicy
```

## Control Strategy

### Fan PI Loop (Primary)

Identical to the [PI controller](unitary-pi.md): a proportional-integral
controller drives the fan mass flow rate based on zone temperature error.  The
integrator resets on mode transitions.

### SAT Trim/Reset (Secondary)

The secondary loop activates when the fan command is pegged at its minimum or
maximum for `sat_saturation_steps` consecutive timesteps **and** the zone
remains outside the deadband:

- **Heating**: increase SAT setpoint by `sat_step_c` (bounded by `sat_max_c`)
- **Cooling**: decrease SAT setpoint by `sat_step_c` (bounded by `sat_min_c`)

SAT changes are rate-limited to `sat_rate_limit_c_per_step` per timestep.
When the mode transitions, the SAT setpoint resets to its mode-specific default.

### Multi-System Support

The controller automatically discovers per-zone unitary systems from
`env.metadata["hvac_equipment"]`.  Each system gets independent PI and SAT
state, so zones are controlled independently even in multi-zone buildings.

## Parameters

```yaml title="configs/policy/unitary_sat.yaml"
type: unitary_sat

# Control objective
target_temp_c: 21.0
deadband_c: 1.0

# Optional time-based target schedule
target_schedule:
  enabled: true
  weekend_days: [1, 7]
  weekend_target_c: 21.0
  weekday_target_c: 21.0
  weekday_setback_target_c: 18.0
  weekday_setback_start_hour: 9.0
  weekday_setback_end_hour: 16.0

availability_on: 2.0

# Supply air temperature defaults [°C]
outlet_temp_heating_c: 50.0
outlet_temp_cooling_c: 14.0

# Fan PI controller [kg/s]
fan_base_kg_s: 0.3
kp: 0.2
ki: 0.0002
integral_limit: 200.0
fan_min_kg_s: null
fan_max_kg_s: null

# SAT trim/reset
sat_min_c: 10.0
sat_max_c: 70.0
sat_step_c: 0.5
sat_rate_limit_c_per_step: 0.5
sat_saturation_steps: 3
```

| Parameter | Default | Description |
|---|---|---|
| `target_temp_c` | 21.0 | Zone temperature setpoint (°C) |
| `deadband_c` | 1.0 | Half-width of the deadband (°C) |
| `fan_base_kg_s` | 0.3 | Base fan mass flow rate (kg/s) |
| `kp` | 0.2 | Proportional gain (fan PI) |
| `ki` | 0.0002 | Integral gain (fan PI) |
| `integral_limit` | 200.0 | Fan PI integrator clamp |
| `sat_min_c` | 10.0 | SAT lower bound (°C) |
| `sat_max_c` | 70.0 | SAT upper bound (°C) |
| `sat_step_c` | 0.5 | SAT adjustment step size (°C) |
| `sat_rate_limit_c_per_step` | 0.5 | Max SAT change per timestep (°C) |
| `sat_saturation_steps` | 3 | Fan-saturation count before SAT adjusts |

## Usage

```bash
# Run a baseline rollout with the SAT controller (default)
python scripts/baselines.py policy=unitary_sat

# Override SAT parameters
python scripts/baselines.py policy=unitary_sat policy.sat_step_c=0.2 policy.sat_saturation_steps=5
```

## API

```python
policy = UnitaryAirflowFirstSatPolicy(cfg.policy)
policy.bind_env(env)
action, _ = policy.predict(obs, deterministic=True)
metrics = policy.step_metrics(obs, action=action)
```

The `step_metrics()` method returns `target_temp_c`, `mean_zone_temp_c`,
`min_zone_temp_c`, and `max_zone_temp_c`.
