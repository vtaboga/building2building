# G36 Controller (`unitary_g36`)

## Overview

`UnitaryG36Policy` implements a G36-inspired supervisory controller for
single-zone PSZ (Packaged Single Zone) unitary systems.  A PI loop drives fan
airflow based on zone temperature error, while a **Trim-and-Respond** (T&R)
algorithm adjusts the supply air temperature setpoint each timestep.

```python
from b2b.baselines.controllers.unitary_g36 import UnitaryG36Policy
```

## Control Strategy

### Airflow PI Loop

A single PI controller converts zone temperature error into a fan mass flow
rate command.  Error is the distance from the nearest deadband edge:

- Zone above cooling setpoint: error = \(T_z\) - `cooling_setpoint_c`
- Zone below heating setpoint: error = `heating_setpoint_c` - \(T_z\)
- In deadband: integrator decays toward zero, fan holds minimum flow.

The PI output scales linearly from `min_fan_fraction * fan_max` to `fan_max`.

### SAT Trim-and-Respond

Each timestep the supply air temperature setpoint is adjusted:

- **Respond down** (zone too warm): if \(T_z\) exceeds the cooling setpoint
  by more than `demand_deadband`, SAT is lowered by `sat_respond`.
- **Respond up** (zone too cold): if \(T_z\) falls below the heating setpoint
  by more than `demand_deadband`, SAT is raised by `sat_respond`.
- **Trim** (zone satisfied): SAT drifts toward `sat_initial_c` at rate
  `sat_trim`, returning to a neutral operating point.

SAT is clamped to [`sat_min_c`, `sat_max_c`] and initialised to
`sat_initial_c` on reset.

### Multi-Zone Support

The controller auto-discovers per-zone unitary systems from the environment
metadata and maintains independent PI and T&R states for each zone.  A
warmup-reset detector resets PI state when zone temperature jumps by more
than 3 C between timesteps (indicating an EnergyPlus warmup phase boundary).

## Parameters

```yaml title="configs/policy/unitary_g36.yaml"
type: unitary_g36

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

target_schedule:
  enabled: false
```

| Parameter | Default | Description |
|---|---|---|
| `heating_setpoint_c` | 20.0 | Heating zone setpoint (C) |
| `cooling_setpoint_c` | 22.0 | Cooling zone setpoint (C) |
| `kp` | 0.25 | Proportional gain for airflow PI |
| `ki` | 0.02 | Integral gain for airflow PI |
| `integral_max` | 200.0 | Anti-windup integrator clamp |
| `min_fan_fraction` | 0.15 | Minimum fan speed (fraction of design max) |
| `sat_min_c` | 12.0 | Lower SAT limit for T&R (C) |
| `sat_max_c` | 35.0 | Upper SAT limit for T&R (C) |
| `sat_initial_c` | 21.0 | Neutral SAT setpoint at reset (C) |
| `sat_trim` | 0.5 | SAT drift rate toward neutral (C/step) |
| `sat_respond` | 1.0 | SAT step on heating/cooling demand (C/step) |
| `demand_deadband` | 0.3 | Zone overshoot threshold for T&R respond (C) |
| `availability_on` | 2.0 | HVAC availability schedule value |

## Usage

```bash
# Run a baseline rollout with the G36 controller
python scripts/baselines.py policy=unitary_g36

# Override setpoints
python scripts/baselines.py policy=unitary_g36 \
    policy.heating_setpoint_c=20.0 policy.cooling_setpoint_c=25.0
```

## API

```python
policy = UnitaryG36Policy(cfg.policy)
policy.bind_env(env)
action, _ = policy.predict(obs, deterministic=True)
metrics = policy.step_metrics(obs, action=action)
```

The `step_metrics()` method returns `heating_setpoint_c`, `cooling_setpoint_c`,
`mean_zone_temp_c`, `min_zone_temp_c`, and `max_zone_temp_c`.
