# PI Controller (`unitary_pi`)

## Overview

`UnitaryPIPolicy` is a proportional-integral controller for unitary HVAC
systems (packaged single-zone, split systems, etc.).  It controls **fan air
mass flow rate** based on zone temperature error and sets a **fixed outlet node
temperature** depending on the current operating mode (heating, cooling, or
deadband).

```python
from b2b.baselines.controllers.unitary_pi import UnitaryPIPolicy
```

## Control Strategy

1. **Mode detection** — compare zone air temperature \(T_z\) against
   the target \(T_{\text{target}}\) and deadband \(\Delta T\):
    - Heating: \(T_z < T_{\text{target}} - \Delta T\)
    - Cooling: \(T_z > T_{\text{target}} + \Delta T\)
    - Deadband: otherwise
2. **Fan PI loop** — compute fan mass flow command:

    \[
    u_{\text{fan}} = m_{\text{base}} + K_p \cdot e + K_i \cdot \int e\, dt
    \]

    where the error \(e\) is signed so that airflow *increases* in both
    heating and cooling modes.  The integrator resets on mode transitions and
    is clamped to `integral_limit`.

3. **Outlet temperature** — set to a fixed value per mode:
    - Heating: `outlet_temp_heating_c` (default 50.0 °C)
    - Cooling: `outlet_temp_cooling_c` (default 12.0 °C)
    - Deadband: target temperature

## Parameters

```yaml title="configs/policy/unitary_pi.yaml"
type: unitary_pi

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

# Outlet temperature setpoints [°C]
outlet_temp_heating_c: 50.0
outlet_temp_cooling_c: 12.0

# PI control on fan mass flow rate [kg/s]
fan_base_kg_s: 1.0
kp: 0.5
ki: 0.001
integral_limit: 1000.0

# Optional clamps (null = use action_space bounds)
fan_min_kg_s: null
fan_max_kg_s: null
```

| Parameter | Default | Description |
|---|---|---|
| `target_temp_c` | 21.0 | Zone temperature setpoint (°C) |
| `deadband_c` | 1.0 | Half-width of the deadband (°C) |
| `kp` | 0.5 | Proportional gain |
| `ki` | 0.001 | Integral gain |
| `fan_base_kg_s` | 1.0 | Base fan mass flow rate (kg/s) |
| `integral_limit` | 1000.0 | Integrator clamp |
| `outlet_temp_heating_c` | 50.0 | Outlet temp in heating mode (°C) |
| `outlet_temp_cooling_c` | 12.0 | Outlet temp in cooling mode (°C) |
| `availability_on` | 2.0 | HVAC availability schedule value |

## Target Schedule

When `target_schedule.enabled` is `true`, the temperature setpoint varies by
time of day and day of week:

- **Weekends** (days in `weekend_days`): use `weekend_target_c`
- **Weekdays outside setback hours**: use `weekday_target_c`
- **Weekdays during setback** (`weekday_setback_start_hour` to
  `weekday_setback_end_hour`): use `weekday_setback_target_c`

## Usage

```bash
# Run a baseline rollout with the PI controller
python scripts/baselines.py policy=unitary_pi

# Override gains
python scripts/baselines.py policy=unitary_pi policy.kp=1.0 policy.ki=0.01
```

## API

`UnitaryPIPolicy` exposes an SB3-compatible interface:

```python
policy = UnitaryPIPolicy(cfg.policy)
policy.bind_env(env)          # discover obs/action indices
action, _ = policy.predict(obs, deterministic=True)
metrics = policy.step_metrics(obs, action=action)
```

The `step_metrics()` method returns `target_temp_c` and
`conditioned_zone_temp_c` for logging.
