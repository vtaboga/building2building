# G36 Controller (`unitary_g36`)

## Overview

`UnitaryG36Policy` implements ASHRAE Guideline 36, Section 5.18 for
single-zone VAV unitary systems.  Two normalised PI demand signals (heating
and cooling) are mapped through **piecewise-linear functions** to determine fan
speed and supply air temperature — with no trim-and-respond delay.

```python
from b2b.baselines.controllers.unitary_g36 import UnitaryG36Policy
```

## Control Strategy

### Demand Signals

Two independent PI controllers produce normalised demand signals in \([0, 1]\):

- **Heating demand** \(u_{\text{heat}}\): error = `heating_setpoint_c` - \(T_z\)
- **Cooling demand** \(u_{\text{cool}}\): error = \(T_z\) - `cooling_setpoint_c`

Both use back-calculation anti-windup with a non-negative integral floor.

### Fan Speed Mapping (G36 Table 5.18.4)

Fan speed is expressed as a fraction of design-maximum airflow:

| Condition | Fan fraction |
|---|---|
| Heating 0–50 % | `min_fan_fraction` |
| Heating 50–100 % | `min_fan_fraction` ... 1.0 |
| Cooling 0–25 % | `min_fan_fraction` |
| Cooling 25–50 % | `min_fan_fraction` ... `med_fan_fraction` |
| Cooling 50–75 % | `med_fan_fraction` |
| Cooling 75–100 % | `med_fan_fraction` ... 1.0 |
| Deadband | `min_fan_fraction` |

### Supply Air Temperature Mapping (G36 Table 5.18.4)

| Condition | SAT setpoint |
|---|---|
| Heating 0–50 % | `sat_dead` ... `sat_max_c` |
| Heating 50–100 % | `sat_max_c` |
| Cooling 0–25 % | `sat_dead` |
| Cooling 25–75 % | `sat_dead` ... `sat_min_c` |
| Cooling 75–100 % | `sat_min_c` |
| Deadband | `sat_dead` |

Where `sat_dead` is the midpoint of the heating and cooling setpoints
(clamped to 21–24 °C).

### Multi-Zone Support

The controller auto-discovers per-zone unitary systems from the environment
metadata and maintains independent PI states for each zone.  A warmup-reset
detector resets PI state when zone temperature jumps by more than 3 °C
between timesteps (indicating an EnergyPlus warmup phase boundary).

## Parameters

```yaml title="configs/policy/unitary_g36.yaml"
type: unitary_g36

# Zone setpoints [°C]
heating_setpoint_c: 21.0
cooling_setpoint_c: 24.0

# PI gains (tuned for 15-min timesteps)
kp: 2.0
ki: 0.1

# Fan speed fractions (of design-max)
min_fan_fraction: 0.70
med_fan_fraction: 0.85

# SAT bounds [°C]
sat_min_c: 13.0
sat_max_c: null          # null = read from action space

availability_on: 2.0

target_schedule:
  enabled: false
```

| Parameter | Default | Description |
|---|---|---|
| `heating_setpoint_c` | 21.0 | Heating zone setpoint (°C) |
| `cooling_setpoint_c` | 24.0 | Cooling zone setpoint (°C) |
| `kp` | 2.0 | Proportional gain |
| `ki` | 0.1 | Integral gain |
| `min_fan_fraction` | 0.70 | Minimum fan speed fraction |
| `med_fan_fraction` | 0.85 | Medium fan speed fraction |
| `sat_min_c` | 13.0 | Minimum cooling SAT (°C) |
| `sat_max_c` | `null` | Maximum SAT; `null` reads from action space |
| `availability_on` | 2.0 | HVAC availability schedule value |

!!! note "Minimum fan fraction"
    `min_fan_fraction` must stay above ~58 % to ensure the heating coil can
    overcome outdoor-air dilution at design heating conditions.  At lower fan
    speeds the fixed minimum ventilation volume becomes a larger fraction of
    total airflow, chilling the mixed air and starving heat delivery.

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
