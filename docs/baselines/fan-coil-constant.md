# Fan Coil Constant (`fan_coil_constant`)

## Overview

`FanCoilConstantPolicy` is the simplest baseline controller: it applies
**constant setpoints** for fan mass flow rate and coil temperatures regardless
of zone conditions.  This provides a lower-bound reference for comparing more
sophisticated controllers.

```python
from b2b.baselines.controllers.fan_coil_constant import FanCoilConstantPolicy
```

## Control Strategy

At every timestep the controller outputs the same fixed action:

| Actuator | Setpoint | Default |
|---|---|---|
| Fan mass flow rate | `fan_mass_flow_kg_s` | 4.0 kg/s |
| Heating coil node temperature | `heating_coil_setpoint_c` | 45.0 °C |
| Supplemental coil node temperature | `supplemental_coil_setpoint_c` | 45.0 °C |
| Cooling coil node temperature | `cooling_coil_setpoint_c` | 18.0 °C |
| Outlet node temperature | `outlet_node_setpoint_c` | 40.0 °C |

HVAC availability is set to `availability_on` (default 2.0 = always on).

## Parameters

```yaml title="configs/policy/fan_coil_constant.yaml"
type: fan_coil_constant

# For annotation only (not used for control)
target_temp_c: 21.0
deadband_c: 0.5

availability_on: 2.0

# Fan control [kg/s]
fan_mass_flow_kg_s: 4.0

# Coil control via System Node Setpoint [°C]
heating_coil_setpoint_c: 45.0
supplemental_coil_setpoint_c: 45.0
cooling_coil_setpoint_c: 18.0
outlet_node_setpoint_c: 40.0
```

| Parameter | Default | Description |
|---|---|---|
| `fan_mass_flow_kg_s` | 4.0 | Constant fan mass flow rate (kg/s) |
| `heating_coil_setpoint_c` | 45.0 | Heating coil temperature setpoint (°C) |
| `supplemental_coil_setpoint_c` | 45.0 | Supplemental coil temperature setpoint (°C) |
| `cooling_coil_setpoint_c` | 18.0 | Cooling coil temperature setpoint (°C) |
| `outlet_node_setpoint_c` | 40.0 | Outlet node temperature setpoint (°C) |
| `availability_on` | 2.0 | HVAC availability schedule value |

!!! info
    `target_temp_c` and `deadband_c` are stored for annotation and logging
    purposes but do **not** influence the control output.

## Usage

```bash
# Run a baseline rollout with the fan coil constant controller
python scripts/baselines.py policy=fan_coil_constant

# Override fan flow
python scripts/baselines.py policy=fan_coil_constant policy.fan_mass_flow_kg_s=2.0
```

## When to Use

The fan coil constant policy is useful as:

- A **sanity check** to verify the simulation pipeline runs end-to-end.
- A **lower-bound baseline** for reporting relative improvements.
- A **debugging tool** to isolate whether issues come from the controller or
  the environment.
