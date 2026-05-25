# Action Space

## Structure

Actions are flat `numpy` arrays of continuous values. Each element controls one
HVAC actuator. The action dimension varies by building type and HVAC system.

## Actions by HVAC System Type

=== "Unitary"

    Each zone has 2 actuators:

    | Actuator | Range | Unit |
    |---|---|---|
    | Fan air mass flow rate | `[0, design_max]` | kg/s |
    | Supply air temperature setpoint | `[12, 50]` | C |

    For an `OfficeSmall` with 5 zones: action dim = 10.

=== "VAV"

    Central supply + per-zone terminals:

    | Actuator | Range | Unit |
    |---|---|---|
    | Supply air temperature setpoint (central) | `[12, 50]` | C |
    | Damper position (per zone) | `[0, 1]` | fraction |
    | Heating setpoint (per zone) | `[15, 30]` | C |
    | Cooling setpoint (per zone) | `[15, 30]` | C |

    For `OfficeMedium` with ~10 VAV zones: action dim ~ 33.

=== "Heating-Only"

    Each zone has 1 actuator:

    | Actuator | Range | Unit |
    |---|---|---|
    | Heating setpoint | `[15, 30]` | C |

## Action Dimensions by Building Type

| Building Type | HVAC | Typical Action Dim |
|---|---|---|
| `SingleFamilyHouse` | Unitary | 2 |
| `OfficeSmall` | Unitary | 10 |
| `OfficeMedium` | VAV | ~33 |
| `RetailStandalone` | Unitary | 8 |
| `RestaurantFastFood` | Unitary | 4 |
| `Warehouse` | Unitary | 6 |

## Action Names

Each action channel has a human-readable name:

```python
import building2building as b2b

env = b2b.new_make_env("OfficeSmall", task="task_const_e0")
for i, name in enumerate(env.metadata["action_names"]):
    print(f"  [{i}] {name}")
env.close()
```

## Action Space Bounds

The action space is a Gymnasium `Box` with per-actuator low/high bounds derived
from the HVAC equipment specifications:

```python
env = b2b.new_make_env("OfficeSmall", task="task_const_e0")
print(env.action_space)         # Box(low, high, shape=(10,))
print(env.action_space.low)     # per-actuator lower bounds
print(env.action_space.high)    # per-actuator upper bounds
env.close()
```

## Morphology-Based Actions

For policies that operate on per-node local action spaces (e.g. the Amorpheus
transformer), the morphology graph provides `join_actions()`:

```python
morph = env.metadata["morphology"]
actions_dict = {
    node.node_id: np.zeros(node.node_type.action_dim)
    for node in morph.nodes
    if node.node_type.action_dim > 0
}
flat_action = morph.join_actions(actions_dict)
```

See [Morphology Graph](morphology.md) for details.
