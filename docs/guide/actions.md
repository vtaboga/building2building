# Action Space

## Structure

Actions are flat `numpy` arrays of continuous values. Each element controls one
HVAC actuator. The action dimension varies by building type and HVAC system.

## Actions by HVAC System Type

=== "Unitary"

    Each unitary system (one per conditioned zone) has 2 actuators:

    | Actuator | Range | Unit |
    |---|---|---|
    | Fan air mass flow rate | `[0, design_max]` (fallback max 15) | kg/s |
    | Supply air temperature setpoint | `[5, design_max]` (fallback max 60) | C |

    For an `OfficeSmall` with 5 conditioned zones: action dim = 10.

=== "VAV"

    Central supply (per air loop) + per-zone terminals:

    | Actuator | Range | Unit |
    |---|---|---|
    | Supply air temperature setpoint (per air loop) | `[10, 55]` | C |
    | Outdoor-air mass flow rate (per air loop) | `[0, 5]` | kg/s |
    | Minimum-air-flow (damper) fraction (per zone) | `[0, 1]` | fraction |
    | Heating setpoint (per zone) | `[10, 35]` | C |

    Per-zone VAV *cooling* setpoints are pinned to a fixed 40 C for
    simulation stability and removed from the agent-facing action space.

    For `OfficeMedium` (3 air loops, 15 VAV zones): action dim = 36.

=== "Heating-Only"

    Each heating-only zone (e.g. baseboard heating) has 1 actuator:

    | Actuator | Range | Unit |
    |---|---|---|
    | Heating setpoint | `[10, 35]` | C |

## Action Dimensions by Building Type

| Building Type | HVAC | Typical Action Dim |
|---|---|---|
| `SingleFamilyHouse` | Unitary | 2 |
| `OfficeSmall` | Unitary | 10 |
| `OfficeMedium` | VAV | 36 |
| `RetailStandalone` | Unitary + heating-only | 9 |
| `RestaurantFastFood` | Unitary | 4 |
| `Warehouse` | Unitary + heating-only | 5 |

## Action Names

Each action channel has a human-readable name:

```python
import building2building as b2b

env = b2b.make_env("OfficeSmall", task="task_const_e0")
for i, name in enumerate(env.metadata["action_names"]):
    print(f"  [{i}] {name}")
env.close()
```

## Action Space Bounds

The action space is a Gymnasium `Box` with per-actuator low/high bounds derived
from the HVAC equipment specifications:

```python
env = b2b.make_env("OfficeSmall", task="task_const_e0")
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
