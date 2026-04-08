# Morphology Graph

## Overview

The morphology graph provides a structured decomposition of the flat
observation and action vectors into per-node local spaces. Each node has a
**type** drawn from a fixed universe and maps to specific indices in the flat
vectors. This representation enables type-heterogeneous policies like the
Amorpheus transformer.

## Node Types

The morphological universe is fixed across all B2B environments:

| Node Type | Obs Dim | Act Dim | Description |
|---|---|---|---|
| `weather` | 2 | 0 | Outdoor temperature, humidity |
| `calendar` | 3 | 0 | Time of day, day of week, day of year |
| `energy` | 2 | 0 | HVAC electricity, gas |
| `unitary_zone` | 1 | 2 | Zone temp; fan flow + SAT setpoint |
| `vav_zone` | 1 | 3 | Zone temp; damper + heating/cooling SP |
| `vav_supply` | 0 | 1 | Central SAT setpoint |
| `heating_zone` | 1 | 1 | Zone temp; heating setpoint |
| `uncontrolled_zone` | 1 | 0 | Zone temp (no actuators) |

## Accessing the Morphology

```python
import building2building as b2b

env = b2b.new_make_env("OfficeSmall", task="task1")
morph = env.metadata["morphology"]

print(f"Nodes: {len(morph.nodes)}")
print(f"Edges: {len(morph.edges)}")
print(f"Type counts: {morph.type_counts()}")

for node in morph.nodes:
    print(f"  {node.node_id}: {node.node_type.name} "
          f"(obs={node.obs_indices}, act={node.action_indices})")
env.close()
```

## Splitting Observations

Convert a flat observation into per-node local observations:

```python
obs, _ = env.reset()
local_obs = morph.split_observation(obs)
# dict[str, np.ndarray] keyed by node_id

for node_id, local in local_obs.items():
    print(f"  {node_id}: shape={local.shape}")
```

## Joining Actions

Convert per-node local actions back to a flat action vector:

```python
import numpy as np

actions_dict = {}
for node in morph.nodes:
    if node.node_type.action_dim > 0:
        actions_dict[node.node_id] = np.zeros(node.node_type.action_dim)

flat_action = morph.join_actions(actions_dict)
# flat_action.shape matches env.action_space.shape
```

## Graph Structure

The morphology is a directed graph where edges connect related nodes (e.g. a
VAV supply node connects to its VAV zone nodes). Each `MorphologyEdge` has
a `source` and `target` node ID.

```python
for edge in morph.edges:
    print(f"  {edge.source} -> {edge.target}")
```

## Building the Morphology

For the Amorpheus cross-domain policy, you can build a morphology from
equipment descriptions:

```python
from building2building.morphology import build_morphology

morph = build_morphology(
    hvac_equipment=env.metadata["hvac_equipment"],
    observation_names=env.metadata["observation_names"],
    action_names=env.metadata["action_names"],
)
```

## Use Cases

- **Amorpheus transformer**: per-node-type encoders/decoders with a shared
  transformer backbone. The morphology defines the input/output mapping for each
  node.
- **Transfer learning**: the fixed node type universe allows a policy trained on
  one building type to be applied to another, since the local spaces are
  type-consistent.
- **Visualization**: the graph structure reveals the HVAC topology of a building.
