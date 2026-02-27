# Adding New HVAC Systems

This guide walks through adding support for a new HVAC equipment type in
Building2Building.

## 1. Implement the Equipment Protocol

All HVAC equipment classes must satisfy the `Equipment` protocol defined in
`b2b.types`:

```python
from b2b.types import Equipment, ActuatorDescription

class MyNewEquipment:
    """Equipment implementation for <describe system>."""

    def actuator_descriptions(self) -> list[ActuatorDescription]:
        """Return the EnergyPlus actuators this equipment exposes."""
        return [
            ActuatorDescription(
                component_type="...",
                control_type="...",
                actuator_key="...",
                minimum=...,
                maximum=...,
            ),
        ]

    def zones(self) -> list[str]:
        """Return the thermal zone names served by this equipment."""
        return [...]
```

The `ActuatorDescription` dataclass specifies the EnergyPlus actuator
component, control type, key, and the valid bounds for the control signal.

## 2. Add Discovery Logic

Equipment is discovered automatically from parsed IDF objects. Add a
detection function in `b2b/pipeline/actuators.py`:

```python
def _discover_my_new_equipment(idf) -> list[MyNewEquipment]:
    ...
```

The function should inspect the IDF data structure, find relevant objects,
and return a list of `MyNewEquipment` instances.

## 3. Register in `make_all_equipment`

Wire the discovery function into the central equipment registry in
`b2b/pipeline/actuators.py`:

```python
def make_all_equipment(idf) -> list[Equipment]:
    equipment: list[Equipment] = []
    equipment.extend(_discover_existing_type(idf))
    equipment.extend(_discover_my_new_equipment(idf))  # <-- add here
    return equipment
```

## 4. Define Correct Bounds

Choose actuator bounds carefully:

- `minimum` and `maximum` should reflect physically meaningful ranges
  (e.g., supply air temperature 12–50 °C).
- Bounds that are too wide can cause EnergyPlus warnings or divergence.
- Check the EnergyPlus *Input Output Reference* for the relevant actuator
  to confirm valid ranges.

## Checklist

- [ ] Class satisfies the `Equipment` protocol.
- [ ] Discovery function added to `b2b/pipeline/actuators.py`.
- [ ] Registered in `make_all_equipment`.
- [ ] Bounds validated against EnergyPlus documentation.
- [ ] Quick test for the discovery function with a fixture IDF.
- [ ] Long test running a short simulation with the new equipment.
