# Action-Space Shift Benchmark

## Task Definition

Train a policy with one set of actuators and evaluate it with a **different
set** at test time.  This benchmark measures robustness to changes in the
action space — a common scenario when deploying a pre-trained controller to a
building with different HVAC commissioning or access rights.

## Mechanism: `ActuatorAccessConfig`

```python
from building2building.config.models import ActuatorAccessConfig
```

`ActuatorAccessConfig` is a frozen dataclass that controls which actuators are
exposed to the policy:

```python
@dataclass(frozen=True)
class ActuatorAccessConfig:
    include_zone_heating_setpoints: bool = True
```

When `include_zone_heating_setpoints` is set to `False`, all zone-level
heating setpoint actuators are filtered out of the `BuildingConfig` before the
environment is created.  This is applied per-side (train or test) through the
benchmark configuration.

The filtering logic lives in `building2building.benchmark.problem_multizones_splits` and
removes actuators whose component/control types match zone temperature heating
setpoints, VAV thermostat schedule actuators, or any actuator with "heating
setpoint" in its name.

## Reference Configuration: OfficeMedium Actuator Shift

B2B ships a ready-made config for the OfficeMedium actuator-shift benchmark:

```yaml title="configs/benchmark_multizones_officemedium_actuator_shift.yaml (excerpt)"
benchmark_interface:
  mode: single_type
  building_type: OfficeMedium

  train:
    selection:
      mode: random
      n: 3
      seed: 42
    config:
      actuator_access:
        include_zone_heating_setpoints: false   # train WITHOUT heating setpoints

  test:
    selection:
      mode: random
      n: 3
      seed: 7
    config:
      actuator_access:
        include_zone_heating_setpoints: true    # test WITH heating setpoints
```

### Running

```bash
python scripts/benchmark_multizones_splits.py \
  --config-name benchmark_multizones_officemedium_actuator_shift
```

## Custom Actuator-Shift Experiments

You can create actuator-shift benchmarks for any building type by overriding
the `actuator_access` section on each side:

```bash
python scripts/benchmark_multizones_splits.py \
  benchmark_interface=single_type \
  benchmark_interface.building_type=Warehouse \
  benchmark_interface.train.config.actuator_access.include_zone_heating_setpoints=false \
  benchmark_interface.test.config.actuator_access.include_zone_heating_setpoints=true
```

### Reversing the Shift Direction

To study the opposite direction — training **with** heating setpoints and
testing **without** — simply swap the `include_zone_heating_setpoints` values:

```bash
python scripts/benchmark_multizones_splits.py \
  --config-name benchmark_multizones_officemedium_actuator_shift \
  benchmark_interface.train.config.actuator_access.include_zone_heating_setpoints=true \
  benchmark_interface.test.config.actuator_access.include_zone_heating_setpoints=false
```

## Evaluation Protocol

1. **Select** train and test building IDs (seeded, reproducible).
2. **Build** train environments with restricted actuators and test
   environments with the full (or differently restricted) actuator set.
3. **Train** a policy on the restricted action space.
4. **Evaluate** zero-shot on the shifted action space.
5. Report per-building episode return, comfort violations, energy consumption,
   and compare with a baseline trained directly on the test action space.

## How Actuator Filtering Works

Under the hood, `_apply_actuator_access()` in
`building2building.benchmark.problem_multizones_splits` iterates over each equipment
object's actuator descriptions and removes those identified as zone heating
setpoints.  The check covers:

- `component_type == "Zone Temperature Control"` with `control_type == "Heating Setpoint"`
- Schedule actuators containing `"htg setpoint"` in the component name
- Any actuator with `"heating setpoint"` in its name

If filtering removes **all** actuators, a `ValueError` is raised to prevent
silent misconfiguration.
