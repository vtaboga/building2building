# Adding New Building Types

This guide explains how to integrate a new building type into
Building2Building so it can be selected, prepared, and used in environments.

## 1. Register the Building Type

Building types are defined as a `Literal` union in `b2b/config/models.py`.
Add your new type to the `BuildingType` literal:

```python
BuildingType = Literal[
    "office_small",
    "office_medium",
    ...,
    "your_new_type",
]
```

## 2. Prepare the Dataset

Each building type requires:

| Artefact       | Description                         |
|----------------|-------------------------------------|
| IDF file       | EnergyPlus input definition         |
| EPW files      | One or more weather files           |
| Metadata JSON  | Zone names, floor areas, schedules  |

Place these files according to the existing dataset directory layout so the
selection helpers in `building2building.datasets` can discover them.

## 3. Run the Pipeline

The preparation pipeline transforms a raw IDF into a controllable model:

```
IDF → version upgrade → format conversion → meter injection → make controllable
```

Use the pipeline entry point:

```python
from building2building.pipeline import prepare_building

prepare_building(building_type="your_new_type", ...)
```

The pipeline calls `make_controllable`, which discovers HVAC equipment via
`building2building.pipeline.actuators` and injects the required EnergyPlus actuators.

## 4. Integration with the Selection System

Once the dataset files are in place and the pipeline has run, the new building
type will be available through:

- `building2building.datasets` listing and filtering functions
- `building2building.config.models.EnvConfig` with `building_type="your_new_type"`
- Benchmark split definitions in `building2building.benchmark.problem_multizones_splits`

## Checklist

- [ ] `BuildingType` literal updated.
- [ ] IDF, EPW, and metadata files added to the dataset.
- [ ] Pipeline runs without errors for the new type.
- [ ] At least one quick test and one long test added.
