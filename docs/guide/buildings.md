# Building Types and Climate Zones

## Building Types

B2B includes 6 building types from ASHRAE 90.1-2022 commercial prototypes and
residential archetypes:

| Type | Zones | HVAC | Description |
|---|---|---|---|
| `SingleFamilyHouse` | 1 | Unitary (heat pump) | Residential single-zone houses |
| `OfficeSmall` | 5 | Unitary | Small office prototype |
| `OfficeMedium` | 15 | VAV air-loop | Medium office with central air handling |
| `RetailStandalone` | 4 | Unitary | Standalone retail store |
| `RestaurantFastFood` | 3 | Unitary | Fast food restaurant |
| `Warehouse` | 3 | Unitary + heating-only (baseboard) | Warehouse prototype |

```python
import building2building as b2b

print(b2b.list_building_types())
# ['SingleFamilyHouse', 'Warehouse', 'RetailStandalone',
#  'RestaurantFastFood', 'OfficeMedium', 'OfficeSmall']
```

## Climate Zones

Commercial buildings are distributed across 16 North American locations
spanning ASHRAE climate zones 1--8, with real TMY3 weather files:

| Zone | Locations |
|---|---|
| 1 | Miami, FL |
| 2 | Tampa, FL; Tucson, AZ |
| 3 | Atlanta, GA; El Paso, TX; San Diego, CA |
| 4 | Albuquerque, NM; New York, NY; Port Angeles, WA; Seattle, WA |
| 5 | Buffalo, NY; Denver, CO |
| 6 | Great Falls, MT; Rochester, MN |
| 7 | International Falls, MN |
| 8 | Fairbanks, AK |

The registry stores the climate zone as an integer 1--8; use
`b2b.list_buildings_by_climate_zone()` and `b2b.get_climate_zone()` to query
it. `SingleFamilyHouse` buildings use per-building Canadian (Québec) weather
files and have no ASHRAE climate-zone assignment -- climate-zone queries for
them raise `ClimateZoneUnavailableError`.

## Parametric Generation

Each building type has **1,000 instances** (6,000 buildings total). Commercial
buildings are generated through Latin Hypercube Sampling (LHS) over
construction parameters, with climate-dependent bounds derived from ASHRAE
90.1-2022:

- Insulation levels (envelope conductivity)
- Window properties (U-factor, SHGC) and window-to-wall ratio
- Building size (geometry scaling)
- Building orientation
- Infiltration rates

`SingleFamilyHouse` instances come from a residential building generator
representing the Québec housing stock.

This parametric variation produces buildings with diverse thermal dynamics
within each archetype, enabling the dynamics adaptation benchmark.

## Dataset

All pre-processed buildings are hosted on HuggingFace at
`vtaboga/building2building_dataset`. Each building type is packaged as a
separate zip archive, enabling partial downloads.

Each building directory contains:

| File | Description |
|---|---|
| `building.epjson` | EnergyPlus model |
| `equipment.json` | Discovered HVAC equipment |
| Weather file (`.epw`) | TMY3 weather data |

## Listing and Selecting Buildings

```python
import building2building as b2b

# List all buildings of a type and split
train_ids = b2b.list_buildings("OfficeSmall", split="train")
test_ids = b2b.list_buildings("OfficeSmall", split="test")
print(f"Train: {len(train_ids)}, Test: {len(test_ids)}")

# Select by index
env = b2b.make_env("OfficeSmall", split="train", index=0)

# Select by explicit ID
env = b2b.make_env("OfficeSmall", building_id=train_ids[5])
```

## Train/Test Splits

Each building type ships with three deterministic, reproducible splits stored
in `splits.json` on the HuggingFace repository:

| Split | Per type | Total | Purpose |
|---|---|---|---|
| `train` | 900 | 5,400 | Training |
| `test` | 100 | 600 | Held-out evaluation |
| `test_small` | 8 | 48 | Fast evaluation / smoke tests (subset of `test`) |

Select a split with the `split=` argument to `list_buildings` and `make_env`
(`"train"`, `"test"`, or `"test_small"`).

### Updating the Small Test Split

Use `baselines/scripts/make_test_small_split.py` to create a reviewable
`splits.json` manifest with an added `test_small` split:

```bash
python baselines/scripts/make_test_small_split.py \
    --output outputs/splits_with_test_small.json \
    --seed 0
```

The script preserves the existing `train` and `test` entries. It selects one
commercial test building per climate zone 1 through 8, then samples 8
`SingleFamilyHouse` test buildings with the provided seed. After reviewing the
printed summary and generated JSON, manually upload
`outputs/splits_with_test_small.json` to the
`vtaboga/building2building_dataset` Hugging Face dataset as `splits.json`.

After upload, refresh the Hugging Face cache if needed and verify:

```python
import building2building as b2b

ids = b2b.list_buildings("OfficeSmall", split="test_small")
assert len(ids) == 8
```
