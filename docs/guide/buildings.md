# Building Types and Climate Zones

## Building Types

B2B includes 6 building types from ASHRAE 90.1-2022 commercial prototypes and
residential archetypes:

| Type | Zones | HVAC | Description |
|---|---|---|---|
| `SingleFamilyHouse` | 1 | Unitary | Residential single-zone houses |
| `OfficeSmall` | 5 | Unitary | Small office prototype |
| `OfficeMedium` | 15+ | VAV air-loop | Medium office with central air handling |
| `RetailStandalone` | 4 | Unitary | Standalone retail store |
| `RestaurantFastFood` | 2 | Unitary | Fast food restaurant |
| `Warehouse` | 3 | Unitary | Warehouse prototype |

```python
import building2building as b2b

print(b2b.list_building_types())
# ['SingleFamilyHouse', 'Warehouse', 'RetailStandalone',
#  'RestaurantFastFood', 'OfficeMedium', 'OfficeSmall']
```

## Climate Zones

Buildings are distributed across 16 ASHRAE climate zones with real TMY3 weather
files:

| Zone | Description | Example City |
|---|---|---|
| 1A | Very Hot, Humid | Miami, FL |
| 2A | Hot, Humid | Houston, TX |
| 2B | Hot, Dry | Phoenix, AZ |
| 3A | Warm, Humid | Atlanta, GA |
| 3B | Warm, Dry | Las Vegas, NV |
| 3C | Warm, Marine | San Francisco, CA |
| 4A | Mixed, Humid | Baltimore, MD |
| 4B | Mixed, Dry | Albuquerque, NM |
| 4C | Mixed, Marine | Seattle, WA |
| 5A | Cool, Humid | Chicago, IL |
| 5B | Cool, Dry | Denver, CO |
| 6A | Cold, Humid | Minneapolis, MN |
| 6B | Cold, Dry | Helena, MT |
| 7 | Very Cold | Duluth, MN |
| 8 | Subarctic | Fairbanks, AK |

## Parametric Generation

Each building type has hundreds to thousands of instances generated through
Latin Hypercube Sampling (LHS) over construction parameters:

- Insulation levels (walls, roof, foundation)
- Window-to-wall ratio
- Building orientation
- HVAC equipment sizing
- Infiltration rates

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

Each building type has a fixed train/test split stored in
`splits.json` on the HuggingFace repository. The splits are deterministic and
reproducible across runs.

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
