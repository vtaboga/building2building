# minimal_singlefamilyhouse fixture

Faithful copy of a single real building from the published dataset, used by
the quick test suite so envs build without a HuggingFace download. Regenerate
with `python tests/fixtures/regenerate_minimal_fixtures.py`.

## Provenance

- Dataset: `vtaboga/building2building_dataset` (revision `main`)
- Building type: `SingleFamilyHouse`
- Building ID: `SingleFamilyHouse-0001`
- HVAC type: `unitarysystem` (archetype `Unitary`)
- Climate zone: `None`

`building.epjson`, `equipment.json`, and `metadata.json` are a single
self-consistent pipeline output; `weather.epw` is the building's TMY3 EPW.

## Discovery pins

- `area_m2`: `157.75`
- `warmup_phases`: `1`
- `hvac_actuators`: `2`
