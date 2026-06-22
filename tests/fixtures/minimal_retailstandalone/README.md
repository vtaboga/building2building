# minimal_retailstandalone fixture

Faithful copy of a single real building from the published dataset, used by
the quick test suite so envs build without a HuggingFace download. Regenerate
with `python tests/fixtures/regenerate_minimal_fixtures.py`.

## Provenance

- Dataset: `vtaboga/building2building_dataset` (revision `main`)
- Building type: `RetailStandalone`
- Building ID: `RetailStandalone-2002`
- HVAC type: `heating_only+unitarysystem` (archetype `HeatingOnly`)
- Climate zone: `7`

`building.epjson`, `equipment.json`, and `metadata.json` are a single
self-consistent pipeline output; `weather.epw` is the building's TMY3 EPW.

## Discovery pins

- `area_m2`: `3361.89`
- `warmup_phases`: `3`
- `hvac_actuators`: `9`
