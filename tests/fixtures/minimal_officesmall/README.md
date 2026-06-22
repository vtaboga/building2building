# minimal_officesmall fixture

Faithful copy of a single real building from the published dataset, used by
the quick test suite so envs build without a HuggingFace download. Regenerate
with `python tests/fixtures/regenerate_minimal_fixtures.py`.

## Provenance

- Dataset: `vtaboga/building2building_dataset` (revision `main`)
- Building type: `OfficeSmall`
- Building ID: `OfficeSmall-5003`
- HVAC type: `unitarysystem` (archetype `Unitary`)
- Climate zone: `3`

`building.epjson`, `equipment.json`, and `metadata.json` are a single
self-consistent pipeline output; `weather.epw` is the building's TMY3 EPW.

## Discovery pins

- `area_m2`: `448.67`
- `warmup_phases`: `3`
- `hvac_actuators`: `10`
