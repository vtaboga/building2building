# minimal_vav fixture (bootstrap)

Bootstrap fixture directory used to start T0 migration work.

- HVAC type: `VAV`
- Files are intentionally minimal while shared registry wiring lands.
- `building.epjson` currently comes from the existing committed fixture.
- `equipment.json` is `[]` for API-level tests that patch simulator creation.
