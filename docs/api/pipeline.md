# b2b.pipeline

The building preparation pipeline. This module transforms raw IDF building
files into fully controllable EnergyPlus models by upgrading versions,
converting formats, injecting meters, and discovering HVAC equipment.

::: b2b.pipeline
    options:
      members:
        - prepare_building
        - create_complete_pipeline
        - make_controllable
        - extract_discovery_metadata
