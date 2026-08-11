# building2building.morphology

Structured environment representation (Section 3.4 of the B2B paper). Defines a
morphological universe of node types with local observation and action spaces,
and provides per-environment morphology graphs with split/join operations.

::: building2building.morphology
    options:
      members:
        - NodeType
        - WEATHER
        - CALENDAR
        - ENERGY
        - UNITARY_ZONE
        - VAV_ZONE
        - VAV_ZONE_NO_COOLING
        - VAV_SUPPLY
        - HEATING_ZONE
        - UNCONTROLLED_ZONE
        - ALL_NODE_TYPES
        - MorphologyNode
        - MorphologyEdge
        - Morphology
        - build_morphology
