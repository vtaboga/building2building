# API Stability and Deprecation Policy

This page documents the **public API contract** of Building2Building and the
project's policy for deprecating and removing symbols.

---

## Public API surface (v0.1.0)

The following names are exported from `building2building` (i.e. accessible as
`import building2building as b2b; b2b.<name>`).  Changes to any of these
require a `CHANGELOG.md` entry and must respect the deprecation window below.

### Environment creation

| Symbol | Description |
|---|---|
| `make_env` | Primary entry point: create a Gymnasium env by building type and index or ID. |
| `make_env_from_config` | Low-level factory: build an env from a fully-specified `EnvBuildConfig`. |
| `list_building_types` | Return the list of supported building-type strings. |
| `list_buildings` | Return metadata for all buildings of a given type. |
| `list_buildings_by_climate_zone` | Filter buildings by ASHRAE climate zone. |
| `get_climate_zone` | Return the climate zone for a given building. |
| `ClimateZoneUnavailableError` | Raised when climate zone data is missing. |
| `TYPES_WITHOUT_CLIMATE_ZONE` | Building types that do not have climate zones. |

### Rollout / trajectory capture

| Symbol | Description |
|---|---|
| `Controller` | Protocol type for deterministic controllers. |
| `Trajectory` | Dataclass holding a complete rollout record. |
| `callable_controller` | Decorator to wrap a callable as a `Controller`. |
| `rollout` | Run a controller for one episode and return a `Trajectory`. |

### Scoring

| Symbol | Description |
|---|---|
| `compute_normalized_score` | Compute the normalized score for a trajectory against pre-computed baselines. |

### Benchmarks

| Symbol | Description |
|---|---|
| `benchmarks` | Module exposing `CrossDomain`, `DynamicsAdaptation`, `GoalAdaptation`, `ActionSpaceTransfer` benchmark classes. |

### Morphology (structured representation)

| Symbol | Description |
|---|---|
| `Morphology` | Graph representation of a building's HVAC topology. |
| `MorphologyEdge` | Edge in the morphology graph. |
| `MorphologyNode` | Node in the morphology graph. |
| `NodeType` | Enum of node types. |
| `build_morphology` | Construct a `Morphology` from a `BuildingConfig`. |
| `ALL_NODE_TYPES` | Set of all `NodeType` values. |
| `CALENDAR`, `ENERGY`, `HEATING_ZONE`, `UNCONTROLLED_ZONE`, `UNITARY_ZONE`, `VAV_SUPPLY`, `VAV_ZONE`, `VAV_ZONE_NO_COOLING`, `WEATHER` | Named `NodeType` constants. |

### Equipment types

| Symbol | Description |
|---|---|
| `HeatingOnlyZone` | HVAC equipment type: heating-only zone. |
| `HeatPump` | HVAC equipment type: heat pump. |
| `UnitarySystem` | HVAC equipment type: unitary system. |
| `VAVSystem` | HVAC equipment type: VAV air-handling unit. |
| `VAVTerminal` | HVAC equipment type: VAV terminal box. |

### Type definitions

| Symbol | Description |
|---|---|
| `ActuatorDescription` | Dataclass describing an EnergyPlus actuator. |
| `BuildingConfig` | Dataclass fully specifying a building environment. |
| `Equipment` | Dataclass representing a piece of HVAC equipment. |
| `NormalizedDeadbandRewardConfig` | The sole supported reward config; parameterised by `energy_weight`, `dT`, `tau_T`, `tau_E`. |
| `RewardConfig` | Type alias for `NormalizedDeadbandRewardConfig`. |
| `TaskConfig` | Dataclass bundling reward config + setpoint mode + run period. |

### Wrappers

| Symbol | Description |
|---|---|
| `AugmentObservationWithBuildingParams` | Appends static building parameters to the observation. |
| `NormalizeObservation` | Normalises observation to `[0, 1]` based on known sensor ranges. |
| `PadObservation` | Zero-pads the observation to a fixed size. |
| `ResampleBuildingOnResetWrapper` | Resamples a new building from a pool on each `reset()`. |
| `wrap_env_for_rl` | Convenience wrapper: applies `NormalizeObservation` + action rescaling. |

---

## Deprecation policy

1. **Deprecation window:** a symbol that is deprecated in version `X.Y.0` is
   removed no earlier than version `X.(Y+2).0` (i.e. at least two minor
   releases after the deprecation notice).  For pre-1.0 releases (0.x) the
   window is at least one minor release.

2. **How to deprecate:** add a `DeprecationWarning` at the call site using
   `warnings.warn(..., DeprecationWarning, stacklevel=2)`, add a `CHANGELOG.md`
   entry under `### Deprecated`, and update this page to mark the symbol as
   deprecated.

3. **Breaking changes:** any removal or incompatible signature change to a
   symbol listed in this page requires:
   - A `CHANGELOG.md` entry under `### Removed` or `### Changed`.
   - A bump of the minor version (or major version if post-1.0).
   - A failing `api_contract` test (`pytest -m api_contract`) in the PR that
     introduces the change — so the breakage is explicit and reviewed.
