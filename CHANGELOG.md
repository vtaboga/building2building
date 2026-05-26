# Changelog

All notable changes to Building2Building are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- MIT LICENSE.
- `CHANGELOG.md` and `docs/api/stability.md` (API stability + deprecation policy).
- GitHub Actions CI workflow (`.github/workflows/test.yml`).

---

## [0.1.0] — 2026-06-01 (planned OSS release)

Initial open-source release accompanying the RLJ/RLC paper
*Building2Building: A Large Scale Benchmark for Generalizable Real-World
Reinforcement Learning* (Taboga, Veilleux, Jang, Rankawat, Bacon).

### Added
- 7,000+ EnergyPlus building environments via `new_make_env` / `make_env`.
- Six building types: `SingleFamilyHouse`, `OfficeSmall`, `OfficeMedium`,
  `RetailStandalone`, `RestaurantFastFood`, `Warehouse`.
- Three HVAC system types: VAV, Unitary, HeatingOnly.
- Normalized deadband reward family (`NormalizedDeadbandRewardConfig`) with a
  3 × 3 task-preset grid (`task_<mode>_<level>` where `mode ∈ {const, occ,
  rand}` and `level ∈ {e0, emed, ehigh}`).
- Four benchmark problems: `CrossDomain`, `DynamicsAdaptation`,
  `GoalAdaptation`, `ActionSpaceTransfer`.
- Morphology graph (`build_morphology`) for structured observation/action
  decomposition.
- Normalized scoring against reactive-controller baselines
  (`compute_normalized_score`).
- Pre-processed building dataset on HuggingFace
  (`vtaboga/building2building_dataset`), downloaded automatically on first use.
- SB3-compatible RL wrappers: `NormalizeObservation`, `PadObservation`,
  `AugmentObservationWithBuildingParams`, `ResampleBuildingOnResetWrapper`,
  `wrap_env_for_rl`.
- PPO and SAC specialist training entry points under `baselines/`.
- Reactive baseline controller (`baselines/run_reactive_control.py`).
- `REPRODUCING.md` with full reproduction instructions.

### Changed
- `OfficeMedium` action space: OA-mixer actuator added per loop (Phase M fix).

---

[Unreleased]: https://github.com/vtaboga/Building2Building/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/vtaboga/Building2Building/releases/tag/v0.1.0
