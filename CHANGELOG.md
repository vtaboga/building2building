# Changelog

## Unreleased

### Added

- `sampling` option on `make_env`, `BuildingConfig` and `EnvBuildConfig`,
  reported in `env.metadata["sampling"]`. `"zone_timestep"` makes `step()`
  return once per completed EnergyPlus zone timestep: every step is
  `60 / timesteps_per_hour` minutes, `time_of_day` never goes backwards, and
  the HVAC energy observations cover exactly the step
  (`building2building/simulator/sampling.py`).
- `building2building.bacnet`: a BACnet layer for B2B environments.
  `ChannelMap` describes an environment's observations and actuators as
  role-labelled channels; `PointStore` turns them into BACnet devices,
  `analog-input` sensors and commandable `analog-value` objects with
  priority arrays; `BACnetBuilding` runs an environment with the points in
  the loop (the building's program writes relinquish defaults, clients write
  at priorities 8 to 16); `Gateway`, `WireClient` and `BACnetLink` serve the
  points over BACnet/IP with BACpypes3. Supports unitary-system and
  heating-only equipment (`OfficeSmall`, `RestaurantFastFood`,
  `RetailStandalone`, `Warehouse`, `SingleFamilyHouse`). New `bacnet` extra
  (`bacpypes3`), included in `all`. Guide: `docs/guide/bacnet.md`.
  Use it with `sampling="zone_timestep"`; `BACnetBuilding` warns otherwise.

### Documentation

- Documented that the default `sampling="hvac_timestep"` returns at every
  HVAC system timestep: steps can be shorter than `60 / timesteps_per_hour`
  minutes, `time_of_day` can go backwards, and energy is counted again on
  every extra step. The default is unchanged, so `v1.0.0` results stay
  reproducible.

## v1.0.0 — paper release

Version accompanying *"Building2Building: A Large Scale Benchmark for
Generalizable Real-World Reinforcement Learning"*
([arXiv:2607.16534](https://arxiv.org/abs/2607.16534), Reinforcement Learning
Journal 2026).  All results reported in the paper were produced with this tag.

### Added

- `CITATION.cff` for GitHub's "Cite this repository" button, recording the
  software version alongside the preferred paper citation.
- README arXiv/docs/license badges, an "official implementation" header, a
  reproduction note pinning the paper to `v1.0.0`, and a citation section.

### Changed

- Package version bumped `0.1.0` -> `1.0.0`.
- **Breaking:** the reward API dropped the misleading "deadband" name —
  the reward has no deadband; its comfort term is a plain mean squared
  deviation from target. Renames: `NormalizedDeadbandRewardConfig` ->
  `NormalizedRewardConfig`, `NormalizedDeadbandReward` ->
  `NormalizedReward`, `normalized_deadband_reward_function` ->
  `normalized_reward_function`, `make_normalized_deadband_task` ->
  `make_normalized_task`. The serialized `reward_type` discriminator is
  now `"NormalizedRewardConfig"`; the legacy
  `"NormalizedDeadbandRewardConfig"` string is still accepted by
  `reward_config_from_dict`. The reactive controllers' genuine control
  deadbands (`demand_deadband`, `reheat_sp_deadband`) and the analysis
  scripts' deadband comfort diagnostics are unchanged.

### Removed

- **Breaking:** the vestigial `dT` parameter is gone. It survived an
  earlier reward definition but never entered the current reward: the
  comfort term is a plain mean squared deviation, and `tau_T`/`tau_E`
  calibration ignores it too. Removed from
  `NormalizedRewardConfig`, `NormalizedReward`,
  `normalized_reward_function`,
  `make_normalized_task`, and the serialized dict form
  (`reward_config_from_dict` still accepts and ignores a legacy `dT`
  key). The `dT != 1.0` calibration-mismatch `RuntimeWarning` is gone
  with it; the `mode != "occupancy"` warning is unchanged. No scores,
  normalizer constants, or baselines are affected.

## Unreleased

### Added

- **Documentation site.** MkDocs Material site under `docs/` (user guide,
  benchmark descriptions, baseline guides, tutorials, and an mkdocstrings API
  reference), runnable tutorial scripts under `tutorials/`, hosted at
  <https://building2building.readthedocs.io/> (`.readthedocs.yaml`), with a
  `.github/workflows/docs.yml` CI check that builds with
  `mkdocs build --strict`. Build locally with
  `pip install -e ".[docs]" && mkdocs serve`.

### Changed

- **Task grid is a 3x2 family of six presets** `task_{const,occ,rand}_{e0,e05}`
  (`energy_weight ∈ {0.0, 0.5}`). `e0` is comfort-only; `e05` prices energy
  against comfort.
- **Asymmetric reward normalization.** Comfort is now unnormalized (`tau_T ≡ 1`,
  a raw mean squared deviation from target in °C² — the same physical unit in every
  building, zone and season). Only energy is normalized: `tau_E` is the
  reference reactive controller's per-`(building_type, climate_zone)` energy
  spend, so `power_penalty / tau_E = 1` means "spends like the reference
  controller" and `energy_weight` is a dimensionless price with a fixed meaning
  everywhere. `building2building/data/reward_normalizers.yaml` is produced by
  `baselines.compute_reactive_reward_normalizers`.
- `GoalAdaptation` defaults are now `train_task="task_occ_e05"`,
  `test_task="task_occ_e0"` (the trade-off-transfer axis on the new grid).

### Notes / follow-ups


