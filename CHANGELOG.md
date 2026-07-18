# Changelog

## Unreleased

### Added

- **Documentation site.** MkDocs Material site under `docs/` (user guide,
  benchmark descriptions, baseline guides, tutorials, and an mkdocstrings API
  reference), runnable tutorial scripts under `tutorials/`, and a
  `.github/workflows/docs.yml` workflow that builds with `mkdocs build --strict`
  and deploys to GitHub Pages on pushes to `main`. Build locally with
  `pip install -e ".[docs]" && mkdocs serve`.

### Changed

- **Task grid is a 3x2 family of six presets** `task_{const,occ,rand}_{e0,e05}`
  (`energy_weight ∈ {0.0, 0.5}`). `e0` is comfort-only; `e05` prices energy
  against comfort.
- **Asymmetric reward normalization.** Comfort is now unnormalized (`tau_T ≡ 1`,
  a raw squared out-of-band deviation in °C² — the same physical unit in every
  building, zone and season). Only energy is normalized: `tau_E` is the
  reference reactive controller's per-`(building_type, climate_zone)` energy
  spend, so `power_penalty / tau_E = 1` means "spends like the reference
  controller" and `energy_weight` is a dimensionless price with a fixed meaning
  everywhere. `building2building/data/reward_normalizers.yaml` is produced by
  `baselines.compute_reactive_reward_normalizers`.
- `GoalAdaptation` defaults are now `train_task="task_occ_e05"`,
  `test_task="task_occ_e0"` (the trade-off-transfer axis on the new grid).

### Notes / follow-ups


