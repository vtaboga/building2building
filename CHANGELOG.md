# Changelog

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


