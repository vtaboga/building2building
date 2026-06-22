# Release-tier tests

`tests/release/` contains dataset and artifact integrity checks that are intended
for release-candidate validation, not per-push development loops.

Current tests:

- `test_data_integrity.py`: validates split/metadata consistency invariants.
- `test_reward_normalizers_coverage.py`: ensures normalizer buckets cover the
  metadata climate-zone grid.
- `test_baseline_returns_coverage.py`: ensures baseline returns cover the paper
  evaluation grid.

Run manually with:

```bash
pytest -m release
```

CI automation for this tier is tracked as deferred TODO item `TZ1`.
