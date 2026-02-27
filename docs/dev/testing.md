# Testing

Building2Building uses **pytest** with two test levels to balance fast
iteration with thorough simulation-based verification.

## Test Levels

| Level   | Marker  | EnergyPlus Required | Typical Runtime |
|---------|---------|---------------------|-----------------|
| Quick   | `quick` | No                  | Seconds         |
| Long    | `long`  | Yes                 | Minutes         |

**Quick** tests cover pure-Python logic (config parsing, reward computation,
observation transformations) and should always pass without an EnergyPlus
installation.

**Long** tests launch full EnergyPlus simulations and verify end-to-end
behaviour. They are gated behind an environment variable so they do not run
by default.

## Running Tests

Run only the quick suite:

```bash
pytest -m quick
```

Run the long (simulation-heavy) suite:

```bash
B2B_RUN_LONG_TESTS=1 pytest -m long
```

Run everything:

```bash
B2B_RUN_LONG_TESTS=1 pytest
```

## Writing New Tests

1. Place test files under `tests/` (or `tests/long/` for long tests).
2. Name files `test_<module>.py` and functions `test_<behaviour>`.
3. Mark each test with `@pytest.mark.quick` or `@pytest.mark.long`.
4. Store any fixture data in `tests/data/`.

## Mocking External Dependencies

Mock EnergyPlus, network calls, and file downloads in quick tests so they
remain fast and self-contained:

```python
from unittest.mock import patch

@pytest.mark.quick
def test_reward_without_simulation(mock_obs):
    with patch("b2b.simulator.energyplus.run") as mock_run:
        mock_run.return_value = mock_obs
        result = compute_reward(mock_obs)
        assert result < 0
```
