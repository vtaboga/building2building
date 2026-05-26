from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
FAKE_DATASET_DIR = FIXTURES_DIR / "fake_dataset"


try:
    from building2building.env import setup_energyplus_path

    setup_energyplus_path()
except ModuleNotFoundError:
    pass


def pytest_collection_modifyitems(items: List[pytest.Item]) -> None:
    _API_CONTRACT_GLOBS = {
        "test_api.py",
        "test_api_mode_default.py",
        "test_new_api.py",
        "test_gym_registration.py",
        "test_climate_zones.py",
        "test_data_registry.py",
        "test_selection_and_env_creation.py",
        "test_types.py",
    }
    api_contract_marker = pytest.mark.api_contract
    for item in items:
        if Path(item.fspath).name in _API_CONTRACT_GLOBS:
            item.add_marker(api_contract_marker)
        if "quick" in item.keywords or "long" in item.keywords:
            continue
        item.add_marker(pytest.mark.quick)


@pytest.fixture()
def fake_dataset_dir() -> Path:
    """Return path to the fake dataset fixture directory."""
    return FAKE_DATASET_DIR


@pytest.fixture()
def fake_metadata(fake_dataset_dir: Path) -> pd.DataFrame:
    """Load the fake metadata parquet as a DataFrame."""
    return pd.read_parquet(fake_dataset_dir / "metadata.parquet")


@pytest.fixture()
def fake_splits(fake_dataset_dir: Path) -> dict[str, dict[str, list[str]]]:
    """Load the fake splits.json."""
    return json.loads((fake_dataset_dir / "splits.json").read_text())


@pytest.fixture()
def baseline_csv_path() -> Path:
    """Return path to the baseline_returns fixture CSV."""
    return FIXTURES_DIR / "baseline_returns_fixture.csv"
