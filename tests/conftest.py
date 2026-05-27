from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, cast

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from building2building.data.download import BuildingType
from building2building.data.registry import BuildingInfo


FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
FAKE_DATASET_DIR = FIXTURES_DIR / "fake_dataset"
_MINIMAL_FIXTURE_DIRS: dict[str, Path] = {
    "minimal_vav": FIXTURES_DIR / "minimal_vav",
    "minimal_unitary": FIXTURES_DIR / "minimal_unitary",
    "minimal_heating_only": FIXTURES_DIR / "minimal_heating_only",
}
_HVAC_LABELS: dict[str, str] = {
    "minimal_vav": "VAV",
    "minimal_unitary": "Unitary",
    "minimal_heating_only": "HeatingOnly",
}


try:
    from building2building.env import setup_energyplus_path

    setup_energyplus_path()
except ModuleNotFoundError:
    pass


def pytest_collection_modifyitems(items: List[pytest.Item]) -> None:
    _API_CONTRACT_GLOBS = {
        "test_api.py",
        "test_api_mode_default.py",
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


@dataclass(frozen=True)
class _FixtureRegistry:
    fixture_dir: Path
    hvac_type: str

    def _make_info(self, building_type: BuildingType, building_id: str) -> BuildingInfo:
        return BuildingInfo(
            building_id=building_id,
            building_type=building_type,
            source="tests-fixture",
            num_zones=1,
            action_dim=0,
            observation_dim=1,
            net_conditioned_area_m2=100.0,
            warmup_phases=1,
            weather_file="weather.epw",
            hvac_type=self.hvac_type,
            building_dir=self.fixture_dir,
            climate_zone=5,
        )

    def get_building_by_index(
        self, building_type: BuildingType, _split: str, _index: int
    ) -> BuildingInfo:
        return self._make_info(building_type=building_type, building_id="fixture-0001")

    def get_building_by_id(
        self, building_type: BuildingType, building_id: str
    ) -> BuildingInfo:
        return self._make_info(building_type=building_type, building_id=building_id)


def _resolve_hvac_fixture_key(raw: str) -> str:
    aliases = {
        "vav": "minimal_vav",
        "unitary": "minimal_unitary",
        "heating_only": "minimal_heating_only",
    }
    key = aliases.get(raw, raw)
    if key not in _MINIMAL_FIXTURE_DIRS:
        valid = ", ".join(sorted(_MINIMAL_FIXTURE_DIRS))
        raise ValueError(f"Unknown hvac fixture key {raw!r}. Expected one of: {valid}.")
    return key


@pytest.fixture()
def minimal_building_dir(request: pytest.FixtureRequest) -> Path:
    """Return path to a minimal building fixture directory for an HVAC variant."""
    raw_key = cast(str, getattr(request, "param", "minimal_vav"))
    key = _resolve_hvac_fixture_key(raw_key)
    path = _MINIMAL_FIXTURE_DIRS[key]
    if not path.exists():
        raise FileNotFoundError(f"Missing fixture directory: {path}")
    return path


@pytest.fixture()
def fixture_registry(minimal_building_dir: Path) -> _FixtureRegistry:
    """Return a registry stub backed by a real fixture directory."""
    key = minimal_building_dir.name
    hvac_type = _HVAC_LABELS.get(key)
    if hvac_type is None:
        raise ValueError(f"Unsupported minimal fixture directory name: {key!r}")
    return _FixtureRegistry(fixture_dir=minimal_building_dir, hvac_type=hvac_type)
