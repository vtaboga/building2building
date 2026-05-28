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

# The committed minimal building fixtures (one per building type) are described
# by a manifest written by ``tests/fixtures/regenerate_minimal_fixtures.py``.
# Reading it here keeps a single source of truth for fixture provenance,
# climate zone, and the per-HVAC-type archetype label.
_MANIFEST_PATH = FIXTURES_DIR / "minimal_fixtures.json"
_MANIFEST: dict[str, dict] = json.loads(_MANIFEST_PATH.read_text())["fixtures"]

_MINIMAL_FIXTURE_DIRS: dict[str, Path] = {
    name: FIXTURES_DIR / name for name in _MANIFEST
}
_HVAC_LABELS: dict[str, str] = {
    name: entry["hvac_label"] for name, entry in _MANIFEST.items()
}

# Representative fixture per HVAC archetype, so tests can ask for "the VAV
# fixture" without pinning a specific building type.
_HVAC_ARCHETYPE_DIR: dict[str, str] = {}
for _name, _entry in _MANIFEST.items():
    _HVAC_ARCHETYPE_DIR.setdefault(_entry["hvac_label"], _name)


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
        if (
            "quick" in item.keywords
            or "long" in item.keywords
            or "release" in item.keywords
        ):
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
    entry: dict

    def _make_info(self, building_type: BuildingType, building_id: str) -> BuildingInfo:
        return BuildingInfo(
            building_id=building_id,
            building_type=building_type,
            source="tests-fixture",
            num_zones=int(self.entry["num_zones"]),
            action_dim=int(self.entry["action_dim"]),
            observation_dim=int(self.entry["observation_dim"]),
            net_conditioned_area_m2=float(self.entry["net_conditioned_area_m2"]),
            warmup_phases=int(self.entry["warmup_phases"]),
            weather_file=str(self.entry["weather_file"]),
            hvac_type=str(self.entry["hvac_type"]),
            building_dir=self.fixture_dir,
            climate_zone=self.entry["climate_zone"],
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
    """Resolve a fixture selector to a committed fixture directory name.

    Accepts a fixture directory name (``minimal_officemedium``), an HVAC
    archetype alias (``vav`` / ``unitary`` / ``heating_only``), or a building
    type (``OfficeMedium``).
    """
    if raw in _MINIMAL_FIXTURE_DIRS:
        return raw
    archetype_aliases = {
        "vav": "VAV",
        "unitary": "Unitary",
        "heating_only": "HeatingOnly",
    }
    if raw in archetype_aliases:
        return _HVAC_ARCHETYPE_DIR[archetype_aliases[raw]]
    for name, entry in _MANIFEST.items():
        if entry["building_type"] == raw:
            return name
    valid = ", ".join(sorted(_MINIMAL_FIXTURE_DIRS))
    raise ValueError(f"Unknown fixture selector {raw!r}. Expected one of: {valid}.")


@pytest.fixture()
def minimal_building_dir(request: pytest.FixtureRequest) -> Path:
    """Return path to a minimal building fixture directory for an HVAC variant."""
    raw_key = cast(str, getattr(request, "param", "minimal_officemedium"))
    key = _resolve_hvac_fixture_key(raw_key)
    path = _MINIMAL_FIXTURE_DIRS[key]
    if not path.exists():
        raise FileNotFoundError(f"Missing fixture directory: {path}")
    return path


@pytest.fixture()
def fixture_registry(minimal_building_dir: Path) -> _FixtureRegistry:
    """Return a registry stub backed by a real fixture directory."""
    key = minimal_building_dir.name
    entry = _MANIFEST.get(key)
    if entry is None:
        raise ValueError(f"Unsupported minimal fixture directory name: {key!r}")
    return _FixtureRegistry(fixture_dir=minimal_building_dir, entry=entry)
