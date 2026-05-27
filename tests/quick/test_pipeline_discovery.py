from __future__ import annotations

import re
from pathlib import Path

import pytest

from building2building.env import STORE_PATH
from building2building.pipeline.actuators import make_controllable
from building2building.pipeline.discovery import extract_discovery_metadata
from building2building.store import realize

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"


def _read_pinned_values(readme_path: Path) -> tuple[float, int, int]:
    text = readme_path.read_text()

    area_match = re.search(r"`area_m2`:\s*`([0-9.]+)`", text)
    warmup_match = re.search(r"`warmup_phases`:\s*`([0-9]+)`", text)
    hvac_match = re.search(r"`hvac_actuators`:\s*`([0-9]+)`", text)
    if not area_match or not warmup_match or not hvac_match:
        raise ValueError(f"Missing discovery pins in {readme_path}")

    return float(area_match.group(1)), int(warmup_match.group(1)), int(hvac_match.group(1))


@pytest.mark.quick
@pytest.mark.parametrize(
    "fixture_name",
    ["minimal_vav", "minimal_unitary", "minimal_heating_only"],
)
def test_extract_discovery_metadata_matches_pinned_fixture_values(
    fixture_name: str,
) -> None:
    fixture_dir = FIXTURES_DIR / fixture_name
    expected_area, expected_warmup_phases, expected_hvac_actuators = _read_pinned_values(
        fixture_dir / "README.md"
    )

    metadata = realize(
        STORE_PATH.get(),
        extract_discovery_metadata(
            fixture_dir / "building.epjson",
            fixture_dir / "weather.epw",
            discovery_run_days=1,
        ),
    )
    _, equipment = realize(
        STORE_PATH.get(),
        make_controllable(fixture_dir / "building.epjson"),
    )
    hvac_actuators = sum(len(eq.actuator_descriptions()) for eq in equipment)

    assert metadata.net_conditioned_area == pytest.approx(expected_area, abs=0.01)
    assert metadata.warmup_phases == expected_warmup_phases
    assert hvac_actuators == expected_hvac_actuators
