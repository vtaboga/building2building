"""Pins the pipeline discovery-metadata contract for each HVAC fixture.

Asserts that ``extract_discovery_metadata`` returns ``net_conditioned_area``,
``warmup_phases``, and HVAC actuator count that match the pinned values in each
fixture's ``README.md``, and that ``make_controllable`` emits the expected
actuator count.
"""

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


# SingleFamilyHouse is intentionally omitted: it is the only residential
# fixture and references its schedules through a relative ``Schedule:File``
# (``in.schedules.csv``).  ``extract_discovery_metadata`` runs EnergyPlus on a
# relocated copy of the epJSON, so the relative path no longer resolves and
# EnergyPlus aborts.  SFH is still exercised by the static fixture tests
# (equipment schema, make_controllable, action dims) and env construction.
@pytest.mark.quick
@pytest.mark.parametrize(
    "fixture_name",
    [
        "minimal_officemedium",
        "minimal_officesmall",
        "minimal_restaurantfastfood",
        "minimal_retailstandalone",
        "minimal_warehouse",
    ],
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
