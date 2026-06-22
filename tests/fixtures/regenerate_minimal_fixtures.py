"""Regenerate the committed minimal building fixtures from the real dataset.

Each ``minimal_<buildingtype>/`` fixture under ``tests/fixtures/`` is a faithful,
self-consistent copy of a single real building from the published
``vtaboga/building2building_dataset`` HuggingFace dataset. Committing a real
building dir (rather than a hand-rolled one) guarantees that ``building.epjson``
and ``equipment.json`` come from the *same* pipeline run, so the actuator
``component_name`` handles in ``equipment.json`` actually exist in
``building.epjson`` (a mismatch silently breaks real rollouts while passing the
metadata-only quick tests).

The fixture matrix covers one building of every building type, spanning all
three HVAC archetypes (VAV / Unitary / HeatingOnly) and five distinct climate
zones, so the quick suite exercises a representative cross-section of the
dataset without any HuggingFace download at test time.

Run this whenever the dataset is regenerated or the pinned building IDs change::

    python tests/fixtures/regenerate_minimal_fixtures.py

It rewrites, for each fixture: ``building.epjson``, ``equipment.json``,
``metadata.json``, ``weather.epw`` (renamed from the dataset's hashed EPW),
any ``in.schedules.csv`` (residential buildings), a provenance ``README.md``,
and the shared ``minimal_fixtures.json`` manifest that ``conftest.py`` reads.
The buildings are downloaded via the real registry, so a fresh machine fetches
them from HuggingFace on first run.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from cattrs import structure

from building2building.data.download import (
    REPO_ID,
    REVISION,
    BuildingType,
    download_metadata,
    get_building_path,
)
from building2building.pipeline.actuators import AnyEquipment

FIXTURES_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = FIXTURES_DIR / "minimal_fixtures.json"
MANIFEST_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class FixtureSpec:
    """One committed fixture: a building type pinned to a concrete building."""

    dir_name: str
    building_type: BuildingType
    building_id: str


# One building per type, chosen to span every HVAC archetype and to maximise
# climate-zone coverage across the commercial buildings (CZ 1/3/4/5/7).
# SingleFamilyHouse is residential and carries no climate zone.
FIXTURE_SPECS: tuple[FixtureSpec, ...] = (
    FixtureSpec("minimal_officemedium", "OfficeMedium", "OfficeMedium-4001"),
    FixtureSpec("minimal_officesmall", "OfficeSmall", "OfficeSmall-5003"),
    FixtureSpec("minimal_restaurantfastfood", "RestaurantFastFood", "RestaurantFastFood-3001"),
    FixtureSpec("minimal_retailstandalone", "RetailStandalone", "RetailStandalone-2002"),
    FixtureSpec("minimal_warehouse", "Warehouse", "Warehouse-0006"),
    FixtureSpec("minimal_singlefamilyhouse", "SingleFamilyHouse", "SingleFamilyHouse-0001"),
)

# Map the dataset ``hvac_type`` string to the human archetype label that the
# per-HVAC-type contract tests parametrize over.
_HVAC_LABEL_BY_TYPE: dict[str, str] = {
    "vavsystem": "VAV",
    "unitarysystem": "Unitary",
    "heating_only+unitarysystem": "HeatingOnly",
}


def _hvac_label(hvac_type: str) -> str:
    if hvac_type not in _HVAC_LABEL_BY_TYPE:
        raise ValueError(
            f"Unknown hvac_type {hvac_type!r}; expected one of "
            f"{sorted(_HVAC_LABEL_BY_TYPE)}."
        )
    return _HVAC_LABEL_BY_TYPE[hvac_type]


def _count_actuators(equipment_path: Path) -> int:
    """Sum actuator descriptions exactly as the discovery test computes them."""
    equipment = structure(json.loads(equipment_path.read_text()), list[AnyEquipment])
    return sum(len(eq.actuator_descriptions()) for eq in equipment)


def _copy_building_dir(source_dir: Path, dest_dir: Path) -> None:
    """Copy a real building dir into a fixture dir, renaming the EPW.

    Copies ``building.epjson``, ``equipment.json``, ``metadata.json``, the
    single weather EPW (renamed to ``weather.epw`` to match the registry stub
    in ``conftest.py``), and any ``in.schedules.csv`` referenced by residential
    buildings via relative ``Schedule:File`` paths.
    """
    if dest_dir.exists():
        for existing in dest_dir.iterdir():
            if existing.is_file():
                existing.unlink()
    dest_dir.mkdir(parents=True, exist_ok=True)

    for name in ("building.epjson", "equipment.json", "metadata.json"):
        shutil.copy(source_dir / name, dest_dir / name)

    epws = sorted(source_dir.glob("*.epw"))
    if len(epws) != 1:
        raise RuntimeError(f"Expected exactly one EPW in {source_dir}, found {len(epws)}.")
    shutil.copy(epws[0], dest_dir / "weather.epw")

    schedules = source_dir / "in.schedules.csv"
    if schedules.exists():
        shutil.copy(schedules, dest_dir / "in.schedules.csv")


def _write_readme(dest_dir: Path, *, spec: FixtureSpec, entry: dict) -> None:
    lines = [
        f"# {spec.dir_name} fixture",
        "",
        "Faithful copy of a single real building from the published dataset, used by",
        "the quick test suite so envs build without a HuggingFace download. Regenerate",
        "with `python tests/fixtures/regenerate_minimal_fixtures.py`.",
        "",
        "## Provenance",
        "",
        f"- Dataset: `{REPO_ID}` (revision `{REVISION}`)",
        f"- Building type: `{spec.building_type}`",
        f"- Building ID: `{spec.building_id}`",
        f"- HVAC type: `{entry['hvac_type']}` (archetype `{entry['hvac_label']}`)",
        f"- Climate zone: `{entry['climate_zone']}`",
        "",
        "`building.epjson`, `equipment.json`, and `metadata.json` are a single",
        "self-consistent pipeline output; `weather.epw` is the building's TMY3 EPW.",
        "",
        "## Discovery pins",
        "",
        f"- `area_m2`: `{entry['net_conditioned_area_m2']}`",
        f"- `warmup_phases`: `{entry['warmup_phases']}`",
        f"- `hvac_actuators`: `{entry['hvac_actuators']}`",
        "",
    ]
    (dest_dir / "README.md").write_text("\n".join(lines))


def main() -> None:
    metadata = pd.read_parquet(download_metadata())
    manifest: dict[str, object] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset_repo": REPO_ID,
        "dataset_revision": REVISION,
        "fixtures": {},
    }
    fixtures: dict[str, dict] = manifest["fixtures"]  # type: ignore[assignment]

    for spec in FIXTURE_SPECS:
        source_dir = get_building_path(spec.building_type, spec.building_id)
        dest_dir = FIXTURES_DIR / spec.dir_name
        _copy_building_dir(source_dir, dest_dir)

        row = metadata[metadata["building_id"] == spec.building_id]
        if row.empty:
            raise KeyError(f"{spec.building_id!r} not present in metadata.parquet")
        r = row.iloc[0]
        cz_raw = r.get("climate_zone")
        climate_zone = None if pd.isna(cz_raw) else int(cz_raw)
        hvac_type = str(r["hvac_type"])

        local_meta = json.loads((dest_dir / "metadata.json").read_text())
        entry = {
            "building_type": spec.building_type,
            "building_id": spec.building_id,
            "hvac_type": hvac_type,
            "hvac_label": _hvac_label(hvac_type),
            "climate_zone": climate_zone,
            "weather_file": "weather.epw",
            "net_conditioned_area_m2": float(local_meta["net_conditioned_area"]),
            "warmup_phases": int(local_meta["warmup_phases"]),
            "num_zones": int(r["num_zones"]),
            "action_dim": int(r["action_dim"]),
            "observation_dim": int(r["observation_dim"]),
            "hvac_actuators": _count_actuators(dest_dir / "equipment.json"),
        }
        fixtures[spec.dir_name] = entry
        _write_readme(dest_dir, spec=spec, entry=entry)
        print(
            f"{spec.dir_name:30s} {spec.building_id:24s} "
            f"hvac={entry['hvac_label']:11s} cz={entry['climate_zone']} "
            f"area={entry['net_conditioned_area_m2']:.2f} "
            f"actuators={entry['hvac_actuators']}"
        )

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nWrote manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
