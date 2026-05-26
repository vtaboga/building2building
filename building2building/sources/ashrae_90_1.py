"""ASHRAE 90.1 prototype building source.

Reads the official `ASHRAE901_all.zip` published at
``https://www.energycodes.gov/sites/default/files/2023-10/ASHRAE901_all.zip``,
which packages every prototype IDF (multiple vintages: STD2007, STD2010,
STD2013, STD2016, STD2019, STD2022) across all 16 ASHRAE climate locations,
plus the matching TMY3 EPW files.

This module is the input side of Stage 1 of the dataset generation pipeline
(see ``building2building/pipeline/generate_raw_dataset.py``), which applies
LHS perturbations on top of the STD2022 prototypes to produce
``vtaboga/multizones_reference_buildings.zip``.

Restored from commit ``720ea9a`` ("Add new energycodes.gov data source.",
Sept 2025) which was deleted in the April 2026 refactoring (commit
``1f57775``). Modernized to the current ``store.py`` API: the
``BaseDerivation`` subclass pattern is replaced with the ``@derivation``
decorator, and ``search_buildings`` / ``search_weathers`` no longer
attach a per-row derivation thunk (Stage 1 operates on raw IDF/EPW bytes
and reads from the content-hashed extracted tree directly).
"""

from __future__ import annotations

import re
from pathlib import Path

import duckdb
import pandas as pd

from building2building.env import STORE_PATH
from building2building.store import (
    OUTPUT,
    Derivation,
    DownloadFile,
    ExtractZip,
    derivation,
    realize,
)

# Pinned URL + content hash for the official ASHRAE 90.1 prototype bundle.
# The URL is a stable NREL / energycodes.gov release artefact; the hash
# guarantees that any future re-download fails loud if upstream is
# republished with different content.
_ASHRAE901_URL = (
    "https://www.energycodes.gov/sites/default/files/2023-10/ASHRAE901_all.zip"
)
_ASHRAE901_SHA256 = bytes.fromhex(
    "de35252dada89f6e24f6007e24c2c1796a047c294707f491b65e81cc7ee212ab"
)


def ASHRAE901_all_zip() -> Derivation:
    """Return the (downloaded, content-hash-verified) `ASHRAE901_all.zip`."""
    return DownloadFile(
        "ASHRAE901_all.zip",
        _ASHRAE901_URL,
        _ASHRAE901_SHA256,
    )


def ASHRAE901_all() -> Derivation:
    """Return the extracted-on-disk tree of the prototype bundle."""
    return ExtractZip(ASHRAE901_all_zip())


# Filename pattern: ``ASHRAE901_<BuildingType>_STD<Year>_<Place>.idf``.
# Example: ``ASHRAE901_OfficeMedium_STD2022_Buffalo.idf``.
_IDF_PATTERN = re.compile(r"^ASHRAE901_([^_]+)_STD(\d{4})_([^.]+)\.idf$")

# EPW filename pattern: ``USA_<State>_<rest>.epw``.
# Example: ``USA_NY_Buffalo.Niagara.Intl.AP.725280_TMY3.epw``.
_EPW_PATTERN = re.compile(r"^USA_([A-Z]{2})_(.+)\.epw$")


@derivation("ashrae901_idf_index.parquet")
def _index_buildings(extracted_root: Path) -> None:
    """Walk the extracted ASHRAE901_all tree and emit a parquet index of
    every IDF file recognised by the prototype filename convention.

    Schema: ``(building_type, year, place, path)``.
    """
    out = OUTPUT.get()
    records: list[tuple[str, int, str, str]] = []
    for idf_path in extracted_root.rglob("*.idf"):
        match = _IDF_PATTERN.match(idf_path.name)
        if match is None:
            continue
        building_type, year_str, place = match.groups()
        records.append((building_type, int(year_str), place, str(idf_path)))
    records.sort(key=lambda r: (r[0], r[1], r[2]))
    df = pd.DataFrame(records, columns=["building_type", "year", "place", "path"])
    df.to_parquet(str(out))


@derivation("ashrae901_epw_index.parquet")
def _index_weathers(extracted_root: Path) -> None:
    """Walk the extracted ASHRAE901_all tree and emit a parquet index of
    every EPW file recognised by the ``USA_<state>_<rest>.epw`` convention.

    Schema: ``(state, filename, path)``. The IDF index's ``place`` column
    uses short labels like "Buffalo", "ElPaso", "InternationalFalls", which
    do not parse cleanly out of the EPW basename
    (``El.Paso.Intl.AP.722700_TMY3``). Stage 1 joins (idf -> epw) via an
    explicit place->weather mapping in
    ``building2building.pipeline.generate_raw_dataset.PLACE_TO_WEATHER``
    rather than out of the EPW basename, so this index simply records
    ``filename`` and ``path`` for direct basename lookup.
    """
    out = OUTPUT.get()
    records: list[tuple[str, str, str]] = []
    for epw_path in extracted_root.rglob("*.epw"):
        match = _EPW_PATTERN.match(epw_path.name)
        if match is None:
            continue
        state = match.group(1)
        records.append((state, epw_path.name, str(epw_path)))
    records.sort(key=lambda r: (r[0], r[1]))
    df = pd.DataFrame(records, columns=["state", "filename", "path"])
    df.to_parquet(str(out))


def search_buildings(
    *,
    building_type: str | None = None,
    year: int | None = None,
    place: str | None = None,
) -> pd.DataFrame:
    """Return the IDFs matching ``(building_type, year, place)``.

    Each row carries:
        - ``building_type``, ``year``, ``place``: parsed from the filename
          via :data:`_IDF_PATTERN`.
        - ``path``: absolute filesystem path to the IDF inside the
          extracted, content-hashed store tree (safe to read directly).
    """
    realize(STORE_PATH.get(), ASHRAE901_all())
    idx_path = realize(STORE_PATH.get(), _index_buildings(ASHRAE901_all()))
    rel = duckdb.from_parquet(str(idx_path))
    if building_type is not None:
        rel = rel.filter(
            duckdb.ColumnExpression("building_type")
            == duckdb.ConstantExpression(building_type)
        )
    if year is not None:
        rel = rel.filter(
            duckdb.ColumnExpression("year") == duckdb.ConstantExpression(year)
        )
    if place is not None:
        rel = rel.filter(
            duckdb.ColumnExpression("place") == duckdb.ConstantExpression(place)
        )
    return rel.to_df()


def search_weathers(
    *,
    state: str | None = None,
    filename: str | None = None,
) -> pd.DataFrame:
    """Return the EPWs matching ``(state, filename)``.

    Each row carries:
        - ``state``: 2-letter state code parsed from the filename.
        - ``filename``: the EPW basename.
        - ``path``: absolute filesystem path to the EPW inside the
          extracted, content-hashed store tree.
    """
    realize(STORE_PATH.get(), ASHRAE901_all())
    idx_path = realize(STORE_PATH.get(), _index_weathers(ASHRAE901_all()))
    rel = duckdb.from_parquet(str(idx_path))
    if state is not None:
        rel = rel.filter(
            duckdb.ColumnExpression("state") == duckdb.ConstantExpression(state)
        )
    if filename is not None:
        rel = rel.filter(
            duckdb.ColumnExpression("filename") == duckdb.ConstantExpression(filename)
        )
    return rel.to_df()
