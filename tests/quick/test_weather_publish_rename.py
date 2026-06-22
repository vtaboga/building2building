"""Tests for the weather-file publish-rename helper.

The upstream ASHRAE/NREL TMY3 distribution ships the San Diego Brown Field
file with a typo (``San.Deigo``).  The pipeline locates the source file by its
real (mis-spelled) name via ``PLACE_TO_WEATHER`` but PUBLISHES it under the
corrected spelling so the ``metadata.parquet`` ``weather_file`` column and the
per-building ``.epw`` filename always agree.  These tests pin that contract.
"""

from __future__ import annotations

import pytest

from building2building.pipeline.generate_raw_dataset import (
    PLACE_TO_WEATHER,
    published_weather_filename,
)


@pytest.mark.quick
def test_san_deigo_renamed_to_san_diego() -> None:
    source = "USA_CA_San.Deigo-Brown.Field.Muni.AP.722904_TMY3.epw"
    published = published_weather_filename(source)
    assert published == "USA_CA_San.Diego-Brown.Field.Muni.AP.722904_TMY3.epw"
    assert "San.Deigo" not in published


@pytest.mark.quick
def test_no_published_name_contains_deigo_typo() -> None:
    for source_name in PLACE_TO_WEATHER.values():
        assert "San.Deigo" not in published_weather_filename(source_name)


@pytest.mark.quick
def test_non_renamed_name_passes_through() -> None:
    name = "USA_NY_Buffalo.Niagara.Intl.AP.725280_TMY3.epw"
    assert published_weather_filename(name) == name
