from __future__ import annotations

import pytest

from building2building.sources import hydroquebec


def test_search_configs_errors_if_control_mode_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    # Prevent any expensive work: we only want to exercise config validation.
    monkeypatch.setattr(hydroquebec, "search_buildings", lambda **_kw: (_ for _ in ()))

    with pytest.raises(ValueError, match=r"env\.control_mode must be explicitly set"):
        hydroquebec.search_configs({"env": {}})


def test_search_configs_errors_if_only_legacy_hvac_control_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hydroquebec, "search_buildings", lambda **_kw: (_ for _ in ()))

    with pytest.raises(ValueError, match=r"Found legacy key env\.hvac_control_mode"):
        hydroquebec.search_configs({"env": {"hvac_control_mode": "sensible_load_continuous"}})


def test_search_configs_errors_if_thermostat_setpoints_selected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(hydroquebec, "search_buildings", lambda **_kw: (_ for _ in ()))

    with pytest.raises(ValueError, match=r"thermostat_setpoints.*deprecated"):
        hydroquebec.search_configs({"env": {"control_mode": "thermostat_setpoints"}})


