from __future__ import annotations

from pathlib import Path

from building2building.pipeline import get_sensible_load_actuators
from building2building.sources import hydroquebec


def _count_sensible_load_request_lines(edd_path: Path) -> int:
    """
    Count how many actuator-availability lines in a `.edd` contain "Sensible Load Request".

    We treat this as the ground truth for fixture-based tests.
    """
    count = 0
    with open(edd_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("!"):
                continue
            if "sensible load request" in line.lower():
                count += 1
    return count


def test_get_sensible_load_actuators_from_fixture_edd() -> None:
    """
    Test parsing sensible-load request actuators from the fixture `.edd`.

    The fixture `.edd` is produced from the buildings in `tests/fixtures/`.
    """
    edd_path = Path("tests/fixtures/eplusout.edd")
    assert edd_path.exists(), f"Missing test fixture: {edd_path}"

    actuators = get_sensible_load_actuators(edd_path)
    expected = _count_sensible_load_request_lines(edd_path)

    assert len(actuators) == expected

    for a in actuators:
        assert set(a.keys()) == {"component_name", "component_type", "control_type", "units"}
        assert isinstance(a["component_name"], str)
        assert isinstance(a["component_type"], str)
        assert isinstance(a["control_type"], str)
        assert isinstance(a["units"], str)

        assert "sensible load request" in a["control_type"].lower()


def test_search_configs_supports_sensible_load_control_mode(
    monkeypatch,
) -> None:
    """
    Ensure `hydroquebec.search_configs()` works with env.control_mode="sensible_load".

    This test uses fixtures and monkeypatching to avoid running EnergyPlus.
    """
    import pandas as pd

    fixture_epjson = Path("tests/fixtures/bldg1.epjson")
    fixture_epw = Path("tests/fixtures/weather.epw")
    fixture_edd = Path("tests/fixtures/eplusout.edd")
    fixture_htm = Path("tests/fixtures/eplustbl.htm")

    assert fixture_epjson.exists()
    assert fixture_epw.exists()
    assert fixture_edd.exists()
    assert fixture_htm.exists()

    # Make building search return a single "row" with the same columns the real code expects.
    def _fake_search_buildings(**_kw):
        return pd.DataFrame(
            [
                {
                    "epw_path": str(fixture_epw),
                    # search_configs calls this then passes it to realize(); we keep it simple
                    # and return a Path directly, while monkeypatching realize() to identity.
                    "derivation_thunk": lambda: fixture_epjson,
                }
            ]
        )

    monkeypatch.setattr(hydroquebec, "search_buildings", _fake_search_buildings)

    # Avoid calling the caching/build pipeline; just pass values through.
    monkeypatch.setattr(hydroquebec, "realize", lambda _store, x: x)

    # Avoid running simulations; return fixture output paths directly.
    monkeypatch.setattr(hydroquebec, "eplustbl", lambda _ep, _epjson, _epw: fixture_htm)
    monkeypatch.setattr(hydroquebec, "eddfile", lambda _ep, _epjson, _epw: fixture_edd)

    configs = hydroquebec.search_configs(
        {
            "env": {
                "control_mode": "sensible_load",
                "hvac_action_space": "box",
                "n_bins_continuous": 21,
            }
            ,
            "reward": {
                "reward_type": "BaseRewardConfig",
                "energy_weight": 0.0,
            },
        },
        n=1,
    )

    assert len(configs) == 1
    cfg = configs[0]
    assert isinstance(cfg.hvac_actuators, list)
    assert len(cfg.hvac_actuators) >= 1
    # Must include at least one sensible-load request actuator.
    assert any(
        "sensible load request" in a.get("control_type", "").lower()
        for a in cfg.hvac_actuators
    )
    # Any additional actuators in sensible_load mode must be:
    # - AirLoopHVAC availability override
    # - Zone Temperature Control heating/cooling setpoints (mode enable)
    for a in cfg.hvac_actuators:
        ctrl = a.get("control_type", "").lower()
        if "sensible load request" in ctrl:
            continue
        if a.get("component_type") == "AirLoopHVAC":
            assert a.get("control_type") == "Availability Status"
            continue
        assert a.get("component_type") == "Zone Temperature Control"
        assert a.get("control_type") in ("Heating Setpoint", "Cooling Setpoint")

