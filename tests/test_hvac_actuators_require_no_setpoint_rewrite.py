from __future__ import annotations

import pytest

from building2building.sources import hydroquebec


def test_search_configs_errors_if_setpoint_rewrite_enabled_for_hvac_actuators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Avoid any expensive work; we only want config validation.
    monkeypatch.setattr(hydroquebec, "search_buildings", lambda **_kw: (_ for _ in ()))

    with pytest.raises(
        ValueError, match=r"include_setpoint_control must be false.*hvac_actuators"
    ):
        hydroquebec.search_configs(
            {
                "env": {
                    "control_mode": "hvac_actuators",
                    "include_setpoint_control": True,
                }
            }
        )


