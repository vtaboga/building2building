from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from building2building.pipeline import (
    get_hvac_actuators,
    get_net_conditioned_area,
    get_sensible_load_actuators,
    get_zone_temperature_control_actuators,
)
from building2building.simulator import create_simulator
from building2building.types import BaseRewardConfig, BuildingConfig


def _find_action_index(action_names: list[str], component_type: str, control_type: str) -> int:
    ct = component_type.strip().lower()
    ctrl = control_type.strip().lower()
    for i, name in enumerate(action_names):
        parts = str(name).split("::")
        if len(parts) < 2:
            continue
        if parts[0].strip().lower() == ct and parts[1].strip().lower() == ctrl:
            return int(i)
    raise RuntimeError(f"Could not find action index for {component_type}::{control_type}")


def _find_first_zone_air_temp_obs_index(observation_names: list[str]) -> int:
    for i, name in enumerate(observation_names):
        if str(name).lower().startswith("zone air temperature"):
            return int(i)
    for i, name in enumerate(observation_names):
        if "zone air temperature" in str(name).lower():
            return int(i)
    raise RuntimeError("Could not find a zone air temperature observation")


def _build_sensible_load_actuator_list(edd_path: Path) -> list[dict[str, str]]:
    # Match the same composition used in hydroquebec.search_configs for sensible_load:
    # Zone temp control setpoints + airloop availability + sensible load request.
    zone_setpoints = get_zone_temperature_control_actuators(edd_path)
    hvac = get_hvac_actuators(edd_path)
    availability = [
        a
        for a in hvac
        if a.get("component_type") == "AirLoopHVAC"
        and a.get("control_type") == "Availability Status"
    ]
    load = get_sensible_load_actuators(edd_path)
    return zone_setpoints + availability + load


def _run_rollout(
    *,
    env,
    n_steps: int,
    idx_tz: int,
    idx_heat_sp: int,
    idx_cool_sp: int,
    idx_avail: int,
    idx_load: int,
    mode: str,
    q_w: float,
) -> list[float]:
    temps: list[float] = []
    obs, _ = env.reset()

    for _ in range(int(n_steps)):
        tz = float(np.asarray(obs, dtype=float).reshape(-1)[idx_tz])
        temps.append(tz)

        action = np.zeros((env.action_space.shape[0],), dtype=float)

        # Always force air loop on.
        action[idx_avail] = 2.0

        # Setpoint enabler (same idea as scripts/ha.py): nudge setpoints around the
        # *current* zone temperature so EnergyPlus is consistently in the desired mode.
        if mode == "heat":
            action[idx_heat_sp] = tz + 0.5
            action[idx_cool_sp] = tz + 100.0
            action[idx_load] = float(q_w)
        elif mode == "cool":
            action[idx_heat_sp] = tz - 100.0
            action[idx_cool_sp] = tz - 0.5
            action[idx_load] = -float(q_w)
        else:
            raise ValueError("mode must be 'heat' or 'cool'")

        obs, _reward, terminated, truncated, _info = env.step(action)
        if bool(terminated or truncated):
            break

    return temps


def test_sensible_load_drives_temperature_more_than_setpoint_enabler_only() -> None:
    """
    Compare two simulations:

    1) Only "setpoint enabling" (±0.5C around initial temperature) + AirLoopHVAC availability on,
       with load request forced to 0 W.
    2) Same setpoint enabling + availability on, but with a large sensible load request (W).

    We assert that case (2) changes zone temperature substantially more than case (1),
    demonstrating that the sensible load request (not the tiny setpoint change) is driving
    the thermal response.
    """
    epjson_path = Path("tests/fixtures/bldg1.epjson").resolve()
    epw_path = Path("tests/fixtures/weather.epw").resolve()
    edd_path = Path("tests/fixtures/eplusout.edd").resolve()
    htm_path = Path("tests/fixtures/eplustbl.htm").resolve()

    assert epjson_path.exists()
    assert epw_path.exists()
    assert edd_path.exists()
    assert htm_path.exists()

    hvac_actuators = _build_sensible_load_actuator_list(edd_path)
    area = get_net_conditioned_area(htm_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        eplus_output_dir = Path(tmpdir)
        cfg = BuildingConfig(
            path_to_building=epjson_path,
            path_to_weather=epw_path,
            reward_config=BaseRewardConfig(energy_weight=0.0),
            hvac_actuators=hvac_actuators,
            eplus_output_dir=eplus_output_dir,
            warmup_phases=0,
            area=area,
            hvac_action_space="box",
            n_bins_continuous=21,
        )
        env = create_simulator(cfg)

        obs_names = env.metadata.get("observation_names")
        act_names = env.metadata.get("action_names")
        assert isinstance(obs_names, list)
        assert isinstance(act_names, list)

        idx_tz = _find_first_zone_air_temp_obs_index([str(x) for x in obs_names])
        idx_heat_sp = _find_action_index([str(x) for x in act_names], "Zone Temperature Control", "Heating Setpoint")
        idx_cool_sp = _find_action_index([str(x) for x in act_names], "Zone Temperature Control", "Cooling Setpoint")
        idx_avail = _find_action_index([str(x) for x in act_names], "AirLoopHVAC", "Availability Status")
        idx_load = _find_action_index([str(x) for x in act_names], "Unitary HVAC", "Sensible Load Request")

        # Case A: Enabler only (0 W requested)
        temps_enabler = _run_rollout(
            env=env,
            n_steps=80,
            idx_tz=idx_tz,
            idx_heat_sp=idx_heat_sp,
            idx_cool_sp=idx_cool_sp,
            idx_avail=idx_avail,
            idx_load=idx_load,
            mode="heat",
            q_w=0.0,
        )

        # Case B: Enabler + large heating request
        temps_load = _run_rollout(
            env=env,
            n_steps=80,
            idx_tz=idx_tz,
            idx_heat_sp=idx_heat_sp,
            idx_cool_sp=idx_cool_sp,
            idx_avail=idx_avail,
            idx_load=idx_load,
            mode="heat",
            q_w=20000.0,
        )

        # Basic sanity
        assert len(temps_enabler) > 5
        assert len(temps_load) > 5

        delta_enabler = float(max(temps_enabler) - temps_enabler[0])
        delta_load = float(max(temps_load) - temps_load[0])

        # Enabler-only should not raise the zone temperature by much (<= ~1C).
        assert delta_enabler <= 1.5

        # Load request should produce a materially larger temperature rise.
        assert delta_load >= delta_enabler + 2.0

        env.close()


