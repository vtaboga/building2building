from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from algorithms.baselines import HVACActuatorOnOffPolicy
from building2building.pipeline import get_hvac_actuators, get_net_conditioned_area
from building2building.simulator import create_simulator
from building2building.types import BaseRewardConfig, BuildingConfig


def _zone_temp_indices(env) -> list[int]:
    names = []
    if hasattr(env, "metadata") and isinstance(env.metadata, dict):
        raw = env.metadata.get("observation_names")
        if isinstance(raw, list):
            names = [str(x) for x in raw]
    if not names:
        raise RuntimeError("env.metadata['observation_names'] missing")
    idxs = [i for i, nm in enumerate(names) if "zone air temperature" in nm.lower()]
    if not idxs:
        raise RuntimeError("No zone air temperature observations found")
    return idxs


def _rollout_mean_zone_temp(env, policy, n_steps: int) -> list[float]:
    obs, _ = env.reset()
    tz_idxs = _zone_temp_indices(env)
    temps: list[float] = []
    for _ in range(n_steps):
        act, _ = policy.predict(obs, deterministic=True)
        obs, _reward, terminated, truncated, _info = env.step(np.asarray(act, dtype=float))
        tz = float(np.mean(np.asarray([float(obs[i]) for i in tz_idxs], dtype=float)))
        temps.append(tz)
        if terminated or truncated:
            break
    return temps


def test_hvac_actuator_on_off_policy_tracks_setpoint_and_uses_hvac_actuators() -> None:
    """
    Integration-light test:
    1) Policy reduces temperature error vs "HVAC forced off" baseline.
    2) Env action space is built from HVAC actuators (not Schedule/Zone Temp setpoints).
    """
    epjson_path = Path("tests/fixtures/bldg1.epjson").resolve()
    epw_path = Path("tests/fixtures/weather.epw").resolve()
    edd_path = Path("tests/fixtures/eplusout.edd").resolve()
    htm_path = Path("tests/fixtures/eplustbl.htm").resolve()

    hvac_actuators = get_hvac_actuators(edd_path)
    area = get_net_conditioned_area(htm_path)

    target = 21.0
    deadband = 0.5
    n_steps = 60
    tail = 20

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        cfg = BuildingConfig(
            path_to_building=epjson_path,
            path_to_weather=epw_path,
            reward_config=BaseRewardConfig(energy_weight=0.0),
            hvac_actuators=hvac_actuators,
            eplus_output_dir=out_dir,
            warmup_phases=0,
            area=area,
        )

        env = create_simulator(cfg)

        # --- requirement (2): MUST be HVAC actuators, not thermostat setpoint schedules ---
        action_names = env.metadata.get("action_names", [])
        assert isinstance(action_names, list) and action_names, "Expected env.metadata['action_names']"
        assert all("schedule" not in str(nm).lower() for nm in action_names)
        assert all("zone temperature control" not in str(nm).lower() for nm in action_names)

        # --- baseline: force HVAC off using availability override if present ---
        off_policy = HVACActuatorOnOffPolicy.from_env_metadata(
            env_metadata=env.metadata,
            target_temp_c=target,
            deadband_c=deadband,
            fan_mass_flow_kg_s=0.0,
            coil_speed_heat=0.0,
            coil_speed_cool=0.0,
            supplemental_stage_heat=0.0,
            availability_on=1.0,
            availability_off=1.0,
        )
        temps_off = _rollout_mean_zone_temp(env, off_policy, n_steps=n_steps)

        # --- controller ---
        policy = HVACActuatorOnOffPolicy.from_env_metadata(
            env_metadata=env.metadata,
            target_temp_c=target,
            deadband_c=deadband,
            fan_mass_flow_kg_s=1.0,
            coil_speed_heat=1.0,
            coil_speed_cool=1.0,
            supplemental_stage_heat=0.0,
        )
        temps_ctl = _rollout_mean_zone_temp(env, policy, n_steps=n_steps)

        assert len(temps_off) >= tail and len(temps_ctl) >= tail

        off_tail = np.asarray(temps_off[-tail:], dtype=float)
        ctl_tail = np.asarray(temps_ctl[-tail:], dtype=float)

        err_off = float(np.mean(np.abs(off_tail - target)))
        err_ctl = float(np.mean(np.abs(ctl_tail - target)))

        # requirement (1): maintain around setpoint (operationalized as "better than off")
        assert err_ctl < err_off, f"Expected control to reduce temp error: {err_ctl=} {err_off=}"

        # Loose absolute sanity: should not drift wildly
        assert float(np.max(np.abs(ctl_tail - target))) <= 6.0

        try:
            env.close()
        except Exception:
            pass


