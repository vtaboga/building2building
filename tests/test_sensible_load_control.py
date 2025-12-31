from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from algorithms import baselines
from building2building.pipeline import get_hvac_actuators, get_net_conditioned_area, get_sensible_load_actuators
from building2building.simulator import create_simulator
from building2building.types import BaseRewardConfig, BuildingConfig


def _compute_rollout_metrics(npz_path: Path) -> tuple[float, float]:
    data = np.load(npz_path, allow_pickle=True)
    obs_names = [str(x) for x in data["obs_names"].tolist()]
    obs = np.asarray(data["obs"], dtype=float)

    zone_idxs = [i for i, n in enumerate(obs_names) if "zone air temperature" in n.lower()]
    if not zone_idxs:
        raise RuntimeError("Could not find any 'Zone Air Temperature' entries in rollout obs_names")

    energy_idxs = [
        i
        for i, n in enumerate(obs_names)
        if ("energy_electricity" in n.lower()) or ("energy_gas" in n.lower())
    ]
    if not energy_idxs:
        raise RuntimeError("Could not find any 'energy_electricity'/'energy_gas' entries in rollout obs_names")

    # Temperature metric: mean zone air temp (averaged across zones + timesteps)
    zone_temp_series = np.mean(obs[:, zone_idxs], axis=1)
    mean_zone_temp_c = float(np.mean(zone_temp_series))

    # Energy metric: sum over timesteps of per-area HVAC meters (electricity + gas)
    total_energy = float(np.sum(obs[:, energy_idxs]))

    return mean_zone_temp_c, total_energy


def test_sensible_load_control(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Run the on/off sensible-load baseline controller 3 times with different (q_heat_w, q_cool_w)
    and verify both the simulated zone temperature and energy differ.
    """
    epjson_path = Path("tests/fixtures/bldg1.epjson").resolve()
    epw_path = Path("tests/fixtures/weather.epw").resolve()
    edd_path = Path("tests/fixtures/eplusout.edd").resolve()
    htm_path = Path("tests/fixtures/eplustbl.htm").resolve()

    assert epjson_path.exists()
    assert epw_path.exists()
    assert edd_path.exists()
    assert htm_path.exists()

    area = float(get_net_conditioned_area(htm_path))
    sensible = get_sensible_load_actuators(edd_path)
    assert sensible, "Fixture must expose at least one sensible-load request actuator"

    # Optional: include availability override if present so the HVAC stays available.
    hvac_all = get_hvac_actuators(edd_path)
    availability = [
        a
        for a in hvac_all
        if (a.get("component_type") == "AirLoopHVAC") and (a.get("control_type") == "Availability Status")
    ]
    hvac_actuators = (availability[:1] + sensible) if availability else sensible

    def _make_env(*, eplus_output_dir: str):
        cfg = BuildingConfig(
            path_to_building=epjson_path,
            path_to_weather=epw_path,
            reward_config=BaseRewardConfig(energy_weight=0.0),
            hvac_actuators=hvac_actuators,
            eplus_output_dir=Path(eplus_output_dir),
            warmup_phases=0,
            area=area,
        )
        return create_simulator(cfg)

    # baselines.run_baseline_rollout() calls baselines.make_env() (imported into that module).
    monkeypatch.setattr(baselines, "make_env", lambda config, eplus_output_dir: _make_env(eplus_output_dir=eplus_output_dir))

    # Use noticeably different values to ensure the resulting dynamics differ.
    q_pairs: list[tuple[float, float]] = [
        (1000.0, 0.0),
        (1000.0, 1000.0),
        (10000.0, 10000.0),
    ]

    temps: list[float] = []
    energies: list[float] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)

        for i, (q_heat_w, q_cool_w) in enumerate(q_pairs):
            run_dir = base_dir / f"run_{i}"
            cfg = OmegaConf.create(
                {
                    "env": {"max_steps": 17000, "normalize_obs": False},
                    "n_episodes": 1,
                    "policy": {
                        "type": "on_off",
                        "target_temp_c": 21.0,
                        "deadband_c": 0.5,
                        "q_heat_w": float(q_heat_w),
                        "q_cool_w": float(q_cool_w),
                        "availability_on": 2.0,
                    },
                }
            )

            paths = baselines.run_baseline_rollout(cfg, run_dir=run_dir)
            mean_t_c, tot_e = _compute_rollout_metrics(paths.npz_path)
            temps.append(mean_t_c)
            energies.append(tot_e)

    # Validate they are different across the 3 parameter sets.
    for i in range(len(q_pairs)):
        for j in range(i + 1, len(q_pairs)):
            assert not np.isclose(temps[i], temps[j], atol=0.05), f"Expected temperature to differ: {temps=}"
            assert not np.isclose(energies[i], energies[j], atol=100), f"Expected energy to differ: {energies=}"


