from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from algorithms import baselines
from building2building.pipeline import (
    get_airflow_and_coil_node_setpoint_actuators,
    get_net_conditioned_area,
)
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

    zone_temp_series = np.mean(obs[:, zone_idxs], axis=1)
    mean_zone_temp_c = float(np.mean(zone_temp_series))

    total_energy = float(np.sum(obs[:, energy_idxs]))
    return mean_zone_temp_c, total_energy


def test_constant_fan_and_node_setpoint_control_changes_dynamics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Run the constant baseline controller multiple times with different constant actuator
    values (fan flow + coil node setpoints) and ensure outcomes differ.
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
    hvac_actuators = get_airflow_and_coil_node_setpoint_actuators(edd_path)
    assert hvac_actuators, "Fixture must expose fan/coil-node-setpoint actuators"

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
    monkeypatch.setattr(
        baselines,
        "make_env",
        lambda config, eplus_output_dir: _make_env(eplus_output_dir=eplus_output_dir),
    )

    # Intentionally large differences so the resulting temp/energy differ reliably.
    scenarios: list[dict[str, float]] = [
        # Lower flow, milder coil setpoints
        {
            "fan_mass_flow_kg_s": 0.2,
            "heating_coil_setpoint_c": 30.0,
            "supplemental_coil_setpoint_c": 35.0,
            "cooling_coil_setpoint_c": 16.0,
        },
        # Higher flow, more aggressive heating + colder cooling
        {
            "fan_mass_flow_kg_s": 3.0,
            "heating_coil_setpoint_c": 45.0,
            "supplemental_coil_setpoint_c": 55.0,
            "cooling_coil_setpoint_c": 10.0,
        },
    ]

    temps: list[float] = []
    energies: list[float] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)

        for i, p in enumerate(scenarios):
            run_dir = base_dir / f"run_{i}"
            cfg = OmegaConf.create(
                {
                    # Long enough to cover both winter and summer periods.
                    "env": {"max_steps": 20000, "normalize_obs": False},
                    "n_episodes": 1,
                    "policy": {
                        "type": "fan_coil_constant",
                        "availability_on": 2.0,
                        **p,
                    },
                }
            )
            paths = baselines.run_baseline_rollout(cfg, run_dir=run_dir)
            mean_t_c, tot_e = _compute_rollout_metrics(paths.npz_path)
            temps.append(mean_t_c)
            energies.append(tot_e)

    # Validate outcomes differ between the two constant actuation settings.
    #
    # Depending on the building's native thermostat schedules and initial conditions,
    # zone air temperature may be nearly unchanged over a short rollout even if HVAC
    # energy responds. Therefore we require *at least one* of (temperature, energy) to
    # differ meaningfully.
    temp_diff = not np.isclose(temps[0], temps[1], atol=0.05)
    energy_diff = not np.isclose(energies[0], energies[1], atol=50.0)
    print(f"temps: {temps[0]}, {temps[1]}")
    print(f"energies: {energies[0]}, {energies[1]}")
    assert temp_diff or energy_diff, f"Expected temperature or energy to differ: {temps=}, {energies=}"

