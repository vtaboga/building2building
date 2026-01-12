from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from algorithms import baselines


def _compute_mean_zone_temp_c(npz_path: Path) -> float:
    data = np.load(npz_path, allow_pickle=True)
    obs_names = [str(x) for x in data["obs_names"].tolist()]
    obs = np.asarray(data["obs"], dtype=float)

    zone_idxs = [i for i, n in enumerate(obs_names) if "zone air temperature" in n.lower()]
    if not zone_idxs:
        raise RuntimeError("Could not find any 'Zone Air Temperature' entries in rollout obs_names")

    zone_temp_series = np.mean(obs[:, zone_idxs], axis=1)
    return float(np.mean(zone_temp_series))


def test_baseboard_control_rollouts_have_different_temperatures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Run three rollouts on a Hydro-Québec building that includes electric baseboards:
      1) baseline on/off control (baseboards enabled only when requesting heat)
      2) forced always on (baseboards always available)
      3) forced always off (baseboards always unavailable)

    and verify the resulting zone temperatures differ.
    """

    # Use the same selector as configs/bldg/baseboard_unitary_cooling.yaml.
    # This should pick a building with electric resistance heating (baseboards) and mini-split cooling.
    bldg_selector: dict[str, object] = {
        "geometry_unit_type": "single-family detached",
        "geometry_building_num_units": 1,
        "year_built": 1940,
        "heating_system_type": "electricresistance",
        "cooling_system_type": "mini-split",
    }

    # Baseline policy config (we'll override forced_mode per scenario).
    base_cfg = OmegaConf.create(
        {
            "bldg": bldg_selector,
            "reward": {
                "reward_type": "DeadbandRewardConfig",
                "target_temp": 21.0,
                "dT": 0.5,
                "energy_weight": 0.0,
            },
            "env": {"max_steps": 4000, "normalize_obs": False},
            "n_episodes": 1,
            "policy": {
                "type": "on_off",
                "target_temp_c": 21.0,
                # Slightly wider deadband makes the difference between "availability-gated"
                # vs "always available" baseboards show up more reliably.
                "deadband_c": 1.5,
                "q_heat_pct": 0.8,
                "q_cool_pct": 0.8,
                # Optional override: "heat" / "cool" / "off".
                "forced_mode": None,
            },
        }
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        scenarios: list[tuple[str, str | None]] = [
            ("on_off", None),
            ("always_on", "heat"),
            ("always_off", "off"),
        ]

        temps: list[float] = []
        for name, forced_mode in scenarios:
            run_dir = tmp / f"run_{name}"
            cfg = OmegaConf.merge(
                base_cfg,
                OmegaConf.create({"policy": {"forced_mode": forced_mode}}),
            )
            paths = baselines.run_baseline_rollout(cfg, run_dir=run_dir)
            temps.append(_compute_mean_zone_temp_c(paths.npz_path))

        for i in range(len(scenarios)):
            for j in range(i + 1, len(scenarios)):
                assert not np.isclose(
                    temps[i], temps[j], atol=0.1
                ), f"Expected mean zone temperature to differ: {temps=}"

