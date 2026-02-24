from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from b2b.baselines.common import RolloutPaths, make_rollout_paths
from b2b.baselines.controllers.fan_coil_constant import FanCoilConstantPolicy
from b2b.baselines.controllers.unitary_pi import UnitaryPIPolicy
from b2b.baselines.controllers.unitary_sat import UnitaryAirflowFirstSatPolicy
from b2b.baselines.controllers.zone_temp_21 import ZoneTemp21Policy
from b2b.baselines.wandb_utils import (
    finish_wandb_if_started,
    init_wandb_from_config,
    wandb_log_df_line_series,
)
from b2b.benchmark.runner import run_rollout

logger = logging.getLogger(__name__)


def _require_list_str(meta: dict[str, Any], key: str) -> list[str]:
    raw = meta.get(key)
    if not isinstance(raw, list) or not all(isinstance(x, str) for x in raw):
        raise RuntimeError(f"env.metadata['{key}'] must be a list[str]")
    return list(raw)


def _build_controller_policy(cfg: DictConfig) -> Any:
    policy_type = str(getattr(cfg.policy, "type", "fan_coil_constant")).strip()
    if policy_type == "fan_coil_constant":
        return FanCoilConstantPolicy(cfg.policy)
    if policy_type == "unitary_pi":
        return UnitaryPIPolicy(cfg.policy)
    if policy_type in ("unitary_airflow_first_sat", "unitary_sat"):
        return UnitaryAirflowFirstSatPolicy(cfg.policy)
    if policy_type == "zone_temp_21":
        return ZoneTemp21Policy(cfg.policy)
    raise NotImplementedError(f"Unsupported baseline policy.type={policy_type!r}")


def run_baseline_rollout(
    cfg: DictConfig,
    *,
    make_env: Callable[..., Any],
    run_dir: Path | None = None,
) -> RolloutPaths:
    """
    Rollout a simulation with a baseline controller policy.

    This lives under `b2b.benchmark` so all simulation execution code stays
    centralized outside of `b2b.baselines` (controllers + SB3 interaction only).
    """
    if run_dir is None:
        if HydraConfig.initialized():
            run_dir = Path(HydraConfig.get().runtime.output_dir)
        else:
            run_dir = Path.cwd()

    paths = make_rollout_paths(Path(run_dir))
    with paths.config_path.open("w", encoding="utf-8") as f:
        json.dump(OmegaConf.to_container(cfg, resolve=True), f, indent=2)

    eplus_output_dir = Path(run_dir) / "eplus_outputs"
    env = make_env(config=cfg, eplus_output_dir=str(eplus_output_dir))
    try:
        meta = getattr(env, "metadata", {}) or {}
        meta = meta if isinstance(meta, dict) else {}
        obs_names = _require_list_str(meta, "observation_names")
        act_names = _require_list_str(meta, "action_names")

        policy = _build_controller_policy(cfg)

        max_steps = int(getattr(cfg.env, "max_steps"))
        n_episodes = int(getattr(cfg, "n_episodes", 1))

        wandb_run, started_here = init_wandb_from_config(cfg, run_dir=Path(run_dir))
        try:
            results, data = run_rollout(
                env=env,
                policy=policy,
                n_episodes=n_episodes,
                deterministic=True,
                max_steps=max_steps,
                record=True,
            )
            assert data is not None

            # Build a flat dataframe similar to the historical baselines runner output.
            df_dict: dict[str, Any] = {
                "episode": data.episode.astype(np.int32),
                "step": data.step.astype(np.int32),
                "global_step": (data.episode.astype(np.int64) * int(max_steps) + data.step.astype(np.int64)).astype(np.int64),
                "reward": data.reward.astype(float),
            }
            for k, arr in data.metrics.items():
                df_dict[str(k)] = np.asarray(arr, dtype=float)

            obs_arr = np.asarray(data.obs, dtype=float)
            act_arr = np.asarray(data.action, dtype=float)
            for i, name in enumerate(obs_names):
                if i < obs_arr.shape[1]:
                    df_dict[f"obs::{name}"] = obs_arr[:, i]
            for i, name in enumerate(act_names):
                if i < act_arr.shape[1]:
                    df_dict[f"act::{name}"] = act_arr[:, i]

            df_plot = pd.DataFrame(df_dict)
            df_plot.to_csv(paths.csv_path, index=False)

            # Log a few lightweight timeseries plots to W&B (downsampled).
            if wandb_run is not None:
                try:

                    # Temperature: zone air temps + outdoor temp if present
                    zone_temp_cols = [
                        c
                        for c in df_plot.columns
                        if c.lower().startswith("obs::zone air temperature")
                    ]
                    # Limit to avoid overly crowded plots
                    zone_temp_cols = zone_temp_cols[:10]
                    outdoor_temp_col = next(
                        (
                            c
                            for c in df_plot.columns
                            if c.lower() in ("obs::outdoor_temperature", "outdoor_temperature")
                        ),
                        None,
                    )
                    temp_cols = list(zone_temp_cols)
                    if outdoor_temp_col is not None:
                        temp_cols.append(outdoor_temp_col)

                    if temp_cols:
                        wandb_log_df_line_series(
                            df=df_plot,
                            x="global_step",
                            y_cols=temp_cols,
                            key_prefix="rollout/temperature",
                            title="Temperatures",
                        )

                    # Actions: log a small subset (fan + any schedule setpoints)
                    act_cols = [c for c in df_plot.columns if c.lower().startswith("act::")]
                    preferred = [
                        c
                        for c in act_cols
                        if ("fan air mass flow rate" in c.lower())
                        or ("temperature" in c.lower())
                        or ("schedule value" in c.lower())
                    ]
                    action_cols = (preferred + [c for c in act_cols if c not in preferred])[:12]
                    if action_cols:
                        wandb_log_df_line_series(
                            df=df_plot,
                            x="global_step",
                            y_cols=action_cols,
                            key_prefix="rollout/actions",
                            title="Actions",
                        )

                    # Energy + reward
                    energy_cols = [
                        c
                        for c in ("obs::energy_electricity", "obs::energy_gas")
                        if c in df_plot.columns
                    ]
                    if energy_cols:
                        wandb_log_df_line_series(
                            df=df_plot,
                            x="global_step",
                            y_cols=energy_cols,
                            key_prefix="rollout/energy",
                            title="Energy",
                        )
                    if "reward" in df_plot.columns:
                        wandb_log_df_line_series(
                            df=df_plot,
                            x="global_step",
                            y_cols=["reward"],
                            key_prefix="rollout/reward",
                            title="Reward",
                        )
                except Exception as e:
                    logger.warning("Failed to log rollout plots to wandb: %s", e)

            obs_cols = [c for c in df_plot.columns if c.startswith("obs::")]
            act_cols = [c for c in df_plot.columns if c.startswith("act::")]
            np.savez_compressed(
                paths.npz_path,
                episode=df_plot["episode"].to_numpy(dtype=np.int32),
                step=df_plot["step"].to_numpy(dtype=np.int32),
                reward=df_plot["reward"].to_numpy(dtype=float),
                obs_names=np.asarray(obs_cols, dtype=object),
                act_names=np.asarray(act_cols, dtype=object),
                obs=df_plot[obs_cols].to_numpy(dtype=float)
                if obs_cols
                else np.zeros((len(df_plot), 0)),
                act=df_plot[act_cols].to_numpy(dtype=float)
                if act_cols
                else np.zeros((len(df_plot), 0)),
            )

            # Summary scalars.
            if wandb_run is not None:
                try:
                    import wandb  # type: ignore

                    wandb.summary["rollout/mean_reward"] = float(df_plot["reward"].mean())
                    if results:
                        wandb.summary["rollout/episode_return_mean"] = float(
                            np.mean([r.total_reward for r in results])
                        )
                except Exception as e:
                    logger.warning("Failed to log rollout summary to wandb: %s", e)

            return paths
        finally:
            finish_wandb_if_started(wandb_run, started_here=started_here)
    finally:
        try:
            env.close()
        except Exception:
            pass

