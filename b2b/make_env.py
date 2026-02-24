from __future__ import annotations

import json
import logging
import traceback
import uuid
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from b2b.api import make_env as make_env_typed
from b2b.config.models import DatasetSelectionConfig, EnvBuildConfig
from b2b.types import TaskConfig, reward_config_from_dict

logger = logging.getLogger(__name__)


def _to_plain_dict(config: object) -> dict[str, Any]:
    if isinstance(config, dict):
        return dict(config)
    try:
        raw = OmegaConf.to_container(config, resolve=True)
        if isinstance(raw, dict):
            return dict(raw)
    except Exception:
        pass
    return {}


def _infer_dataset_selection(cfg: dict[str, Any]) -> DatasetSelectionConfig:
    bldg = cfg.get("bldg", {})
    if not isinstance(bldg, dict):
        bldg = {}
    raw_query = bldg.get("bldg", {})
    if not isinstance(raw_query, dict):
        raw_query = {}
    selection = bldg.get("selection", {})
    if not isinstance(selection, dict):
        selection = {}

    if "building_type" in raw_query:
        if selection.get("enabled"):
            return DatasetSelectionConfig(
                dataset="multizones_reference_buildings",
                building_type=str(raw_query["building_type"]),  # type: ignore[arg-type]
                split=str(selection.get("split", "train")),  # type: ignore[arg-type]
                mode="split_index",
                split_index=int(selection.get("index", 0)),
            )
        return DatasetSelectionConfig(
            dataset="multizones_reference_buildings",
            building_type=str(raw_query["building_type"]),  # type: ignore[arg-type]
            split="train",
            mode="metadata_query",
            metadata_query=dict(raw_query),
            sample_size=1,
        )

    if selection.get("enabled"):
        return DatasetSelectionConfig(
            dataset="single_zone_houses",
            split=str(selection.get("split", "train")),  # type: ignore[arg-type]
            mode="split_index",
            split_index=int(selection.get("index", 0)),
        )

    return DatasetSelectionConfig(
        dataset="single_zone_houses",
        split="train",
        mode="metadata_query",
        metadata_query=dict(raw_query),
        sample_size=1,
    )


def make_env(config: object, eplus_output_dir: str | Path):
    out_dir = Path(eplus_output_dir) / str(uuid.uuid4())
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        cfg = _to_plain_dict(config)
        task_section = cfg.get("task", {}) if isinstance(cfg.get("task"), dict) else {}
        reward_section = (
            cfg.get("reward", {}) if isinstance(cfg.get("reward"), dict) else {}
        )
        env_section = cfg.get("env", {}) if isinstance(cfg.get("env"), dict) else {}
        task = TaskConfig.from_dict(task_section)
        reward = reward_config_from_dict(reward_section, area=1.0)
        build = EnvBuildConfig(
            dataset_selection=_infer_dataset_selection(cfg),
            task=task,
            reward=reward,
            env_max_steps=(
                int(env_section["max_steps"])
                if env_section.get("max_steps") is not None
                else None
            ),
        )
        return make_env_typed(build, eplus_output_dir=out_dir)
    except Exception as e:
        err_path = Path(out_dir) / "env_creation_error.json"
        record: dict[str, Any] = {
            "error_type": type(e).__name__,
            "message": str(e),
            "traceback": traceback.format_exc(),
        }
        try:
            with err_path.open("w", encoding="utf-8") as f:
                json.dump(record, f, indent=2)
        except Exception:
            logger.exception("Failed to write env creation error to %s", err_path)
        raise

