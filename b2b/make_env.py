from __future__ import annotations

import json
import logging
import os
import traceback
import uuid
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from b2b.simulator import create_simulator
from b2b.sources import hydroquebec
from b2b.utils import (
    hydroquebec_building_id_from_split_index,
    hydroquebec_filenames_for_building_id,
)

logger = logging.getLogger(__name__)


def make_env(config: object, eplus_output_dir: str | Path):
    """
    Create a single Building2Building Gymnasium environment from a config.

    This is the centralized environment factory used by both:
    - Stable-Baselines training/eval code
    - Benchmarks/experiments that evaluate policies on datasets

    Notes
    -----
    - EnergyPlus needs a unique output dir per run. We create a UUID subfolder.
    - If `bldg.selection.enabled` is true, we deterministically pick a building from
      the stored Hydro-Québec split lists by injecting `idf_filename` and
      `schedule_filename` into `bldg.bldg`.
    """
    # EnergyPlus needs a unique output dir for each run
    out_dir = Path(eplus_output_dir) / str(uuid.uuid4())
    out_dir.mkdir(parents=True, exist_ok=True)

    # Fetch exactly one BuildingConfig and build a single simulator env
    try:
        cfg_any: object = config
        if not isinstance(cfg_any, dict):
            try:
                cfg_any = OmegaConf.to_container(cfg_any, resolve=True)
            except Exception:
                cfg_any = config

        if isinstance(cfg_any, dict):
            bldg_section = cfg_any.get("bldg")
            if isinstance(bldg_section, dict):
                sel = bldg_section.get("selection")
                if isinstance(sel, dict) and bool(sel.get("enabled", False)):
                    split = str(sel.get("split", "train")).strip().lower()
                    if split not in ("train", "test"):
                        raise ValueError(
                            f"bldg.selection.split must be 'train' or 'test', got {split!r}"
                        )
                    split_idx_raw = sel.get("index", 0)
                    if not isinstance(split_idx_raw, int):
                        raise TypeError(
                            "bldg.selection.index must be int, got "
                            f"{type(split_idx_raw).__name__}"
                        )

                    building_id = hydroquebec_building_id_from_split_index(
                        split=split, split_index=split_idx_raw
                    )
                    idf_filename, schedule_filename = hydroquebec_filenames_for_building_id(
                        building_id
                    )

                    # Override any existing building query: pick exactly this building.
                    bldg_section["bldg"] = {
                        "idf_filename": idf_filename,
                        "schedule_filename": schedule_filename,
                    }
                    cfg_any["bldg"] = bldg_section
                    config = cfg_any

        # Let the pipeline copy discovery `eplusout.err` into this run folder.
        prev = os.environ.get("B2B_PIPELINE_DEBUG_DIR")
        os.environ["B2B_PIPELINE_DEBUG_DIR"] = str(out_dir)
        configs = hydroquebec.search_configs(
            config=config, n=1, eplus_output_dir=Path(out_dir)
        )
        if not configs:
            raise RuntimeError(
                "No building configurations found for the provided config "
                "(see pipeline_errors.jsonl in the EnergyPlus output dir if present)."
            )
        env_config = configs[0]
        env = create_simulator(env_config)
        return env
    except Exception as e:
        # Persist a structured error record to make batch runs debuggable.
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
    finally:
        # Restore previous value to avoid leaking run-specific state.
        if "prev" in locals():
            if prev is None:
                os.environ.pop("B2B_PIPELINE_DEBUG_DIR", None)
            else:
                os.environ["B2B_PIPELINE_DEBUG_DIR"] = prev

