"""
Utilities for multi-building training with diverse building sampling.
"""

import gc
import json
import logging
import os
import random
import shutil
import traceback
import uuid
from pathlib import Path
from typing import Any

from b2b.simulator import create_simulator
from b2b.simulator.observation_spaces import flat_observation_info
from b2b.sources import hydroquebec
from minergym.ontology import Ontology

logger = logging.getLogger(__name__)


def make_diverse_env(
    config, eplus_output_dir: str, building_pool: list[Any] | None = None
):
    """
    Create an environment with a randomly sampled building from a diverse pool.

    This differs from make_env() by:
    1. Sampling from a pre-fetched pool of diverse buildings
    2. Ensuring each environment gets a different building
    3. Re-sampling a new building on each episode reset (future enhancement)

    Args:
        config: Hydra configuration
        eplus_output_dir: Directory for EnergyPlus outputs
        building_pool: Pre-fetched list of BuildingConfig objects. REQUIRED.

    Returns:
        gym.Env: EnergyPlus environment

    Raises:
        ValueError: If building_pool is None or empty
    """
    # Require building pool
    if not building_pool or len(building_pool) == 0:
        raise ValueError(
            "building_pool is required and must not be empty. "
            "Call fetch_diverse_building_pool() first to create a building pool."
        )

    # Create a unique directory for this environment instance
    # NOTE: We create ONE directory per environment, not per episode
    # This prevents disk space exhaustion from thousands of directories
    output_path = Path(eplus_output_dir) / str(uuid.uuid4())

    # Clean up if it already exists (shouldn't happen with UUID, but be safe)
    if output_path.exists():
        shutil.rmtree(output_path, ignore_errors=True)

    output_path.mkdir(parents=True, exist_ok=True)

    prev = os.environ.get("B2B_PIPELINE_DEBUG_DIR")
    try:
        os.environ["B2B_PIPELINE_DEBUG_DIR"] = str(output_path)

        # Randomly sample from the pool
        env_config = random.choice(building_pool)
        logger.info(
            "Sampled building from pool: area=%.1fm², warmup=%s, actuators=%d",
            env_config.area,
            env_config.warmup_phases,
            len(env_config.hvac_actuators),
        )

        # Update the eplus_output_dir to the unique one we created
        env_config.eplus_output_dir = output_path

        env = create_simulator(env_config)
        return env

    except Exception as e:
        err_path = output_path / "env_creation_error.json"
        record = {
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
        if prev is None:
            os.environ.pop("B2B_PIPELINE_DEBUG_DIR", None)
        else:
            os.environ["B2B_PIPELINE_DEBUG_DIR"] = prev


def fetch_diverse_building_pool(config, n_buildings: int = 10) -> list[Any]:
    """
    Fetch a diverse pool of buildings for multi-building training.

    IMPORTANT: All buildings in the pool must have the same observation space shape
    (same number of zones) for vectorized environments to work.

    Args:
        config: Hydra configuration with building selection criteria
        n_buildings: Number of diverse buildings to fetch

    Returns:
        List of BuildingConfig objects with compatible observation spaces
    """
    logger.info("Fetching pool of %d diverse buildings...", n_buildings)

    # Fetch more buildings than needed to ensure diversity and filtering
    # Use 2x instead of 3x to reduce memory usage during filtering
    configs = hydroquebec.search_configs(
        config=config,
        n=n_buildings * 2,  # Fetch 2x to account for failures and filtering
        eplus_output_dir=Path("/tmp/building_pool_discovery"),
    )

    if len(configs) == 0:
        raise RuntimeError("No buildings found matching the criteria")

    # Group buildings by (observation_size, action_size) tuple
    # All buildings in a vectorized environment must have the same obs AND action space shapes
    buildings_by_space_size: dict[tuple[int, int], list[Any]] = {}
    for i, cfg in enumerate(configs):
        try:
            # Determine observation space size for this building
            with open(cfg.path_to_building, "r") as f:
                epjson = json.load(f)
            ont = Ontology.from_object(epjson)
            obs_info = flat_observation_info(ont, area=cfg.area)
            obs_size = obs_info.space.shape[0]

            # Determine action space size (number of actuators)
            action_size = len(cfg.hvac_actuators)

            # Group by (obs_size, action_size) tuple
            space_key = (obs_size, action_size)
            if space_key not in buildings_by_space_size:
                buildings_by_space_size[space_key] = []
            buildings_by_space_size[space_key].append(cfg)

            # Explicitly clean up large objects to prevent memory leak
            del epjson
            del ont
            del obs_info

            # Force garbage collection every 10 buildings
            if (i + 1) % 10 == 0:
                gc.collect()

        except Exception as e:
            logger.warning("Failed to determine space sizes for building: %s", e)
            continue

    # Final garbage collection after processing all buildings
    gc.collect()

    if not buildings_by_space_size:
        raise RuntimeError("No buildings could be processed successfully")

    # Find the group with the most buildings
    largest_group_key = max(
        buildings_by_space_size.keys(), key=lambda k: len(buildings_by_space_size[k])
    )
    compatible_buildings = buildings_by_space_size[largest_group_key]
    obs_size, action_size = largest_group_key

    logger.info(
        "Found %d buildings with observation size %d and action size %d",
        len(compatible_buildings),
        obs_size,
        action_size,
    )
    other_sizes = [
        (k, len(v))
        for k, v in buildings_by_space_size.items()
        if k != largest_group_key
    ]
    logger.info("Other space sizes available: %s", other_sizes)

    if len(compatible_buildings) < n_buildings:
        logger.warning(
            "Only found %d buildings with compatible observation spaces, "
            "requested %d. Using all %d available.",
            len(compatible_buildings),
            n_buildings,
            len(compatible_buildings),
        )
        building_pool = compatible_buildings
    else:
        # Take the first n_buildings from the compatible group
        building_pool = compatible_buildings[:n_buildings]

    # Log diversity statistics
    areas = [c.area for c in building_pool]
    warmups = [c.warmup_phases for c in building_pool]
    actuators = [len(c.hvac_actuators) for c in building_pool]

    logger.info("Building pool diversity:")
    logger.info(
        "  - Areas: min=%.1f, max=%.1f, mean=%.1f m²",
        min(areas),
        max(areas),
        sum(areas) / len(areas),
    )
    logger.info(
        "  - Warmup phases: min=%s, max=%s, mean=%.1f",
        min(warmups),
        max(warmups),
        sum(warmups) / len(warmups),
    )
    logger.info(
        "  - Actuators: min=%d, max=%d, mean=%.1f",
        min(actuators),
        max(actuators),
        sum(actuators) / len(actuators),
    )

    # Log to WandB if available
    try:
        import wandb

        if wandb.run is not None:
            wandb.run.summary.update(
                {
                    "building_pool/size": len(building_pool),
                    "building_pool/obs_size": obs_size,
                    "building_pool/action_size": action_size,
                    "building_pool/area_min": min(areas),
                    "building_pool/area_max": max(areas),
                    "building_pool/area_mean": sum(areas) / len(areas),
                    "building_pool/warmup_min": min(warmups),
                    "building_pool/warmup_max": max(warmups),
                    "building_pool/warmup_mean": sum(warmups) / len(warmups),
                    "building_pool/actuators_min": min(actuators),
                    "building_pool/actuators_max": max(actuators),
                    "building_pool/actuators_mean": sum(actuators) / len(actuators),
                }
            )
    except Exception:
        pass  # WandB logging is optional

    return building_pool
