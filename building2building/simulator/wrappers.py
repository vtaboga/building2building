import logging
import random
import warnings
from typing import Any, Callable

import gymnasium as gym
import numpy as np

logger = logging.getLogger(__name__)


class NormalizeObservation(gym.ObservationWrapper):
    """
    Observation wrapper that normalizes observations to a [0, 1] range according to the observation space bounds.
    Values may be outside of this range if they are out of the environment's observation space bounds.
    """

    def __init__(self, env: gym.Env, dtype: np.dtype = np.float32):
        """
        Args:
            env: The environment to wrap
            dtype: The dtype of the observation space
        """
        super().__init__(env)

        # Ensure the observation space is a Box
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise ValueError(
                f"Expected observation space to be Box, got {type(env.observation_space)}"
            )

        # Initialize observation space with the specified dtype
        # (will be properly set by _update_bounds)
        self.observation_space = gym.spaces.Box(
            low=np.zeros(env.observation_space.shape, dtype=dtype),
            high=np.ones(env.observation_space.shape, dtype=dtype),
            dtype=dtype,
        )

        # Initialize bounds using the shared method
        self._update_bounds()

    def _update_bounds(self) -> None:
        """Re-read observation space bounds from the inner env and recompute derived state."""
        self.obs_low = self.env.observation_space.low
        self.obs_high = self.env.observation_space.high

        self.obs_range = self.obs_high - self.obs_low

        if (self.obs_range == 0).any():
            raise ValueError("Observation space range is zero for at least one feature")

        dtype = self.observation_space.dtype
        target_low = np.zeros(self.obs_low.shape, dtype=dtype)
        target_high = np.ones(self.obs_high.shape, dtype=dtype)
        self.observation_space = gym.spaces.Box(
            low=target_low, high=target_high, dtype=dtype
        )

    def reset(self, **kwargs):  # type: ignore[override]
        """Reset and re-read observation space bounds (inner env may have changed)."""
        obs, info = self.env.reset(**kwargs)
        self._update_bounds()
        return self.observation(obs), info

    def observation(self, observation: Any) -> np.ndarray:
        """
        Normalize the observation to the target range.

        Args:
            observation: The original observation from the environment

        Returns:
            The normalized observation
        """
        observation = np.asarray(observation, dtype=self.observation_space.dtype)
        normalized_observation = (observation - self.obs_low) / self.obs_range
        return normalized_observation

    def denormalize(self, observation: np.ndarray) -> np.ndarray:
        """
        Denormalize the observation to the original range.
        """
        return observation * self.obs_range + self.obs_low


class PadObservation(gym.ObservationWrapper):
    """Pad observations to a fixed target size with zone-aware padding.

    When buildings produce observations of different sizes due to varying zone
    counts, this wrapper intelligently pads the zone temperatures to a fixed
    size while keeping all other features (outdoor temp, time, energy) in
    consistent positions across buildings.

    Observation structure (from flat_observation_info):
        - Zone Air Temperatures (variable count) [-50°C, 50°C]
        - Outdoor Air Temperature [-50°C, 50°C]
        - Outdoor Air Relative Humidity [0%, 100%]
        - Current Time of Day [1, 25]
        - Day of Week [1, 7]
        - Day of Year [1, 366]
        - HVAC Electricity Consumption [0, 50] Wh/m²/15min
        - HVAC Natural Gas Consumption [0, 50] Wh/m²/15min

    The wrapper pads zone temperatures to a fixed count, ensuring that outdoor
    temp, time features, and energy consumption are always at the same indices
    across all buildings. This is critical for multi-building generalization.

    Padded zone temperature dimensions have bounds [0, 0] so normalization
    layers treat them as constants.
    """

    # Fallback number of non-zone features (legacy layout):
    # outdoor temp, outdoor humid, 3 time features, 2 energy.
    NUM_NON_ZONE_FEATURES = 7

    def __init__(self, env: gym.Env, target_size: int):
        super().__init__(env)

        if not isinstance(env.observation_space, gym.spaces.Box):
            raise ValueError(
                "Expected observation space to be Box, "
                f"got {type(env.observation_space)}"
            )

        self._target_size = target_size
        orig_size = env.observation_space.shape[0]

        if orig_size > target_size:
            raise ValueError(
                f"Inner observation size ({orig_size}) exceeds "
                f"target_size ({target_size})"
            )

        # Calculate max zones that can fit
        self._max_zones = target_size - self.NUM_NON_ZONE_FEATURES
        if self._max_zones < 1:
            raise ValueError(
                f"target_size ({target_size}) too small to accommodate "
                f"at least 1 zone + {self.NUM_NON_ZONE_FEATURES} non-zone features"
            )

        self._rebuild_observation_space()

        logger.info(
            "PadObservation: inner obs %d → padded to %d (max %d zones)",
            orig_size,
            target_size,
            self._max_zones,
        )

    # ------------------------------------------------------------------
    def _zone_air_temperature_indices(self, obs_size: int) -> list[int]:
        meta = getattr(self.env, "metadata", None)
        if not isinstance(meta, dict):
            return list(range(max(0, obs_size - self.NUM_NON_ZONE_FEATURES)))
        names = meta.get("observation_names")
        if not isinstance(names, list) or len(names) != obs_size:
            return list(range(max(0, obs_size - self.NUM_NON_ZONE_FEATURES)))

        idx: list[int] = []
        for i, name in enumerate(names):
            if str(name).strip().lower().startswith("zone air temperature"):
                idx.append(i)
        if idx:
            return idx

        return list(range(max(0, obs_size - self.NUM_NON_ZONE_FEATURES)))

    def _split_zone_non_zone(
        self, values: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        obs_size = values.shape[0]
        zone_idx = self._zone_air_temperature_indices(obs_size)
        if any(i < 0 or i >= obs_size for i in zone_idx):
            raise ValueError("Invalid zone index inferred from observation names")

        zone_mask = np.zeros(obs_size, dtype=bool)
        zone_mask[np.array(zone_idx, dtype=int)] = True
        zone_vals = values[zone_mask]
        non_zone_vals = values[~zone_mask]
        return zone_vals, non_zone_vals, zone_mask

    # ------------------------------------------------------------------
    def _rebuild_observation_space(self) -> None:
        """Rebuild padded observation space with zone-aware padding.

        Pads zone temperatures to max_zones, keeping non-zone features at
        consistent positions across buildings.
        """
        inner_low = self.env.observation_space.low
        inner_high = self.env.observation_space.high
        inner_size = inner_low.shape[0]
        dtype = self.env.observation_space.dtype

        if inner_size > self._target_size:
            raise ValueError(
                f"Inner observation size ({inner_size}) exceeds "
                f"target_size ({self._target_size}). "
                "Increase target_obs_size in your config."
            )

        zone_low, non_zone_low, zone_mask = self._split_zone_non_zone(inner_low)
        zone_high = inner_high[zone_mask]
        non_zone_high = inner_high[~zone_mask]
        current_num_zones = zone_low.shape[0]

        if current_num_zones > self._max_zones:
            raise ValueError(
                f"Building has {current_num_zones} zones, exceeds max_zones "
                f"({self._max_zones}). Increase target_obs_size in your config."
            )

        # Pad zone temperatures to max_zones with [0, 0] bounds
        num_pad_zones = self._max_zones - current_num_zones
        if num_pad_zones > 0:
            pad_zeros = np.zeros(num_pad_zones, dtype=dtype)
            padded_zone_low = np.concatenate([zone_low, pad_zeros])
            padded_zone_high = np.concatenate([zone_high, pad_zeros])
        else:
            padded_zone_low = zone_low
            padded_zone_high = zone_high

        # Concatenate: padded zones + non-zone features
        new_low = np.concatenate([padded_zone_low, non_zone_low])
        new_high = np.concatenate([padded_zone_high, non_zone_high])

        self.observation_space = gym.spaces.Box(low=new_low, high=new_high, dtype=dtype)

    # ------------------------------------------------------------------
    def reset(self, **kwargs):  # type: ignore[override]
        """Reset and re-read inner observation space (may have changed)."""
        obs, info = self.env.reset(**kwargs)
        self._rebuild_observation_space()
        return self.observation(obs), info

    def observation(self, obs: np.ndarray) -> np.ndarray:
        """Apply zone-aware padding to observation.

        Pads zone temperatures to max_zones, keeping non-zone features at
        consistent positions.
        """
        obs = np.asarray(obs, dtype=self.observation_space.dtype)
        obs_size = obs.shape[0]

        if obs_size > self._target_size:
            raise ValueError(
                f"Observation size ({obs_size}) exceeds target_size "
                f"({self._target_size}). Increase target_obs_size in "
                "your config."
            )

        zone_temps, non_zone_features, _zone_mask = self._split_zone_non_zone(obs)
        current_num_zones = zone_temps.shape[0]

        if current_num_zones > self._max_zones:
            raise ValueError(
                f"Building has {current_num_zones} zones, exceeds max_zones "
                f"({self._max_zones}). Increase target_obs_size."
            )

        # Pad zone temperatures to max_zones
        num_pad_zones = self._max_zones - current_num_zones
        if num_pad_zones > 0:
            padded_zones = np.concatenate(
                [zone_temps, np.zeros(num_pad_zones, dtype=obs.dtype)]
            )
        else:
            padded_zones = zone_temps

        # Concatenate: padded zones + non-zone features
        return np.concatenate([padded_zones, non_zone_features])


class AugmentObservationWithBuildingParams(gym.ObservationWrapper):
    """
    Augment observations with normalized building parameters.

    This wrapper adds building-specific parameters to the observation space,
    allowing a single policy to generalize across multiple buildings.

    Building parameters included:
    - area: Building floor area (m²)
    - warmup_phases: Number of warmup phases
    - num_actuators: Number of HVAC actuators
    - year_built: Year the building was constructed
    - num_units: Number of units in the building
    """

    # Down-scale for the normalized building params. They are normalized to
    # [-1, 1], but the base observation (after NormalizeObservation /
    # PadObservation) has per-step std ~0.02-0.15. Building params change only
    # at episode boundaries, so within a rollout batch their std is roughly
    # 0.20-0.25 — about 5× larger than the base obs. Without rescaling they
    # dominate the first-layer activations and the policy overfits to a coarse
    # signal. Dividing by this constant brings the per-batch std into the same
    # ballpark as the base obs (~0.04-0.05).
    PARAM_OUTPUT_SCALE: float = 5.0

    def __init__(
        self,
        env: gym.Env,
        building_params: dict[str, float] | None = None,
        *,
        allow_defaults: bool = False,
    ):
        """
        Args:
            env: The environment to wrap
            building_params: Dictionary of building parameters to append.
                           If None, will try to extract from env metadata.
            allow_defaults: If True and metadata is missing, keep the legacy
                           default-filling behavior. If False (default),
                           missing required keys raise KeyError.
        """
        super().__init__(env)

        # Ensure the observation space is a Box
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise ValueError(
                f"Expected observation space to be Box, "
                f"got {type(env.observation_space)}"
            )

        # Extract or use provided building parameters
        if building_params is None:
            building_params = self._extract_building_params(
                env, allow_defaults=allow_defaults
            )

        self.building_params = building_params
        self._allow_defaults = allow_defaults

        # Normalize building parameters for better learning
        self.normalized_params = self._normalize_params(building_params)

        # Create augmented observation space
        orig_low = env.observation_space.low
        orig_high = env.observation_space.high

        # Add parameter dimensions (normalized to [-1, 1] then scaled by
        # 1 / PARAM_OUTPUT_SCALE so per-batch std matches the base obs).
        _param_max = 1.0 / float(self.PARAM_OUTPUT_SCALE) if self.PARAM_OUTPUT_SCALE else 1.0
        param_low = np.full(len(self.normalized_params), -_param_max, dtype=orig_low.dtype)
        param_high = np.full(len(self.normalized_params), _param_max, dtype=orig_high.dtype)

        new_low = np.concatenate([orig_low, param_low])
        new_high = np.concatenate([orig_high, param_high])

        self.observation_space = gym.spaces.Box(
            low=new_low,
            high=new_high,
            dtype=env.observation_space.dtype,
        )

        logger.info(
            "Augmented observation space from %d to %d dimensions. "
            "Building params: %s",
            len(orig_low),
            len(new_low),
            list(building_params.keys()),
        )

    def _extract_building_params(
        self, env: gym.Env, *, allow_defaults: bool
    ) -> dict[str, float]:
        """Extract building parameters from environment metadata."""
        params: dict[str, float] = {}

        # Try to get parameters from env metadata
        unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env

        if hasattr(unwrapped, "metadata") and isinstance(unwrapped.metadata, dict):
            metadata = unwrapped.metadata
            # Extract area
            if "area" in metadata:
                params["area"] = float(metadata["area"])
            # Extract warmup_phases
            if "warmup_phases" in metadata:
                params["warmup_phases"] = float(metadata["warmup_phases"])
            # Extract num_actuators from hvac_actuators list
            if "hvac_actuators" in metadata:
                params["num_actuators"] = float(len(metadata["hvac_actuators"]))

            # Extract additional parameters from building_source_metadata
            source_meta = metadata.get("building_source_metadata", {})
            if isinstance(source_meta, dict):
                if (
                    "year_built" in source_meta
                    and source_meta["year_built"] is not None
                ):
                    try:
                        params["year_built"] = float(source_meta["year_built"])
                    except (TypeError, ValueError):
                        pass
                num_units_key = "geometry_building_num_units"
                if (
                    num_units_key in source_meta
                    and source_meta[num_units_key] is not None
                ):
                    try:
                        params["num_units"] = float(source_meta[num_units_key])
                    except (TypeError, ValueError):
                        pass

        # Use defaults for any missing parameters
        defaults: dict[str, float] = {
            "area": 100.0,
            "warmup_phases": 3.0,
            "num_actuators": 1.0,
            "year_built": 1980.0,
            "num_units": 1.0,
        }
        for key, default_val in defaults.items():
            if key not in params:
                if not allow_defaults:
                    raise KeyError(key)
                logger.warning(
                    "Could not extract %r from env metadata, using default",
                    key,
                )
                params[key] = default_val

        return params

    def _normalize_params(self, params: dict[str, float]) -> np.ndarray:
        """Normalize building parameters to reasonable ranges.

        Logs warnings when values are clipped (outside expected ranges).
        """
        param_ranges = {
            "area": (25.0, 1200.0),  # m² — spans SFH + OfficeSmall
            "warmup_phases": (1.0, 10.0),
            "num_actuators": (1.0, 20.0),
            "year_built": (1940.0, 2025.0),  # year range for building stock
            "num_units": (1.0, 10.0),  # number of units in the building
        }

        normalized = []

        for key, value in params.items():
            if key in param_ranges:
                min_val, max_val = param_ranges[key]
                norm_val_unclipped = 2.0 * (value - min_val) / (max_val - min_val) - 1.0
                norm_val = np.clip(norm_val_unclipped, -1.0, 1.0)

                if norm_val != norm_val_unclipped:
                    logger.warning(
                        "Building parameter %r clipped: value=%.2f outside range [%.2f, %.2f], "
                        "normalized from %.3f to %.3f",
                        key,
                        value,
                        min_val,
                        max_val,
                        norm_val_unclipped,
                        norm_val,
                    )
            else:
                # Unknown parameter, just clip to reasonable range
                norm_val_unclipped = value / 100.0
                norm_val = np.clip(norm_val_unclipped, -1.0, 1.0)

                if norm_val != norm_val_unclipped:
                    logger.warning(
                        "Unknown building parameter %r clipped: value=%.2f, "
                        "normalized from %.3f to %.3f",
                        key,
                        value,
                        norm_val_unclipped,
                        norm_val,
                    )

            normalized.append(norm_val)

        arr = np.array(normalized, dtype=np.float32)
        if self.PARAM_OUTPUT_SCALE and self.PARAM_OUTPUT_SCALE != 1.0:
            arr = arr / float(self.PARAM_OUTPUT_SCALE)
        return arr

    def reset(self, **kwargs):  # type: ignore[override]
        """Reset and re-extract building parameters (inner env may have changed)."""
        obs, info = self.env.reset(**kwargs)

        # Re-extract in case the inner env was swapped (e.g. by ResampleBuildingOnResetWrapper)
        self.building_params = self._extract_building_params(
            self.env, allow_defaults=self._allow_defaults
        )
        self.normalized_params = self._normalize_params(self.building_params)

        # Rebuild observation space in case inner env's obs shape changed
        orig_low = self.env.observation_space.low
        orig_high = self.env.observation_space.high

        _param_max = 1.0 / float(self.PARAM_OUTPUT_SCALE) if self.PARAM_OUTPUT_SCALE else 1.0
        param_low = np.full(len(self.normalized_params), -_param_max, dtype=orig_low.dtype)
        param_high = np.full(len(self.normalized_params), _param_max, dtype=orig_high.dtype)

        new_low = np.concatenate([orig_low, param_low])
        new_high = np.concatenate([orig_high, param_high])

        self.observation_space = gym.spaces.Box(
            low=new_low,
            high=new_high,
            dtype=self.env.observation_space.dtype,
        )

        return self.observation(obs), info

    def observation(self, obs: np.ndarray) -> np.ndarray:
        """Augment observation with normalized building parameters."""
        return np.concatenate([obs, self.normalized_params])

    def denormalize(self, obs: np.ndarray) -> np.ndarray:
        """
        Remove building parameters and denormalize the original observation.

        This is used for logging/visualization purposes.
        """
        # Split off the building parameters (last N dimensions)
        n_params = len(self.normalized_params)
        obs_without_params = obs[:-n_params] if n_params > 0 else obs

        # If the wrapped env has a denormalize method, use it
        if hasattr(self.env, "denormalize"):
            return self.env.denormalize(obs_without_params)

        # Otherwise, just return the observation without building params
        return obs_without_params


class ResampleBuildingOnResetWrapper(gym.Wrapper):
    """Resample a new building environment on each episode reset.

    On every call to ``reset()``, a new index is drawn uniformly from
    ``available_indices`` and, if it differs from the current one, the
    old environment is closed and a fresh one is created via
    ``env_factory``.

    At each reset the wrapper also:

    * logs the **previous** episode's summary statistics (total reward,
      episode length) to W&B, and
    * logs the **new** building's environment parameters to W&B.

    W&B logging is best-effort: if ``wandb`` is not installed or no
    active run exists the wrapper still works normally.

    The wrapper intentionally catches ``IndexError`` in ``step()`` to
    avoid hard-crashing multi-building runs on actuator mismatches. This
    branch emits a ``RuntimeWarning``, returns a terminal transition with
    zero reward, and forces environment recreation on the next ``reset()``.

    Args:
        env_factory: ``env_factory(index) -> gym.Env``.  Called to
            create a new environment for the given index.
        available_indices: Non-empty sequence of integer indices that
            ``env_factory`` accepts.
        wandb_prefix: Prefix for all W&B log keys.  Use ``"train"``
            for training environments and ``"eval"`` for evaluation
            environments so that their metrics appear in separate
            W&B panels.
    """

    def __init__(
        self,
        env_factory: Callable[[int], gym.Env],
        available_indices: list[int],
        wandb_prefix: str = "train",
        log_interval_steps: int = 2048,
    ):
        if not available_indices:
            raise ValueError("available_indices must not be empty")

        self._env_factory = env_factory
        self._available_indices = list(available_indices)
        self._current_index = random.choice(self._available_indices)
        self._wandb_prefix = wandb_prefix
        self._log_interval_steps = log_interval_steps

        initial_env = env_factory(self._current_index)
        super().__init__(initial_env)

        # Episode tracking
        self._episode_reward: float = 0.0
        self._episode_steps: int = 0
        self._episode_count: int = 0
        self._has_stepped: bool = False
        self._total_steps: int = 0
        self._steps_since_last_log: int = 0
        self._force_resample_next_reset: bool = False
        self._last_obs: Any = None

        logger.info(
            "ResampleBuildingOnResetWrapper(%s): %d buildings available, log_interval=%d",
            wandb_prefix,
            len(self._available_indices),
            log_interval_steps,
        )

    # ------------------------------------------------------------------
    # wandb helpers (best-effort, never raise)
    # ------------------------------------------------------------------

    @staticmethod
    def _wandb_is_active() -> bool:
        try:
            import wandb  # type: ignore[import-untyped]

            return getattr(wandb, "run", None) is not None
        except Exception:
            return False

    def _log_episode_summary(self) -> None:
        """Log previous episode reward / length to wandb."""
        if not self._wandb_is_active():
            return
        try:
            import wandb  # type: ignore[import-untyped]

            p = self._wandb_prefix
            wandb.log(
                {
                    f"{p}/episode/reward": self._episode_reward,
                    f"{p}/episode/length": self._episode_steps,
                    f"{p}/episode/number": self._episode_count,
                    f"{p}/episode/building_index": self._current_index,
                },
            )
        except Exception as exc:
            logger.warning("wandb episode log failed: %s", exc)

    def _log_building_params(self) -> None:
        """Log the current building index to wandb."""
        if not self._wandb_is_active():
            return
        try:
            import wandb  # type: ignore[import-untyped]

            meta = getattr(self.env, "metadata", None)
            src = (
                meta.get("building_source_metadata", {})
                if isinstance(meta, dict)
                else {}
            )

            p = self._wandb_prefix
            payload: dict[str, Any] = {
                f"{p}/building/split_index": self._current_index,
            }
            bid = src.get("building_id")
            if bid is not None:
                payload[f"{p}/building/id"] = str(bid)

            wandb.log(payload)
        except Exception as exc:
            logger.warning("wandb building-param log failed: %s", exc)

    def _log_intermediate_reward(self) -> None:
        """Log cumulative reward during long episodes (before completion)."""
        if not self._wandb_is_active():
            return
        try:
            import wandb  # type: ignore[import-untyped]

            p = self._wandb_prefix
            wandb.log(
                {
                    f"{p}/episode/cumulative_reward": self._episode_reward,
                    f"{p}/episode/current_length": self._episode_steps,
                    f"{p}/episode/current_number": self._episode_count,
                },
            )
        except Exception as exc:
            logger.warning("wandb intermediate reward log failed: %s", exc)

    # ------------------------------------------------------------------
    # gym.Wrapper overrides
    # ------------------------------------------------------------------

    def step(self, action):  # type: ignore[override]
        try:
            obs, reward, terminated, truncated, info = self.env.step(action)
            self._last_obs = obs
            self._episode_reward += float(reward)
            self._episode_steps += 1
            self._total_steps += 1
            self._steps_since_last_log += 1
            self._has_stepped = True

            # Log intermediate rewards every N steps (for long episodes)
            if self._steps_since_last_log >= self._log_interval_steps:
                self._log_intermediate_reward()
                self._steps_since_last_log = 0

            return obs, reward, terminated, truncated, info

        except IndexError as exc:
            msg = (
                "IndexError in building "
                f"{self._current_index} during step (likely actuator mismatch): {exc}. "
                "Episode terminated and building will be resampled on next reset."
            )
            # warnings.warn is deduplicated per call-site by the default filter,
            # so it surfaces only the first occurrence in a long run; the logger
            # line ensures every actuator mismatch is recorded.
            warnings.warn(msg, RuntimeWarning)
            logger.warning(msg)
            self._force_resample_next_reset = True
            if self._last_obs is None:
                self._last_obs = self.observation_space.sample()
            return self._last_obs, 0.0, True, False, {"resample_pending": True}

    def reset(self, **kwargs):  # type: ignore[override]
        """Reset with a newly sampled building."""
        # Log the *previous* episode's summary (skip the very first reset)
        if self._has_stepped:
            self._log_episode_summary()

        # Sample a new building
        new_index = random.choice(self._available_indices)

        should_recreate = self._force_resample_next_reset or (
            new_index != self._current_index
        )
        if should_recreate:
            logger.info(
                "Resampling building: index %d -> %d",
                self._current_index,
                new_index,
            )
            self.env.close()
            self.env = self._env_factory(new_index)
            self._current_index = new_index
            self._force_resample_next_reset = False

        # Reset episode counters
        self._episode_reward = 0.0
        self._episode_steps = 0
        self._episode_count += 1
        self._has_stepped = False

        obs_info = self.env.reset(**kwargs)
        self._last_obs = obs_info[0]

        # Log the *new* building's parameters
        self._log_building_params()

        return obs_info
