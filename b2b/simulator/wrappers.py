import logging
import random
from typing import Any, Callable

import gymnasium as gym
import numpy as np

logger = logging.getLogger(__name__)


class SetpointDeltaActionWrapper(gym.Wrapper):
    """
    Interpret Zone Temperature Control setpoint actions as *deltas* from the current zone air temperature.

    Motivation: we want actions to be "add ΔT to current temperature" rather than absolute setpoints.
    This keeps policies invariant to absolute temperature scales and avoids hardcoding setpoint ranges.
    """

    _ABS_SETPOINT_MIN_C = 0.0
    _ABS_SETPOINT_MAX_C = 50.0
    _MIN_DEADBAND_C = 0.5
    _DELTA_MIN_C = -10.0
    _DELTA_MAX_C = 10.0

    def __init__(self, env: gym.Env):
        super().__init__(env)

        if not hasattr(env, "metadata") or not isinstance(env.metadata, dict):
            raise RuntimeError(
                "env.metadata missing; required for setpoint delta mapping"
            )
        obs_names = env.metadata.get("observation_names")
        act_names = env.metadata.get("action_names")
        if not isinstance(obs_names, list) or not all(
            isinstance(x, str) for x in obs_names
        ):
            raise RuntimeError("env.metadata['observation_names'] must be a list[str]")
        if not isinstance(act_names, list) or not all(
            isinstance(x, str) for x in act_names
        ):
            raise RuntimeError("env.metadata['action_names'] must be a list[str]")

        self._obs_names = obs_names
        self._act_names = act_names
        self._last_obs: np.ndarray | None = None

        # Per-zone mapping so we can enforce heating/cooling ordering.
        # zone_name -> {"obs_idx": int, "heat_act_idx": int?, "cool_act_idx": int?}
        self._zones: dict[str, dict[str, int]] = {}
        for i, name in enumerate(act_names):
            parts = str(name).split("::")
            if len(parts) < 3:
                continue
            component_type = parts[0].strip().lower()
            control_type = parts[1].strip().lower()
            zone_name = parts[2].strip()
            if component_type != "zone temperature control":
                continue
            if control_type not in ("heating setpoint", "cooling setpoint"):
                continue
            obs_idx = self._find_zone_air_temp_index(zone_name)
            z = self._zones.setdefault(zone_name, {"obs_idx": int(obs_idx)})
            if control_type == "heating setpoint":
                z["heat_act_idx"] = int(i)
            else:
                z["cool_act_idx"] = int(i)

        # Narrow the delta action bounds for setpoint actuators to make random sampling safe.
        if isinstance(getattr(env, "action_space", None), gym.spaces.Box):
            low = np.asarray(env.action_space.low, dtype=float).reshape(-1).copy()
            high = np.asarray(env.action_space.high, dtype=float).reshape(-1).copy()

            for z in self._zones.values():
                for k in ("heat_act_idx", "cool_act_idx"):
                    if k in z:
                        idx = int(z[k])
                        low[idx] = self._DELTA_MIN_C
                        high[idx] = self._DELTA_MAX_C

            self.action_space = gym.spaces.Box(
                low=low.astype(env.action_space.dtype),
                high=high.astype(env.action_space.dtype),
                dtype=env.action_space.dtype,
            )

        # Keep metadata visible on the wrapper (gymnasium uses env.metadata for rendering, etc).
        self.metadata = getattr(env, "metadata", {})

    def _find_zone_air_temp_index(self, zone_name: str) -> int:
        zn = zone_name.strip().lower()
        prefix = "zone air temperature"

        # Exact-ish match: "Zone Air Temperature {ZONE}"
        for j, n in enumerate(self._obs_names):
            s = str(n).strip()
            sl = s.lower()
            if not sl.startswith(prefix):
                continue
            zone_part = sl[len(prefix) :].strip()
            if zone_part == zn:
                return int(j)

        # Substring match fallback (EnergyPlus sometimes decorates names).
        for j, n in enumerate(self._obs_names):
            sl = str(n).strip().lower()
            if not sl.startswith(prefix):
                continue
            zone_part = sl[len(prefix) :].strip()
            if zn in zone_part or zone_part in zn:
                return int(j)

        raise RuntimeError(
            f"Could not map Zone Temperature Control zone '{zone_name}' to a Zone Air Temperature obs"
        )

    def reset(self, **kwargs):  # type: ignore[override]
        obs, info = self.env.reset(**kwargs)
        self._last_obs = np.asarray(obs, dtype=float).reshape(-1)
        return obs, info

    def step(self, action):  # type: ignore[override]
        if self._last_obs is None:
            raise RuntimeError("SetpointDeltaActionWrapper.step called before reset()")

        act = np.asarray(action, dtype=float).reshape(-1).copy()
        obs_arr = self._last_obs

        for z in self._zones.values():
            tz = float(obs_arr[int(z["obs_idx"])])
            heat_idx = z.get("heat_act_idx")
            cool_idx = z.get("cool_act_idx")

            heat_abs: float | None = None
            cool_abs: float | None = None

            if heat_idx is not None:
                heat_abs = float(
                    np.clip(
                        tz + float(act[int(heat_idx)]),
                        self._ABS_SETPOINT_MIN_C,
                        self._ABS_SETPOINT_MAX_C,
                    )
                )
            if cool_idx is not None:
                cool_abs = float(
                    np.clip(
                        tz + float(act[int(cool_idx)]),
                        self._ABS_SETPOINT_MIN_C,
                        self._ABS_SETPOINT_MAX_C,
                    )
                )

            if heat_abs is not None and cool_abs is not None:
                if cool_abs < heat_abs + self._MIN_DEADBAND_C:
                    cool_abs = min(
                        heat_abs + self._MIN_DEADBAND_C, self._ABS_SETPOINT_MAX_C
                    )
                    if cool_abs < heat_abs + self._MIN_DEADBAND_C:
                        heat_abs = max(
                            cool_abs - self._MIN_DEADBAND_C, self._ABS_SETPOINT_MIN_C
                        )

            if heat_idx is not None and heat_abs is not None:
                act[int(heat_idx)] = heat_abs
            if cool_idx is not None and cool_abs is not None:
                act[int(cool_idx)] = cool_abs

        obs, reward, terminated, truncated, info = self.env.step(act)
        self._last_obs = np.asarray(obs, dtype=float).reshape(-1)
        return obs, reward, terminated, truncated, info


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

        # Store the original observation space bounds
        self.obs_low = self.env.observation_space.low
        self.obs_high = self.env.observation_space.high

        # Handle infinite bounds by replacing them with large finite values
        self.obs_low = np.where(np.isinf(self.obs_low), -1e10, self.obs_low)
        self.obs_high = np.where(np.isinf(self.obs_high), 1e10, self.obs_high)

        # Calculate the range of the observation space
        self.obs_range = self.obs_high - self.obs_low
        # Avoid division by zero for dimensions with zero range
        self.obs_range = np.where(self.obs_range == 0, 1.0, self.obs_range)

        # Set the new observation space to be in the target range
        target_low = np.zeros(self.obs_low.shape, dtype=dtype)
        target_high = np.ones(self.obs_high.shape, dtype=dtype)

        self.observation_space = gym.spaces.Box(
            low=target_low, high=target_high, dtype=dtype
        )

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

    def __init__(self, env: gym.Env, building_params: dict[str, float] | None = None):
        """
        Args:
            env: The environment to wrap
            building_params: Dictionary of building parameters to append.
                           If None, will try to extract from env metadata.
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
            building_params = self._extract_building_params(env)

        self.building_params = building_params

        # Normalize building parameters for better learning
        self.normalized_params = self._normalize_params(building_params)

        # Create augmented observation space
        orig_low = env.observation_space.low
        orig_high = env.observation_space.high

        # Add parameter dimensions (normalized to [-1, 1])
        param_low = -np.ones(len(self.normalized_params), dtype=orig_low.dtype)
        param_high = np.ones(len(self.normalized_params), dtype=orig_high.dtype)

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

    def _extract_building_params(self, env: gym.Env) -> dict[str, float]:
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
                logger.warning(
                    "Could not extract %r from env metadata, using default",
                    key,
                )
                params[key] = default_val

        return params

    def _normalize_params(self, params: dict[str, float]) -> np.ndarray:
        """Normalize building parameters to reasonable ranges."""
        # Define normalization ranges (min, max) for each parameter
        param_ranges = {
            "area": (50.0, 500.0),  # m² - typical range for buildings
            "warmup_phases": (1.0, 10.0),
            "num_actuators": (1.0, 20.0),
            "num_zones": (1.0, 10.0),
            "year_built": (1940.0, 2025.0),  # year range for building stock
            "num_units": (1.0, 10.0),  # number of units in the building
        }

        normalized = []
        for key, value in params.items():
            if key in param_ranges:
                min_val, max_val = param_ranges[key]
                # Normalize to [-1, 1]
                norm_val = 2.0 * (value - min_val) / (max_val - min_val) - 1.0
                norm_val = np.clip(norm_val, -1.0, 1.0)
            else:
                # Unknown parameter, just clip to reasonable range
                norm_val = np.clip(value / 100.0, -1.0, 1.0)

            normalized.append(norm_val)

        return np.array(normalized, dtype=np.float32)

    def reset(self, **kwargs):  # type: ignore[override]
        """Reset and re-extract building parameters (inner env may have changed)."""
        obs, info = self.env.reset(**kwargs)

        # Re-extract in case the inner env was swapped (e.g. by ResampleBuildingOnResetWrapper)
        self.building_params = self._extract_building_params(self.env)
        self.normalized_params = self._normalize_params(self.building_params)

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

    Args:
        env_factory: ``env_factory(index) -> gym.Env``.  Called to
            create a new environment for the given index.
        available_indices: Non-empty sequence of integer indices that
            ``env_factory`` accepts.
    """

    def __init__(
        self,
        env_factory: Callable[[int], gym.Env],
        available_indices: list[int],
    ):
        if not available_indices:
            raise ValueError("available_indices must not be empty")

        self._env_factory = env_factory
        self._available_indices = list(available_indices)
        self._current_index = random.choice(self._available_indices)

        initial_env = env_factory(self._current_index)
        super().__init__(initial_env)

        # Episode tracking
        self._episode_reward: float = 0.0
        self._episode_steps: int = 0
        self._episode_count: int = 0
        self._has_stepped: bool = False

        logger.info(
            "ResampleBuildingOnResetWrapper: %d buildings available",
            len(self._available_indices),
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

            wandb.log(
                {
                    "episode/reward": self._episode_reward,
                    "episode/length": self._episode_steps,
                    "episode/number": self._episode_count,
                    "episode/building_index": self._current_index,
                },
            )
        except Exception as exc:
            logger.debug("wandb episode log failed: %s", exc)

    def _log_building_params(self) -> None:
        """Log the current building's environment parameters to wandb."""
        if not self._wandb_is_active():
            return
        try:
            import wandb  # type: ignore[import-untyped]

            meta = getattr(self.env, "metadata", None)
            if not isinstance(meta, dict):
                return

            payload: dict[str, Any] = {
                "building/split_index": self._current_index,
            }

            # Core building parameters
            for key in ("area", "warmup_phases"):
                if key in meta:
                    payload[f"building/{key}"] = float(meta[key])

            if "hvac_actuators" in meta:
                payload["building/num_actuators"] = len(meta["hvac_actuators"])

            if "controlled_zones" in meta:
                payload["building/num_controlled_zones"] = len(meta["controlled_zones"])

            # Source metadata (year_built, num_units, …)
            src = meta.get("building_source_metadata")
            if isinstance(src, dict):
                for src_key in (
                    "year_built",
                    "geometry_building_num_units",
                ):
                    val = src.get(src_key)
                    if val is not None:
                        try:
                            payload[f"building/{src_key}"] = float(val)
                        except (TypeError, ValueError):
                            pass

            wandb.log(payload)
        except Exception as exc:
            logger.debug("wandb building-param log failed: %s", exc)

    # ------------------------------------------------------------------
    # gym.Wrapper overrides
    # ------------------------------------------------------------------

    def step(self, action):  # type: ignore[override]
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._episode_reward += float(reward)
        self._episode_steps += 1
        self._has_stepped = True
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):  # type: ignore[override]
        """Reset with a newly sampled building."""
        # Log the *previous* episode's summary (skip the very first reset)
        if self._has_stepped:
            self._log_episode_summary()

        # Sample a new building
        new_index = random.choice(self._available_indices)

        if new_index != self._current_index:
            logger.info(
                "Resampling building: index %d -> %d",
                self._current_index,
                new_index,
            )
            self.env.close()
            self.env = self._env_factory(new_index)
            self._current_index = new_index

        # Reset episode counters
        self._episode_reward = 0.0
        self._episode_steps = 0
        self._episode_count += 1
        self._has_stepped = False

        obs_info = self.env.reset(**kwargs)

        # Log the *new* building's parameters
        self._log_building_params()

        return obs_info
