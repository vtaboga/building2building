"""Named task presets from the Building2Building.

Each preset fully specifies the reward function, target temperature mode,
and temperature setpoints for a reproducible benchmark task.

Nine presets form a 3x3 grid parameterized along
``(setpoint_mode, energy_weight_level)``.  All use
:class:`~building2building.types.NormalizedDeadbandRewardConfig`
with ``dT=1.0`` and ``(tau_T, tau_E) = (None, None)`` (the unfilled
sentinel state); :func:`building2building.api.make_env` resolves
the per-(building_type, climate_zone) constants from
:file:`building2building/data/reward_normalizers.yaml` at env-build
time via :func:`make_normalized_deadband_task`.

Naming convention:

* mode component (calibration regime is ``occ``):
    * ``const`` -- ``target_temperature_mode="constant"``, 21 C / 21 C.
    * ``occ``   -- ``target_temperature_mode="occupancy"`` with the
      seasonal unoccupied policy (winter 18 C / shoulder 21 C /
      summer 26 C).  **This is the calibration regime**.
    * ``rand``  -- ``target_temperature_mode="random_schedule"``
      (per-day arrival/departure + setpoints).
* energy-weight component:
    * ``e0``    -- ``energy_weight = 0.0`` (comfort-only).
    * ``emed``  -- ``energy_weight = 1.0`` (balanced under the
      calibration random policy by construction).
    * ``ehigh`` -- ``energy_weight = 5.0`` (energy emphasis).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from building2building.types import (
    DEFAULT_SEASONAL_UNOCCUPIED_C,
    NormalizedDeadbandRewardConfig,
    RewardConfig,
    SeasonName,
    TargetTemperatureMode,
    UnoccupiedPolicy,
)

SetpointMode = Literal["constant", "occupancy", "random_schedule"]
WeightLevel = Literal["e0", "emed", "ehigh"]
ModeShort = Literal["const", "occ", "rand"]

#: Energy-weight values for the 3-level grid.  ``emed = 1.0`` is the
#: "balanced under the calibration random policy" anchor; the two
#: endpoints are chosen to span comfort-only and energy-emphasis regimes
#: without making the comparison degenerate.
NORMALIZED_WEIGHT_LEVELS: dict[WeightLevel, float] = {
    "e0": 0.0,
    "emed": 1.0,
    "ehigh": 5.0,
}

#: Map between short preset names and full :class:`TaskConfig` modes.
NORMALIZED_MODES: dict[ModeShort, SetpointMode] = {
    "const": "constant",
    "occ": "occupancy",
    "rand": "random_schedule",
}

#: Calibration regime baked into ``reward_normalizers.yaml`` -- used
#: only to flag the ``task_occ_*`` presets as the "in-distribution"
#: subgroup in this module's docstring; the actual runtime check
#: lives in :mod:`building2building.simulator`.
CALIBRATION_MODE: SetpointMode = "occupancy"
CALIBRATION_DT: float = 1.0


@dataclass(frozen=True)
class TaskPreset:
    """A named combination of reward and task parameters.

    Attributes:
        reward: Reward function configuration.
        target_temperature_mode: How target temperature varies
            (``"constant"``, ``"occupancy"``-based, or
            ``"random_schedule"``).
        target_temperature_occupied: Target when zone is occupied
            (C).  Used as the *fallback* occupied setpoint for
            ``"random_schedule"``.
        target_temperature_unoccupied: Target when zone is unoccupied
            (C).  Used when ``unoccupied_policy == "fixed"``; acts
            as fallback otherwise.
        unoccupied_policy: ``"fixed"`` (default) keeps
            :attr:`target_temperature_unoccupied` constant;
            ``"seasonal"`` dispatches to :attr:`seasonal_unoccupied_c`
            based on the simulation month.
        seasonal_unoccupied_c: Per-season unoccupied setpoints (C)
            used when ``unoccupied_policy == "seasonal"``.  ``None``
            means "fall back to :data:`DEFAULT_SEASONAL_UNOCCUPIED_C`"
            at the resolution site.
    """

    reward: RewardConfig
    target_temperature_mode: TargetTemperatureMode
    target_temperature_occupied: float
    target_temperature_unoccupied: float
    unoccupied_policy: UnoccupiedPolicy = "fixed"
    seasonal_unoccupied_c: dict[SeasonName, float] | None = field(default=None)


def _build_normalized_preset(
    *,
    mode_short: ModeShort,
    level: WeightLevel,
) -> TaskPreset:
    """Build one of the 9 normalized presets.

    The returned preset stores
    :class:`~building2building.types.NormalizedDeadbandRewardConfig`
    with ``tau_T = tau_E = None``: the env factory fills these in
    based on the resolved ``(building_type, building_id)`` at env-build
    time.  ``dT`` is fixed at the calibration value (``1.0``) for the
    preset; users wanting to sweep ``dT`` should construct a custom
    :class:`TaskPreset` directly.
    """
    mode = NORMALIZED_MODES[mode_short]
    energy_weight = NORMALIZED_WEIGHT_LEVELS[level]

    reward = NormalizedDeadbandRewardConfig(
        energy_weight=energy_weight,
        dT=CALIBRATION_DT,
        tau_T=None,
        tau_E=None,
    )

    if mode == "constant":
        return TaskPreset(
            reward=reward,
            target_temperature_mode="constant",
            target_temperature_occupied=21.0,
            target_temperature_unoccupied=21.0,
        )
    if mode == "occupancy":
        # Seasonal unoccupied policy for the occupancy calibration regime.
        # Fallback unoccupied_c=18 is used only when
        # unoccupied_policy="fixed" overrides at the env factory site.
        return TaskPreset(
            reward=reward,
            target_temperature_mode="occupancy",
            target_temperature_occupied=21.0,
            target_temperature_unoccupied=18.0,
            unoccupied_policy="seasonal",
            seasonal_unoccupied_c=dict(DEFAULT_SEASONAL_UNOCCUPIED_C),
        )
    # Random schedule: occupied/unoccupied act as fallbacks; the
    # per-day generator overrides them at runtime.
    return TaskPreset(
        reward=reward,
        target_temperature_mode="random_schedule",
        target_temperature_occupied=21.0,
        target_temperature_unoccupied=18.0,
    )


def _normalized_preset_name(mode_short: ModeShort, level: WeightLevel) -> str:
    return f"task_{mode_short}_{level}"


_NORMALIZED_TASK_PRESETS: dict[str, TaskPreset] = {
    _normalized_preset_name(mode_short, level): _build_normalized_preset(
        mode_short=mode_short, level=level
    )
    for mode_short in ("const", "occ", "rand")
    for level in ("e0", "emed", "ehigh")
}


TASK_PRESETS: dict[str, TaskPreset] = dict(_NORMALIZED_TASK_PRESETS)


def resolve_task_preset(task: str) -> TaskPreset:
    """Look up a named task preset.

    Recognised names:

    * Normalized 3x3 family: ``"task_<mode>_<level>"`` where ``mode ∈
      {const, occ, rand}`` and ``level ∈ {e0, emed, ehigh}`` -- 9
      combinations total.

    Args:
        task: Preset name.

    Returns:
        The corresponding :class:`TaskPreset`.

    Raises:
        KeyError: If *task* is not a recognised preset name.
    """
    if task not in TASK_PRESETS:
        raise KeyError(
            f"Unknown task preset {task!r}. "
            f"Available: {sorted(TASK_PRESETS.keys())}"
        )
    return TASK_PRESETS[task]


def make_normalized_deadband_task(
    building_type: str,
    building_id: str,
    *,
    w_E: float,
    mode: SetpointMode = "occupancy",
    dT: float = CALIBRATION_DT,
) -> TaskPreset:
    """Return a *filled* :class:`TaskPreset` for a normalized deadband task.

    Useful for ad-hoc experiments that want to choose ``w_E`` and
    ``mode`` outside the 9-preset grid (e.g. ``w_E = 2.5`` for a finer
    sweep, or a non-calibration ``dT`` value).

    The ``(tau_T, tau_E)`` constants are looked up via
    :func:`building2building.data.reward_normalizers.resolve_reward_normalizer`
    using the building's ``(building_type, climate_zone)`` bucket.

    Args:
        building_type: e.g. ``"OfficeMedium"``.
        building_id: Dataset building identifier.
        w_E: Dimensionless energy-trade-off weight.  See
            :class:`~building2building.types.NormalizedDeadbandRewardConfig`.
        mode: Setpoint mode.  Calibration assumes ``"occupancy"``;
            other values are accepted but cause the simulator to emit
            a calibration-mismatch :class:`RuntimeWarning`.
        dT: Comfort deadband (C); same caveat -- calibration uses 1.0.

    Returns:
        A :class:`TaskPreset` whose reward is a *filled*
        :class:`~building2building.types.NormalizedDeadbandRewardConfig`
        ready for the simulator dispatch site.

    Raises:
        building2building.data.reward_normalizers.RewardNormalizersUnavailableError:
            If ``reward_normalizers.yaml`` has not been generated yet.
    """
    # Lazy import to keep ``config.tasks`` free of dataset-loading
    # side effects at import time.
    from building2building.data.reward_normalizers import resolve_reward_normalizer

    normalizer = resolve_reward_normalizer(building_type, building_id)
    reward = NormalizedDeadbandRewardConfig(
        energy_weight=float(w_E),
        dT=float(dT),
        tau_T=None,
        tau_E=None,
    ).filled(normalizer.tau_T, normalizer.tau_E)

    if mode == "constant":
        return TaskPreset(
            reward=reward,
            target_temperature_mode="constant",
            target_temperature_occupied=21.0,
            target_temperature_unoccupied=21.0,
        )
    if mode == "occupancy":
        return TaskPreset(
            reward=reward,
            target_temperature_mode="occupancy",
            target_temperature_occupied=21.0,
            target_temperature_unoccupied=18.0,
            unoccupied_policy="seasonal",
            seasonal_unoccupied_c=dict(DEFAULT_SEASONAL_UNOCCUPIED_C),
        )
    return TaskPreset(
        reward=reward,
        target_temperature_mode="random_schedule",
        target_temperature_occupied=21.0,
        target_temperature_unoccupied=18.0,
    )
