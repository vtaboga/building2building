"""Core domain types for Building2Building.

Defines task, reward, actuator, and building configuration dataclasses
used across the simulation and RL pipeline.

The only supported reward type is
:class:`NormalizedDeadbandRewardConfig`, which carries per-bucket
``(tau_T, tau_E)`` normalizers so that ``energy_weight`` is
dimensionless and comparable across buildings.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal, Protocol, Sequence

RunPeriodName = Literal["full_year", "winter", "summer"]
TargetTemperatureMode = Literal["constant", "occupancy", "random_schedule"]
SeasonName = Literal["winter", "shoulder", "summer"]
UnoccupiedPolicy = Literal["fixed", "seasonal"]

DEFAULT_TIMESTEPS_PER_HOUR: int = 12

VALID_TARGET_TEMPERATURE_MODES: frozenset[str] = frozenset(
    {"constant", "occupancy", "random_schedule"}
)
VALID_SEASON_NAMES: frozenset[str] = frozenset({"winter", "shoulder", "summer"})
VALID_UNOCCUPIED_POLICIES: frozenset[str] = frozenset({"fixed", "seasonal"})


@dataclass(frozen=True)
class RunPeriodConfig:
    """Simulation run period defined by a named season or full year.

    Attributes:
        name: Canonical period name.
        begin_day_of_month: Start day (inclusive).
        begin_month: Start month (1-12).
        end_day_of_month: End day (inclusive).
        end_month: End month (1-12).
    """

    name: RunPeriodName
    begin_day_of_month: int
    begin_month: int
    end_day_of_month: int
    end_month: int

    @classmethod
    def from_name(cls, name: str | RunPeriodName) -> "RunPeriodConfig":
        """Look up a predefined run period by name.

        Args:
            name: One of ``"full_year"``, ``"winter"``, or ``"summer"``.

        Returns:
            The corresponding ``RunPeriodConfig``.

        Raises:
            ValueError: If *name* is not a recognised period.
        """
        normalized = str(name).strip().lower()
        mapping: dict[str, RunPeriodConfig] = {
            "full_year": cls(
                name="full_year",
                begin_day_of_month=1,
                begin_month=1,
                end_day_of_month=31,
                end_month=12,
            ),
            "winter": cls(
                name="winter",
                begin_day_of_month=1,
                begin_month=1,
                end_day_of_month=31,
                end_month=3,
            ),
            "summer": cls(
                name="summer",
                begin_day_of_month=1,
                begin_month=6,
                end_day_of_month=31,
                end_month=8,
            ),
        }
        if normalized not in mapping:
            raise ValueError(
                "task.run_period must be one of {'full_year', 'winter', 'summer'}, "
                f"got {name!r}"
            )
        return mapping[normalized]

    def expected_steps(
        self, timesteps_per_hour: int = DEFAULT_TIMESTEPS_PER_HOUR
    ) -> int:
        """Return the expected number of simulation steps for this period.

        Args:
            timesteps_per_hour: Number of simulation steps per hour.
                Defaults to ``DEFAULT_TIMESTEPS_PER_HOUR`` (5-minute
                intervals).

        Returns:
            Total number of simulation steps.

        Raises:
            ValueError: If *timesteps_per_hour* is not positive.
        """
        if timesteps_per_hour <= 0:
            raise ValueError("timesteps_per_hour must be > 0")
        day_counts: dict[RunPeriodName, int] = {
            "full_year": 365,
            "winter": 90,
            "summer": 92,
        }
        return day_counts[self.name] * 24 * timesteps_per_hour


DEFAULT_SEASONAL_UNOCCUPIED_C: dict[SeasonName, float] = {
    "winter": 18.0,
    "shoulder": 21.0,
    "summer": 26.0,
}


@dataclass(frozen=True)
class ZoneTargetTemperatureConfig:
    """Target temperature setpoints for a single thermal zone.

    Attributes:
        occupied_c: Target temperature when the zone is occupied (°C).
        unoccupied_c: Target temperature when the zone is unoccupied
            (°C).  Used only when ``unoccupied_policy == "fixed"``.
        unoccupied_policy: ``"fixed"`` always uses :attr:`unoccupied_c`;
            ``"seasonal"`` dispatches to :attr:`seasonal_unoccupied_c`
            based on the current simulation month.
        seasonal_unoccupied_c: Per-season unoccupied setpoints (°C),
            keyed by ``"winter"`` / ``"shoulder"`` / ``"summer"``.
            Required when ``unoccupied_policy == "seasonal"``.
    """

    occupied_c: float
    unoccupied_c: float
    unoccupied_policy: UnoccupiedPolicy = "fixed"
    seasonal_unoccupied_c: dict[SeasonName, float] | None = None

    def __post_init__(self) -> None:
        if self.unoccupied_policy not in VALID_UNOCCUPIED_POLICIES:
            raise ValueError(
                "unoccupied_policy must be one of "
                f"{sorted(VALID_UNOCCUPIED_POLICIES)}, "
                f"got {self.unoccupied_policy!r}"
            )
        if self.unoccupied_policy == "seasonal":
            if self.seasonal_unoccupied_c is None:
                raise ValueError(
                    "seasonal_unoccupied_c is required when "
                    "unoccupied_policy == 'seasonal'"
                )
            missing = VALID_SEASON_NAMES - set(self.seasonal_unoccupied_c.keys())
            if missing:
                raise ValueError(
                    "seasonal_unoccupied_c must define all seasons "
                    f"{sorted(VALID_SEASON_NAMES)}; missing {sorted(missing)}"
                )

    def unoccupied_for_season(self, season: SeasonName) -> float:
        """Return the unoccupied setpoint for a given season.

        For ``"fixed"`` policies this is just :attr:`unoccupied_c`.
        For ``"seasonal"`` policies this looks up the per-season map.
        """
        if self.unoccupied_policy == "seasonal":
            assert self.seasonal_unoccupied_c is not None
            return float(self.seasonal_unoccupied_c[season])
        return float(self.unoccupied_c)

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], *, fallback_temperature_c: float
    ) -> "ZoneTargetTemperatureConfig":
        """Create from a dictionary, using a fallback for missing values.

        Args:
            data: Mapping with optional keys ``"occupied_c"``,
                ``"unoccupied_c"``, ``"unoccupied_policy"``, and
                ``"seasonal_unoccupied_c"``.
            fallback_temperature_c: Value used when ``"occupied_c"`` is
                absent. ``"unoccupied_c"`` falls back to the occupied
                value.

        Returns:
            A new ``ZoneTargetTemperatureConfig``.
        """
        occupied = data.get("occupied_c", fallback_temperature_c)
        unoccupied = data.get("unoccupied_c", occupied)
        policy_raw = str(data.get("unoccupied_policy", "fixed")).strip().lower()
        if policy_raw not in VALID_UNOCCUPIED_POLICIES:
            raise ValueError(
                "unoccupied_policy must be one of "
                f"{sorted(VALID_UNOCCUPIED_POLICIES)}, got {policy_raw!r}"
            )
        policy: UnoccupiedPolicy = policy_raw  # type: ignore[assignment]
        seasonal_raw = data.get("seasonal_unoccupied_c")
        seasonal: dict[SeasonName, float] | None = None
        if seasonal_raw is not None:
            if not isinstance(seasonal_raw, dict):
                raise TypeError(
                    "seasonal_unoccupied_c must be a mapping of season → °C"
                )
            unknown_keys = set(seasonal_raw.keys()) - VALID_SEASON_NAMES
            if unknown_keys:
                raise ValueError(
                    "seasonal_unoccupied_c keys must be subset of "
                    f"{sorted(VALID_SEASON_NAMES)}, got unknown {sorted(unknown_keys)}"
                )
            seasonal = {
                str(k): float(v) for k, v in seasonal_raw.items()  # type: ignore[misc]
            }
        if policy == "seasonal" and seasonal is None:
            seasonal = dict(DEFAULT_SEASONAL_UNOCCUPIED_C)
        return cls(
            occupied_c=float(occupied),
            unoccupied_c=float(unoccupied),
            unoccupied_policy=policy,
            seasonal_unoccupied_c=seasonal,
        )


@dataclass(frozen=True)
class RandomScheduleConfig:
    """Configuration for the per-day random occupancy schedule.

    This drives ``target_temperature_mode == "random_schedule"``: each
    simulated day a fresh arrival time, departure time, occupied
    setpoint, and unoccupied setpoint are drawn from the distribution
    associated with :attr:`building_type`.

    Attributes:
        building_type: Building type key (e.g. ``"OfficeSmall"``) used
            to pick default per-type distributions. ``None`` falls
            back to a generic office profile.
        seed: Base RNG seed for reproducibility.  The effective
            per-day seed is derived from ``(seed, year, day_of_year)``.
    """

    building_type: str | None = None
    seed: int = 0


@dataclass
class TaskConfig:
    """High-level task specification for a simulation episode.

    Attributes:
        run_period: Simulation run period (season or full year).
        target_temperature_mode: How target temperatures are determined
            (``"constant"``, ``"occupancy"``-dependent, or
            ``"random_schedule"`` with a Python-side daily generator).
        default_zone_target_temperature: Fallback target temperature
            used for zones without a zone-specific override.
        zone_target_temperatures: Per-zone target temperature overrides,
            keyed by lower-cased zone name.
        timesteps_per_hour: Number of EnergyPlus simulation timesteps
            per hour.  Determines the control resolution (e.g. 4 → 15 min,
            12 → 5 min).  Must be a divisor of 60 accepted by EnergyPlus
            (1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60).
        random_schedule_config: Parameters for the random daily
            schedule generator.  Only used when
            ``target_temperature_mode == "random_schedule"``; ignored
            otherwise.
    """

    run_period: RunPeriodConfig
    target_temperature_mode: TargetTemperatureMode
    default_zone_target_temperature: ZoneTargetTemperatureConfig
    zone_target_temperatures: dict[str, ZoneTargetTemperatureConfig] = field(
        default_factory=dict
    )
    timesteps_per_hour: int = DEFAULT_TIMESTEPS_PER_HOUR
    random_schedule_config: RandomScheduleConfig | None = None

    VALID_TIMESTEPS_PER_HOUR: ClassVar[frozenset[int]] = frozenset(
        {1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60}
    )

    @classmethod
    def from_dict(cls, task_section: dict[str, Any]) -> "TaskConfig":
        """Parse a task configuration from a raw dictionary.

        Args:
            task_section: Dictionary with optional keys ``"run_period"``,
                ``"target_temperature_mode"``,
                ``"default_zone_target_temperature"``,
                ``"zone_target_temperatures"``, and
                ``"timesteps_per_hour"``.

        Returns:
            A fully validated ``TaskConfig``.

        Raises:
            ValueError: If ``"target_temperature_mode"`` has an invalid
                value, or ``"timesteps_per_hour"`` is not an accepted
                EnergyPlus divisor of 60.
            TypeError: If ``"zone_target_temperatures"`` is not a
                mapping or contains non-string keys.
        """
        run_period = RunPeriodConfig.from_name(
            task_section.get("run_period", "full_year")
        )

        mode_raw = (
            str(task_section.get("target_temperature_mode", "constant")).strip().lower()
        )
        if mode_raw not in VALID_TARGET_TEMPERATURE_MODES:
            raise ValueError(
                "task.target_temperature_mode must be one of "
                f"{sorted(VALID_TARGET_TEMPERATURE_MODES)}, got {mode_raw!r}"
            )
        mode: TargetTemperatureMode = mode_raw  # type: ignore[assignment]

        default_temp = ZoneTargetTemperatureConfig.from_dict(
            task_section.get("default_zone_target_temperature", {}),
            fallback_temperature_c=21.0,
        )

        raw_zone_targets = task_section.get("zone_target_temperatures", {})
        if not isinstance(raw_zone_targets, dict):
            raise TypeError("task.zone_target_temperatures must be a mapping")

        zone_targets: dict[str, ZoneTargetTemperatureConfig] = {}
        for zone_name, zone_cfg in raw_zone_targets.items():
            if not isinstance(zone_name, str):
                raise TypeError("task.zone_target_temperatures keys must be strings")
            if not isinstance(zone_cfg, dict):
                raise TypeError("task.zone_target_temperatures values must be mappings")
            zone_targets[zone_name.strip().lower()] = (
                ZoneTargetTemperatureConfig.from_dict(
                    zone_cfg,
                    fallback_temperature_c=default_temp.occupied_c,
                )
            )

        timesteps_per_hour = int(
            task_section.get("timesteps_per_hour", DEFAULT_TIMESTEPS_PER_HOUR)
        )
        if timesteps_per_hour not in cls.VALID_TIMESTEPS_PER_HOUR:
            raise ValueError(
                f"task.timesteps_per_hour must be a divisor of 60 accepted by "
                f"EnergyPlus {sorted(cls.VALID_TIMESTEPS_PER_HOUR)}, "
                f"got {timesteps_per_hour}"
            )

        random_schedule_cfg: RandomScheduleConfig | None = None
        rs_raw = task_section.get("random_schedule")
        if rs_raw is not None:
            if not isinstance(rs_raw, dict):
                raise TypeError("task.random_schedule must be a mapping")
            bt_raw = rs_raw.get("building_type")
            random_schedule_cfg = RandomScheduleConfig(
                building_type=str(bt_raw) if bt_raw is not None else None,
                seed=int(rs_raw.get("seed", 0)),
            )
        elif mode == "random_schedule":
            random_schedule_cfg = RandomScheduleConfig()

        return cls(
            run_period=run_period,
            target_temperature_mode=mode,
            default_zone_target_temperature=default_temp,
            zone_target_temperatures=zone_targets,
            timesteps_per_hour=timesteps_per_hour,
            random_schedule_config=random_schedule_cfg,
        )

    def expected_steps(self) -> int:
        """Return the expected number of simulation steps for this task.

        Delegates to ``run_period.expected_steps`` using the configured
        :attr:`timesteps_per_hour`.
        """
        return self.run_period.expected_steps(self.timesteps_per_hour)

    def target_for_zone(self, zone_name: str) -> ZoneTargetTemperatureConfig:
        """Return the target temperature config for a given zone.

        Falls back to :attr:`default_zone_target_temperature` when no
        zone-specific override exists.

        Args:
            zone_name: EnergyPlus zone name (case-insensitive).

        Returns:
            The zone-specific or default target temperature config.
        """
        key = zone_name.strip().lower()
        return self.zone_target_temperatures.get(
            key, self.default_zone_target_temperature
        )


@dataclass(frozen=True)
class NormalizedDeadbandRewardConfig:
    """Deadband reward with per-bucket comfort/energy normalizers.

    The reward is

    .. math::

        r = -\\Big(\\tfrac{\\text{temp\\_penalty}}{\\tau_T}
                  + w_E \\cdot \\tfrac{\\text{power\\_penalty}}{\\tau_E}\\Big)

    where ``(tau_T, tau_E)`` come from a random-policy calibration
    rollout on the train split (see
    :file:`building2building/data/reward_normalizers.yaml`).  After
    normalization, ``mean(temp_penalty / tau_T) ≈ 1`` and
    ``mean(power_penalty / tau_E) ≈ 1`` at the median train building of
    each ``(building_type, climate_zone)`` bucket under the calibration
    random policy, so ``energy_weight`` becomes a *dimensionless*
    trade-off knob:

    * ``energy_weight < 1`` → temperature priority,
    * ``energy_weight ≈ 1`` → balanced trade-off,
    * ``energy_weight > 1`` → energy priority.

    Sentinel state
    --------------
    Task presets store this config with ``tau_T = tau_E = None`` (the
    *unfilled* sentinel state) because the constants depend on the
    chosen building.  At env-construction time
    :func:`building2building.api.make_env` resolves the bucket via
    :func:`building2building.data.reward_normalizers.resolve_reward_normalizer`
    and replaces the unfilled config with a filled one.  The simulator
    dispatch site rejects unfilled configs with a clear error.

    Attributes:
        energy_weight: Dimensionless trade-off weight ``w_E``.  See
            class docstring.
        dT: Half-width of the temperature deadband (°C).  Calibration
            assumes ``dT = 1.0``; other values are accepted but emit a
            calibration-mismatch :class:`RuntimeWarning` at dispatch.
        tau_T: Comfort normalizer.  ``None`` means "preset-time
            sentinel; resolve me at env build time".
        tau_E: Energy normalizer.  Same convention.
    """

    energy_weight: float
    dT: float
    tau_T: float | None = None
    tau_E: float | None = None

    def __post_init__(self) -> None:
        # Both-None (unfilled preset) and both-set (filled, ready for
        # the simulator) are valid; the mixed case is always a bug.
        if (self.tau_T is None) != (self.tau_E is None):
            raise ValueError(
                "NormalizedDeadbandRewardConfig: tau_T and tau_E must be set "
                "together (both None for an unfilled preset, both float for "
                f"a filled config). Got tau_T={self.tau_T!r}, tau_E={self.tau_E!r}."
            )
        if self.is_filled:
            assert self.tau_T is not None and self.tau_E is not None
            if self.tau_T <= 0 or self.tau_E <= 0:
                raise ValueError(
                    "NormalizedDeadbandRewardConfig: tau_T and tau_E must be "
                    f"strictly positive when filled, got tau_T={self.tau_T!r}, "
                    f"tau_E={self.tau_E!r}."
                )

    @property
    def is_filled(self) -> bool:
        """``True`` iff both ``tau_T`` and ``tau_E`` are set."""
        return self.tau_T is not None and self.tau_E is not None

    def filled(self, tau_T: float, tau_E: float) -> "NormalizedDeadbandRewardConfig":
        """Return a copy of this config with concrete ``(tau_T, tau_E)``."""
        from dataclasses import replace

        return replace(self, tau_T=float(tau_T), tau_E=float(tau_E))


RewardConfig = NormalizedDeadbandRewardConfig


VALID_REWARD_TYPES = frozenset({"NormalizedDeadbandRewardConfig"})


def reward_config_from_dict(
    reward_section: dict[str, Any],
) -> RewardConfig:
    """Instantiate a reward config from a raw dictionary.

    The ``"reward_type"`` key must be ``"NormalizedDeadbandRewardConfig"``.

    Args:
        reward_section: Dictionary with a mandatory ``"reward_type"`` key
            and type-specific parameters (``energy_weight``, ``dT``,
            and optionally ``tau_T`` / ``tau_E``).

    Returns:
        A :class:`NormalizedDeadbandRewardConfig`.

    Raises:
        ValueError: If ``"reward_type"`` is missing or not recognised.
    """
    reward_type = reward_section.get("reward_type")
    if reward_type is None:
        raise ValueError(
            "reward.reward_type is required — must be "
            "'NormalizedDeadbandRewardConfig'. "
            "Pass an explicit reward config to avoid silent defaults."
        )
    if reward_type == "NormalizedDeadbandRewardConfig":
        # Both tau_T and tau_E are optional in the dict form: missing
        # values produce an *unfilled* sentinel config (consistent with
        # the preset construction path), so YAML reward configs can
        # opt into auto-resolution at env-build time without naming
        # specific (bt, cz) constants.
        raw_tau_T = reward_section.get("tau_T")
        raw_tau_E = reward_section.get("tau_E")
        return NormalizedDeadbandRewardConfig(
            energy_weight=float(reward_section.get("energy_weight", 1.0)),
            dT=float(reward_section.get("dT", 1.0)),
            tau_T=float(raw_tau_T) if raw_tau_T is not None else None,
            tau_E=float(raw_tau_E) if raw_tau_E is not None else None,
        )

    raise ValueError(
        f"Unknown reward_type: {reward_type!r} — "
        f"must be one of {sorted(VALID_REWARD_TYPES)}"
    )


@dataclass(frozen=True)
class ActuatorDescription:
    """Metadata for a single EnergyPlus actuator.

    Attributes:
        component_type: EnergyPlus component type string.
        control_type: EnergyPlus control type string.
        component_name: Name of the controlled component.
        units: Physical units of the actuator value.
        lower_bound: Minimum allowed actuator value.
        upper_bound: Maximum allowed actuator value.
    """

    component_type: str
    control_type: str
    component_name: str
    units: str

    lower_bound: float
    upper_bound: float


class Equipment(Protocol):
    """This protocol is meant to describe what a 'controlled piece of equipment'
    provides: a set of actuators that we can control (some equipments will
    provide more than one) and a list of zones this actuator influences.

    """

    def actuator_descriptions(self) -> list[ActuatorDescription]: ...
    def zones(self) -> list[str]: ...


@dataclass
class BuildingConfig:
    """Full configuration required to instantiate an EnergyPlus simulation.

    Attributes:
        path_to_building: Path to the EnergyPlus IDF / epJSON file.
        path_to_weather: Path to the EPW weather file.
        reward_config: Reward function configuration.
        eplus_output_dir: Directory for EnergyPlus output artefacts.
        warmup_phases: Number of EnergyPlus warmup phases.
        area: Building conditioned floor area (m²), used to normalise
            energy readings.
        hvac_equipment: Sequence of controlled HVAC equipment providing
            actuator descriptions and zone mappings.
        source_metadata: Free-form metadata for logging / debugging
            (e.g. dataset row id, original IDF filename).
        task_config: Task specification (run period, target temps, …).
    """

    path_to_building: Path
    path_to_weather: Path
    reward_config: RewardConfig
    eplus_output_dir: Path
    warmup_phases: int
    area: float
    hvac_equipment: Sequence[Equipment]
    source_metadata: dict[str, Any] = field(default_factory=dict)
    task_config: TaskConfig = field(default_factory=lambda: TaskConfig.from_dict({}))
    expose_heating_only_zones: bool = True
    fixed_actuator_overrides: dict[str, float] = field(default_factory=dict)
