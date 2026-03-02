"""Core domain types for Building2Building.

Defines task, reward, actuator, and building configuration dataclasses
used across the simulation and RL pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, Sequence, Union


RunPeriodName = Literal["full_year", "winter", "summer"]
TargetTemperatureMode = Literal["constant", "occupancy"]


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

    def expected_steps(self, timesteps_per_hour: int = 4) -> int:
        """Return the expected number of simulation steps for this period.

        Args:
            timesteps_per_hour: Number of simulation steps per hour.
                Defaults to 4 (15-minute intervals).

        Returns:
            Total number of simulation steps.

        Raises:
            ValueError: If *timesteps_per_hour* is not positive.
        """
        # 15-minute simulation step is the default in this repository.
        if timesteps_per_hour <= 0:
            raise ValueError("timesteps_per_hour must be > 0")
        day_counts: dict[RunPeriodName, int] = {
            "full_year": 365,
            "winter": 90,
            "summer": 92,
        }
        return day_counts[self.name] * 24 * timesteps_per_hour


@dataclass(frozen=True)
class ZoneTargetTemperatureConfig:
    """Target temperature setpoints for a single thermal zone.

    Attributes:
        occupied_c: Target temperature when the zone is occupied (°C).
        unoccupied_c: Target temperature when the zone is unoccupied (°C).
    """

    occupied_c: float
    unoccupied_c: float

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], *, fallback_temperature_c: float
    ) -> "ZoneTargetTemperatureConfig":
        """Create from a dictionary, using a fallback for missing values.

        Args:
            data: Mapping with optional keys ``"occupied_c"`` and
                ``"unoccupied_c"``.
            fallback_temperature_c: Value used when ``"occupied_c"`` is
                absent. ``"unoccupied_c"`` falls back to the occupied
                value.

        Returns:
            A new ``ZoneTargetTemperatureConfig``.
        """
        occupied = data.get("occupied_c", fallback_temperature_c)
        unoccupied = data.get("unoccupied_c", occupied)
        return cls(occupied_c=float(occupied), unoccupied_c=float(unoccupied))


@dataclass
class TaskConfig:
    """High-level task specification for a simulation episode.

    Attributes:
        run_period: Simulation run period (season or full year).
        target_temperature_mode: How target temperatures are determined
            (``"constant"`` or ``"occupancy"``-dependent).
        default_zone_target_temperature: Fallback target temperature
            used for zones without a zone-specific override.
        zone_target_temperatures: Per-zone target temperature overrides,
            keyed by lower-cased zone name.
    """

    run_period: RunPeriodConfig
    target_temperature_mode: TargetTemperatureMode
    default_zone_target_temperature: ZoneTargetTemperatureConfig
    zone_target_temperatures: dict[str, ZoneTargetTemperatureConfig] = field(
        default_factory=dict
    )

    @classmethod
    def from_dict(cls, task_section: dict[str, Any]) -> "TaskConfig":
        """Parse a task configuration from a raw dictionary.

        Args:
            task_section: Dictionary with optional keys ``"run_period"``,
                ``"target_temperature_mode"``,
                ``"default_zone_target_temperature"``, and
                ``"zone_target_temperatures"``.

        Returns:
            A fully validated ``TaskConfig``.

        Raises:
            ValueError: If ``"target_temperature_mode"`` has an invalid
                value.
            TypeError: If ``"zone_target_temperatures"`` is not a
                mapping or contains non-string keys.
        """
        run_period = RunPeriodConfig.from_name(task_section.get("run_period", "full_year"))

        mode_raw = str(task_section.get("target_temperature_mode", "constant")).strip().lower()
        if mode_raw not in {"constant", "occupancy"}:
            raise ValueError(
                "task.target_temperature_mode must be one of {'constant', 'occupancy'}, "
                f"got {mode_raw!r}"
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
                raise TypeError(
                    "task.zone_target_temperatures values must be mappings"
                )
            zone_targets[zone_name.strip().lower()] = ZoneTargetTemperatureConfig.from_dict(
                zone_cfg,
                fallback_temperature_c=default_temp.occupied_c,
            )

        return cls(
            run_period=run_period,
            target_temperature_mode=mode,
            default_zone_target_temperature=default_temp,
            zone_target_temperatures=zone_targets,
        )

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
        return self.zone_target_temperatures.get(key, self.default_zone_target_temperature)


@dataclass
class BaseRewardConfig:
    """Reward configuration using only weighted energy consumption.

    Attributes:
        energy_weight: Multiplicative weight applied to energy cost.
    """

    energy_weight: float


@dataclass
class BarrierRewardConfig:
    """Reward with a barrier penalty for temperature-band violations.

    Attributes:
        energy_weight: Multiplicative weight applied to energy cost.
        deadband_c: Half-width of the acceptable temperature band (°C).
        violation_penalty: Penalty magnitude when temperature exits the
            deadband.
    """

    energy_weight: float
    deadband_c: float = 0.5
    violation_penalty: float = 100.0


@dataclass
class DeadbandRewardConfig:
    """Reward that penalises deviations from a temperature deadband.

    Attributes:
        energy_weight: Multiplicative weight applied to energy cost.
        dT: Half-width of the temperature deadband (°C).
    """

    energy_weight: float
    dT: float


RewardConfig = Union[DeadbandRewardConfig, BaseRewardConfig, BarrierRewardConfig]


def reward_config_from_dict(
    reward_section: dict[str, Any],
) -> RewardConfig:
    """Instantiate a reward config from a raw dictionary.

    The ``"reward_type"`` key selects the concrete config class:

    * ``"DeadbandRewardConfig"`` -> :class:`DeadbandRewardConfig`
    * ``"BarrierRewardConfig"``  -> :class:`BarrierRewardConfig`
    * ``None`` / ``"BaseRewardConfig"`` -> :class:`BaseRewardConfig`

    Args:
        reward_section: Dictionary with a ``"reward_type"`` key and
            type-specific parameters.

    Returns:
        The appropriate ``RewardConfig`` variant.

    Raises:
        ValueError: If ``"reward_type"`` is not recognised.
    """
    reward_type = reward_section.get("reward_type")
    if reward_type == "DeadbandRewardConfig":
        return DeadbandRewardConfig(
            energy_weight=float(reward_section.get("energy_weight", 0.0)),
            dT=float(reward_section.get("dT", 0.5)),
        )
    if reward_type == "BarrierRewardConfig":
        return BarrierRewardConfig(
            energy_weight=float(reward_section.get("energy_weight", 0.0)),
            deadband_c=float(reward_section.get("deadband_c", 0.5)),
            violation_penalty=float(reward_section.get("violation_penalty", 100.0)),
        )
    if reward_type in (None, "BaseRewardConfig"):
        return BaseRewardConfig(energy_weight=float(reward_section.get("energy_weight", 0.0)))

    raise ValueError(f"Unknown reward type: {reward_type}")


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
    task_config: TaskConfig = field(
        default_factory=lambda: TaskConfig.from_dict({})
    )
    expose_heating_only_zones: bool = True
