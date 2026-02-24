from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, Sequence, Union


RunPeriodName = Literal["full_year", "winter", "summer"]
TargetTemperatureMode = Literal["constant", "occupancy"]


@dataclass(frozen=True)
class RunPeriodConfig:
    name: RunPeriodName
    begin_day_of_month: int
    begin_month: int
    end_day_of_month: int
    end_month: int

    @classmethod
    def from_name(cls, name: str | RunPeriodName) -> "RunPeriodConfig":
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
    occupied_c: float
    unoccupied_c: float

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], *, fallback_temperature_c: float
    ) -> "ZoneTargetTemperatureConfig":
        occupied = data.get("occupied_c", fallback_temperature_c)
        unoccupied = data.get("unoccupied_c", occupied)
        return cls(occupied_c=float(occupied), unoccupied_c=float(unoccupied))


@dataclass
class TaskConfig:
    run_period: RunPeriodConfig
    target_temperature_mode: TargetTemperatureMode
    default_zone_target_temperature: ZoneTargetTemperatureConfig
    zone_target_temperatures: dict[str, ZoneTargetTemperatureConfig] = field(
        default_factory=dict
    )

    @classmethod
    def from_dict(cls, task_section: dict[str, Any]) -> "TaskConfig":
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
        key = zone_name.strip().lower()
        return self.zone_target_temperatures.get(key, self.default_zone_target_temperature)


@dataclass
class BaseRewardConfig:
    energy_weight: float


@dataclass
class BarrierRewardConfig:
    energy_weight: float
    deadband_c: float = 0.5
    violation_penalty: float = 100.0


@dataclass
class DeadbandRewardConfig:
    area: float
    energy_weight: float
    target_temp: float
    dT: float


RewardConfig = Union[DeadbandRewardConfig, BaseRewardConfig, BarrierRewardConfig]


def reward_config_from_dict(
    reward_section: dict[str, Any],
    *,
    area: float,
) -> RewardConfig:
    reward_type = reward_section.get("reward_type")
    if reward_type == "DeadbandRewardConfig":
        return DeadbandRewardConfig(
            area=area,
            energy_weight=float(reward_section.get("energy_weight", 0.0)),
            target_temp=float(reward_section.get("target_temp", 21.0)),
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
    path_to_building: Path
    path_to_weather: Path
    reward_config: RewardConfig
    eplus_output_dir: Path
    warmup_phases: int
    area: float
    hvac_equipment: Sequence[Equipment]
    # Optional metadata describing the building source/selection (e.g. dataset row id,
    # original IDF filename, weather station, etc.). This is meant for logging/debug.
    source_metadata: dict[str, Any] = field(default_factory=dict)
    task_config: TaskConfig = field(
        default_factory=lambda: TaskConfig.from_dict({})
    )
