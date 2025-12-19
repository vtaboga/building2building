import json
import logging
import subprocess
import tempfile
from importlib.resources import files
from pathlib import Path

import duckdb
from building2building import pipeline
from building2building.env import STORE_PATH, energyplus_path
from building2building.pipeline import (
    create_complete_pipeline,
    eplustbl,
    eddfile,
    get_net_conditioned_area,
    get_hvac_actuators,
    get_warmup_days,
    link_in_schedule,
)
from building2building.store import (
    OUTPUT,
    Constant,
    Derivation,
    ExtractZip,
    LocalFile,
    derivation,
    realize,
)
from building2building.types import (
    DeadbandRewardConfig,
    BaseRewardConfig,
    BarrierRewardConfig,
    BuildingConfig,
)
from pandas import DataFrame
import itertools

logger = logging.getLogger(__name__)


def extracted() -> Derivation:
    place = files("building2building.sources.data") / "hydroquebec.zip"
    if not isinstance(place, Path):
        raise Exception("error")

    return ExtractZip(LocalFile(place))


def table_index():
    @derivation("table.parquet")
    def build_database(root: Path):
        out = OUTPUT.get()

        df = duckdb.from_csv_auto(
            str(root / "2025-10-09_building-stock-100-mila.csv")
        ).to_df()

        df = df.assign(
            epw_path=df.weather_station_epw_filepath.apply(
                lambda name: str(root / "weather" / name)
            )
        )

        df = df.assign(
            idf_path=[
                str(root / "IDFsAndSchedules" / str(i) / "in.idf")
                for i in range(1, len(df) + 1)
            ]
        )

        df = df.assign(
            schedule_path=[
                str(root / "IDFsAndSchedules" / str(i) / "in.schedules.csv")
                for i in range(1, len(df) + 1)
            ]
        )

        df = df.drop(columns=["geometry_roof_pitch"])

        df.to_parquet(str(out))

    return build_database(extracted())


def search_weathers() -> DataFrame:
    index = realize(STORE_PATH.get(), table_index())
    return duckdb.from_parquet(str(index)).to_df()


def search_buildings(**query) -> DataFrame:
    index = realize(STORE_PATH.get(), table_index())
    ep = energyplus_path()

    include_setpoint_control = bool(query.pop("include_setpoint_control", True))

    def trans(idf_path, schedule_path):
        return lambda: link_in_schedule(
            create_complete_pipeline(
                Constant(Path(idf_path)),
                ep,
                src_version="24.2.0",
                include_setpoint_control=include_setpoint_control,
            ),
            Path(schedule_path),
        )

    db = duckdb.from_parquet(str(index))

    for k, v in query.items():
        if isinstance(v, str):
            db = db.filter(
                duckdb.FunctionExpression("lower", duckdb.ColumnExpression(k))
                == duckdb.ConstantExpression(v)
            )
        elif isinstance(v, (int, float)):
            db.order(f"abs({k} - {v})")

    df = db.to_df()

    df = df.assign(derivation_thunk=list(map(trans, df.idf_path, df.schedule_path)))
    return df


def search_configs(
    config: dict | object | None = None,
    n: int = 2,
    eplus_output_dir: Path = Path("eplus_out"),
) -> list[BuildingConfig]:
    """
    Return a BuildingConfig per selected building, using each row's weather_path.
    """
    cfg_any = config or {}
    if not isinstance(cfg_any, dict):
        # Hydra passes OmegaConf objects; convert to plain dict so `.get()` and
        # `isinstance(..., dict)` logic behaves as expected.
        try:
            from omegaconf import OmegaConf  # type: ignore

            cfg = OmegaConf.to_container(cfg_any, resolve=True)  # type: ignore[assignment]
            if not isinstance(cfg, dict):
                cfg = {}
        except Exception:
            cfg = {}
    else:
        cfg = cfg_any

    config_nn = cfg.get("bldg", {})
    # Our bldg group configs are nested like: bldg: { bldg: {...} }
    if isinstance(config_nn, dict) and "bldg" in config_nn and isinstance(
        config_nn["bldg"], dict
    ):
        config_nn = config_nn["bldg"]
    ep_path = energyplus_path()

    env_cfg = cfg.get("env", {}) if isinstance(cfg, dict) else {}

    # Backward-compatible: if user provides hvac_control_mode (older configs),
    # interpret that as wanting direct HVAC actuator control.
    if env_cfg.get("hvac_control_mode") is not None and "control_mode" not in env_cfg:
        control_mode = "hvac_actuators"
    else:
        control_mode = env_cfg.get("control_mode", "thermostat_setpoints")
    if control_mode not in ("thermostat_setpoints", "hvac_actuators"):
        raise ValueError(
            "env.control_mode must be one of: 'thermostat_setpoints', 'hvac_actuators'. "
            f"Got: {control_mode}"
        )

    def _filter_for_hvac_component_control(
        actuators: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        """
        Keep only the small set of actuators we actively command for direct HVAC control.

        Important: `hvac_actuators_transform()` will *set every actuator we include*.
        Including "multiplier" style actuators (e.g., frost multipliers) and leaving
        them at the default (0.0) can unintentionally zero out heating/cooling output.
        """

        keep: list[dict[str, str]] = []
        for a in actuators:
            ct = a.get("component_type", "")
            ctrl = a.get("control_type", "")

            # Air loop availability override (force system on)
            if ct == "AirLoopHVAC" and ctrl == "Availability Status":
                keep.append(a)
                continue

            # Fan air mass flow override
            if ct == "Fan" and ctrl == "Fan Air Mass Flow Rate":
                keep.append(a)
                continue

            # Terminal mass flow override (deliver air to zone)
            if ct.startswith("AirTerminal:") and ctrl == "Mass Flow Rate":
                keep.append(a)
                continue

            # Unitary coil speed + supplemental stage control
            if ct == "Coil Speed Control" and ctrl in (
                "Unitary System DX Coil Speed Value",
                "Unitary System Supplemental Coil Stage Level",
            ):
                keep.append(a)
                continue

        return keep

    # If we intend to control HVAC components directly, avoid rewriting thermostat
    # schedules in the pipeline (it can force HVAC off for some models).
    include_setpoint_control = control_mode == "thermostat_setpoints"

    rows = search_buildings(
        **config_nn, include_setpoint_control=include_setpoint_control
    )
    configs: list[BuildingConfig] = []
    for _, row in itertools.islice(rows.iterrows(), n):
        epw = Path(row.epw_path)  # use the weather file for THIS building
        derivation = row.derivation_thunk()
        epjson = realize(STORE_PATH.get(), derivation)

        metrics_path = realize(STORE_PATH.get(), eplustbl(ep_path, derivation, epw))
        area = get_net_conditioned_area(metrics_path)
        # warmup_phases = get_warmup_days(metrics_path)

        # Search actuators
        ems_file = realize(STORE_PATH.get(), eddfile(ep_path, derivation, epw))
        hvac_actuators = get_hvac_actuators(ems_file)
        if control_mode == "hvac_actuators":
            hvac_actuators = _filter_for_hvac_component_control(hvac_actuators)

        reward_section = cfg.get("reward", {}) if isinstance(cfg, dict) else {}
        reward_type = reward_section.get("reward_type")

        # Sensible defaults if reward is not configured
        if reward_type is None:
            reward_type = "DeadbandRewardConfig"

        if reward_type == "DeadbandRewardConfig":
            energy_weight = reward_section.get("energy_weight", 0.1)
            target_temp = reward_section.get("target_temp", 21.0)
            dT = reward_section.get("dT", 0.5)
            reward_config = DeadbandRewardConfig(
                area=area,
                energy_weight=energy_weight,
                target_temp=target_temp,
                dT=dT,
            )
        elif reward_type == "BaseRewardConfig":
            energy_weight = reward_section.get("energy_weight", 1.0)
            reward_config = BaseRewardConfig(
                energy_weight=energy_weight,
            )
        elif reward_type == "BarrierRewardConfig":
            energy_weight = reward_section.get("energy_weight", 1.0)
            reward_config = BarrierRewardConfig(
                energy_weight=energy_weight,
            )
        else:
            raise ValueError(f"Unknown reward type: {reward_type}")

        configs.append(
            BuildingConfig(
                path_to_building=epjson,
                path_to_weather=epw,
                reward_config=reward_config,
                hvac_actuators=hvac_actuators,
                eplus_output_dir=eplus_output_dir,
                warmup_phases=1,  # keep consistent with existing search_config
                area=area,
            )
        )

    return configs
