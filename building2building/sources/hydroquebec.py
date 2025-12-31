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
    get_sensible_load_actuators,
    get_zone_temperature_control_actuators,
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


def search_buildings(*, include_setpoint_control: bool = True, **query) -> DataFrame:
    index = realize(STORE_PATH.get(), table_index())
    ep = energyplus_path()

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
    control_mode = env_cfg["control_mode"]

    # --- Action space configuration ---
    hvac_action_space = env_cfg.get("hvac_action_space", "multidiscrete")
    if hvac_action_space not in ("box", "multidiscrete"):
        raise ValueError(
            "env.hvac_action_space must be one of: 'box', 'multidiscrete'. "
            f"Got: {hvac_action_space}"
        )
    n_bins_continuous = int(env_cfg.get("n_bins_continuous"))
    if n_bins_continuous <= 1 or n_bins_continuous is None:
        raise ValueError("env.n_bins_continuous must be >= 2")

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

            # Direct load request (bypasses thermostat demand for supported unitary systems)
            if ct == "Unitary HVAC" and ctrl in ("Sensible Load Request", "Moisture Load Request"):
                keep.append(a)
                continue

        return keep

    # For direct sensible-load control, we *do not* rewrite thermostat setpoint schedules
    # to extreme values (10C/40C). Those setpoints are used by EnergyPlus to compute
    # zone predicted/remaining loads; extreme setpoints can eliminate the load signal and
    # keep unitary coils off.
    include_setpoint_control = control_mode not in ("hvac_actuators", "sensible_load")
    rows = search_buildings(**config_nn, include_setpoint_control=include_setpoint_control)
    
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
        if control_mode == "hvac_actuators":
            actuators = get_hvac_actuators(ems_file)
            actuators = _filter_for_hvac_component_control(actuators)
        elif control_mode == "sensible_load":
            # For sensible-load control we primarily use:
            # - Unitary HVAC :: Sensible Load Request
            #
            # Many of our unitary systems are wrapped in an AirLoopHVAC that can still
            # be turned off by availability logic; include the AirLoopHVAC availability
            # actuator when present so policies can force CycleOn (2.0).
            actuators = get_sensible_load_actuators(ems_file)
            hvac_actuators = get_hvac_actuators(ems_file)
            availability = [
                a
                for a in hvac_actuators
                if a.get("component_type") == "AirLoopHVAC"
                and a.get("control_type") == "Availability Status"
            ]
            zone_setpoints = get_zone_temperature_control_actuators(ems_file)
            # Prepend availability so action_names are stable/readable.
            actuators = zone_setpoints + availability + actuators
        else:
            raise ValueError(f"Unknown control mode: {control_mode}")

        reward_section = cfg.get("reward", {}) if isinstance(cfg, dict) else {}
        reward_type = reward_section.get("reward_type")

        if reward_type == "DeadbandRewardConfig":
            energy_weight = reward_section.get("energy_weight")
            target_temp = reward_section.get("target_temp")
            dT = reward_section.get("dT")
            reward_config = DeadbandRewardConfig(
                area=area,
                energy_weight=energy_weight,
                target_temp=target_temp,
                dT=dT,
            )
        elif reward_type == "BaseRewardConfig":
            energy_weight = reward_section.get("energy_weight")
            reward_config = BaseRewardConfig(
                energy_weight=energy_weight,
            )
        elif reward_type == "BarrierRewardConfig":
            energy_weight = reward_section.get("energy_weight")
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
                hvac_actuators=actuators,
                eplus_output_dir=eplus_output_dir,
                warmup_phases=1,  # keep consistent with existing search_config
                area=area,
                hvac_action_space=hvac_action_space,
                n_bins_continuous=n_bins_continuous,
            )
        )

    return configs
