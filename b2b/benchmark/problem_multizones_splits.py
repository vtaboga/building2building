from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

from b2b.sources import building_access, multizones_reference_buildings as mz
from b2b.sources.multizones_reference_buildings import BuildingType
from b2b.types import ActuatorDescription, BuildingConfig, Equipment

SelectionMode = Literal["random", "indices", "search_config"]
SplitName = Literal["train", "test"]


@dataclass(frozen=True)
class SelectionSpec:
    mode: SelectionMode = "random"
    n: int = 1
    seed: int | None = None
    replace: bool = False
    indices: list[int] = field(default_factory=list)
    queries: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SelectionSpec":
        mode_raw = str(raw.get("mode", "random")).strip().lower()
        if mode_raw not in {"random", "indices", "search_config"}:
            raise ValueError(
                "selection.mode must be one of {'random', 'indices', 'search_config'}, "
                f"got {mode_raw!r}"
            )

        n = int(raw.get("n", 1))
        if n < 0:
            raise ValueError("selection.n must be >= 0")

        indices_raw = raw.get("indices", [])
        if not isinstance(indices_raw, list) or not all(
            isinstance(x, int) for x in indices_raw
        ):
            raise TypeError("selection.indices must be a list[int]")

        queries_raw = raw.get("queries", [])
        if not isinstance(queries_raw, list) or not all(
            isinstance(x, dict) for x in queries_raw
        ):
            raise TypeError("selection.queries must be a list[dict]")

        seed_raw = raw.get("seed", None)
        seed = int(seed_raw) if seed_raw is not None else None
        replace = bool(raw.get("replace", False))

        return cls(
            mode=cast(SelectionMode, mode_raw),
            n=n,
            seed=seed,
            replace=replace,
            indices=list(indices_raw),
            queries=list(queries_raw),
        )


@dataclass(frozen=True)
class SplitBenchmarkResult:
    train_building_ids: list[int]
    test_building_ids: list[int]
    train_configs: list[BuildingConfig]
    test_configs: list[BuildingConfig]


@dataclass(frozen=True)
class FilteredEquipment:
    _zones: list[str]
    _actuators: list[ActuatorDescription]

    def actuator_descriptions(self) -> list[ActuatorDescription]:
        return list(self._actuators)

    def zones(self) -> list[str]:
        return list(self._zones)


def _deep_merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in extra.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _is_zone_heating_setpoint_actuator(a: ActuatorDescription) -> bool:
    comp_type = str(a.component_type).strip().lower()
    control_type = str(a.control_type).strip().lower()
    name = str(a.component_name).strip().lower()
    if comp_type == "zone temperature control" and control_type == "heating setpoint":
        return True
    # VAV thermostat control path in this repo often uses schedule actuators.
    if control_type == "schedule value" and "htg setpoint" in name:
        return True
    if "heating setpoint" in name:
        return True
    return False


def _apply_actuator_access(
    bcfg: BuildingConfig,
    *,
    actuator_access: dict[str, Any],
) -> BuildingConfig:
    include_zone_heating = bool(
        actuator_access.get("include_zone_heating_setpoints", True)
    )
    if include_zone_heating:
        return bcfg

    filtered_equipment: list[Equipment] = []
    for eq in bcfg.hvac_equipment:
        zones = list(eq.zones())
        kept = [
            a
            for a in eq.actuator_descriptions()
            if not _is_zone_heating_setpoint_actuator(a)
        ]
        if kept:
            filtered_equipment.append(
                FilteredEquipment(_zones=zones, _actuators=kept)
            )

    total_acts = sum(len(eq.actuator_descriptions()) for eq in filtered_equipment)
    if total_acts == 0:
        raise ValueError(
            "Actuator filtering removed all actuators. "
            "Set include_zone_heating_setpoints=true or relax filters."
        )

    return BuildingConfig(
        path_to_building=bcfg.path_to_building,
        path_to_weather=bcfg.path_to_weather,
        reward_config=bcfg.reward_config,
        eplus_output_dir=bcfg.eplus_output_dir,
        warmup_phases=bcfg.warmup_phases,
        area=bcfg.area,
        hvac_equipment=filtered_equipment,
        source_metadata=dict(bcfg.source_metadata),
        task_config=bcfg.task_config,
    )


def _filter_to_split_ids(
    building_type: BuildingType,
    split: SplitName,
    *,
    query: dict[str, Any],
) -> list[int]:
    split_ids = set(mz.load_split_ids(building_type, split))
    rows = mz.search_buildings(building_type=building_type, **query)
    if rows.empty:
        return []
    out: list[int] = []
    for _, row in rows.iterrows():
        bid = int(row.building_id)
        if bid in split_ids and bid not in out:
            out.append(bid)
    return out


def _select_ids_for_type(
    *,
    building_type: BuildingType,
    split: SplitName,
    spec: SelectionSpec,
) -> list[int]:
    if spec.mode == "random":
        return building_access.sample_building_ids(
            dataset="multizones_reference_buildings",
            split=split,
            n=spec.n,
            seed=spec.seed,
            replace=spec.replace,
            building_type=building_type,
        )

    if spec.mode == "indices":
        return building_access.building_ids_from_split_indices(
            dataset="multizones_reference_buildings",
            split=split,
            split_indices=spec.indices,
            building_type=building_type,
        )

    # search_config mode: metadata query + split filtering.
    ids: list[int] = []
    queries = spec.queries if spec.queries else [{}]
    for query in queries:
        candidates = _filter_to_split_ids(
            building_type=building_type,
            split=split,
            query=query,
        )
        for bid in candidates:
            if bid not in ids:
                ids.append(bid)
            if spec.n > 0 and len(ids) >= spec.n:
                return ids[: spec.n]
    return ids[: spec.n] if spec.n > 0 else ids


def _build_configs_for_ids(
    *,
    building_type: BuildingType,
    building_ids: list[int],
    side_config: dict[str, Any],
    eplus_output_dir: Path,
) -> list[BuildingConfig]:
    out: list[BuildingConfig] = []
    actuator_access = (
        side_config.get("actuator_access", {})
        if isinstance(side_config.get("actuator_access", {}), dict)
        else {}
    )
    for bid in building_ids:
        cfg = _deep_merge(
            {"bldg": {"bldg": {"building_type": building_type, "building_id": int(bid)}}},
            side_config,
        )
        built = (
            building_access.search_config(
                dataset="multizones_reference_buildings",
                building_type=building_type,
                config=cfg,
                eplus_output_dir=eplus_output_dir,
                building_id=int(bid),
            )
        )
        out.append(_apply_actuator_access(built, actuator_access=actuator_access))
    return out


@dataclass(frozen=True)
class SingleTypeTrainTestBenchmark:
    building_type: BuildingType
    train_selection: SelectionSpec
    test_selection: SelectionSpec
    train_config: dict[str, Any] = field(default_factory=dict)
    test_config: dict[str, Any] = field(default_factory=dict)

    def select_building_ids(self) -> tuple[list[int], list[int]]:
        train_ids = _select_ids_for_type(
            building_type=self.building_type,
            split="train",
            spec=self.train_selection,
        )
        test_ids = _select_ids_for_type(
            building_type=self.building_type,
            split="test",
            spec=self.test_selection,
        )
        return train_ids, test_ids

    def build_configs(self, *, eplus_output_dir: Path) -> SplitBenchmarkResult:
        train_ids, test_ids = self.select_building_ids()
        train_cfgs = _build_configs_for_ids(
            building_type=self.building_type,
            building_ids=train_ids,
            side_config=self.train_config,
            eplus_output_dir=eplus_output_dir / "train",
        )
        test_cfgs = _build_configs_for_ids(
            building_type=self.building_type,
            building_ids=test_ids,
            side_config=self.test_config,
            eplus_output_dir=eplus_output_dir / "test",
        )
        return SplitBenchmarkResult(
            train_building_ids=train_ids,
            test_building_ids=test_ids,
            train_configs=train_cfgs,
            test_configs=test_cfgs,
        )


@dataclass(frozen=True)
class MultiTypeTrainTestBenchmark:
    train_types: list[BuildingType]
    test_types: list[BuildingType]
    train_selection: SelectionSpec
    test_selection: SelectionSpec
    train_config: dict[str, Any] = field(default_factory=dict)
    test_config: dict[str, Any] = field(default_factory=dict)

    def select_building_ids(self) -> tuple[list[int], list[int]]:
        train_ids: list[int] = []
        test_ids: list[int] = []
        for bt in self.train_types:
            train_ids.extend(
                _select_ids_for_type(
                    building_type=bt,
                    split="train",
                    spec=self.train_selection,
                )
            )
        for bt in self.test_types:
            test_ids.extend(
                _select_ids_for_type(
                    building_type=bt,
                    split="test",
                    spec=self.test_selection,
                )
            )
        return train_ids, test_ids

    def build_configs(self, *, eplus_output_dir: Path) -> SplitBenchmarkResult:
        train_cfgs: list[BuildingConfig] = []
        test_cfgs: list[BuildingConfig] = []
        train_ids: list[int] = []
        test_ids: list[int] = []

        for bt in self.train_types:
            ids = _select_ids_for_type(
                building_type=bt,
                split="train",
                spec=self.train_selection,
            )
            train_ids.extend(ids)
            train_cfgs.extend(
                _build_configs_for_ids(
                    building_type=bt,
                    building_ids=ids,
                    side_config=self.train_config,
                    eplus_output_dir=eplus_output_dir / "train" / bt,
                )
            )

        for bt in self.test_types:
            ids = _select_ids_for_type(
                building_type=bt,
                split="test",
                spec=self.test_selection,
            )
            test_ids.extend(ids)
            test_cfgs.extend(
                _build_configs_for_ids(
                    building_type=bt,
                    building_ids=ids,
                    side_config=self.test_config,
                    eplus_output_dir=eplus_output_dir / "test" / bt,
                )
            )

        return SplitBenchmarkResult(
            train_building_ids=train_ids,
            test_building_ids=test_ids,
            train_configs=train_cfgs,
            test_configs=test_cfgs,
        )


def _as_dict(cfg: object) -> dict[str, Any]:
    if isinstance(cfg, dict):
        return dict(cfg)
    try:
        from omegaconf import OmegaConf  # type: ignore

        c = OmegaConf.to_container(cfg, resolve=True)
        if isinstance(c, dict):
            return dict(c)
    except Exception:
        pass
    return {}


def build_interface_from_cfg(
    cfg: object,
) -> SingleTypeTrainTestBenchmark | MultiTypeTrainTestBenchmark:
    data = _as_dict(cfg)
    bench = data.get("benchmark_interface", {})
    if not isinstance(bench, dict):
        raise TypeError("benchmark_interface must be a mapping")

    mode = str(bench.get("mode", "single_type")).strip().lower()
    train_raw = bench.get("train", {})
    test_raw = bench.get("test", {})
    if not isinstance(train_raw, dict) or not isinstance(test_raw, dict):
        raise TypeError("benchmark_interface.train/test must be mappings")

    train_selection = SelectionSpec.from_dict(
        train_raw.get("selection", {})
        if isinstance(train_raw.get("selection", {}), dict)
        else {}
    )
    test_selection = SelectionSpec.from_dict(
        test_raw.get("selection", {})
        if isinstance(test_raw.get("selection", {}), dict)
        else {}
    )
    train_cfg = (
        train_raw.get("config", {}) if isinstance(train_raw.get("config", {}), dict) else {}
    )
    test_cfg = (
        test_raw.get("config", {}) if isinstance(test_raw.get("config", {}), dict) else {}
    )

    if mode == "single_type":
        bt = str(bench.get("building_type", "OfficeSmall"))
        return SingleTypeTrainTestBenchmark(
            building_type=cast(BuildingType, bt),
            train_selection=train_selection,
            test_selection=test_selection,
            train_config=train_cfg,
            test_config=test_cfg,
        )

    if mode == "multi_type":
        train_types_raw = train_raw.get("types", [])
        test_types_raw = test_raw.get("types", [])
        if not isinstance(train_types_raw, list) or not all(
            isinstance(x, str) for x in train_types_raw
        ):
            raise TypeError("benchmark_interface.train.types must be list[str]")
        if not isinstance(test_types_raw, list) or not all(
            isinstance(x, str) for x in test_types_raw
        ):
            raise TypeError("benchmark_interface.test.types must be list[str]")
        if len(train_types_raw) < 1 or len(test_types_raw) < 1:
            raise ValueError(
                "multi_type benchmark requires at least one train type and one test type"
            )
        return MultiTypeTrainTestBenchmark(
            train_types=cast(list[BuildingType], list(train_types_raw)),
            test_types=cast(list[BuildingType], list(test_types_raw)),
            train_selection=train_selection,
            test_selection=test_selection,
            train_config=train_cfg,
            test_config=test_cfg,
        )

    raise ValueError(
        "benchmark_interface.mode must be one of {'single_type', 'multi_type'}"
    )
