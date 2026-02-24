from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from b2b.benchmark.problem_multizones_splits import (
    _apply_actuator_access,
    _is_zone_heating_setpoint_actuator,
    FilteredEquipment,
    SelectionSpec,
    SingleTypeTrainTestBenchmark,
    build_interface_from_cfg,
)
from b2b.types import ActuatorDescription, BaseRewardConfig, BuildingConfig


def test_single_type_select_ids_from_indices() -> None:
    bench = SingleTypeTrainTestBenchmark(
        building_type="OfficeSmall",
        train_selection=SelectionSpec(mode="indices", indices=[0, 2]),
        test_selection=SelectionSpec(mode="indices", indices=[1]),
    )

    with patch(
        "b2b.sources.building_access.building_ids_from_split_indices"
    ) as mock_indices:
        mock_indices.side_effect = [[101, 103], [202]]
        train_ids, test_ids = bench.select_building_ids()
        assert train_ids == [101, 103]
        assert test_ids == [202]


def test_search_config_selection_filters_to_split() -> None:
    bench = SingleTypeTrainTestBenchmark(
        building_type="OfficeSmall",
        train_selection=SelectionSpec(
            mode="search_config",
            n=2,
            queries=[{"place": "austin"}],
        ),
        test_selection=SelectionSpec(mode="indices", indices=[0]),
    )
    df = pd.DataFrame(
        [
            {"building_id": 1, "place": "austin"},
            {"building_id": 2, "place": "austin"},
            {"building_id": 3, "place": "austin"},
        ]
    )
    with patch(
        "b2b.sources.multizones_reference_buildings.load_split_ids"
    ) as mock_split, patch(
        "b2b.sources.multizones_reference_buildings.search_buildings"
    ) as mock_search, patch(
        "b2b.sources.building_access.building_ids_from_split_indices"
    ) as mock_indices:
        mock_split.return_value = [2, 3]
        mock_search.return_value = df
        mock_indices.return_value = [999]  # test side
        train_ids, test_ids = bench.select_building_ids()
        assert train_ids == [2, 3]
        assert test_ids == [999]


@patch("b2b.sources.building_access.search_config")
def test_build_configs_uses_train_and_test_configs(mock_search_config: MagicMock) -> None:
    mock_cfg = BuildingConfig(
        path_to_building=Path("a"),
        path_to_weather=Path("b"),
        reward_config=BaseRewardConfig(0.0),
        eplus_output_dir=Path("eplus"),
        warmup_phases=1,
        area=1.0,
        hvac_equipment=[],
    )
    mock_search_config.return_value = mock_cfg

    bench = SingleTypeTrainTestBenchmark(
        building_type="OfficeSmall",
        train_selection=SelectionSpec(mode="indices", indices=[0]),
        test_selection=SelectionSpec(mode="indices", indices=[0]),
        train_config={"task": {"run_period": "winter"}},
        test_config={"task": {"run_period": "summer"}},
    )

    with patch(
        "b2b.sources.building_access.building_ids_from_split_indices"
    ) as mock_indices:
        mock_indices.side_effect = [[11], [22]]
        result = bench.build_configs(eplus_output_dir=Path("out"))

    assert result.train_building_ids == [11]
    assert result.test_building_ids == [22]
    assert len(result.train_configs) == 1
    assert len(result.test_configs) == 1


def test_build_interface_from_cfg_multi_type() -> None:
    cfg = {
        "benchmark_interface": {
            "mode": "multi_type",
            "train": {
                "types": ["OfficeSmall"],
                "selection": {"mode": "random", "n": 1},
                "config": {},
            },
            "test": {
                "types": ["Warehouse"],
                "selection": {"mode": "random", "n": 1},
                "config": {},
            },
        }
    }
    iface = build_interface_from_cfg(cfg)
    assert type(iface).__name__ == "MultiTypeTrainTestBenchmark"


def test_zone_heating_setpoint_detector() -> None:
    a1 = ActuatorDescription(
        component_type="Zone Temperature Control",
        control_type="Heating Setpoint",
        component_name="Zone A",
        units="C",
        lower_bound=0.0,
        upper_bound=1.0,
    )
    a2 = ActuatorDescription(
        component_type="Schedule:Constant",
        control_type="Schedule Value",
        component_name="B2B vav htg setpoint zone",
        units="C",
        lower_bound=0.0,
        upper_bound=1.0,
    )
    a3 = ActuatorDescription(
        component_type="Fan",
        control_type="Fan Air Mass Flow Rate",
        component_name="Fan A",
        units="[kg/s]",
        lower_bound=0.0,
        upper_bound=1.0,
    )
    assert _is_zone_heating_setpoint_actuator(a1)
    assert _is_zone_heating_setpoint_actuator(a2)
    assert not _is_zone_heating_setpoint_actuator(a3)


def test_apply_actuator_access_removes_zone_heating_setpoints() -> None:
    keep = ActuatorDescription(
        component_type="Fan",
        control_type="Fan Air Mass Flow Rate",
        component_name="fan-a",
        units="[kg/s]",
        lower_bound=0.0,
        upper_bound=1.0,
    )
    drop = ActuatorDescription(
        component_type="Zone Temperature Control",
        control_type="Heating Setpoint",
        component_name="zone-a",
        units="C",
        lower_bound=10.0,
        upper_bound=35.0,
    )
    cfg = BuildingConfig(
        path_to_building=Path("a"),
        path_to_weather=Path("b"),
        reward_config=BaseRewardConfig(0.0),
        eplus_output_dir=Path("eplus"),
        warmup_phases=1,
        area=1.0,
        hvac_equipment=[FilteredEquipment(_zones=["Z1"], _actuators=[keep, drop])],
    )

    out = _apply_actuator_access(
        cfg,
        actuator_access={"include_zone_heating_setpoints": False},
    )
    acts = []
    for eq in out.hvac_equipment:
        acts.extend(eq.actuator_descriptions())
    assert len(acts) == 1
    assert acts[0].component_name == "fan-a"
