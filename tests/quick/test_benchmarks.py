"""Tests for building2building.benchmarks — all 4 benchmark classes."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pytest
from cattrs import structure

from building2building.benchmarks import (
    ActionSpaceTransfer,
    CrossDomainGeneralization,
    DynamicsAdaptation,
    GoalAdaptation,
)
from building2building.benchmarks.action_space_transfer import (
    _DEFAULT_BUILDING_TYPE,
    _DEFAULT_CENTRAL_SAT_VALUE,
    _DEFAULT_UNITARY_SAT_VALUE,
    _central_sat_overrides,
    _unitary_sat_overrides,
)
from building2building.benchmarks.cross_domain import CROSS_DOMAIN_PRESETS
from building2building.benchmarks.dynamics_adaptation import DYNAMICS_ADAPTATION_PRESETS
from building2building.pipeline.actuators import (
    AnyEquipment,
    UnitarySystem,
    VAVSystem,
    VAVTerminal,
)
from building2building.simulator.action_spaces import hvac_action_space
from building2building.types import ActuatorDescription

_FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"


@pytest.mark.quick
class TestGoalAdaptation:
    def test_defaults(self) -> None:
        bm = GoalAdaptation()
        assert bm.building_type == "OfficeSmall"
        # Defaults moved to the trade-off-transfer axis on the
        # normalized 3x3 task family (see goal_adaptation.py docstring).
        assert bm.train_task == "task_occ_emed"
        assert bm.test_task == "task_occ_ehigh"
        assert bm.run_period == "full_year"

    def test_custom_params(self) -> None:
        bm = GoalAdaptation(
            building_type="Warehouse",
            train_task="task_occ_emed",
            test_task="task_occ_ehigh",
            run_period="winter",
        )
        assert bm.building_type == "Warehouse"
        assert bm.train_task == "task_occ_emed"
        assert bm.test_task == "task_occ_ehigh"


@pytest.mark.quick
class TestDynamicsAdaptation:
    @pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
    def test_valid_difficulties(self, difficulty: str) -> None:
        bm = DynamicsAdaptation(difficulty=difficulty)  # type: ignore[arg-type]
        preset = DYNAMICS_ADAPTATION_PRESETS[difficulty]
        assert bm.building_type == preset["building_type"]

    def test_invalid_difficulty_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown difficulty"):
            DynamicsAdaptation(difficulty="extreme")  # type: ignore[arg-type]

    def test_easy_is_single_family_house(self) -> None:
        bm = DynamicsAdaptation(difficulty="easy")
        assert bm.building_type == "SingleFamilyHouse"

    def test_hard_is_office_medium(self) -> None:
        bm = DynamicsAdaptation(difficulty="hard")
        assert bm.building_type == "OfficeMedium"

    def test_custom_n_train_test(self) -> None:
        bm = DynamicsAdaptation(n_train=10, n_test=5)
        assert bm.n_train == 10
        assert bm.n_test == 5


# ---------------------------------------------------------------------------
# ActionSpaceTransfer
# ---------------------------------------------------------------------------


def _make_fan_actuator(name: str) -> ActuatorDescription:
    return ActuatorDescription(
        component_type="Fan",
        control_type="Fan Air Mass Flow Rate",
        component_name=name,
        units="[kg/s]",
        lower_bound=0.0,
        upper_bound=1.5,
    )


def _make_sat_actuator(name: str) -> ActuatorDescription:
    return ActuatorDescription(
        component_type="Schedule:Constant",
        control_type="Schedule Value",
        component_name=name,
        units="Temperature",
        lower_bound=5.0,
        upper_bound=60.0,
    )


def _make_flow_fraction_actuator(name: str) -> ActuatorDescription:
    return ActuatorDescription(
        component_type="Schedule:Constant",
        control_type="Schedule Value",
        component_name=name,
        units="[frac]",
        lower_bound=0.0,
        upper_bound=1.0,
    )


def _make_heating_sp_actuator(name: str) -> ActuatorDescription:
    return ActuatorDescription(
        component_type="Schedule:Constant",
        control_type="Schedule Value",
        component_name=name,
        units="[C]",
        lower_bound=10.0,
        upper_bound=35.0,
    )


def _make_cooling_sp_actuator(name: str) -> ActuatorDescription:
    return ActuatorDescription(
        component_type="Schedule:Constant",
        control_type="Schedule Value",
        component_name=name,
        units="[C]",
        lower_bound=18.0,
        upper_bound=40.0,
    )


def _make_oa_mass_flow_actuator(name: str) -> ActuatorDescription:
    """Stub OA mixer actuator for VAVSystem fixtures."""
    return ActuatorDescription(
        component_type="Outdoor Air Controller",
        control_type="Air Mass Flow Rate",
        component_name=name,
        units="[kg/s]",
        lower_bound=0.0,
        upper_bound=5.0,
    )


@pytest.mark.quick
class TestActionSpaceTransfer:
    def test_defaults(self) -> None:
        bm = ActionSpaceTransfer()
        assert bm.system_type == "unitary"
        assert bm.direction == "expand"
        assert bm.task == "task_const_e0"
        assert bm.building_type == "OfficeSmall"
        assert bm.split == "train"
        assert bm.split_index == 0

    def test_custom_system_type(self) -> None:
        bm = ActionSpaceTransfer(system_type="central", direction="reduce")
        assert bm.system_type == "central"
        assert bm.direction == "reduce"

    def test_default_building_type_from_system_type(self) -> None:
        bm_unitary = ActionSpaceTransfer(system_type="unitary")
        assert bm_unitary.building_type == _DEFAULT_BUILDING_TYPE["unitary"]
        assert bm_unitary.building_type == "OfficeSmall"

        bm_central = ActionSpaceTransfer(system_type="central")
        assert bm_central.building_type == _DEFAULT_BUILDING_TYPE["central"]
        assert bm_central.building_type == "OfficeMedium"

    def test_explicit_building_type_overrides_default(self) -> None:
        bm = ActionSpaceTransfer(system_type="unitary", building_type="Warehouse")
        assert bm.building_type == "Warehouse"

    def test_split_param_stored(self) -> None:
        bm = ActionSpaceTransfer(split="test", split_index=5)
        assert bm.split == "test"
        assert bm.split_index == 5


@pytest.mark.quick
class TestUnitarySatOverrides:
    """Tests for the _unitary_sat_overrides helper."""

    def test_returns_sat_actuators_only(self) -> None:
        fan = _make_fan_actuator("fan_zone1")
        sat = _make_sat_actuator("sat_zone1")
        equipment = [UnitarySystem(zone="Zone1", actuators=[fan, sat])]

        overrides = _unitary_sat_overrides(equipment)
        assert overrides == {"sat_zone1": _DEFAULT_UNITARY_SAT_VALUE}

    def test_skips_non_unitary_equipment(self) -> None:
        central_sat = _make_sat_actuator("central_sat")
        terminal = VAVTerminal(
            zone="Zone1",
            flow_fraction=_make_flow_fraction_actuator("ff_z1"),
            heating_setpoint=_make_heating_sp_actuator("htg_z1"),
            cooling_setpoint=_make_cooling_sp_actuator("clg_z1"),
        )
        vav = VAVSystem(
            supply_temp_setpoint=central_sat,
            terminals=[terminal],
            oa_mass_flow=_make_oa_mass_flow_actuator("oa_loop1"),
        )
        overrides = _unitary_sat_overrides([vav])
        assert overrides == {}

    def test_multiple_zones(self) -> None:
        equipment = [
            UnitarySystem(
                zone=f"Zone{i}",
                actuators=[
                    _make_fan_actuator(f"fan_z{i}"),
                    _make_sat_actuator(f"sat_z{i}"),
                ],
            )
            for i in range(5)
        ]
        overrides = _unitary_sat_overrides(equipment)
        assert len(overrides) == 5
        for i in range(5):
            assert f"sat_z{i}" in overrides
            assert overrides[f"sat_z{i}"] == _DEFAULT_UNITARY_SAT_VALUE

    def test_custom_fixed_value(self) -> None:
        fan = _make_fan_actuator("fan")
        sat = _make_sat_actuator("sat")
        equipment = [UnitarySystem(zone="Z", actuators=[fan, sat])]
        overrides = _unitary_sat_overrides(equipment, fixed_value=30.0)
        assert overrides == {"sat": 30.0}

    def test_empty_equipment_returns_empty(self) -> None:
        assert _unitary_sat_overrides([]) == {}


@pytest.mark.quick
class TestCentralSatOverrides:
    """Tests for the _central_sat_overrides helper."""

    def test_returns_supply_temp_actuator(self) -> None:
        central_sat = _make_sat_actuator("central_sat_loop1")
        terminal = VAVTerminal(
            zone="Zone1",
            flow_fraction=_make_flow_fraction_actuator("ff_z1"),
            heating_setpoint=_make_heating_sp_actuator("htg_z1"),
            cooling_setpoint=_make_cooling_sp_actuator("clg_z1"),
        )
        equipment = [
            VAVSystem(
                supply_temp_setpoint=central_sat,
                terminals=[terminal],
                oa_mass_flow=_make_oa_mass_flow_actuator("oa_loop1"),
            )
        ]

        overrides = _central_sat_overrides(equipment)
        assert overrides == {"central_sat_loop1": _DEFAULT_CENTRAL_SAT_VALUE}

    def test_skips_unitary_equipment(self) -> None:
        fan = _make_fan_actuator("fan")
        sat = _make_sat_actuator("sat")
        equipment = [UnitarySystem(zone="Z", actuators=[fan, sat])]
        assert _central_sat_overrides(equipment) == {}

    def test_multiple_loops(self) -> None:
        equipment = []
        for i in range(3):
            central_sat = _make_sat_actuator(f"central_sat_loop{i}")
            terminal = VAVTerminal(
                zone=f"Zone{i}",
                flow_fraction=_make_flow_fraction_actuator(f"ff_z{i}"),
                heating_setpoint=_make_heating_sp_actuator(f"htg_z{i}"),
                cooling_setpoint=_make_cooling_sp_actuator(f"clg_z{i}"),
            )
            equipment.append(
                VAVSystem(
                    supply_temp_setpoint=central_sat,
                    terminals=[terminal],
                    oa_mass_flow=_make_oa_mass_flow_actuator(f"oa_loop{i}"),
                )
            )

        overrides = _central_sat_overrides(equipment)
        assert len(overrides) == 3
        for i in range(3):
            assert overrides[f"central_sat_loop{i}"] == _DEFAULT_CENTRAL_SAT_VALUE

    def test_custom_fixed_value(self) -> None:
        central_sat = _make_sat_actuator("csat")
        terminal = VAVTerminal(
            zone="Z",
            flow_fraction=_make_flow_fraction_actuator("ff"),
            heating_setpoint=_make_heating_sp_actuator("htg"),
            cooling_setpoint=_make_cooling_sp_actuator("clg"),
        )
        equipment = [
            VAVSystem(
                supply_temp_setpoint=central_sat,
                terminals=[terminal],
                oa_mass_flow=_make_oa_mass_flow_actuator("oa"),
            )
        ]
        overrides = _central_sat_overrides(equipment, fixed_value=15.0)
        assert overrides == {"csat": 15.0}

    def test_empty_equipment_returns_empty(self) -> None:
        assert _central_sat_overrides([]) == {}


@pytest.mark.quick
class TestHvacActionSpaceAdditionalFixed:
    """Tests for the additional_fixed parameter on hvac_action_space."""

    def test_additional_fixed_removes_from_agent_space(self) -> None:
        actuators = [
            _make_fan_actuator("fan_z1"),
            _make_sat_actuator("sat_z1"),
            _make_fan_actuator("fan_z2"),
            _make_sat_actuator("sat_z2"),
        ]
        result = hvac_action_space(
            actuators, additional_fixed={"sat_z1": 22.0, "sat_z2": 22.0}
        )
        assert len(result.agent_actuators) == 2
        agent_names = [a.component_name for a in result.agent_actuators]
        assert agent_names == ["fan_z1", "fan_z2"]

    def test_additional_fixed_values_in_assembly(self) -> None:
        actuators = [
            _make_fan_actuator("fan"),
            _make_sat_actuator("sat"),
        ]
        result = hvac_action_space(actuators, additional_fixed={"sat": 22.0})
        assert len(result.agent_actuators) == 1
        assert result.fixed_indices == [1]
        assert result.fixed_values == [22.0]

        full_action = result.assemble_full_action(np.array([0.5]))
        assert len(full_action) == 2
        assert full_action[0] == pytest.approx(0.5)
        assert full_action[1] == pytest.approx(22.0)

    def test_additional_fixed_none_is_noop(self) -> None:
        actuators = [_make_fan_actuator("fan"), _make_sat_actuator("sat")]
        result = hvac_action_space(actuators, additional_fixed=None)
        assert len(result.agent_actuators) == 2

    def test_additional_fixed_empty_dict_is_noop(self) -> None:
        actuators = [_make_fan_actuator("fan"), _make_sat_actuator("sat")]
        result = hvac_action_space(actuators, additional_fixed={})
        assert len(result.agent_actuators) == 2

    def test_additional_fixed_combined_with_builtin_fixed(self) -> None:
        """VAV CLG setpoints are always fixed; additional_fixed adds more."""
        vav_clg = ActuatorDescription(
            component_type="Schedule:Constant",
            control_type="Schedule Value",
            component_name="B2B VAV CLG SETPOINT zone1",
            units="[C]",
            lower_bound=18.0,
            upper_bound=40.0,
        )
        fan = _make_fan_actuator("fan")
        sat = _make_sat_actuator("sat")

        result = hvac_action_space([fan, sat, vav_clg], additional_fixed={"sat": 22.0})
        assert len(result.agent_actuators) == 1
        assert result.agent_actuators[0].component_name == "fan"
        assert len(result.fixed_indices) == 2
        assert set(result.fixed_indices) == {1, 2}


@pytest.mark.quick
class TestHvacActionSpaceFixtureCoverage:
    @pytest.mark.parametrize(
        ("fixture_name", "expected_action_dim"),
        [
            ("minimal_officemedium", 36),
            ("minimal_officesmall", 10),
            ("minimal_restaurantfastfood", 4),
            ("minimal_retailstandalone", 9),
            ("minimal_warehouse", 5),
            ("minimal_singlefamilyhouse", 2),
        ],
    )
    def test_hvac_action_space_supports_all_advertised_hvac_types(
        self,
        fixture_name: str,
        expected_action_dim: int,
    ) -> None:
        equipment_path = _FIXTURES_DIR / fixture_name / "equipment.json"
        equipment = structure(json.loads(equipment_path.read_text()), list[AnyEquipment])
        actuators = list(
            itertools.chain.from_iterable(eq.actuator_descriptions() for eq in equipment)
        )
        action_space = hvac_action_space(actuators)
        assert len(action_space.agent_actuators) == expected_action_dim


@pytest.mark.quick
class TestActionSpaceTransferComputeOverrides:
    """Tests for ActionSpaceTransfer._compute_overrides direction logic."""

    def _sample_unitary_equipment(self) -> list[UnitarySystem]:
        return [
            UnitarySystem(
                zone="Z1",
                actuators=[_make_fan_actuator("fan_z1"), _make_sat_actuator("sat_z1")],
            ),
        ]

    def _sample_vav_equipment(self) -> list[VAVSystem]:
        central_sat = _make_sat_actuator("central_sat")
        terminal = VAVTerminal(
            zone="Z1",
            flow_fraction=_make_flow_fraction_actuator("ff_z1"),
            heating_setpoint=_make_heating_sp_actuator("htg_z1"),
            cooling_setpoint=_make_cooling_sp_actuator("clg_z1"),
        )
        return [
            VAVSystem(
                supply_temp_setpoint=central_sat,
                terminals=[terminal],
                oa_mass_flow=_make_oa_mass_flow_actuator("oa"),
            )
        ]

    def test_unitary_reduced_fixes_sat(self) -> None:
        bm = ActionSpaceTransfer(system_type="unitary")
        eq = self._sample_unitary_equipment()
        overrides = bm._compute_overrides(eq, reduced=True)
        assert "sat_z1" in overrides
        assert overrides["sat_z1"] == _DEFAULT_UNITARY_SAT_VALUE

    def test_unitary_full_is_empty(self) -> None:
        bm = ActionSpaceTransfer(system_type="unitary")
        eq = self._sample_unitary_equipment()
        assert bm._compute_overrides(eq, reduced=False) == {}

    def test_central_reduced_fixes_supply_temp(self) -> None:
        bm = ActionSpaceTransfer(system_type="central")
        eq = self._sample_vav_equipment()
        overrides = bm._compute_overrides(eq, reduced=True)
        assert "central_sat" in overrides
        assert overrides["central_sat"] == _DEFAULT_CENTRAL_SAT_VALUE

    def test_central_full_is_empty(self) -> None:
        bm = ActionSpaceTransfer(system_type="central")
        eq = self._sample_vav_equipment()
        assert bm._compute_overrides(eq, reduced=False) == {}

    def test_expand_direction_reduces_train(self) -> None:
        bm = ActionSpaceTransfer(system_type="unitary", direction="expand")
        eq = self._sample_unitary_equipment()
        train_overrides = bm._compute_overrides(eq, reduced=(bm.direction == "expand"))
        test_overrides = bm._compute_overrides(eq, reduced=(bm.direction == "reduce"))
        assert len(train_overrides) > 0
        assert len(test_overrides) == 0

    def test_reduce_direction_reduces_test(self) -> None:
        bm = ActionSpaceTransfer(system_type="unitary", direction="reduce")
        eq = self._sample_unitary_equipment()
        train_overrides = bm._compute_overrides(eq, reduced=(bm.direction == "expand"))
        test_overrides = bm._compute_overrides(eq, reduced=(bm.direction == "reduce"))
        assert len(train_overrides) == 0
        assert len(test_overrides) > 0


@pytest.mark.quick
class TestCrossDomainGeneralization:
    @pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
    def test_valid_difficulties(self, difficulty: str) -> None:
        bm = CrossDomainGeneralization(difficulty=difficulty)  # type: ignore[arg-type]
        preset = CROSS_DOMAIN_PRESETS[difficulty]
        assert bm.train_type == preset["train_type"]
        assert bm.test_type == preset["test_type"]

    def test_invalid_difficulty_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown difficulty"):
            CrossDomainGeneralization(difficulty="ultra")  # type: ignore[arg-type]

    def test_easy_preset(self) -> None:
        bm = CrossDomainGeneralization(difficulty="easy")
        assert bm.train_type == "RetailStandalone"
        assert bm.test_type == "OfficeSmall"

    def test_custom_n_train_test(self) -> None:
        bm = CrossDomainGeneralization(n_train=2, n_test=3)
        assert bm.n_train == 2
        assert bm.n_test == 3
