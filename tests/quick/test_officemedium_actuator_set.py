"""Regression test for the OfficeMedium OA-mixer actuator emission.

Pins:
- ``make_vav_system_controllable`` emits exactly one
  ``Outdoor Air Controller × Air Mass Flow Rate`` actuator per air loop,
  on top of the existing SAT + per-zone (flow, htg, clg) actuators.
- The resulting epJSON has no schedule, EMS program, or availability
  manager that can silently override any agent-facing actuator (SAT,
  flow, htg, clg, OA mass flow) -- i.e. every actuator is on the
  always-on regime installed by the existing fan/availability path.

The fixture ``tests/fixtures/minimal_officemedium/building.epjson`` is the DOE
Reference OfficeMedium prototype (3 air loops PACU_VAV_bot/mid/top, each
with a single Controller:OutdoorAir). 
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from building2building.pipeline.actuators import (
    OA_MASS_FLOW_MAX_KGS,
    VAVSystem,
    make_all_equipment,
    gensym,
)

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1] / "fixtures" / "minimal_officemedium" / "building.epjson"
)


@pytest.fixture
def vav_pipeline_output() -> tuple[dict, list]:
    """Run ``make_all_equipment`` on the OfficeMedium fixture and return
    the mutated epJSON together with the resulting equipment list."""
    with open(FIXTURE_PATH, "r") as f:
        raw = json.load(f)
    gensym.reset()
    obj, equipment = make_all_equipment(raw)
    return obj, list(equipment)


def test_three_vav_loops_yield_three_oa_actuators(
    vav_pipeline_output: tuple[dict, list],
) -> None:
    """Exactly one ``Outdoor Air Controller × Air Mass Flow Rate``
    actuator per VAV loop, matching the 3 ``Controller:OutdoorAir``
    objects in the DOE OfficeMedium prototype."""
    _, equipment = vav_pipeline_output
    vav_systems = [e for e in equipment if isinstance(e, VAVSystem)]
    # The fixture is a 3-loop OfficeMedium.  We do not encode this as a
    # constant elsewhere in the pipeline (Q1) but we assert it here so
    # that any regression that drops or duplicates a loop is caught.
    assert len(vav_systems) == 3, (
        f"Expected 3 VAV loops on the OfficeMedium fixture, got " f"{len(vav_systems)}"
    )

    oa_actuators = []
    for vav in vav_systems:
        # The new field must be present on every VAV loop.
        assert vav.oa_mass_flow is not None
        oa_actuators.append(vav.oa_mass_flow)

    # All three OA actuators reference one of the canonical
    # Controller:OutdoorAir names from the DOE OfficeMedium prototype
    # (one packaged air-conditioning unit per floor: bot / mid / top).
    expected_controller_names = {
        "PACU_VAV_bot_OA_Controller",
        "PACU_VAV_mid_OA_Controller",
        "PACU_VAV_top_OA_Controller",
    }
    assert {a.component_name for a in oa_actuators} == expected_controller_names

    for act in oa_actuators:
        assert act.component_type == "Outdoor Air Controller"
        assert act.control_type == "Air Mass Flow Rate"
        assert act.units == "[kg/s]"
        assert act.lower_bound == 0.0
        assert act.upper_bound == OA_MASS_FLOW_MAX_KGS


def test_oa_actuator_appears_in_actuator_descriptions(
    vav_pipeline_output: tuple[dict, list],
) -> None:
    """The OA actuator is exposed through ``VAVSystem.actuator_descriptions()``
    -- this is the surface ``hvac_action_space`` reads in
    ``building2building/simulator``."""
    _, equipment = vav_pipeline_output
    vav_systems = [e for e in equipment if isinstance(e, VAVSystem)]
    for vav in vav_systems:
        descs = vav.actuator_descriptions()
        oa_descs = [
            d
            for d in descs
            if d.component_type == "Outdoor Air Controller"
            and d.control_type == "Air Mass Flow Rate"
        ]
        assert len(oa_descs) == 1, (
            f"VAVSystem.actuator_descriptions() should expose exactly one "
            f"OA actuator, got {len(oa_descs)}"
        )


def test_oa_controller_schedules_pinned_to_always_on(
    vav_pipeline_output: tuple[dict, list],
) -> None:
    """The ``Controller:OutdoorAir`` schedule fields (which would clamp
    or override the EMS write) must point to the always-on schedule
    installed by ``make_vav_system_controllable``."""
    obj, _ = vav_pipeline_output
    controllers = obj.get("Controller:OutdoorAir", {})
    assert controllers, "Fixture has no Controller:OutdoorAir objects"
    schedule_constants = obj.get("Schedule:Constant", {})
    schedule_fields = (
        "minimum_outdoor_air_schedule_name",
        "minimum_fraction_of_outdoor_air_schedule_name",
        "maximum_fraction_of_outdoor_air_schedule_name",
        "time_of_day_economizer_control_schedule_name",
    )
    for ctrl_name, ctrl in controllers.items():
        for field in schedule_fields:
            if field not in ctrl:
                continue
            sched_name = ctrl[field]
            # Must be one of B2B's installed Schedule:Constant objects,
            # not the DOE prototype's ``MinOA_MotorizedDamper_Sched``.
            assert sched_name in schedule_constants, (
                f"Controller {ctrl_name!r}.{field} = {sched_name!r} "
                f"is not a B2B-installed Schedule:Constant"
            )
            sched = schedule_constants[sched_name]
            assert float(sched.get("hourly_value", 0.0)) == 1.0, (
                f"Controller {ctrl_name!r}.{field} points to "
                f"{sched_name!r} but its hourly_value is "
                f"{sched.get('hourly_value')!r}, expected 1.0"
            )


def test_air_loop_availability_pinned_to_always_on(
    vav_pipeline_output: tuple[dict, list],
) -> None:
    """No ``AvailabilityManager:NightCycle`` survives on a VAV loop; the
    air-loop is on the always-on schedule (per
    ``_ensure_always_on_availability``).  This is a precondition for the
    OA actuator EMS write to be honored at all (EnergyPlus ignores EMS
    overrides when the air loop is off)."""
    obj, _ = vav_pipeline_output
    night_cycle = obj.get("AvailabilityManager:NightCycle", {})
    assert not night_cycle, (
        f"Expected zero AvailabilityManager:NightCycle entries after "
        f"make_all_equipment, found {list(night_cycle.keys())}"
    )


def test_no_ems_program_overrides_agent_actuators(
    vav_pipeline_output: tuple[dict, list],
) -> None:
    """The ASHRAE 90.1 OfficeMedium EMS optimum-start programs target
    CLGSETP_SCH / HTGSETP_SCH and would override the thermostat setpoint
    actuators.  ``remove_thermostat_ems_overrides`` must have purged
    them; assert it did."""
    obj, equipment = vav_pipeline_output
    ems_actuators = obj.get("EnergyManagementSystem:Actuator", {})
    for name, act in ems_actuators.items():
        comp_name = act.get("actuated_component_unique_name", "")
        assert (
            "CLGSETP_SCH" not in comp_name
        ), f"EMS actuator {name!r} still targets CLGSETP_SCH"
        assert (
            "HTGSETP_SCH" not in comp_name
        ), f"EMS actuator {name!r} still targets HTGSETP_SCH"


def test_actuator_count_matches_design(
    vav_pipeline_output: tuple[dict, list],
) -> None:
    """The total VAV actuator count (across all loops) is
    n_loops × (1 SAT) + n_zones × (flow + htg + clg) + n_loops × (OA).
    For the 3-loop / 15-zone DOE OfficeMedium fixture this is
    3 + 45 + 3 = 51 actuators on the VAV systems."""
    _, equipment = vav_pipeline_output
    vav_systems = [e for e in equipment if isinstance(e, VAVSystem)]
    total = sum(len(v.actuator_descriptions()) for v in vav_systems)
    assert total == 51, (
        f"Expected 51 VAV-side actuators for the OfficeMedium fixture, " f"got {total}"
    )
