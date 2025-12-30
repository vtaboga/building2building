from __future__ import annotations

import pytest

pytest.skip(
    "Deprecated: PatchedEnergyPlusSimulation was upstreamed to minergym; this repo no longer "
    "ships building2building.simulator.patched_minergym_simulation.",
    allow_module_level=True,
)


def test_get_hvac_actuators_includes_fan_coil_and_airloop_availability() -> None:
    edd_path = Path("tests/fixtures/eplusout.edd").resolve()
    assert edd_path.exists()

    acts = get_hvac_actuators(edd_path)
    assert acts, "Expected at least one HVAC actuator from fixture .edd"

    # Fan actuator
    assert any(
        a.get("component_type") == "Fan" and "Fan Air Mass Flow Rate" in a.get("control_type", "")
        for a in acts
    ), "Expected to find a Fan Air Mass Flow Rate actuator"

    # Coil/Unitary coil speed actuator
    assert any(
        a.get("component_type") == "Coil Speed Control" and "DX Coil Speed" in a.get("control_type", "")
        for a in acts
    ), "Expected to find a coil speed control actuator"

    # AirLoop availability override actuator (needed to prevent HVAC being forced off)
    assert any(
        a.get("component_type") == "AirLoopHVAC"
        and a.get("control_type") == "Availability Status"
        for a in acts
    ), "Expected to find AirLoopHVAC Availability Status actuator"


def test_patched_simulation_registers_late_hvac_loop_callback(monkeypatch: Any) -> None:
    """
    Unit test: verify we register both:
    - BeginSystemTimestepBeforePredictor (minergym step protocol)
    - InsideSystemIterationLoop (late re-apply so HVAC doesn't overwrite)
    """
    calls: list[tuple[str, Any]] = []

    import minergym.simulation as ms

    def _recorder(name: str):
        def _inner(ep_state: c_void_p, cb: Any) -> None:
            calls.append((name, cb))

        return _inner

    monkeypatch.setattr(
        ms.api.runtime,
        "callback_begin_system_timestep_before_predictor",
        _recorder("begin_system_timestep_before_predictor"),
    )
    monkeypatch.setattr(
        ms.api.runtime,
        "callback_inside_system_iteration_loop",
        _recorder("inside_system_iteration_loop"),
    )

    sim = PatchedEnergyPlusSimulation(
        building_path=Path("tests/fixtures/bldg1.epjson").resolve(),
        weather_path=Path("tests/fixtures/weather.epw").resolve(),
        observation_template=[],
        actuators=[],
        verbose=False,
        log_dir=Path("."),
        warmup_phases=0,
    )

    sim.register_callbacks(c_void_p(1))

    names = [n for (n, _cb) in calls]
    assert "begin_system_timestep_before_predictor" in names
    assert "inside_system_iteration_loop" in names


