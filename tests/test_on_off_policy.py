from __future__ import annotations

import pytest

pytest.skip(
    "Deprecated: this test targets legacy pipeline helpers and a BuildingConfig field "
    "(hvac_control_mode) that are no longer present in this repo.",
    allow_module_level=True,
)


def _find_zone_air_temp_index(env) -> int:
    names = []
    if hasattr(env, "metadata") and isinstance(env.metadata, dict):
        raw = env.metadata.get("observation_names")
        if isinstance(raw, list):
            names = [str(x) for x in raw]
    if not names:
        raise RuntimeError("env.metadata['observation_names'] missing")

    for i, name in enumerate(names):
        if name.lower().startswith("zone air temperature"):
            return i
    for i, name in enumerate(names):
        if "zone air temperature" in name.lower():
            return i
    raise RuntimeError("Could not find zone air temperature index")


def test_on_off_policy_controls_zone_temperature_with_sensible_load_request():
    """Sanity test: on/off controller keeps zone air temperature near target.

    This uses the fixture building and the Unitary HVAC 'Sensible Load Request'
    actuator (continuous Box action space).
    """
    epjson_path = Path("tests/fixtures/bldg1.epjson").resolve()
    epw_path = Path("tests/fixtures/weather.epw").resolve()
    edd_path = Path("tests/fixtures/eplusout.edd").resolve()
    htm_path = Path("tests/fixtures/eplustbl.htm").resolve()

    hvac_actuators = get_hvac_actuators(edd_path)
    hvac_actuators = _infer_unitary_actuator_limits_from_epjson(epjson_path, hvac_actuators)
    hvac_actuators = select_hvac_actuators_for_mode(
        hvac_actuators, "sensible_load_continuous"
    )

    assert len(hvac_actuators) == 1
    assert "sensible load request" in hvac_actuators[0]["control_type"].lower()

    area = get_net_conditioned_area(htm_path)

    target = 21.0
    deadband = 0.5
    q_heat = 6000.0
    q_cool = 6000.0

    with tempfile.TemporaryDirectory() as tmpdir:
        eplus_output_dir = Path(tmpdir)
        building_config = BuildingConfig(
            path_to_building=epjson_path,
            path_to_weather=epw_path,
            reward_config=BaseRewardConfig(energy_weight=0.0),
            hvac_actuators=hvac_actuators,
            eplus_output_dir=eplus_output_dir,
            warmup_phases=0,
            area=area,
            hvac_control_mode="sensible_load_continuous",
            n_bins_continuous=21,
        )

        env = create_simulator(building_config)
        tz_idx = _find_zone_air_temp_index(env)

        policy = OnOffSensibleLoadPolicy(
            target_temp_c=target,
            deadband_c=deadband,
            q_heat_w=q_heat,
            q_cool_w=q_cool,
            temp_obs_index=tz_idx,
        )

        obs, _ = env.reset()
        temps: list[float] = []
        actions: list[float] = []
        print(
            f"[on_off_test] start Tz={float(obs[tz_idx]):.2f}C target={target:.2f}C deadband=±{deadband:.2f}C",
            flush=True,
        )

        # Run a short rollout (fast). We only assert behavior on the tail.
        n_steps = 10000
        for step in range(n_steps):
            act, _ = policy.predict(obs, deterministic=True)
            actions.append(float(act[0]))
            obs, _reward, terminated, truncated, _info = env.step(
                np.asarray(act, dtype=float)
            )
            temps.append(float(obs[tz_idx]))
            # Progress logging (captured by pytest unless -s is used)
            if step % 1000 == 0:
                err = temps[-1] - target
                print(
                    f"[on_off_test] step={step} Tz={temps[-1]:.2f}C "
                    f"err={err:+.2f}C "
                    f"action={actions[-1]:.1f}W mode={policy.last_mode} "
                    f"terminated={terminated} truncated={truncated}",
                    flush=True,
                )
            if terminated or truncated:
                print(
                    f"[on_off_test] finished early at step={step} "
                    f"terminated={terminated} truncated={truncated}",
                    flush=True,
                )
                break

        # Ensure we got enough samples to judge steady behavior
        assert len(temps) >= 200

        tail = np.asarray(temps[-200:], dtype=float)
        abs_err = np.abs(tail - target)
        print(
            "[on_off_test] tail stats "
            f"Tz_mean={float(np.mean(tail)):.2f}C "
            f"Tz_min={float(np.min(tail)):.2f}C "
            f"Tz_max={float(np.max(tail)):.2f}C "
            f"abs_err_mean={float(np.mean(abs_err)):.2f}C "
            f"abs_err_max={float(np.max(abs_err)):.2f}C",
            flush=True,
        )

        # Loose but meaningful: keep within +/- 2°C most of the time
        assert float(np.max(abs_err)) <= 3.0
        assert float(np.mean(abs_err)) <= 1.5

        # Ensure controller actually did something at least once
        assert any(a > 0 for a in actions) or any(a < 0 for a in actions)

        try:
            env.close()
        except Exception:
            # Best-effort shutdown
            if hasattr(env, "ep") and env.ep is not None:
                env.ep.try_stop()


