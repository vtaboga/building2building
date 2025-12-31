from __future__ import annotations

import numpy as np

from algorithms.baselines import PIDSensibleLoadPolicy, TrimAndRespondSensibleLoadPolicy


def test_pid_sensible_load_policy_heats_and_cools_and_respects_deadband() -> None:
    p = PIDSensibleLoadPolicy(
        target_temp_c=21.0,
        deadband_c=0.5,
        temp_obs_index=0,
        kp=1000.0,
        ki=0.0,
        kd=0.0,
        q_heat_max_w=5000.0,
        q_cool_max_w=5000.0,
        dt_s=1.0,
    )

    # Below target -> heating (positive)
    a, _ = p.predict(np.asarray([19.0]))
    assert a.shape == (1,)
    assert float(a[0]) > 0.0

    # Above target -> cooling (negative)
    a, _ = p.predict(np.asarray([24.0]))
    assert float(a[0]) < 0.0

    # Inside deadband -> off
    a, _ = p.predict(np.asarray([21.2]))
    assert float(a[0]) == 0.0


def test_pid_sensible_load_policy_saturates() -> None:
    p = PIDSensibleLoadPolicy(
        target_temp_c=21.0,
        deadband_c=0.0,
        temp_obs_index=0,
        kp=1e9,
        ki=0.0,
        kd=0.0,
        q_heat_max_w=123.0,
        q_cool_max_w=456.0,
        dt_s=1.0,
    )

    a, _ = p.predict(np.asarray([0.0]))
    assert float(a[0]) == 123.0

    a, _ = p.predict(np.asarray([100.0]))
    assert float(a[0]) == -456.0


def test_trim_and_respond_increases_and_trims() -> None:
    p = TrimAndRespondSensibleLoadPolicy(
        target_temp_c=21.0,
        deadband_c=0.5,
        temp_obs_index=0,
        respond_step_w=1000.0,
        trim_step_w=500.0,
        q_heat_max_w=3000.0,
        q_cool_max_w=3000.0,
    )

    # Two steps of heating response
    a1, _ = p.predict(np.asarray([18.0]))
    a2, _ = p.predict(np.asarray([18.0]))
    assert float(a1[0]) == 1000.0
    assert float(a2[0]) == 2000.0

    # Saturate at max
    a3, _ = p.predict(np.asarray([18.0]))
    a4, _ = p.predict(np.asarray([18.0]))
    assert float(a3[0]) == 3000.0
    assert float(a4[0]) == 3000.0

    # Inside deadband -> trim toward zero
    a5, _ = p.predict(np.asarray([21.1]))
    assert float(a5[0]) == 2500.0
    a6, _ = p.predict(np.asarray([21.1]))
    assert float(a6[0]) == 2000.0


