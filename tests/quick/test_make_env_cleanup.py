"""Pins the staging-directory cleanup contract for ``make_env``.

Asserts that seasonal run periods register a cleanup callback for the
EnergyPlus staging directory on ``env.close()``, and that full-year periods
skip the cleanup (the staging dir is reused across episodes).
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest

import building2building.api as api_mod
from building2building.types import RewardConfig


def _patch_registry(monkeypatch: pytest.MonkeyPatch, fixture_registry: Any) -> None:
    monkeypatch.setattr(
        "building2building.data.registry.get_registry", lambda: fixture_registry
    )


@pytest.mark.quick
def test_make_env_registers_staging_dir_cleanup_for_seasonal_period(
    monkeypatch: pytest.MonkeyPatch,
    fixture_registry: Any,
) -> None:
    assert hasattr(api_mod, "weakref")
    _patch_registry(monkeypatch, fixture_registry)

    calls: list[tuple[object, object, tuple[object, ...], dict[str, object]]] = []
    real_finalize = api_mod.weakref.finalize

    def _recording_finalize(
        target: object,
        callback: object,
        *args: object,
        **kwargs: object,
    ) -> object:
        calls.append((target, callback, args, kwargs))
        return real_finalize(target, callback, *args, **kwargs)

    monkeypatch.setattr(api_mod.weakref, "finalize", _recording_finalize)

    env = api_mod.make_env(
        "OfficeSmall",
        task="task_occ_emed",
        reward=RewardConfig(energy_weight=1.0, dT=1.0, tau_T=1.0, tau_E=1.0),
        run_period="winter",
        max_episode_steps=5,
    )
    try:
        assert len(calls) == 1
        _target, callback, args, kwargs = calls[0]
        assert callback is shutil.rmtree
        assert kwargs == {}
        assert len(args) == 2
        assert args[1] is True
        staging_dir = args[0]
        assert isinstance(staging_dir, Path)
        assert staging_dir.is_dir()

        callback(*args)
        assert not staging_dir.exists()
    finally:
        env.close()


@pytest.mark.quick
def test_make_env_skips_staging_dir_cleanup_for_full_year(
    monkeypatch: pytest.MonkeyPatch,
    fixture_registry: Any,
) -> None:
    _patch_registry(monkeypatch, fixture_registry)

    calls: list[tuple[object, object, tuple[object, ...], dict[str, object]]] = []
    real_finalize = api_mod.weakref.finalize

    def _recording_finalize(
        target: object,
        callback: object,
        *args: object,
        **kwargs: object,
    ) -> object:
        calls.append((target, callback, args, kwargs))
        return real_finalize(target, callback, *args, **kwargs)

    monkeypatch.setattr(api_mod.weakref, "finalize", _recording_finalize)

    env = api_mod.make_env(
        "OfficeSmall",
        task="task_occ_emed",
        reward=RewardConfig(energy_weight=1.0, dT=1.0, tau_T=1.0, tau_E=1.0),
        run_period="full_year",
        max_episode_steps=5,
    )
    try:
        assert calls == []
    finally:
        env.close()
