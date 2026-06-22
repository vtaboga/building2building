"""Gated long tests: verify that env.close() is leak-free.

Requires a working EnergyPlus installation and network/dataset access.
These tests are excluded from quick CI runs via the ``long`` marker.

Acceptance criteria:
  (i)   The parent ``eplus_output_dir`` contains zero leftover subdirs
        after each ``env.close()``.
  (ii)  ``threading.active_count()`` returns exactly to its baseline
        after each ``env.close()``.
  (iii) RSS growth across N=20 create/reset/close cycles is bounded by
        the known EnergyPlus-native residual (~14 MB/cycle) plus a small
        Python-side margin (requires ``psutil``).

The key assertion is that *plain* ``env.close()`` — without
``close_env_aggressively`` — satisfies all three criteria.
"""

from __future__ import annotations

import gc
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

import pytest

pytestmark = pytest.mark.long

try:
    import psutil

    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False

@dataclass(frozen=True)
class _LeakProfile:
    building_type: str
    n_cycles: int


# Measured on 2026-05-27 (interactive node):
# OfficeMedium create/reset/close cycle (winter, timesteps_per_hour=4)
# took ~4.75 s total. We therefore keep OfficeMedium at N=10 so the
# parametrized leak suite remains around the 5-minute budget while still
# providing N >= 5 signal strength.
_LEAK_PROFILES = (
    _LeakProfile("SingleFamilyHouse", 20),
    _LeakProfile("OfficeMedium", 10),
)
# EnergyPlus accumulates ~14 MB/cycle of RSS that is not attributable to
# Python objects.  Investigation shows this comes from C++ global/static
# objects inside the EnergyPlus DLL
# that grow with each run_energyplus() call regardless of whether
# delete_state() or reset_state() is used.  ManagedState's weakref.finalize
# callback IS fired correctly after every env.close() (confirmed by direct
# weakref tracking), so our Python fix is correct.  The EnergyPlus-level
# growth requires either an upstream EnergyPlus fix (moving globals into
# EnergyPlusData) or subprocess isolation to eliminate fully.
#
# The limit below is a regression guard: 14 MB/cycle (measured residual) +
# 2 MB/cycle Python-side margin × N.  If something catastrophically broke
# our fix, growth would be far larger.  This constant should be re-derived
# from a controlled measurement procedure when the residual is re-measured.
_RSS_PER_CYCLE_BYTES = 16 * 1024 * 1024  # 14 MB native + 2 MB margin
def _rss_limit_bytes(n_cycles: int) -> int:
    return _RSS_PER_CYCLE_BYTES * n_cycles


def _env_kwargs(building_type: str) -> dict[str, object]:
    return dict(
        building_type=building_type,
        split="train",
        index=0,
        task="task_occ_emed",
        run_period="winter",
        timesteps_per_hour=4,
    )


@pytest.mark.parametrize(
    "profile",
    _LEAK_PROFILES,
    ids=lambda p: f"{p.building_type}_n{p.n_cycles}",
)
class TestEnvLeakClose:
    """Plain env.close() must release all EnergyPlus resources."""

    def test_close_removes_output_dir(self, profile: _LeakProfile) -> None:
        """(i) Output directory is removed after env.close()."""
        from building2building.api import make_env

        env_kwargs = _env_kwargs(profile.building_type)
        with tempfile.TemporaryDirectory(prefix="b2b_leak_test_") as tmpdir:
            parent = Path(tmpdir)
            for i in range(profile.n_cycles):
                out_dir = parent / f"run_{i}"
                out_dir.mkdir()
                env = make_env(**env_kwargs, eplus_output_dir=out_dir)
                env.reset()
                env.close()

                leftover = [d for d in parent.iterdir() if d.is_dir()]
                assert (
                    not leftover
                ), f"After close() #{i}, leftover output dirs: {leftover}"

    def test_close_joins_thread(self, profile: _LeakProfile) -> None:
        """(ii) Thread count returns to baseline after env.close()."""
        from building2building.api import make_env

        env_kwargs = _env_kwargs(profile.building_type)
        gc.collect()
        baseline = threading.active_count()

        for i in range(profile.n_cycles):
            env = make_env(**env_kwargs)
            env.reset()
            env.close()

            # gc.collect() before counting so short-lived Python-internal
            # threads have had a chance to finish.
            gc.collect()
            count = threading.active_count()
            assert count == baseline, (
                f"After close() #{i}: thread count {count} != baseline "
                f"{baseline}; EnergyPlus thread was not joined."
            )

    @pytest.mark.skipif(not _PSUTIL_AVAILABLE, reason="psutil not installed")
    def test_close_bounds_rss_growth(self, profile: _LeakProfile) -> None:
        """(iii) RSS growth across N cycles is bounded by _RSS_MAX_GROWTH_BYTES."""
        import psutil

        from building2building.api import make_env

        env_kwargs = _env_kwargs(profile.building_type)
        rss_max_growth_bytes = _rss_limit_bytes(profile.n_cycles)
        proc = psutil.Process()
        gc.collect()
        rss_before = proc.memory_info().rss

        for _ in range(profile.n_cycles):
            env = make_env(**env_kwargs)
            env.reset()
            env.close()

        gc.collect()
        rss_after = proc.memory_info().rss
        growth = rss_after - rss_before
        assert growth < rss_max_growth_bytes, (
            f"RSS grew by {growth / 1e6:.1f} MB across {profile.n_cycles} env cycles "
            f"(limit: {rss_max_growth_bytes / 1e6:.0f} MB, "
            f"= {_RSS_PER_CYCLE_BYTES // (1024 * 1024)} MB/cycle × {profile.n_cycles}); "
            "EnergyPlus-native growth of ~14 MB/cycle is expected and "
            "accounted for; this failure means extra leakage beyond that."
        )

    def test_plain_close_without_close_env_aggressively(
        self, profile: _LeakProfile
    ) -> None:
        """Regression: close() alone is sufficient — no helper needed."""
        from building2building.api import make_env

        env_kwargs = _env_kwargs(profile.building_type)
        with tempfile.TemporaryDirectory(prefix="b2b_plain_close_") as tmpdir:
            out_dir = Path(tmpdir) / "run"
            out_dir.mkdir()
            env = make_env(**env_kwargs, eplus_output_dir=out_dir)
            env.reset()
            env.close()

            assert not out_dir.exists(), (
                "Output dir still exists after plain env.close(); "
                "EnergyPlusEnvironment.close() did not clean it up."
            )


@pytest.mark.parametrize(
    "profile",
    _LEAK_PROFILES,
    ids=lambda p: f"{p.building_type}_n{p.n_cycles}",
)
class TestEnvLeakReset:
    """env.reset() must be leak-free on a single persistent env instance."""

    def test_reset_does_not_accumulate_threads(self, profile: _LeakProfile) -> None:
        """(i) reset() may keep one worker thread alive, but must not accumulate."""
        from building2building.api import make_env

        env_kwargs = _env_kwargs(profile.building_type)
        gc.collect()
        baseline = threading.active_count()
        steady_state_count: int | None = None

        env = make_env(**env_kwargs)
        try:
            for i in range(profile.n_cycles):
                env.reset()

                gc.collect()
                count = threading.active_count()
                if steady_state_count is None:
                    # First reset can spawn/retain a worker thread while the
                    # env stays open; this is acceptable as long as the count
                    # stays stable over subsequent resets.
                    steady_state_count = count
                    assert steady_state_count >= baseline, (
                        f"After reset() #{i}: thread count {count} < baseline "
                        f"{baseline}, unexpected thread accounting."
                    )
                else:
                    assert count == steady_state_count, (
                        f"After reset() #{i}: thread count {count} != steady state "
                        f"{steady_state_count}; reset() appears to leak threads."
                    )
        finally:
            env.close()
            gc.collect()

        after_close = threading.active_count()
        assert after_close == baseline, (
            f"After close(): thread count {after_close} != baseline {baseline}; "
            "worker thread was not released."
        )

    def test_reset_does_not_accumulate_output_dirs(self, profile: _LeakProfile) -> None:
        """(ii) Parent eplus_output_dir contains exactly one run-dir at any time."""
        from building2building.api import make_env

        env_kwargs = _env_kwargs(profile.building_type)
        with tempfile.TemporaryDirectory(prefix="b2b_reset_leak_") as tmpdir:
            parent = Path(tmpdir)
            out_dir = parent / "run"
            out_dir.mkdir()
            env = make_env(**env_kwargs, eplus_output_dir=out_dir)
            try:
                for i in range(profile.n_cycles):
                    env.reset()

                    subdirs = [d for d in parent.iterdir() if d.is_dir()]
                    assert len(subdirs) == 1, (
                        f"After reset() #{i}, expected exactly 1 subdir in "
                        f"{parent}, found {len(subdirs)}: {subdirs}"
                    )
            finally:
                env.close()

    @pytest.mark.skipif(not _PSUTIL_AVAILABLE, reason="psutil not installed")
    def test_reset_bounds_rss_growth(self, profile: _LeakProfile) -> None:
        """(iii) RSS growth across N cycles is bounded by _RSS_MAX_GROWTH_BYTES."""
        import psutil

        from building2building.api import make_env

        env_kwargs = _env_kwargs(profile.building_type)
        rss_max_growth_bytes = _rss_limit_bytes(profile.n_cycles)
        proc = psutil.Process()
        gc.collect()
        rss_before = proc.memory_info().rss

        env = make_env(**env_kwargs)
        try:
            for _ in range(profile.n_cycles):
                env.reset()
        finally:
            env.close()

        gc.collect()
        rss_after = proc.memory_info().rss
        growth = rss_after - rss_before
        assert growth < rss_max_growth_bytes, (
            f"RSS grew by {growth / 1e6:.1f} MB across {profile.n_cycles} reset cycles "
            f"(limit: {rss_max_growth_bytes / 1e6:.0f} MB, "
            f"= {_RSS_PER_CYCLE_BYTES // (1024 * 1024)} MB/cycle × {profile.n_cycles}); "
            "EnergyPlus-native growth of ~14 MB/cycle is expected and "
            "accounted for; this failure means extra leakage beyond that."
        )

    def test_double_reset_same_env(self, profile: _LeakProfile) -> None:
        """Two consecutive reset() calls without close() must not crash."""
        from building2building.api import make_env

        env_kwargs = _env_kwargs(profile.building_type)
        with tempfile.TemporaryDirectory(prefix="b2b_double_reset_") as tmpdir:
            parent = Path(tmpdir)
            out_dir = parent / "run"
            out_dir.mkdir()
            env = make_env(**env_kwargs, eplus_output_dir=out_dir)
            try:
                env.reset()
                env.reset()

                subdirs = [d for d in parent.iterdir() if d.is_dir()]
                assert len(subdirs) == 1, (
                    f"After double reset(), expected exactly 1 subdir in "
                    f"{parent}, found {len(subdirs)}: {subdirs}"
                )
                assert out_dir.exists(), (
                    "Output dir does not exist after double reset(); "
                    "reset() should recreate it."
                )
            finally:
                env.close()
