"""Gated long tests: verify that env.close() is leak-free (TODO B0).

Requires a working EnergyPlus installation and network/dataset access.
These tests are excluded from quick CI runs via the ``long`` marker.

Acceptance criteria (TODO.md § B0):
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
from pathlib import Path

import pytest

pytestmark = pytest.mark.long

try:
    import psutil

    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False

_N = 20
# EnergyPlus accumulates ~14 MB/cycle of RSS that is not attributable to
# Python objects.  Investigation (see notes.md § "EnergyPlus RSS growth")
# shows this comes from C++ global/static objects inside the EnergyPlus DLL
# that grow with each run_energyplus() call regardless of whether
# delete_state() or reset_state() is used.  ManagedState's weakref.finalize
# callback IS fired correctly after every env.close() (confirmed by direct
# weakref tracking), so our Python fix is correct.  The EnergyPlus-level
# growth requires either an upstream EnergyPlus fix (moving globals into
# EnergyPlusData) or subprocess isolation to eliminate fully.
#
# The limit below is a regression guard: 14 MB/cycle (measured residual) +
# 2 MB/cycle Python-side margin × N.  If something catastrophically broke
# our fix, growth would be far larger.  See TODO B0.1.c for the controlled
# measurement procedure that this constant should be re-derived from.
_RSS_PER_CYCLE_BYTES = 16 * 1024 * 1024  # 14 MB native + 2 MB margin
_RSS_MAX_GROWTH_BYTES = _RSS_PER_CYCLE_BYTES * _N  # 320 MB for N=20

_BUILDING_TYPE = "SingleFamilyHouse"
_ENV_KWARGS: dict = dict(
    building_type=_BUILDING_TYPE,
    split="train",
    index=0,
    task="task1",
    run_period="winter",
    timesteps_per_hour=4,
)


class TestEnvLeakClose:
    """Plain env.close() must release all EnergyPlus resources."""

    def test_close_removes_output_dir(self) -> None:
        """(i) Output directory is removed after env.close()."""
        from building2building.api import new_make_env

        with tempfile.TemporaryDirectory(prefix="b2b_leak_test_") as tmpdir:
            parent = Path(tmpdir)
            for i in range(_N):
                out_dir = parent / f"run_{i}"
                out_dir.mkdir()
                env = new_make_env(**_ENV_KWARGS, eplus_output_dir=out_dir)
                env.reset()
                env.close()

                leftover = [d for d in parent.iterdir() if d.is_dir()]
                assert (
                    not leftover
                ), f"After close() #{i}, leftover output dirs: {leftover}"

    def test_close_joins_thread(self) -> None:
        """(ii) Thread count returns to baseline after env.close()."""
        from building2building.api import new_make_env

        gc.collect()
        baseline = threading.active_count()

        for i in range(_N):
            env = new_make_env(**_ENV_KWARGS)
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
    def test_close_bounds_rss_growth(self) -> None:
        """(iii) RSS growth across N cycles is bounded by _RSS_MAX_GROWTH_BYTES."""
        import psutil

        from building2building.api import new_make_env

        proc = psutil.Process()
        gc.collect()
        rss_before = proc.memory_info().rss

        for _ in range(_N):
            env = new_make_env(**_ENV_KWARGS)
            env.reset()
            env.close()

        gc.collect()
        rss_after = proc.memory_info().rss
        growth = rss_after - rss_before
        assert growth < _RSS_MAX_GROWTH_BYTES, (
            f"RSS grew by {growth / 1e6:.1f} MB across {_N} env cycles "
            f"(limit: {_RSS_MAX_GROWTH_BYTES / 1e6:.0f} MB, "
            f"= {_RSS_PER_CYCLE_BYTES // (1024 * 1024)} MB/cycle × {_N}); "
            "EnergyPlus-native growth of ~14 MB/cycle is expected and "
            "accounted for; this failure means extra leakage beyond that."
        )

    def test_plain_close_without_close_env_aggressively(self) -> None:
        """Regression: close() alone is sufficient — no helper needed."""
        from building2building.api import new_make_env

        with tempfile.TemporaryDirectory(prefix="b2b_plain_close_") as tmpdir:
            out_dir = Path(tmpdir) / "run"
            out_dir.mkdir()
            env = new_make_env(**_ENV_KWARGS, eplus_output_dir=out_dir)
            env.reset()
            env.close()

            assert not out_dir.exists(), (
                "Output dir still exists after plain env.close(); "
                "EnergyPlusEnvironment.close() did not clean it up."
            )


class TestEnvLeakReset:
    """env.reset() must be leak-free on a single persistent env instance."""

    def test_reset_joins_thread(self) -> None:
        """(i) Thread count returns to baseline after every env.reset()."""
        from building2building.api import new_make_env

        gc.collect()
        baseline = threading.active_count()

        env = new_make_env(**_ENV_KWARGS)
        try:
            for i in range(_N):
                env.reset()

                gc.collect()
                count = threading.active_count()
                assert count == baseline, (
                    f"After reset() #{i}: thread count {count} != baseline "
                    f"{baseline}; EnergyPlus thread was not joined."
                )
        finally:
            env.close()

    def test_reset_does_not_accumulate_output_dirs(self) -> None:
        """(ii) Parent eplus_output_dir contains exactly one run-dir at any time."""
        from building2building.api import new_make_env

        with tempfile.TemporaryDirectory(prefix="b2b_reset_leak_") as tmpdir:
            parent = Path(tmpdir)
            out_dir = parent / "run"
            out_dir.mkdir()
            env = new_make_env(**_ENV_KWARGS, eplus_output_dir=out_dir)
            try:
                for i in range(_N):
                    env.reset()

                    subdirs = [d for d in parent.iterdir() if d.is_dir()]
                    assert len(subdirs) == 1, (
                        f"After reset() #{i}, expected exactly 1 subdir in "
                        f"{parent}, found {len(subdirs)}: {subdirs}"
                    )
            finally:
                env.close()

    @pytest.mark.skipif(not _PSUTIL_AVAILABLE, reason="psutil not installed")
    def test_reset_bounds_rss_growth(self) -> None:
        """(iii) RSS growth across N cycles is bounded by _RSS_MAX_GROWTH_BYTES."""
        import psutil

        from building2building.api import new_make_env

        proc = psutil.Process()
        gc.collect()
        rss_before = proc.memory_info().rss

        env = new_make_env(**_ENV_KWARGS)
        try:
            for _ in range(_N):
                env.reset()
        finally:
            env.close()

        gc.collect()
        rss_after = proc.memory_info().rss
        growth = rss_after - rss_before
        assert growth < _RSS_MAX_GROWTH_BYTES, (
            f"RSS grew by {growth / 1e6:.1f} MB across {_N} reset cycles "
            f"(limit: {_RSS_MAX_GROWTH_BYTES / 1e6:.0f} MB, "
            f"= {_RSS_PER_CYCLE_BYTES // (1024 * 1024)} MB/cycle × {_N}); "
            "EnergyPlus-native growth of ~14 MB/cycle is expected and "
            "accounted for; this failure means extra leakage beyond that."
        )

    def test_double_reset_same_env(self) -> None:
        """Two consecutive reset() calls without close() must not crash."""
        from building2building.api import new_make_env

        with tempfile.TemporaryDirectory(prefix="b2b_double_reset_") as tmpdir:
            parent = Path(tmpdir)
            out_dir = parent / "run"
            out_dir.mkdir()
            env = new_make_env(**_ENV_KWARGS, eplus_output_dir=out_dir)
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
