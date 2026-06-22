"""Long-running smoke test across all dataset buildings.

For each building present in the unified metadata, this test constructs an
environment through ``make_env_from_config``, runs one ``reset`` and one ``step``, then
records any failures by building ID.

Buildings are processed in small batches (``BATCH_SIZE``), each in a fresh
subprocess.  At most ``NUM_WORKERS`` batches run concurrently.  This avoids
two observed problems:

* **Memory leaks** – the EnergyPlus C runtime leaks resources; after ~100
  simulations a long-lived process is OOM-killed (exit code -9) or EnergyPlus
  refuses to start new simulations.  Fresh processes every ``BATCH_SIZE``
  buildings keep peak RSS bounded.
* **Thread-safety** – the EnergyPlus singleton and ``optree`` are not
  thread-safe, so processes (not threads) are required.

Each batch writes incremental results to a JSON file, so a crash in one batch
does not destroy results from other batches.
"""

from __future__ import annotations

import json
import logging
import math
import multiprocessing as mp
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest

from building2building.data.download import BuildingType, download_building_type
from building2building.data.registry import get_registry

logger = logging.getLogger(__name__)

pytestmark = pytest.mark.long

NUM_WORKERS = 3
BATCH_SIZE = 50
BATCH_TIMEOUT_S = 600


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sanitize_for_path(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)


def _trimmed_traceback_str(max_lines: int = 30) -> str:
    """Return a shortened traceback string for the current exception."""
    text = traceback.format_exc()
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    return "\n".join(["... (traceback trimmed) ...", *lines[-max_lines:]])


def _prefetch_building_types(metadata: pd.DataFrame) -> dict[str, str]:
    """Download each building-type archive once before env creation."""
    failures: dict[str, str] = {}
    building_types = sorted(
        {str(value) for value in metadata["building_type"].tolist()}
    )
    total = len(building_types)
    logger.info("Prefetching %d building type archive(s)...", total)
    for i, building_type in enumerate(building_types, 1):
        logger.info("[prefetch %d/%d] Downloading %s ...", i, total, building_type)
        t0 = time.monotonic()
        try:
            download_building_type(cast(BuildingType, building_type))
            elapsed = time.monotonic() - t0
            logger.info(
                "[prefetch %d/%d] %s ready (%.1fs)", i, total, building_type, elapsed
            )
        except Exception as exc:  # noqa: BLE001 - report all prefetch failures
            elapsed = time.monotonic() - t0
            logger.warning(
                "[prefetch %d/%d] %s FAILED after %.1fs: %s",
                i,
                total,
                building_type,
                elapsed,
                exc,
            )
            failures[building_type] = (
                f"{type(exc).__name__}: {exc}\n{_trimmed_traceback_str()}"
            )
    logger.info(
        "Prefetch complete: %d ok, %d failed", total - len(failures), len(failures)
    )
    return failures


# ---------------------------------------------------------------------------
# Batch worker (runs in a child process)
# ---------------------------------------------------------------------------


def _run_batch(
    batch_id: int,
    building_rows: list[tuple[str, str]],
    total_buildings: int,
    global_offset: int,
    output_root: str,
    result_path: str,
) -> None:
    """Smoke-test a small batch of buildings in a fresh subprocess.

    Results are flushed to *result_path* (JSON) after every building so
    data survives a mid-batch crash.
    """
    import gymnasium as gym

    from building2building.api import make_env_from_config
    from building2building.config.models import EnvBuildConfig

    logging.basicConfig(level=logging.INFO, force=True)
    wlog = logging.getLogger(f"{__name__}.batch-{batch_id}")

    def _build_env_config(building_type: str, building_id: str) -> EnvBuildConfig:
        return EnvBuildConfig.from_dict(
            {
                "dataset_selection": {
                    "building_type": building_type,
                    "mode": "building_id",
                    "building_id": building_id,
                },
                "task": {
                    "run_period": "winter",
                    "timesteps_per_hour": 4,
                },
                # Smoke test only builds/reset/steps the env, so the exact
                # normalizer scale is irrelevant; supply filled tau_T/tau_E
                # (the EnvBuildConfig path does not autofill them — only
                # make_env does).
                "reward": {
                    "reward_type": "NormalizedDeadbandRewardConfig",
                    "tau_T": 1.0,
                    "tau_E": 1.0,
                },
            }
        )

    output_path = Path(output_root)
    ok_count = 0
    failures: list[str] = []
    failures_by_type: defaultdict[str, list[str]] = defaultdict(list)

    for local_idx, (building_type, building_id) in enumerate(building_rows):
        global_idx = global_offset + local_idx + 1
        env: gym.Env | None = None
        t0 = time.monotonic()
        try:
            config = _build_env_config(
                building_type=building_type, building_id=building_id
            )
            output_dir = output_path / _sanitize_for_path(
                f"{building_type}_{building_id}"
            )
            env = make_env_from_config(config=config, eplus_output_dir=output_dir)
            env.reset()
            action = env.action_space.sample()
            env.step(action)
            ok_count += 1
            elapsed = time.monotonic() - t0
            wlog.info(
                "[%d/%d] OK  %s / %s  (%.1fs)",
                global_idx,
                total_buildings,
                building_type,
                building_id,
                elapsed,
            )
        except Exception as exc:  # noqa: BLE001 - collect all environment failures
            elapsed = time.monotonic() - t0
            tb_str = _trimmed_traceback_str()
            wlog.warning(
                "[%d/%d] FAIL %s / %s  (%.1fs): %s: %s",
                global_idx,
                total_buildings,
                building_type,
                building_id,
                elapsed,
                type(exc).__name__,
                exc,
            )
            failures.append(
                "\n".join(
                    [
                        f"- {building_type} / {building_id}",
                        f"  {type(exc).__name__}: {exc}",
                        f"  Traceback:",
                        tb_str,
                    ]
                )
            )
            failures_by_type[building_type].append(building_id)
        finally:
            if env is not None:
                env.close()

        _write_result_json(
            result_path,
            ok_count,
            failures,
            dict(failures_by_type),
            global_offset,
            local_idx + 1,
        )


# ---------------------------------------------------------------------------
# Result I/O
# ---------------------------------------------------------------------------


def _write_result_json(
    path: str,
    ok_count: int,
    failures: list[str],
    failures_by_type: dict[str, list[str]],
    global_offset: int,
    processed: int,
) -> None:
    payload: dict[str, Any] = {
        "ok_count": ok_count,
        "failures": failures,
        "failures_by_type": failures_by_type,
        "global_offset": global_offset,
        "processed": processed,
    }
    Path(path).write_text(json.dumps(payload))


def _read_result_json(path: str) -> dict[str, Any] | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())  # type: ignore[no-any-return]
    except (json.JSONDecodeError, OSError):
        return None


# ---------------------------------------------------------------------------
# Batch result collector
# ---------------------------------------------------------------------------


def _collect_batch(
    batch_id: int,
    proc: mp.Process,
    result_file: str,
    n_rows: int,
    all_ok: list[int],
    all_failures: list[str],
    all_failures_by_type: defaultdict[str, list[str]],
    all_processed: list[int],
    crashed_batches: list[str],
) -> None:
    """Harvest a finished (or crashed) batch process."""
    data = _read_result_json(result_file)
    processed = 0
    if data is not None:
        all_ok.append(data["ok_count"])
        all_failures.extend(data["failures"])
        for btype, bids in data["failures_by_type"].items():
            all_failures_by_type[btype].extend(bids)
        processed = data.get("processed", 0)
        all_processed.append(processed)

    if proc.exitcode is None:
        logger.error("[batch-%d] timed out after %ds", batch_id, BATCH_TIMEOUT_S)
        proc.kill()
        proc.join(timeout=10)
        crashed_batches.append(
            f"- batch-{batch_id}: TIMED OUT after {BATCH_TIMEOUT_S}s"
            f" ({processed}/{n_rows} processed)"
        )
    elif proc.exitcode != 0:
        logger.error(
            "[batch-%d] crashed with exit code %d (%d/%d processed)",
            batch_id,
            proc.exitcode,
            processed,
            n_rows,
        )
        crashed_batches.append(
            f"- batch-{batch_id}: CRASHED exit code {proc.exitcode}"
            f" ({processed}/{n_rows} processed)"
        )
    else:
        logger.info(
            "[batch-%d] done: %d ok, %d failed",
            batch_id,
            data["ok_count"] if data else 0,
            len(data["failures"]) if data else 0,
        )


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


def test_all_buildings_make_env_reset_and_step(tmp_path: Path) -> None:
    """Ensure each building can run one reset/step cycle without crashing."""
    registry = get_registry()
    metadata = registry.metadata
    total_buildings = len(metadata)
    logger.info("Found %d buildings across the registry", total_buildings)

    # ------------------------------------------------------------------
    # Prefetch
    # ------------------------------------------------------------------
    prefetch_failures = _prefetch_building_types(metadata)
    if prefetch_failures:
        pf_messages: list[str] = []
        for building_type, reason in prefetch_failures.items():
            ids = metadata.loc[
                metadata["building_type"] == building_type, "building_id"
            ]
            sample = ", ".join(str(item) for item in ids.head(5).tolist())
            pf_messages.append(
                "\n".join(
                    [
                        f"- {building_type} ({len(ids)} building(s))",
                        f"  Sample IDs: {sample}",
                        f"  {reason}",
                    ]
                )
            )
        pytest.fail(
            "Failed to prefetch one or more building archives; smoke test aborted early:\n"
            + "\n\n".join(pf_messages)
        )

    # ------------------------------------------------------------------
    # Build batches
    # ------------------------------------------------------------------
    all_rows: list[tuple[str, str]] = [
        (str(r.building_type), str(r.building_id))
        for r in metadata.itertuples(index=False)  # type: ignore[call-arg]
    ]
    n_batches = math.ceil(len(all_rows) / BATCH_SIZE)
    logger.info(
        "Split %d buildings into %d batches of up to %d, running %d at a time",
        total_buildings,
        n_batches,
        BATCH_SIZE,
        NUM_WORKERS,
    )

    test_start = time.monotonic()

    # Accumulators (wrapped in lists so _collect_batch can mutate)
    all_ok: list[int] = []
    all_failures: list[str] = []
    all_failures_by_type: defaultdict[str, list[str]] = defaultdict(list)
    all_processed: list[int] = []
    crashed_batches: list[str] = []

    # ------------------------------------------------------------------
    # Sliding window of NUM_WORKERS concurrent batch processes
    # ------------------------------------------------------------------
    # Each entry: (batch_id, Process, result_file, n_rows)
    active: list[tuple[int, mp.Process, str, int]] = []
    next_batch = 0

    while next_batch < n_batches or active:
        # Fill up to NUM_WORKERS slots
        while next_batch < n_batches and len(active) < NUM_WORKERS:
            start = next_batch * BATCH_SIZE
            end = min(start + BATCH_SIZE, len(all_rows))
            batch_rows = all_rows[start:end]
            result_file = str(tmp_path / f"batch_{next_batch}_result.json")
            proc = mp.Process(
                target=_run_batch,
                args=(
                    next_batch,
                    batch_rows,
                    total_buildings,
                    start,
                    str(tmp_path),
                    result_file,
                ),
                daemon=True,
            )
            proc.start()
            active.append((next_batch, proc, result_file, len(batch_rows)))
            next_batch += 1

        # Check for completed batches
        still_active: list[tuple[int, mp.Process, str, int]] = []
        for batch_id, proc, result_file, n_rows in active:
            if proc.is_alive():
                still_active.append((batch_id, proc, result_file, n_rows))
            else:
                proc.join(timeout=10)
                _collect_batch(
                    batch_id,
                    proc,
                    result_file,
                    n_rows,
                    all_ok,
                    all_failures,
                    all_failures_by_type,
                    all_processed,
                    crashed_batches,
                )
        active = still_active

        if active:
            time.sleep(0.5)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total_ok = sum(all_ok)
    total_processed = sum(all_processed)
    total_failed = len(all_failures)
    total_elapsed = time.monotonic() - test_start
    logger.info(
        "Smoke test complete: %d ok, %d failed, %d crashed-before-finish, "
        "%d/%d processed (%.1fs total)",
        total_ok,
        total_failed,
        total_buildings - total_processed,
        total_processed,
        total_buildings,
        total_elapsed,
    )

    report_sections: list[str] = []

    if crashed_batches:
        report_sections.append(
            "Batch processes that crashed or timed out:\n" + "\n".join(crashed_batches)
        )

    if all_failures:
        overview = "\n".join(
            f"- {building_type}: {len(ids)} failure(s)"
            for building_type, ids in sorted(all_failures_by_type.items())
        )
        report_sections.append(
            "Building environments that failed during make_env/reset/step:\n"
            + f"{overview}\n\n"
            + "\n\n".join(all_failures)
        )

    if report_sections:
        pytest.fail("\n\n".join(report_sections))
