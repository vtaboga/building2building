from __future__ import annotations

import datetime as dt
import json
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from building2building.sources import hydroquebec


def _parse_hq_building_id(idf_filename: str) -> str | None:
    """
    HydroQuebec idf paths look like: IDFsAndSchedules/<N>/in.idf
    """
    m = re.search(r"IDFsAndSchedules/(\d+)/in\.idf$", idf_filename)
    if not m:
        return None
    return m.group(1)


def _sanitize_for_path(s: str) -> str:
    s2 = re.sub(r"[^A-Za-z0-9._-]+", "_", s.strip())
    return s2.strip("._-") or "run"


def _find_default_venv(project_root: Path) -> Path | None:
    for rel in (".venv", "venv", ".env"):
        cand = project_root / rel
        if (cand / "bin" / "activate").is_file():
            return cand
    return None


def _build_baselines_cmd_bash(
    *,
    venv_dir: Path,
    overrides: list[str],
    extra_overrides: list[str],
) -> list[str]:
    """
    Return a command that activates a venv before running baselines.
    """
    activate = venv_dir / "bin" / "activate"
    if not activate.is_file():
        raise FileNotFoundError(f"Virtualenv activate script not found: {activate}")

    parts = ["python", "-m", "scripts.baselines", *overrides, *extra_overrides]
    shell_cmd = f"source {shlex.quote(str(activate))} && " + " ".join(
        shlex.quote(p) for p in parts
    )
    return ["bash", "-lc", shell_cmd]


def _row_to_overrides(*, row: Any, run_dir: Path, policy: str) -> list[str]:
    idf_filename = str(row["idf_filename"])
    schedule_filename = str(row["schedule_filename"])

    # Important: baseline configs in this repo are nested like `bldg.bldg.*`
    # because the bldg config group YAML also has a `bldg:` root key.
    return [
        f"policy={policy}",
        # Fetch *all* single-family detached rows: avoid narrowing by year/units.
        'bldg.bldg.geometry_unit_type="single-family detached"',
        "bldg.bldg.geometry_building_num_units=null",
        "bldg.bldg.year_built=null",
        # Select a single row deterministically.
        f"+bldg.bldg.idf_filename={idf_filename}",
        f"+bldg.bldg.schedule_filename={schedule_filename}",
        # Put each run in its own output directory.
        # Quote the path because our run dirs may contain '=' (e.g. "policy=..."),
        # which would otherwise break Hydra's override grammar parsing.
        f'hydra.run.dir="{run_dir.as_posix()}"',
    ]


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="experiment_hydroquebec_single_family_detached",
)
def main(cfg: DictConfig) -> int:
    exp = getattr(cfg, "experiment", None)
    if exp is None:
        raise RuntimeError("Missing `experiment` section in config.")

    # Hydra changes cwd; for output + subprocess execution we want the repo root.
    repo_root = Path(get_original_cwd()).resolve()

    venv_raw = getattr(exp, "venv", None)
    venv_dir: Path | None
    if venv_raw is None:
        venv_dir = _find_default_venv(repo_root)
    else:
        venv_dir = Path(str(venv_raw))
        if not venv_dir.is_absolute():
            venv_dir = (repo_root / venv_dir).resolve()
    if venv_dir is None:
        raise RuntimeError(
            "No virtualenv detected. Create one at .venv/ (recommended) "
            "or set experiment.venv=/path/to/venv."
        )

    df = hydroquebec.search_buildings(geometry_unit_type="single-family detached")
    if df.empty:
        print("No buildings found for geometry_unit_type=single-family detached")
        return 1

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root_raw = Path(str(getattr(exp, "output_root")))
    output_root = (
        output_root_raw if output_root_raw.is_absolute() else (repo_root / output_root_raw)
    )

    policy_name = str(getattr(exp, "policy"))
    out_root: Path = output_root / f"policy={_sanitize_for_path(policy_name)}" / timestamp
    out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = out_root / "runs_manifest.jsonl"

    start = int(getattr(exp, "start", 0))
    max_buildings = getattr(exp, "max_buildings", None)
    max_buildings_int = int(max_buildings) if max_buildings is not None else None
    end = len(df) if max_buildings_int is None else min(len(df), start + max_buildings_int)
    selected_idxs = list(range(start, end))

    skip_existing = bool(getattr(exp, "skip_existing", True))
    dry_run = bool(getattr(exp, "dry_run", False))
    continue_on_error = bool(getattr(exp, "continue_on_error", True))
    extra_overrides_raw = getattr(exp, "extra_overrides", []) or []
    extra_overrides = [str(x) for x in list(extra_overrides_raw)]

    failures = 0
    ran = 0

    for i in selected_idxs:
        row = df.iloc[i]
        idf_filename = str(row["idf_filename"])
        schedule_filename = str(row["schedule_filename"])
        epw_filename = str(row["epw_filename"])

        hq_id = _parse_hq_building_id(idf_filename) or str(i)
        run_name = f"hq_{hq_id}_{_sanitize_for_path(idf_filename)}"
        run_dir = out_root / run_name

        if skip_existing and run_dir.exists():
            print(f"[skip] {run_name} (exists)")
            continue

        overrides = _row_to_overrides(row=row, run_dir=run_dir, policy=policy_name)
        cmd = _build_baselines_cmd_bash(
            venv_dir=venv_dir,
            overrides=overrides,
            extra_overrides=extra_overrides,
        )

        record = {
            "run_name": run_name,
            "run_dir": str(run_dir),
            "idf_filename": idf_filename,
            "schedule_filename": schedule_filename,
            "epw_filename": epw_filename,
            "venv": str(venv_dir),
            "cmd": cmd,
        }
        with manifest_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        print(f"[run] {run_name}")
        if dry_run:
            print(" ".join(cmd))
            continue

        ran += 1
        try:
            subprocess.run(cmd, check=True, cwd=str(repo_root))
        except subprocess.CalledProcessError as e:
            failures += 1
            print(f"[fail] {run_name} (exit={e.returncode})")
            if not continue_on_error:
                break

    print(
        json.dumps(
            {
                "total_matched": int(len(df)),
                "selected": int(len(selected_idxs)),
                "ran": int(ran),
                "failures": int(failures),
                "output_root": str(out_root),
                "manifest": str(manifest_path),
            },
            indent=2,
        )
    )
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

