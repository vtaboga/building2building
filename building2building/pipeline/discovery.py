import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from cattrs import structure, unstructure

from building2building.pipeline.parse_reports import (
    get_net_conditioned_area,
    get_warmup_days,
)
from building2building.pipeline.steps.outputs import add_all_outputs, modify_run_period
from building2building.store import (
    OUTPUT,
    Expression,
    Realizable,
    derivation,
    expression,
)


@dataclass(frozen=True)
class Metadata:
    """Discovery metadata extracted from EnergyPlus simulation."""

    source_path: Path

    net_conditioned_area: float
    warmup_phases: int
    warmup_days: int


@derivation("discovery-metadata")
def DiscoveryMetadata(epjson: Path, epw: Path):
    """Run simulation and write metadata JSON."""
    import pyenergyplus.api

    out = OUTPUT.get()

    warmup_count = 0

    def warmup_callback(state):
        nonlocal warmup_count
        warmup_count += 1

    api = pyenergyplus.api.EnergyPlusAPI()
    state = api.state_manager.new_state()

    api.runtime.callback_after_new_environment_warmup_complete(state, warmup_callback)

    out.mkdir()
    metadata_out = out / "metadata.json"
    eplusout = out / "eplusout"

    tmpdir = Path(tempfile.mkdtemp())

    api.runtime.run_energyplus(
        state,
        [
            "-d",
            str(tmpdir),
            "-w",
            str(epw),
            str(epjson),
        ],
    )

    # Always persist the EnergyPlus outputs directory, even when parsing fails.
    # This is critical for debugging fatal E+ errors (e.g. when `eplustbl.htm` is missing).
    if eplusout.exists():
        shutil.rmtree(eplusout, ignore_errors=True)
    shutil.move(str(tmpdir), str(eplusout))

    # If a per-run debug directory is provided, copy `eplusout.err` there so it is
    # easily accessible from experiment output folders.
    debug_dir_raw = os.environ.get("B2B_PIPELINE_DEBUG_DIR")
    if debug_dir_raw:
        debug_dir = Path(debug_dir_raw).expanduser().resolve()
        try:
            debug_dir.mkdir(parents=True, exist_ok=True)
            dbg = debug_dir / "discovery-metadata"
            dbg.mkdir(parents=True, exist_ok=True)
            err_src = eplusout / "eplusout.err"
            if err_src.exists():
                shutil.copy(err_src, dbg / "eplusout.err")
        except Exception:
            # Best-effort only; never mask the underlying failure.
            pass

    # Extract metadata from simulation outputs (may raise if EnergyPlus terminated early)
    htm_file = eplusout / "eplustbl.htm"
    if not htm_file.exists():
        # Provide a helpful error message; the detailed reason is in `eplusout.err`.
        err_path = eplusout / "eplusout.err"
        hint = f"EnergyPlus did not produce {htm_file.name} (see {err_path})."
        if debug_dir_raw:
            hint += f" Copied to {Path(debug_dir_raw) / 'discovery-metadata' / 'eplusout.err'}."
        raise FileNotFoundError(hint)

    area = get_net_conditioned_area(htm_file)
    warmup_days = get_warmup_days(htm_file)

    # Create Metadata instance and serialize with cattrs
    metadata = Metadata(
        eplusout,
        net_conditioned_area=area,
        warmup_phases=warmup_count,
        warmup_days=warmup_days,
    )

    with open(metadata_out, "w") as f:
        json.dump(unstructure(metadata), f, indent=2)


def extract_discovery_metadata(
    epjson: Realizable,
    epw: Realizable,
    discovery_run_days: int = 1,
) -> Expression[Metadata]:
    """
    Prepare epJSON for discovery simulation and extract all metadata.

    This function:
    1. Adds all necessary output configurations (meters, EDD, tabular, SQLite)
    2. Runs a discovery simulation with warmup phase detection
    3. Extracts metadata from simulation outputs

    Metadata extracted:
    - net_conditioned_area: Building floor area in m²
    - warmup_phases: Number of warmup phases (for minergym)
    - warmup_days: Number of warmup days EnergyPlus used

    Args:
        epjson: Path to the building epJSON file
        epw: Path to the weather file

    Returns:
        Expression that resolves to Metadata instance

    Note:
        Uses pyenergyplus.api which requires setup_energyplus_path() to be called first.
    """

    # Add all output configurations needed for discovery simulation and keep
    # the run period intentionally short to speed up metadata extraction.
    if discovery_run_days < 1 or discovery_run_days > 31:
        raise ValueError(
            f"discovery_run_days must be in [1, 31], got {discovery_run_days}"
        )
    epjson_with_outputs = add_all_outputs(epjson)
    epjson_with_outputs = modify_run_period(
        epjson_with_outputs,
        begin_day_of_month=1,
        begin_month=1,
        end_day_of_month=discovery_run_days,
        end_month=1,
    )

    @expression()
    def parse_metadata(base_path: Path) -> Metadata:
        """Parse metadata JSON into Metadata dataclass."""
        meta_path = base_path / "metadata.json"
        try:
            with open(meta_path, "r") as f:
                metadata_dict = json.load(f)
        except FileNotFoundError as e:
            # When EnergyPlus terminates early, `DiscoveryMetadata` may have persisted
            # `eplusout/` but not produced `metadata.json`. Copy `eplusout.err` into the
            # per-run debug directory (if configured) so the underlying E+ fatal is easy
            # to inspect even when the derivation output is cached.
            debug_dir_raw = os.environ.get("B2B_PIPELINE_DEBUG_DIR")
            if debug_dir_raw:
                try:
                    debug_dir = Path(debug_dir_raw).expanduser().resolve()
                    dbg = debug_dir / "discovery-metadata"
                    dbg.mkdir(parents=True, exist_ok=True)
                    err_src = base_path / "eplusout" / "eplusout.err"
                    if err_src.exists():
                        shutil.copy(err_src, dbg / "eplusout.err")
                except Exception:
                    pass
            raise FileNotFoundError(
                f"Discovery metadata missing at {meta_path}. "
                f"EnergyPlus likely terminated early; check "
                f"{base_path / 'eplusout' / 'eplusout.err'}"
                + (
                    f" (also copied to {Path(debug_dir_raw) / 'discovery-metadata' / 'eplusout.err'})."
                    if debug_dir_raw
                    else "."
                )
            ) from e

        return structure(metadata_dict, Metadata)

    return parse_metadata(DiscoveryMetadata(epjson_with_outputs, epw))
