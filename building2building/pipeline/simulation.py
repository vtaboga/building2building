import shutil
import subprocess
import tempfile
from pathlib import Path

from building2building.store import OUTPUT, ChildFile, Derivation, derivation


@derivation("simulation-outputs")
def run_simulation(ep_path: Path, epjson: Path, epw: Path):
    """
    Run an EnergyPlus simulation and save the output files.
    Use to run a dummy simulation
    """
    out = OUTPUT.get()

    tmp = Path(tempfile.mkdtemp())
    cmd = [
        str(ep_path / "energyplus"),
        "-d",
        str(tmp),
        "-w",
        str(epw),
        "-x",
        str(epjson),
    ]

    subprocess.run(cmd, check=True)

    htm_file = tmp / "eplustbl.htm"
    edd_file = tmp / "eplusout.edd"
    eio_file = tmp / "eplusout.eio"
    sql_file = tmp / "eplusout.sql"

    if not htm_file.exists():
        raise Exception("EnergyPlus simulation did not produce eplustbl.htm")
    if not edd_file.exists():
        raise Exception("EnergyPlus simulation did not produce eplusout.edd")
    if not eio_file.exists():
        raise Exception("EnergyPlus simulation did not produce eplusout.eio")
    if not sql_file.exists():
        raise Exception("EnergyPlus simulation did not produce eplusout.sql")

    # Ensure output directory exists
    out.mkdir(parents=True, exist_ok=True)

    shutil.copy(htm_file, out / "eplustbl.htm")
    shutil.copy(edd_file, out / "eplusout.edd")
    shutil.copy(eio_file, out / "eplusout.eio")
    shutil.copy(sql_file, out / "eplusout.sql")


def eplustbl(ep_path: Path, epjson: Path, epw: Path) -> Derivation:
    """Get the eplustbl.htm file from a simulation"""
    sim = run_simulation(ep_path, epjson, epw)
    return ChildFile(sim, "eplustbl.htm")


def eddfile(ep_path: Path, epjson: Path, epw: Path) -> Derivation:
    """Get the eplusout.edd file from a simulation"""
    sim = run_simulation(ep_path, epjson, epw)
    return ChildFile(sim, "eplusout.edd")


def eiofile(ep_path: Path, epjson: Path, epw: Path) -> Derivation:
    """Get the eplusout.eio file from a simulation"""
    sim = run_simulation(ep_path, epjson, epw)
    return ChildFile(sim, "eplusout.eio")


@derivation("warmup-phases.json")
def detect_warmup_phases(epjson: Path, epw: Path):
    """
    Run a minimal EnergyPlus simulation to detect the number of warmup phases.

    This function runs EnergyPlus with the API and counts how many times the
    warmup complete callback is triggered, which corresponds to the number of
    warmup phases (typically zone sizing + system sizing + plant sizing = 3).

    The result is cached by the derivation system.

    Args:
        epjson: Path to the building epJSON file
        epw: Path to the weather file

    Note:
        Uses pyenergyplus.api which requires setup_energyplus_path() to be called first.
    """
    import json
    import tempfile

    import pyenergyplus.api

    out = OUTPUT.get()

    warmup_count = 0

    def warmup_callback(state):
        print("warmup")
        nonlocal warmup_count
        warmup_count += 1

    with tempfile.TemporaryDirectory() as tmpdir:
        api = pyenergyplus.api.EnergyPlusAPI()
        state = api.state_manager.new_state()

        api.runtime.callback_after_new_environment_warmup_complete(
            state, warmup_callback
        )

        api.runtime.run_energyplus(
            state,
            [
                "-d",
                tmpdir,
                "-w",
                str(epw),
                str(epjson),
            ],
        )

    with open(out, "w") as f:
        json.dump(warmup_count, f)
