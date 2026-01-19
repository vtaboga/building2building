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

