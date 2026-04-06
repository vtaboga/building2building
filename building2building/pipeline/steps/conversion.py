import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Literal, TypeAlias

from building2building.env import STORE_PATH
from building2building.pipeline.common import chdir
from building2building.store import (
    OUTPUT,
    ChildFile,
    Derivation,
    Realizable,
    derivation,
    expression,
    realize,
)
from pandas import DataFrame

logger = logging.getLogger(__name__)

Transition: TypeAlias = Literal[
    "9.4.0-to-9.5.0",
    "9.5.0-to-9.6.0",
    "9.6.0-to-22.1.0",
    "22.1.0-to-22.2.0",
    "22.2.0-to-23.1.0",
    "23.1.0-to-23.2.0",
    "23.2.0-to-24.1.0",
]


@derivation("upgraded.idf")
def upgrade_idf(input: Path, transition_exe: Path, from_idd: Path, to_idd: Path):
    dst = OUTPUT.get()
    with tempfile.TemporaryDirectory() as tempdir:
        with chdir(Path(tempdir)):
            Path(from_idd.name).symlink_to(from_idd)
            Path(to_idd.name).symlink_to(to_idd)
            shutil.copy(input, "in.idf")

            temp_idf_out = Path(tempdir) / "in.idfnew"
            # Run the transition executable from the temp_dir
            cmdline = [
                str(transition_exe),
                "in.idf",
            ]

            logging.debug(f"Using cmd: {cmdline}")

            result = subprocess.run(
                cmdline,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env={"DISPLAY": ""},
                cwd=tempdir,
            )

        if result.stdout:
            logger.debug(f"Transition output: {result.stdout}")
        if result.stderr:
            logger.warning(f"Transition errors: {result.stderr}")

        if not temp_idf_out.exists():
            raise Exception(
                f"failed to upgrade idf file from {from_idd.name} to {to_idd.name}"
            )

        shutil.copy(temp_idf_out, dst)


@derivation("building.epjson")
def ConvertIDF(input: Path, converter: Path):
    dst = OUTPUT.get()

    with tempfile.TemporaryDirectory() as temp:
        temp_path = Path(temp)
        temp_idf_path = temp_path / "in.idf"
        temp_epjson_path = temp_idf_path.with_suffix(".epJSON")

        shutil.copy(input, temp_idf_path)

        result = subprocess.run(
            [str(converter), str(temp_idf_path)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=temp,
        )

        if result.stdout:
            logger.debug(f"conversion output: {result.stdout}")
        if result.stderr:
            logger.warning(f"conversion errors: {result.stderr}")

        if not temp_epjson_path.exists():
            raise Exception("failed to convert idf to json")
        shutil.copy(temp_epjson_path, dst)


def convert_idf(idf_in: Derivation, energyplus_path: Realizable) -> Derivation:
    """Create an IDF to epJSON conversion build step."""
    converter = ChildFile(energyplus_path, "ConvertInputFormat")
    return ConvertIDF(idf_in, converter)


all_transitions: list[Transition] = [
    "9.4.0-to-9.5.0",
    "9.5.0-to-9.6.0",
    "9.6.0-to-22.1.0",
    "22.1.0-to-22.2.0",
    "22.2.0-to-23.1.0",
    "23.1.0-to-23.2.0",
    "23.2.0-to-24.1.0",
]


@expression()
def scan_upgraders(energyplus: Path) -> DataFrame:
    ivu = energyplus / "PreProcess/IDFVersionUpdater"

    pattern = re.compile(r"^Transition-V(\d+)-(\d+)-(\d+)-to-V(\d+)-(\d+)-(\d+)$")

    records = []

    for p in ivu.glob("Transition*"):
        if m := pattern.match(p.name):
            M1, m1, p1, M2, m2, p2 = m.groups()

            v1 = f"{M1}.{m1}.{p1}"
            v2 = f"{M2}.{m2}.{p2}"

            idd1 = ivu / f"V{M1}-{m1}-{p1}-Energy+.idd"
            idd2 = ivu / f"V{M2}-{m2}-{p2}-Energy+.idd"

            records.append((str(p), v1, v2, str(idd1), str(idd2)))

    return DataFrame(
        records,
        columns=["path", "src_version", "dst_version", "src_idd", "dst_idd"],
    )


def upgrade(
    input_file: Realizable, energyplus_path: Realizable, src_version: str
) -> Derivation:
    # Multi-step upgrade process

    upgraders: DataFrame = realize(STORE_PATH.get(), scan_upgraders(energyplus_path))

    # Chain upgrades
    current = input_file
    current_version = src_version

    # Ensure the given src_version is even a valid version at all
    valid_versions = set(upgraders.src_version) | set(upgraders.dst_version)

    if src_version not in valid_versions:
        raise Exception(
            f"src_version ({src_version}) is not valid. valid versions: {valid_versions}"
        )

    while True:
        possible_upgraders = upgraders[upgraders.src_version == current_version]
        if len(possible_upgraders) == 0:
            break
        upgrader = possible_upgraders.iloc[0]

        current = upgrade_idf(
            current, Path(upgrader.path), Path(upgrader.src_idd), Path(upgrader.dst_idd)
        )

        current_version = upgrader.dst_version

    return current
