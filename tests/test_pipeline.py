import logging
from pathlib import Path

import building2building.sources.autobem as autobem
import building2building.sources.canmet as canmet
import building2building.sources.energycodes as energycodes
from building2building.env import STORE_PATH, binaries, energyplus_path
from building2building.pipeline import create_complete_pipeline
from building2building.store import LocalFile, realize

here = Path(__file__).parent.resolve()

logging.basicConfig(level=logging.DEBUG)


def test_linux_x86_24_1():
    realize(STORE_PATH.get(), binaries["linux-x86_64"]["24.1.0"])


def test_linux_x86_24_2():
    realize(STORE_PATH.get(), binaries["linux-x86_64"]["24.2.0"])


def test_linux_x86_25_1():
    realize(STORE_PATH.get(), binaries["linux-x86_64"]["25.1.0"])


def test_pipeline():
    x = LocalFile(
        here / "data/1002000371360.idf",
    )

    realize(
        STORE_PATH.get(),
        create_complete_pipeline(x, energyplus_path(), src_version="9.4.0"),
    )


def test_energycodes():
    energycodes.search_config()


def test_autobem():
    autobem.search_config("AK", "Anchorage")


def test_canmet():
    realize(
        STORE_PATH.get(),
        canmet.search_buildings(region="QUEBEC").iloc[0].derivation_thunk(),
    )
