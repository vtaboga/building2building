"""
Centralized configuration management for EnergyPlus paths and other settings.
This module is used by both setup scripts and runtime code.
"""

import contextlib
import json
import logging
import os
import platform
import sys
import sysconfig
from contextvars import ContextVar
from functools import cache
from importlib.util import find_spec
from pathlib import Path
from typing import Literal, get_args

from building2building.store import (
    ChildFile,
    Derivation,
    DownloadFile,
    ExtractTarball,
    LocalSymlink,
    Realizable,
    realize,
)

logger = logging.getLogger(__name__)


def get_cache_dir() -> Path:
    if os.name == "nt":  # Windows
        return Path(os.environ.get("LOCALAPPDATA", "~")) / "building2building"
    elif os.name == "posix":  # Linux/macOS
        if "XDG_CACHE_HOME" in os.environ:
            return Path(os.environ["XDG_CACHE_HOME"]) / "building2building"
        else:
            return Path.home() / ".cache" / "building2building"
    else:
        return Path.home() / ".building2building"  # Fallback


def store_path() -> Path:
    if p := os.getenv("STORE_PATH"):
        path = Path(p).resolve()
        return path
    else:
        return get_cache_dir()


STORE_PATH: ContextVar[Path] = ContextVar("STORE_PATH")
STORE_PATH.set(store_path())


Platform = Literal["linux-x86_64"]
Version = Literal["24.1.0", "24.2.0", "25.1.0"]

binaries: dict[Platform, dict[Version, Derivation]] = {
    "linux-x86_64": {
        "24.1.0": ChildFile(
            ExtractTarball(
                DownloadFile(
                    "energyplus-24.1.0.tar.gz",
                    "https://github.com/NREL/EnergyPlus/releases/download/v24.1.0/EnergyPlus-24.1.0-9d7789a3ac-Linux-Ubuntu20.04-x86_64.tar.gz",
                    bytes.fromhex(
                        "7b90fb1d6b1e58875217eedbc745e8c6d1476321d6fa4ca1d5833414770096cc"
                    ),
                )
            ),
            "EnergyPlus-24.1.0-9d7789a3ac-Linux-Ubuntu20.04-x86_64",
        ),
        "24.2.0": ChildFile(
            ExtractTarball(
                DownloadFile(
                    "energyplus-24.2.0.tar.gz",
                    "https://github.com/NREL/EnergyPlus/releases/download/v24.2.0a/EnergyPlus-24.2.0-94a887817b-Linux-Ubuntu22.04-x86_64.tar.gz",
                    bytes.fromhex(
                        "f1404a264f6483b1e4c9c6f37d174974a39a9c6cd2282acc2110f48e60755e30",
                    ),
                )
            ),
            "EnergyPlus-24.2.0-94a887817b-Linux-Ubuntu22.04-x86_64",
        ),
        "25.1.0": ChildFile(
            ExtractTarball(
                DownloadFile(
                    "energyplus-25.1.0",
                    "https://github.com/NREL/EnergyPlus/releases/download/v25.1.0/EnergyPlus-25.1.0-68a4a7c774-Linux-Ubuntu24.04-x86_64.tar.gz",
                    bytes.fromhex(
                        "faee846457ce450e8b2434e918cf70b551c4f59bea2b282711a498a6c878495a",
                    ),
                )
            ),
            "EnergyPlus-25.1.0-68a4a7c774-Linux-Ubuntu24.04-x86_64",
        ),
    },
}


def energyplus_path(version: Version = "25.1.0") -> Realizable:
    current_platform = sysconfig.get_platform()
    assert current_platform in get_args(Platform)
    platform: Platform = current_platform  # type: ignore

    if p := os.getenv("ENERGYPLUS_PATH"):
        path = Path(p).resolve()
        return LocalSymlink("energyplus-path", path)
    else:
        return binaries[platform][version]


def setup_energyplus_path():
    ep = realize(STORE_PATH.get(), energyplus_path())

    sys.path.append(str(ep))
