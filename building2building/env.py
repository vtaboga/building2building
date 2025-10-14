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
from typing import Any, Literal, get_args

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
GlibcVersion = Literal["2.35", "2.38"]


binaries: dict[Platform, dict[GlibcVersion, Derivation]] = {
    "linux-x86_64": {
        "2.38": ChildFile(
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
        "2.35": ChildFile(
            ExtractTarball(
                DownloadFile(
                    "energyplus-25.1.0",
                    "https://github.com/NREL/EnergyPlus/releases/download/v25.1.0/EnergyPlus-25.1.0-68a4a7c774-Linux-Ubuntu22.04-x86_64.tar.gz",
                    bytes.fromhex(
                        "bb12f522f8b5a6144f68f19c637d3790e2d1c948a7cb3ebeda479fd0e8a33f7e"
                    ),
                )
            ),
            "EnergyPlus-25.1.0-68a4a7c774-Linux-Ubuntu22.04-x86_64",
        ),
    },
}


def energyplus_path() -> Realizable:
    current_platform: Platform = sysconfig.get_platform()  # type: ignore
    assert current_platform in get_args(Platform)

    if p := os.getenv("ENERGYPLUS_PATH"):
        path = Path(p).resolve()
        return LocalSymlink("energyplus-path", path)
    else:
        glibc_version: GlibcVersion = platform.libc_ver()[1]  # type: ignore
        assert glibc_version in get_args(GlibcVersion), (
            f"glibc version not supported: {glibc_version}"
        )

        return binaries[current_platform][glibc_version]


def setup_energyplus_path():
    ep = realize(STORE_PATH.get(), energyplus_path())

    sys.path.append(str(ep))
