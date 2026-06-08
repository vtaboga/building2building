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
    DownloadFile,
    ExtractTarball,
    FileLike,
    LocalSymlink,
    Realizable,
    realize,
)

logger = logging.getLogger(__name__)


def get_cache_dir() -> Path:
    if os.name == "nt":  # Windows
        return Path(os.environ.get("LOCALAPPDATA", "~")) / "b2b"
    elif os.name == "posix":  # Linux/macOS
        # Prefer persistent scratch space if available.
        for var in ["SCRATCH"]:
            val = os.environ.get(var)
            if val:
                return Path(val) / "b2b"
        # Otherwise, use standard cache locations
        if "XDG_CACHE_HOME" in os.environ:
            return Path(os.environ["XDG_CACHE_HOME"]) / "b2b"
        else:
            return Path.home() / ".cache" / "b2b"
    else:
        return Path.home() / ".b2b"  # Fallback


def store_path() -> Path:
    if p := os.getenv("STORE_PATH"):
        path = Path(p).resolve()
        return path
    else:
        return get_cache_dir()


STORE_PATH: ContextVar[Path] = ContextVar("STORE_PATH")
STORE_PATH.set(store_path())

Platform = Literal[
    "linux-x86_64",
    "macosx-10.9-x86_64",
    "macosx-11.0-arm64",
    "macosx-15.6-arm64",
]
GlibcVersion = Literal["2.35", "2.38"]

# Note to future selves: under linux, the main constraint on the binary we use
# is the distribution's glibc version. Because glibc is backwards-compatible, it
# is advantageous to pick the binary compiled for the distribution with the
# oldest glibc (in our case, ubuntu 22).
binaries: dict[Platform, FileLike] = {
    "linux-x86_64": ChildFile(
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
    "macosx-15.6-arm64": ChildFile(
        ExtractTarball(
            DownloadFile(
                "energyplus-25.1.0",
                "https://github.com/NREL/EnergyPlus/releases/download/v25.1.0/EnergyPlus-25.1.0-68a4a7c774-Darwin-macOS13-arm64.tar.gz",
                bytes.fromhex(
                    "fdd54d1450cbefbd572f4c2f3e3230b50af912d889333af2c8fd679a5ad9f088"
                ),
            )
        ),
        "EnergyPlus-25.1.0-68a4a7c774-Darwin-macOS13-arm64",
    ),
}


def energyplus_path() -> Realizable:
    # 1. Check for manual override FIRST
    if p := os.getenv("ENERGYPLUS_PATH"):
        path = Path(p).resolve()
        return LocalSymlink("energyplus-path", path)

    # 2. If no manual path, then check platform for auto-download
    current_platform: str = sysconfig.get_platform()

    # If we are on Mac and didn't provide a path, this script doesn't
    # have a download link for Mac anyway, so let's provide a clear error.
    if "macosx" in current_platform:
        raise RuntimeError(
            f"EnergyPlus auto-download not supported on macOS. "
            f"Please set ENERGYPLUS_PATH='/Applications/EnergyPlus-25-1-0' in your terminal."
        )

    assert current_platform in get_args(Platform)
    return binaries[current_platform]  # type: ignore


def setup_energyplus_path():
    ep = realize(STORE_PATH.get(), energyplus_path())

    sys.path.append(str(ep))
