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
from contextvars import ContextVar
from functools import cache
from importlib.util import find_spec
from pathlib import Path
from typing import TypeVar

from building2building.store import (
    Child,
    Derivation,
    DownloadFile,
    ExtractTarball,
    Symlink,
    build,
)

logger = logging.getLogger(__name__)


def get_cache_dir():
    if os.name == "nt":  # Windows
        return Path(os.environ.get("LOCALAPPDATA", "~")) / "building2building"
    elif os.name == "posix":  # Linux/macOS
        if "XDG_CACHE_HOME" in os.environ:
            return Path(os.environ["XDG_CACHE_HOME"]) / "building2building"
        else:
            return Path.home() / ".cache" / "building2building"
    else:
        return Path.home() / ".building2building"  # Fallback


def energyplus_path() -> Derivation:
    if p := os.getenv("ENERGYPLUS_PATH"):
        path = Path(p).resolve()
        return Symlink(path)
    else:
        return Child(
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
        )


STORE_PATH: ContextVar[Path] = ContextVar("STORE_PATH")
STORE_PATH.set(get_cache_dir())


def setup_energyplus_path():
    ep = build(STORE_PATH.get(), energyplus_path())
    sys.path.append(str(ep))
