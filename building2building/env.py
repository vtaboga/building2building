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
from importlib.util import find_spec
from pathlib import Path
from typing import TypeVar

ENERGYPLUS_PATH: ContextVar[Path] = ContextVar("ENERGYPLUS_PATH")

T = TypeVar("T")


@contextlib.contextmanager
def ctxvar_set(var: ContextVar[T], val: T):
    token = var.set(val)
    try:
        yield token
    finally:
        var.reset(token)


logger = logging.getLogger(__name__)


def common_energyplus_paths() -> list[Path]:
    """
    Get the default EnergyPlus installation paths based on the operating system.
    """
    system = platform.system()
    if system == "Linux":
        return [Path("/usr/local/EnergyPlus-24-1-0")]
    elif system == "Darwin":  # macOS
        return [
            Path("/Applications/EnergyPlus-24-1-0"),
            Path("/Applications/EnergyPlus-24.1.0"),
        ]
    elif system == "Windows":
        return [
            Path("C:\\Program Files\\EnergyPlus-24-1-0"),
            Path("C:\\EnergyPlus-24-1-0"),
        ]
    return []


def find_energyplus_path(manual_path=None) -> Path:
    """
    Setup the path to EnergyPlus Python API.
    This should be called before any EnergyPlus-related imports.

    Args:
        manual_path (str, optional): Manually specified path to EnergyPlus

    Returns:
        str: Path to EnergyPlus if found, None otherwise
    """
    # We want this procedure to be idempotent. If we have already set
    # ENERGYPLUS_PATH, we should do nothing.
    if (p := ENERGYPLUS_PATH.get(None)) is not None:
        return p

    # First, check if pyenergyplus is in the load path. If that is the case, we
    # don't need env variables to find where the various binaries are.

    if (spec := find_spec("pyenergyplus")) is not None:
        if spec.origin is not None:
            # will give us something like 'PLACE/lib/python3.11/site-packages/pyenergyplus/__init__.py'
            ep_path = Path(spec.origin).parent.parent.parent.parent.parent
            return ep_path

    # Then, we want to look at common default places

    for path in common_energyplus_paths():
        if path.exists():
            return path

    # Then, we want to look at the ENERGYPLUS_PATH environment variable.

    if (p := os.getenv("ENERGYPLUS_PATH")) is not None:
        return Path(p)

    raise Exception("EnergyPlus installation not found")


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


def setup_energyplus_path():
    energyplus_path = find_energyplus_path()
    if energyplus_path not in sys.path:
        ENERGYPLUS_PATH.set(energyplus_path)
        sys.path.append(str(energyplus_path))
        logger.info(f"Added EnergyPlus path: {energyplus_path}")


class DataPaths:
    """Singleton-like class managing processing context variables."""

    _data_path: ContextVar[Path] = ContextVar("data_path")

    _processed: str = "processed"

    _unprocessed: str = "unprocessed"

    _metadata: str = "metadata"

    _weather: str = "weather"

    _intermediate: str = "intermediate"

    @classmethod
    def setup(cls):
        cls._data_path.set(get_cache_dir())

    @classmethod
    def data_dir(cls) -> Path:
        p = cls._data_path.get()
        p.mkdir(parents=True, exist_ok=True)
        return p

    @classmethod
    def weather_dir(cls) -> Path:
        p = cls.data_dir() / cls._weather
        p.mkdir(parents=True, exist_ok=True)
        return p

    @classmethod
    def metadata_dir(cls) -> Path:
        p = cls.data_dir() / cls._metadata
        p.mkdir(parents=True, exist_ok=True)
        return p

    @classmethod
    def processed_dir(cls) -> Path:
        p = cls.data_dir() / cls._processed
        p.mkdir(parents=True, exist_ok=True)
        return p

    @classmethod
    def unprocessed_dir(cls) -> Path:
        p = cls.data_dir() / cls._unprocessed
        p.mkdir(parents=True, exist_ok=True)
        return p

    @classmethod
    def intermediate_dir(cls) -> Path:
        p = cls.data_dir() / cls._intermediate
        p.mkdir(parents=True, exist_ok=True)
        return p
