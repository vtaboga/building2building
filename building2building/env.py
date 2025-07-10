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
from typing import TypeVar, reveal_type

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


def setup_energyplus_path():
    energyplus_path = find_energyplus_path()
    if energyplus_path not in sys.path:
        ENERGYPLUS_PATH.set(energyplus_path)
        sys.path.append(str(energyplus_path))
        logger.info(f"Added EnergyPlus path: {energyplus_path}")
