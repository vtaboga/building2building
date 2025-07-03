"""
Centralized configuration management for EnergyPlus paths and other settings.
This module is used by both setup scripts and runtime code.
"""


import os
import sys
import platform
from pathlib import Path
import json
import logging
import contextlib
from importlib.util import find_spec
from contextvars import ContextVar
from typing import TypeVar

ENERGYPLUS_PATH: ContextVar[Path] = ContextVar("ENERGYPLUS_PATH")

T = TypeVar("T")

@contextlib.contextmanager
def ctxvar_set(var: ContextVar[T], val: T):
    try:
        token = var.set(val)
        yield token
    finally:
        var.reset(token)

logger = logging.getLogger(__name__)

# Get the project root directory
spec = find_spec("building2building")
if spec is None or spec.origin is None:
    raise Exception("building2building can't find itself. Maybe it hasen't been installed properly.")

PROJECT_ROOT: Path = Path(spec.origin).parent

# Configuration file path in the configs directory
CONFIG_DIR = PROJECT_ROOT / "configs"
CONFIG_FILE = CONFIG_DIR / "energyplus_simulator.json"

def get_default_energyplus_paths():
    """
    Get the default EnergyPlus installation paths based on the operating system.
    """
    system = platform.system()
    if system == 'Linux':
        return ['/usr/local/EnergyPlus-24-1-0']
    elif system == 'Darwin':  # macOS
        return [
            '/Applications/EnergyPlus-24-1-0',
            '/Applications/EnergyPlus-24.1.0'
        ]
    elif system == 'Windows':
        return [
            'C:\\Program Files\\EnergyPlus-24-1-0',
            'C:\\EnergyPlus-24-1-0'
        ]
    return []

def find_energyplus_path(manual_path: Path|None = None) -> Path:
    """
    Find the EnergyPlus installation path by checking:
    1. Manually specified path
    2. Configuration file
    3. Environment variable
    4. Default system paths

    Args:
        manual_path (str, optional): Manually specified path to EnergyPlus

    Returns:
        str: Path to EnergyPlus installation or None if not found
    """
    # Check manually specified path first
    if manual_path and manual_path.exists():
        return manual_path

    # Check configuration file
    config = load_config()
    if config and 'energyplus_path' in config and Path(config['energyplus_path']).exists():
        config_path = config['energyplus_path']
        return Path(config_path)

    # Check environment variable
    if 'ENERGYPLUS_PATH' in os.environ and Path(os.environ['ENERGYPLUS_PATH']).exists():
        return Path(os.environ['ENERGYPLUS_PATH'])

    # Check default paths
    for path in get_default_energyplus_paths():
        if os.path.exists(path):
            return path

    # Add home directory path as a fallback
    home_path = Path.home() / 'EnergyPlus-24-1-0'
    if home_path.exists():
        return home_path

    raise Exception("couldn't find energyplus path")

def save_config(config_data):
    """
    Save configuration to the config file.

    Args:
        config_data (dict): Configuration data to save

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Ensure the configs directory exists
        os.makedirs(CONFIG_DIR, exist_ok=True)

        with open(CONFIG_FILE, 'w') as f:
            json.dump(config_data, f, indent=4)
        logger.info(f"EnergyPlus configuration saved to {CONFIG_FILE}")
        return True
    except Exception as e:
        logger.error(f"Failed to save EnergyPlus configuration: {e}")
        return False

def load_config() -> dict | None:
    """
    Load configuration from the config file.

    Returns:
        dict: Configuration data or empty dict if file doesn't exist
    """
    if not os.path.exists(CONFIG_FILE):
        return {}

    try:
        with open(CONFIG_FILE, 'r') as f:
            o = json.load(f)
            if not isinstance(o, dict):
                e = f"Failed to load EnergyPlus configuration: {CONFIG_FILE} is not a dict"
                logger.error(e)
                raise Exception(e)
    except Exception as e:
        logger.error(f"Failed to load EnergyPlus configuration: {e}")
        return None


def update_energyplus_path(path: Path):
    """
    Update the EnergyPlus path in the configuration file.
    
    Args:
        path (str): Path to EnergyPlus installation
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not path or not os.path.exists(path):
        return False
        
    config = load_config()
    config['energyplus_path'] = str(path)
    return save_config(config)

def setup_energyplus_path(manual_path=None):
    """
    Setup the path to EnergyPlus Python API.
    This should be called before any EnergyPlus-related imports.
    
    Args:
        manual_path (str, optional): Manually specified path to EnergyPlus
        
    Returns:
        str: Path to EnergyPlus if found, None otherwise
    """

    if ENERGYPLUS_PATH.get(None) is not None:
        return

    try:
        energyplus_path = find_energyplus_path(manual_path)
        if energyplus_path not in sys.path:
            ENERGYPLUS_PATH.set(energyplus_path)
            sys.path.append(str(energyplus_path))
            logger.info(f"Added EnergyPlus path: {energyplus_path}")

    except Exception as e:
        logger.warning(
            "EnergyPlus installation not found. If you're not running in a container, "
            "please set the path using setup.sh --energyplus_path=<path>"
        )
        return None

