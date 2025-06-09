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

logger = logging.getLogger(__name__)

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

# Configuration file path in the configs directory
CONFIG_DIR = os.path.join(PROJECT_ROOT, "configs")
CONFIG_FILE = os.path.join(CONFIG_DIR, "energyplus_simulator.json")

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

def find_energyplus_path(manual_path=None):
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
    if manual_path and os.path.exists(manual_path):
        return manual_path
        
    # Check configuration file
    config = load_config()
    if config and 'energyplus_path' in config and os.path.exists(config['energyplus_path']):
        return config['energyplus_path']
    
    # Check environment variable
    if 'ENERGYPLUS_PATH' in os.environ and os.path.exists(os.environ['ENERGYPLUS_PATH']):
        return os.environ['ENERGYPLUS_PATH']
    
    # Check default paths
    for path in get_default_energyplus_paths():
        if os.path.exists(path):
            return path
            
    # Add home directory path as a fallback
    home_path = str(Path.home() / 'EnergyPlus-24-1-0')
    if os.path.exists(home_path):
        return home_path
            
    return None

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

def load_config():
    """
    Load configuration from the config file.
    
    Returns:
        dict: Configuration data or empty dict if file doesn't exist
    """
    if not os.path.exists(CONFIG_FILE):
        return {}
        
    try:
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load EnergyPlus configuration: {e}")
        return {}

def update_energyplus_path(path):
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
    config['energyplus_path'] = path
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
    energyplus_path = find_energyplus_path(manual_path)
    
    if energyplus_path:
        if energyplus_path not in sys.path:
            sys.path.append(energyplus_path)
            logger.info(f"Added EnergyPlus path: {energyplus_path}")
        return energyplus_path
    else:
        logger.warning(
            "EnergyPlus installation not found. If you're not running in a container, "
            "please set the path using setup.sh --energyplus_path=<path>"
        )
        return None 