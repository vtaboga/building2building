import sys
import os
from pathlib import Path

def find_energyplus_path():
    """
    Find the EnergyPlus installation path by checking common locations
    and environment variables.
    """
    # Check environment variable first
    if 'ENERGYPLUS_PATH' in os.environ:
        return os.environ['ENERGYPLUS_PATH']
    
    # Common installation paths
    possible_paths = [
        # Container/Linux path
        '/usr/local/EnergyPlus-24-1-0',
        # Default Linux path
        '/usr/local/energy-plus-24.1.0',
        # Default macOS path
        '/Applications/EnergyPlus-24-1-0',
        # Default Windows path
        'C:\\EnergyPlus-24-1-0',
        # Local development path
        str(Path.home() / 'EnergyPlus-24-1-0'),
    ]
    
    # Check each path
    for path in possible_paths:
        if os.path.exists(path):
            return path
            
    raise RuntimeError(
        "EnergyPlus installation not found. Please either:\n"
        "1. Set ENERGYPLUS_PATH environment variable\n"
        "2. Install EnergyPlus in one of the standard locations:\n"
        f"   {possible_paths}"
    )

def setup_energyplus_path():
    """
    Setup the path to EnergyPlus Python API.
    This should be called before any EnergyPlus-related imports.
    """
    energyplus_path = find_energyplus_path()
    
    if energyplus_path not in sys.path:
        sys.path.append(energyplus_path)
        print(f"Added EnergyPlus path: {energyplus_path}")

# Call it when the module is imported
setup_energyplus_path()

