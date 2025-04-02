import sys
import os

def setup_energyplus_path():
    """
    Setup the path to EnergyPlus Python API.
    This should be called before any EnergyPlus-related imports.
    """
    energyplus_path = "/usr/local/EnergyPlus-24-1-0"
    
    if not os.path.exists(energyplus_path):
        raise RuntimeError(f"EnergyPlus installation not found at {energyplus_path}")
    
    if energyplus_path not in sys.path:
        sys.path.append(energyplus_path)

# Call it when the module is imported
setup_energyplus_path()

