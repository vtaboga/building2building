from __future__ import annotations

import sys
from pathlib import Path


# Ensure repository root is importable in tests.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# Setup EnergyPlus path for pyenergyplus.api
from b2b.env import setup_energyplus_path

setup_energyplus_path()
