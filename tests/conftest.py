from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import pytest


# Ensure repository root is importable in tests.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# Setup EnergyPlus path for pyenergyplus.api when runtime deps are available.
try:
    from b2b.env import setup_energyplus_path

    setup_energyplus_path()
except ModuleNotFoundError:
    pass


def pytest_collection_modifyitems(items: List[pytest.Item]) -> None:
    for item in items:
        if "quick" in item.keywords or "long" in item.keywords:
            continue
        item.add_marker(pytest.mark.quick)
