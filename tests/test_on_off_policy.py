from __future__ import annotations

import pytest

pytest.skip(
    "Deprecated: this test targeted legacy pipeline helpers and a BuildingConfig field "
    "(hvac_control_mode) that are no longer present in this repo.",
    allow_module_level=True,
)


