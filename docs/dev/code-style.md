# Code Style

Building2Building follows strict coding conventions to keep the codebase
consistent and to catch bugs early through static analysis.

## Formatter

All Python code is formatted with **Black**:

- Line length: 88
- Target version: Python 3.10+

Run the formatter before committing:

```bash
black .
```

## Type Hints

Type hints are required on **all public functions and methods**.

```python
def compute_reward(
    temperatures: list[float],
    setpoint: float,
    energy_kwh: float,
) -> float:
    ...
```

Use imports from `typing` for complex types (`Optional`, `Literal`, `Protocol`,
etc.).

## Paths

Always use `pathlib.Path`—never bare `str`—for filesystem paths:

```python
from pathlib import Path

model_dir: Path = DataPaths.models / "checkpoint.pt"
```

## Data Structures

Prefer `@dataclass` (or `@dataclass(frozen=True)`) over ad-hoc dictionaries
for structured data. Provide `load_json` / `save_json` class methods when
the dataclass needs to be serialised.

## Literal Types

When only a finite set of string values is valid, use `Literal` instead of
plain `str`:

```python
from typing import Literal

AggMethod = Literal["mean", "sum", "max"]
```

## Input Validation

Validate inputs early and fail fast. Do not silently accept unexpected types
or values:

```python
def set_temperature(value: float) -> None:
    if not (15.0 <= value <= 30.0):
        raise ValueError(f"Temperature {value} out of bounds [15, 30]")
    ...
```

## Whitespace

- No trailing whitespace at the end of lines.
- No lines consisting only of whitespace.
