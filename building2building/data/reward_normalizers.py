"""Per-(building_type, climate_zone) reward normalization constants.

This module wraps :file:`reward_normalizers.yaml`, which stores
``(tau_T, tau_E)`` constants computed from SAC-warmup uniform-random
policy rollouts on the train split (calibration regime: occupancy-based
deadband, ``dT=1.0``, seasonal unoccupied policy — i.e. the ``task_occ_*`` regime).
The random controller was chosen because it is policy-independent —
it bakes in no RBC-specific bias into the normalizers.  The constants
are consumed at training time by
:class:`building2building.simulator.rewards.NormalizedDeadbandReward`
so that

.. math::

    r = -\\Big(\\frac{\\text{temp\\_penalty}}{\\tau_T}
              + w_E \\cdot \\frac{\\text{power\\_penalty}}{\\tau_E}\\Big)

is approximately balanced (mean-1 on each axis) at the median building
of each ``(building_type, climate_zone)`` bucket under the calibration
random policy.

The YAML file is *committed to git* (small) and produced by
:mod:`baselines.compute_random_policy_reward_normalizers`
(``--mode aggregate``).

Numerical floor
---------------
HVAC penalties under the random policy are non-trivial in practice
(random actions still exercise the HVAC), so neither ``tau_T`` nor
``tau_E`` should be near zero on the actual calibration data.  As a
defensive guardrail we clip up to
``max(epsilon_abs, epsilon_rel * median(tau over buckets))`` and set a
``floor_applied_*`` flag on the resulting :class:`RewardNormalizer`.
Tests assert that the floor is dormant on the committed YAML.

Independent of ``baseline_returns.csv``
---------------------------------------
This file is *not* the same thing as
:file:`building2building/scores/baseline_returns.csv`:

* ``baseline_returns.csv`` stores **per-episode** RBC returns and is
  consumed by
  :func:`building2building.scoring.compute_normalized_score` at
  evaluation time.
* ``reward_normalizers.yaml`` stores **per-step mean** penalty
  components and is consumed by
  :class:`~building2building.simulator.rewards.NormalizedDeadbandReward`
  at training time.

Neither file replaces the other.
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from building2building.data.climate_zones import TYPES_WITHOUT_CLIMATE_ZONE

logger = logging.getLogger(__name__)


SUPPORTED_SCHEMA_VERSION: int = 1

DEFAULT_REWARD_NORMALIZERS_PATH: Path = (
    Path(__file__).resolve().parent / "reward_normalizers.yaml"
)

_SEASON_KEYS: frozenset[str] = frozenset({"winter", "summer", "full_year"})


class RewardNormalizersUnavailableError(FileNotFoundError):
    """Raised when ``reward_normalizers.yaml`` cannot be located.

    This is distinct from a generic ``FileNotFoundError`` so callers
    can distinguish "the calibration constants have not been produced
    yet" from arbitrary I/O errors.
    """


@dataclass(frozen=True)
class RewardNormalizer:
    """Resolved ``(tau_T, tau_E)`` pair for one ``(bt, cz)`` bucket.

    Attributes:
        tau_T: Comfort-penalty normalizer (mean ``temp_penalty`` of the
            random policy under the calibration task at the median train
            building of this bucket).
        tau_E: Energy-penalty normalizer (mean ``power_penalty`` of the
            random policy under the calibration task at the median train
            building of this bucket).
        tau_T_iqr: Inter-quartile range of ``tau_T`` across the bucket
            (a tightness measure).
        tau_E_iqr: Inter-quartile range of ``tau_E`` across the bucket.
        n_buildings: Number of train buildings the median was computed
            from.
        floor_applied_T: ``True`` if ``tau_T`` was clipped up to the
            numerical floor (should be ``False`` on the committed
            calibration data).
        floor_applied_E: Same, for ``tau_E``.
    """

    tau_T: float
    tau_E: float
    tau_T_iqr: float
    tau_E_iqr: float
    n_buildings: int
    floor_applied_T: bool = False
    floor_applied_E: bool = False


@dataclass(frozen=True)
class RewardNormalizerSource:
    """Provenance metadata copied from the YAML header."""

    controller: str
    calibration_task: str
    run_period: str
    split: str
    aggregation: str
    b2b_version: str | None = None
    git_sha: str | None = None


@dataclass(frozen=True)
class RewardNormalizerFloor:
    """Numerical-floor configuration parsed from the YAML header."""

    epsilon_abs: float = 1e-6
    epsilon_rel: float = 0.05


@dataclass(frozen=True)
class RewardNormalizerTable:
    """Fully-loaded ``reward_normalizers.yaml`` after floor application.

    Attributes:
        schema_version: Schema-version field from the YAML.
        source: Provenance header.
        floor: Floor configuration that was applied.
        constants: Mapping ``building_type -> {cz_key -> RewardNormalizer}``.
            ``cz_key`` is ``"cz<N>"`` for types with a climate zone and
            ``"cz0"`` for :data:`TYPES_WITHOUT_CLIMATE_ZONE` (single
            bucket per type).
    """

    schema_version: int
    source: RewardNormalizerSource
    floor: RewardNormalizerFloor
    constants: dict[str, dict[str, RewardNormalizer]] = field(default_factory=dict)


def _coerce_float(value: Any, *, key: str) -> float:
    if value is None:
        raise ValueError(f"reward_normalizers.yaml: {key!r} is required")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"reward_normalizers.yaml: {key!r} must be a number, got {value!r}"
        ) from exc


def _coerce_int(value: Any, *, key: str) -> int:
    if value is None:
        raise ValueError(f"reward_normalizers.yaml: {key!r} is required")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"reward_normalizers.yaml: {key!r} must be an integer, got {value!r}"
        ) from exc


def _parse_source(raw: Any) -> RewardNormalizerSource:
    if not isinstance(raw, dict):
        raise TypeError("reward_normalizers.yaml: 'source' must be a mapping")
    required_without_period = {"controller", "calibration_task", "split", "aggregation"}
    has_run_period = "run_period" in raw
    has_run_periods = "run_periods" in raw
    if not has_run_period and not has_run_periods:
        required_without_period.add("run_period")
    missing = required_without_period - set(raw.keys())
    if missing:
        raise ValueError(
            f"reward_normalizers.yaml: 'source' is missing keys {sorted(missing)}"
        )
    if has_run_period:
        run_period_str = str(raw["run_period"])
    elif has_run_periods:
        rp_val = raw["run_periods"]
        run_period_str = (
            ",".join(str(v) for v in rp_val)
            if isinstance(rp_val, list)
            else str(rp_val)
        )
    else:
        run_period_str = ""
    return RewardNormalizerSource(
        controller=str(raw["controller"]),
        calibration_task=str(raw["calibration_task"]),
        run_period=run_period_str,
        split=str(raw["split"]),
        aggregation=str(raw["aggregation"]),
        b2b_version=(
            str(raw["b2b_version"]) if raw.get("b2b_version") is not None else None
        ),
        git_sha=str(raw["git_sha"]) if raw.get("git_sha") is not None else None,
    )


def _parse_floor(raw: Any) -> RewardNormalizerFloor:
    if raw is None:
        return RewardNormalizerFloor()
    if not isinstance(raw, dict):
        raise TypeError("reward_normalizers.yaml: 'floor' must be a mapping")
    return RewardNormalizerFloor(
        epsilon_abs=_coerce_float(
            raw.get("epsilon_abs", 1e-6), key="floor.epsilon_abs"
        ),
        epsilon_rel=_coerce_float(
            raw.get("epsilon_rel", 0.05), key="floor.epsilon_rel"
        ),
    )


def _parse_bucket(raw: Any, *, location: str) -> dict[str, float | int]:
    if not isinstance(raw, dict):
        raise TypeError(
            f"reward_normalizers.yaml: bucket at {location} must be a mapping"
        )
    return {
        "tau_T": _coerce_float(raw.get("tau_T"), key=f"{location}.tau_T"),
        "tau_E": _coerce_float(raw.get("tau_E"), key=f"{location}.tau_E"),
        "tau_T_iqr": _coerce_float(
            raw.get("tau_T_iqr", 0.0), key=f"{location}.tau_T_iqr"
        ),
        "tau_E_iqr": _coerce_float(
            raw.get("tau_E_iqr", 0.0), key=f"{location}.tau_E_iqr"
        ),
        "n_buildings": _coerce_int(
            raw.get("n_buildings", 0), key=f"{location}.n_buildings"
        ),
    }


def _compute_floor(
    raw_buckets: list[tuple[str, str, dict[str, float | int]]],
    floor: RewardNormalizerFloor,
) -> tuple[float, float]:
    """Return ``(tau_T_floor, tau_E_floor)`` from the raw, unclipped buckets.

    The floor is ``max(epsilon_abs, epsilon_rel * median across buckets)``,
    computed on the raw values (before any clipping) so that one
    pathologically small bucket cannot pull the rest down with it.
    """
    tau_T_vals = [float(b["tau_T"]) for _, _, b in raw_buckets if float(b["tau_T"]) > 0]
    tau_E_vals = [float(b["tau_E"]) for _, _, b in raw_buckets if float(b["tau_E"]) > 0]
    median_T = statistics.median(tau_T_vals) if tau_T_vals else 0.0
    median_E = statistics.median(tau_E_vals) if tau_E_vals else 0.0
    tau_T_floor = max(floor.epsilon_abs, floor.epsilon_rel * median_T)
    tau_E_floor = max(floor.epsilon_abs, floor.epsilon_rel * median_E)
    return tau_T_floor, tau_E_floor


def _apply_floor_one(
    raw: dict[str, float | int],
    *,
    tau_T_floor: float,
    tau_E_floor: float,
    bucket_label: str,
) -> RewardNormalizer:
    raw_T = float(raw["tau_T"])
    raw_E = float(raw["tau_E"])
    floor_applied_T = raw_T < tau_T_floor
    floor_applied_E = raw_E < tau_E_floor
    if floor_applied_T:
        logger.warning(
            "reward_normalizers: clipping tau_T for bucket %s up from %.6g to floor %.6g",
            bucket_label,
            raw_T,
            tau_T_floor,
        )
    if floor_applied_E:
        logger.warning(
            "reward_normalizers: clipping tau_E for bucket %s up from %.6g to floor %.6g",
            bucket_label,
            raw_E,
            tau_E_floor,
        )
    return RewardNormalizer(
        tau_T=max(raw_T, tau_T_floor),
        tau_E=max(raw_E, tau_E_floor),
        tau_T_iqr=float(raw["tau_T_iqr"]),
        tau_E_iqr=float(raw["tau_E_iqr"]),
        n_buildings=int(raw["n_buildings"]),
        floor_applied_T=floor_applied_T,
        floor_applied_E=floor_applied_E,
    )


def _is_seasonal(constants_raw: dict) -> bool:
    """Return True if the constants dict uses a season → bt → cz schema."""
    return bool(constants_raw) and set(constants_raw).issubset(_SEASON_KEYS)


def parse_reward_normalizers(
    raw: dict[str, Any],
    *,
    run_period: str | None = None,
) -> RewardNormalizerTable:
    """Parse a deserialized YAML mapping into a typed table.

    Centralizing the parsing here lets unit tests build synthetic
    inputs (good and bad) without writing temporary files.

    Args:
        raw: Deserialized YAML top-level mapping.
        run_period: Season to select from seasonal YAMLs (``"winter"``,
            ``"summer"``, ``"full_year"``).  When ``None`` and the YAML
            is seasonal, ``"full_year"`` is used.  Ignored for flat
            (non-seasonal) YAMLs.
    """
    if not isinstance(raw, dict):
        raise TypeError("reward_normalizers.yaml: top-level must be a mapping")

    schema_version = _coerce_int(raw.get("schema_version"), key="schema_version")
    if schema_version != SUPPORTED_SCHEMA_VERSION:
        raise ValueError(
            f"reward_normalizers.yaml: unsupported schema_version {schema_version}; "
            f"this build of building2building expects {SUPPORTED_SCHEMA_VERSION}"
        )

    source = _parse_source(raw.get("source"))
    floor = _parse_floor(raw.get("floor"))

    constants_raw = raw.get("constants")
    if not isinstance(constants_raw, dict):
        raise TypeError("reward_normalizers.yaml: 'constants' must be a mapping")

    if _is_seasonal(constants_raw):
        season = run_period if run_period is not None else "full_year"
        if season not in constants_raw:
            raise KeyError(
                f"reward_normalizers.yaml: seasonal YAML has no section for "
                f"run_period={season!r}; available: {sorted(constants_raw)}"
            )
        constants_raw = constants_raw[season]
        if not isinstance(constants_raw, dict):
            raise TypeError(
                f"reward_normalizers.yaml: constants[{season!r}] must be a mapping"
            )

    flat: list[tuple[str, str, dict[str, float | int]]] = []
    for bt, by_cz in constants_raw.items():
        if not isinstance(by_cz, dict):
            raise TypeError(
                f"reward_normalizers.yaml: constants[{bt!r}] must be a mapping"
            )
        for cz_key, bucket in by_cz.items():
            if not isinstance(cz_key, str):
                raise TypeError(
                    f"reward_normalizers.yaml: cz key under {bt!r} must be a string, "
                    f"got {cz_key!r}"
                )
            location = f"constants.{bt}.{cz_key}"
            flat.append((str(bt), cz_key, _parse_bucket(bucket, location=location)))

    tau_T_floor, tau_E_floor = _compute_floor(flat, floor)

    constants: dict[str, dict[str, RewardNormalizer]] = {}
    for bt, cz_key, bucket in flat:
        constants.setdefault(bt, {})[cz_key] = _apply_floor_one(
            bucket,
            tau_T_floor=tau_T_floor,
            tau_E_floor=tau_E_floor,
            bucket_label=f"{bt}/{cz_key}",
        )

    return RewardNormalizerTable(
        schema_version=schema_version,
        source=source,
        floor=floor,
        constants=constants,
    )


def load_reward_normalizers(
    path: Path | None = None,
    *,
    run_period: str | None = None,
) -> RewardNormalizerTable:
    """Load and validate ``reward_normalizers.yaml``.

    Args:
        path: Override the default YAML location.  When ``None``
            (default), reads
            :data:`DEFAULT_REWARD_NORMALIZERS_PATH`.
        run_period: Season slice for seasonal YAMLs (``"winter"``,
            ``"summer"``, ``"full_year"``).  Ignored for flat YAMLs.

    Raises:
        RewardNormalizersUnavailableError: If the YAML file does not
            exist.  This fires when
            :class:`~building2building.types.NormalizedDeadbandRewardConfig`
            tries to resolve ``(tau_T, tau_E)`` constants.
        ValueError, TypeError: On schema-validation failures.
    """
    target = path if path is not None else DEFAULT_REWARD_NORMALIZERS_PATH
    if not target.exists():
        raise RewardNormalizersUnavailableError(
            f"Reward-normalizer constants not found at {target!s}. "
            "Generate them via "
            "`python -m analysis.task_study.compute_reward_normalizers --mode aggregate` "
            "and commit the resulting YAML to the repo."
        )
    raw = yaml.safe_load(target.read_text())
    return parse_reward_normalizers(raw, run_period=run_period)


@lru_cache(maxsize=12)
def _cached_load(path_str: str | None, run_period: str | None) -> RewardNormalizerTable:
    """Cached loader keyed by (path, run_period) so the LRU cache hashes cleanly."""
    return load_reward_normalizers(
        Path(path_str) if path_str else None,
        run_period=run_period,
    )


def get_reward_normalizers_cached(
    path: Path | None = None,
    *,
    run_period: str | None = None,
) -> RewardNormalizerTable:
    """Same as :func:`load_reward_normalizers` but cached.

    Reading ~6 building-types' worth of YAML on every env reset would
    be wasteful in long PPO runs.  This helper memoizes the parsed
    table per (resolved path, run_period).
    """
    path_str = str(path.resolve()) if path is not None else None
    return _cached_load(path_str, run_period)


def clear_reward_normalizers_cache() -> None:
    """Clear the path-keyed loader cache.

    Useful in tests that want to swap in a synthetic YAML between
    cases; production code should not need this.
    """
    _cached_load.cache_clear()


def _cz_key_for(building_type: str, building_id: str) -> str:
    """Return the YAML key for the ``(building_type, building_id)`` bucket.

    ``"cz0"`` for :data:`TYPES_WITHOUT_CLIMATE_ZONE` (single-bucket
    per type, mirroring the
    ``unitary_hvac_singlefamilyhouse_cz0.yaml`` controller filename),
    and ``"cz<N>"`` otherwise.
    """
    if building_type in TYPES_WITHOUT_CLIMATE_ZONE:
        return "cz0"
    from building2building.api import get_climate_zone

    cz = get_climate_zone(building_type, building_id)
    return f"cz{cz}"


def resolve_reward_normalizer(
    building_type: str,
    building_id: str,
    *,
    run_period: str | None = None,
    path: Path | None = None,
) -> RewardNormalizer:
    """Look up the ``(tau_T, tau_E)`` for a given building.

    Single-bucket types (currently :data:`TYPES_WITHOUT_CLIMATE_ZONE`,
    which contains ``SingleFamilyHouse``) ignore *building_id* and
    return the single ``cz0`` entry.  Other types dispatch on the
    metadata ``climate_zone`` column.

    Args:
        building_type: e.g. ``"OfficeMedium"``.
        building_id: Dataset building identifier (used only for types
            with a climate zone).
        run_period: Season slice for seasonal YAMLs (``"winter"``,
            ``"summer"``, ``"full_year"``).  Passed through to
            :func:`get_reward_normalizers_cached`.
        path: Optional override for the YAML location (testing).

    Raises:
        RewardNormalizersUnavailableError: If the YAML is missing.
        KeyError: If the building type or climate zone is not present
            in the YAML.
    """
    table = get_reward_normalizers_cached(path, run_period=run_period)
    by_cz = table.constants.get(building_type)
    if by_cz is None:
        raise KeyError(
            f"reward_normalizers.yaml has no entry for building_type "
            f"{building_type!r}; available: {sorted(table.constants.keys())}"
        )
    cz_key = _cz_key_for(building_type, building_id)
    bucket = by_cz.get(cz_key)
    if bucket is None:
        raise KeyError(
            f"reward_normalizers.yaml has no entry for "
            f"{building_type}/{cz_key} (building_id={building_id!r}); "
            f"available cz keys for this type: {sorted(by_cz.keys())}"
        )
    return bucket


__all__ = [
    "DEFAULT_REWARD_NORMALIZERS_PATH",
    "SUPPORTED_SCHEMA_VERSION",
    "RewardNormalizer",
    "RewardNormalizerFloor",
    "RewardNormalizerSource",
    "RewardNormalizerTable",
    "RewardNormalizersUnavailableError",
    "clear_reward_normalizers_cache",
    "get_reward_normalizers_cached",
    "load_reward_normalizers",
    "parse_reward_normalizers",
    "resolve_reward_normalizer",
]
