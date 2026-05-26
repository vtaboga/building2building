"""Configuration models for datasets, benchmarks, and environment building.

All configs are frozen dataclasses with ``from_dict`` class methods that
validate and normalise raw dictionaries (e.g. from YAML / JSON files).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from building2building.data.download import ALL_BUILDING_TYPES, BuildingType
from building2building.types import (
    NormalizedDeadbandRewardConfig,
    RewardConfig,
    TaskConfig,
    reward_config_from_dict,
)

SplitName = Literal["train", "test", "test_small"]
SelectionMode = Literal[
    "split_index",
    "building_id",
]
BenchmarkMode = Literal["single_type", "multi_type"]


def _require_mapping(name: str, value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a mapping, got {type(value).__name__}")
    return value


@dataclass(frozen=True)
class DatasetSelectionConfig:
    """Specifies which building to select from the unified dataset.

    Attributes:
        building_type: Building archetype (e.g. ``"OfficeSmall"``).
        split: Train / test split (``None`` to ignore splits).
        mode: Selection strategy: ``"split_index"`` or ``"building_id"``.
        split_index: Index when ``mode="split_index"``.
        building_id: Explicit building ID string when ``mode="building_id"``.
    """

    building_type: BuildingType
    split: SplitName | None = "train"
    mode: SelectionMode = "split_index"
    split_index: int = 0
    building_id: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatasetSelectionConfig":
        """Parse and validate from a raw dictionary.

        Args:
            data: Mapping with dataset-selection parameters.

        Returns:
            A validated ``DatasetSelectionConfig``.

        Raises:
            ValueError: On invalid split, mode, or building_type values.
        """
        raw = _require_mapping("dataset_selection", data)
        building_type_raw = raw.get("building_type")
        if building_type_raw is None:
            raise ValueError("dataset_selection.building_type is required")
        building_type = str(building_type_raw)
        if building_type not in ALL_BUILDING_TYPES:
            raise ValueError(
                f"dataset_selection.building_type must be one of "
                f"{ALL_BUILDING_TYPES}, got {building_type!r}"
            )

        split_raw = raw.get("split", "train")
        split: SplitName | None
        if split_raw is None:
            split = None
        else:
            split_norm = str(split_raw).strip().lower()
            if split_norm not in {"train", "test", "test_small"}:
                raise ValueError(
                    "dataset_selection.split must be one of "
                    "{'train', 'test', 'test_small'} or null"
                )
            split = split_norm  # type: ignore[assignment]

        mode = str(raw.get("mode", "split_index")).strip().lower()
        if mode not in {"split_index", "building_id"}:
            raise ValueError(
                "dataset_selection.mode must be one of "
                "{'split_index', 'building_id'}"
            )

        building_id_raw = raw.get("building_id")
        building_id = str(building_id_raw) if building_id_raw is not None else None
        split_index = int(raw.get("split_index", 0))

        return cls(
            building_type=building_type,  # type: ignore[arg-type]
            split=split,
            mode=mode,  # type: ignore[arg-type]
            split_index=split_index,
            building_id=building_id,
        )


@dataclass(frozen=True)
class ActuatorAccessConfig:
    """Controls which actuators the agent is allowed to manipulate.

    Attributes:
        include_zone_heating_setpoints: If ``True``, zone-level heating
            setpoint actuators are included in the action space.
    """

    include_zone_heating_setpoints: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ActuatorAccessConfig":
        """Parse from a raw dictionary (``None`` yields defaults).

        Args:
            data: Optional mapping with actuator-access flags.

        Returns:
            A validated ``ActuatorAccessConfig``.
        """
        raw = _require_mapping("actuator_access", data or {})
        return cls(
            include_zone_heating_setpoints=bool(
                raw.get("include_zone_heating_setpoints", True)
            )
        )


@dataclass(frozen=True)
class BenchmarkSelectionConfig:
    """How buildings are selected within a benchmark split.

    Attributes:
        mode: Selection strategy (``"random"``, ``"indices"``, or
            ``"search_config"``).
        n: Number of buildings to select when ``mode="random"``.
        seed: RNG seed for reproducibility.
        replace: Sample with replacement when ``mode="random"``.
        indices: Explicit building indices when ``mode="indices"``.
        queries: Search-config filter dicts when
            ``mode="search_config"``.
    """

    mode: Literal["random", "indices", "search_config"] = "random"
    n: int = 1
    seed: int | None = None
    replace: bool = False
    indices: list[int] = field(default_factory=list)
    queries: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BenchmarkSelectionConfig":
        """Parse and validate from a raw dictionary.

        Args:
            data: Optional mapping with selection parameters.

        Returns:
            A validated ``BenchmarkSelectionConfig``.

        Raises:
            ValueError: On invalid mode or negative *n*.
            TypeError: On wrong-typed ``indices`` or ``queries``.
        """
        raw = _require_mapping("benchmark selection", data or {})
        mode = str(raw.get("mode", "random")).strip().lower()
        if mode not in {"random", "indices", "search_config"}:
            raise ValueError(
                "selection.mode must be one of {'random', 'indices', 'search_config'}"
            )
        indices = raw.get("indices", [])
        if not isinstance(indices, list) or not all(
            isinstance(x, int) for x in indices
        ):
            raise TypeError("selection.indices must be list[int]")
        queries = raw.get("queries", [])
        if not isinstance(queries, list) or not all(
            isinstance(x, dict) for x in queries
        ):
            raise TypeError("selection.queries must be list[dict]")
        seed_raw = raw.get("seed")
        seed = int(seed_raw) if seed_raw is not None else None
        n = int(raw.get("n", 1))
        if n < 0:
            raise ValueError("selection.n must be >= 0")
        return cls(
            mode=mode,  # type: ignore[arg-type]
            n=n,
            seed=seed,
            replace=bool(raw.get("replace", False)),
            indices=[int(i) for i in indices],
            queries=[dict(q) for q in queries],
        )


@dataclass(frozen=True)
class BenchmarkSideConfig:
    """Configuration for one side (train or test) of a benchmark.

    Attributes:
        selection: Building selection strategy.
        task: Task specification (run period, target temperatures, …).
        reward: Reward function configuration.
        actuator_access: Which actuators the agent may control.
        raw: Original un-parsed config dict, kept for round-tripping.
    """

    selection: BenchmarkSelectionConfig
    task: TaskConfig
    reward: RewardConfig
    actuator_access: ActuatorAccessConfig
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BenchmarkSideConfig":
        """Parse a benchmark side from a raw dictionary.

        Args:
            data: Optional mapping with ``"selection"`` and ``"config"``
                sub-sections.

        Returns:
            A validated ``BenchmarkSideConfig``.
        """
        raw = _require_mapping("benchmark side", data or {})
        selection = BenchmarkSelectionConfig.from_dict(raw.get("selection", {}))
        config = _require_mapping("benchmark side config", raw.get("config", {}))
        task_raw = _require_mapping("benchmark side task", config.get("task", {}))
        reward_raw = _require_mapping("benchmark side reward", config.get("reward", {}))
        task = TaskConfig.from_dict(task_raw)
        reward = reward_config_from_dict(reward_raw)
        actuator_access = ActuatorAccessConfig.from_dict(
            config.get("actuator_access", {})
        )
        return cls(
            selection=selection,
            task=task,
            reward=reward,
            actuator_access=actuator_access,
            raw=dict(config),
        )


def _default_benchmark_side() -> BenchmarkSideConfig:
    return BenchmarkSideConfig.from_dict({})


@dataclass(frozen=True)
class SingleTypeBenchmarkConfig:
    """Benchmark that trains and tests on a single building archetype.

    Attributes:
        mode: Always ``"single_type"``.
        building_type: The reference-building archetype to use.
        train: Training-side configuration.
        test: Test-side configuration.
    """

    mode: Literal["single_type"] = "single_type"
    building_type: BuildingType = "OfficeSmall"
    train: BenchmarkSideConfig = field(default_factory=_default_benchmark_side)
    test: BenchmarkSideConfig = field(default_factory=_default_benchmark_side)


@dataclass(frozen=True)
class MultiTypeBenchmarkConfig:
    """Benchmark spanning multiple building archetypes.

    Attributes:
        mode: Always ``"multi_type"``.
        train_types: Building archetypes used for training.
        test_types: Building archetypes used for testing.
        train: Training-side configuration.
        test: Test-side configuration.
    """

    mode: Literal["multi_type"] = "multi_type"
    train_types: list[BuildingType] = field(default_factory=list)
    test_types: list[BuildingType] = field(default_factory=list)
    train: BenchmarkSideConfig = field(default_factory=_default_benchmark_side)
    test: BenchmarkSideConfig = field(default_factory=_default_benchmark_side)


BenchmarkConfig = SingleTypeBenchmarkConfig | MultiTypeBenchmarkConfig


@dataclass(frozen=True)
class EnvBuildConfig:
    """Complete specification for building a Gymnasium environment.

    Attributes:
        dataset_selection: Which building(s) to load.
        task: Task specification (run period, target temperatures, …).
        reward: Reward function configuration.
        actuator_access: Actuator access restrictions.
        env_max_steps: Optional hard cap on episode length. When
            ``None``, the run period's expected step count is used.
    """

    dataset_selection: DatasetSelectionConfig
    task: TaskConfig
    reward: RewardConfig
    actuator_access: ActuatorAccessConfig = field(default_factory=ActuatorAccessConfig)
    env_max_steps: int | None = None
    expose_heating_only_zones: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EnvBuildConfig":
        """Parse and validate from a raw dictionary.

        Args:
            data: Mapping with top-level keys ``"dataset_selection"``,
                ``"task"``, ``"reward"``, ``"actuator_access"``, and
                optionally ``"env_max_steps"``.

        Returns:
            A validated ``EnvBuildConfig``.
        """
        raw = _require_mapping("env build config", data)
        dataset_selection = DatasetSelectionConfig.from_dict(
            _require_mapping("dataset_selection", raw.get("dataset_selection", {}))
        )
        task = TaskConfig.from_dict(_require_mapping("task", raw.get("task", {})))
        reward = reward_config_from_dict(
            _require_mapping("reward", raw.get("reward", {})),
        )
        actuator_access = ActuatorAccessConfig.from_dict(raw.get("actuator_access", {}))
        max_steps_raw = raw.get("env_max_steps")
        env_max_steps = int(max_steps_raw) if max_steps_raw is not None else None
        expose_heating_only_zones = bool(raw.get("expose_heating_only_zones", True))
        return cls(
            dataset_selection=dataset_selection,
            task=task,
            reward=reward,
            actuator_access=actuator_access,
            env_max_steps=env_max_steps,
            expose_heating_only_zones=expose_heating_only_zones,
        )


def parse_benchmark_config(data: dict[str, Any]) -> BenchmarkConfig:
    """Parse a benchmark configuration from a raw dictionary.

    The ``"mode"`` key selects the concrete config type:

    * ``"single_type"`` -> :class:`SingleTypeBenchmarkConfig`
    * ``"multi_type"``  -> :class:`MultiTypeBenchmarkConfig`

    Args:
        data: Top-level benchmark configuration mapping.

    Returns:
        A ``SingleTypeBenchmarkConfig`` or ``MultiTypeBenchmarkConfig``.

    Raises:
        ValueError: If ``"mode"`` is not recognised or required fields
            are missing.
        TypeError: If building-type lists have wrong element types.
    """
    raw = _require_mapping("benchmark_interface", data)
    mode = str(raw.get("mode", "single_type")).strip().lower()
    train = BenchmarkSideConfig.from_dict(raw.get("train", {}))
    test = BenchmarkSideConfig.from_dict(raw.get("test", {}))
    if mode == "single_type":
        bt = str(raw.get("building_type", "OfficeSmall"))
        return SingleTypeBenchmarkConfig(
            building_type=bt,  # type: ignore[arg-type]
            train=train,
            test=test,
        )
    if mode == "multi_type":
        train_raw = _require_mapping("benchmark_interface.train", raw.get("train", {}))
        test_raw = _require_mapping("benchmark_interface.test", raw.get("test", {}))
        train_types_raw = train_raw.get("types", [])
        test_types_raw = test_raw.get("types", [])
        if not isinstance(train_types_raw, list) or not all(
            isinstance(x, str) for x in train_types_raw
        ):
            raise TypeError("benchmark_interface.train.types must be list[str]")
        if not isinstance(test_types_raw, list) or not all(
            isinstance(x, str) for x in test_types_raw
        ):
            raise TypeError("benchmark_interface.test.types must be list[str]")
        if len(train_types_raw) < 1 or len(test_types_raw) < 1:
            raise ValueError("multi_type requires at least one train and one test type")
        return MultiTypeBenchmarkConfig(
            train_types=train_types_raw,  # type: ignore[arg-type]
            test_types=test_types_raw,  # type: ignore[arg-type]
            train=train,
            test=test,
        )
    raise ValueError(
        "benchmark_interface.mode must be one of {'single_type','multi_type'}"
    )


def reward_to_dict(reward: RewardConfig) -> dict[str, Any]:
    """Serialise a reward config back to a plain dictionary.

    Args:
        reward: Any ``RewardConfig`` variant.

    Returns:
        A dictionary suitable for JSON / YAML serialisation,
        including a ``"reward_type"`` discriminator key.

    Raises:
        TypeError: If *reward* is not a known ``RewardConfig`` type.
    """
    if isinstance(reward, NormalizedDeadbandRewardConfig):
        d: dict[str, Any] = {
            "reward_type": "NormalizedDeadbandRewardConfig",
            "energy_weight": reward.energy_weight,
            "dT": reward.dT,
        }
        if reward.tau_T is not None:
            d["tau_T"] = reward.tau_T
        if reward.tau_E is not None:
            d["tau_E"] = reward.tau_E
        return d
    raise TypeError(f"Unsupported reward type: {type(reward).__name__}")
