"""Structured environment representation (Section 3.4 of the B2B paper).

Defines a morphological universe of node types with local observation and
action spaces, and provides per-environment morphology graphs with
``split`` / ``join`` operations for mapping between the flat global
spaces and per-node local spaces.

The morphological universe is fixed across all B2B environments::

    Node type              | Local obs                          | Local action
    -----------------------|------------------------------------|---------------------------
    weather                | outdoor_temp, outdoor_humidity     | (none)
    calendar               | time_of_day, day_of_week, day_year | (none)
    energy                 | hvac_electricity, hvac_gas         | (none)
    unitary_zone           | zone_temp                          | fan_flow, sat_setpoint
    vav_zone               | zone_temp                          | damper, heating_sp, cooling_sp
    vav_zone_no_cooling    | zone_temp                          | damper, heating_sp
    vav_supply             | (none)                             | sat_setpoint
    heating_zone           | zone_temp                          | heating_sp
    uncontrolled_zone      | zone_temp                          | (none)

Usage::

    env = building2building.make_env("OfficeSmall", task="task_const_e0")
    morphology = env.metadata["morphology"]
    obs, _ = env.reset()
    local_obs = morphology.split_observation(obs)
    # local_obs is a dict[str, ndarray] keyed by node_id
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Sequence

import numpy as np
from gymnasium.spaces import Box

from building2building.pipeline.actuators import (
    FLOW_FRACTION,
    HEATING_ONLY_SP_C,
    OA_MASS_FLOW_KGS,
    VAV_COOLING_SP_C,
    VAV_HEATING_SP_C,
    VAV_SUPPLY_TEMP_C,
)


def _empty_array() -> np.ndarray:
    """Default factory for the per-node `attributes` and per-morphology
    `common_attributes` arrays. ndarrays aren't hashable, so the
    corresponding fields are also marked `field(compare=False)` to keep
    the frozen dataclass's auto-generated `__eq__`/`__hash__` working —
    node identity stays determined by the discrete fields (node_id,
    node_type, indices) rather than attribute values.
    """
    return np.empty(0, dtype=np.float32)


if TYPE_CHECKING:
    from building2building.geometry import ZoneGeometry
    from building2building.types import Equipment

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Node types -- the fixed morphological universe shared across all B2B envs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NodeType:
    """A node type in the morphological universe.

    Each type carries its local observation and action spaces (with
    physical bounds) so that type-specific encoders / decoders have stable
    input sizes and policies can rescale outputs without depending on
    per-environment metadata.

    Types may also declare a local *attribute* space — static per-node
    metadata that doesn't change within an episode (e.g. zone geometry
    for zone-typed nodes). Per-node attribute values live on
    :class:`MorphologyNode`; per-morphology common attributes live on
    :class:`Morphology`.

    The bounds are stored as tuples of floats to keep the dataclass
    frozen and hashable.  Use the :attr:`local_observation_space`,
    :attr:`local_action_space`, and :attr:`local_attribute_space`
    properties to obtain ``gymnasium.spaces.Box`` objects.
    """

    name: str
    _obs_low: tuple[float, ...] = ()
    _obs_high: tuple[float, ...] = ()
    _act_low: tuple[float, ...] = ()
    _act_high: tuple[float, ...] = ()
    _attr_low: tuple[float, ...] = ()
    _attr_high: tuple[float, ...] = ()

    @property
    def observation_dim(self) -> int:
        return len(self._obs_low)

    @property
    def action_dim(self) -> int:
        return len(self._act_low)

    @property
    def attribute_dim(self) -> int:
        return len(self._attr_low)

    @property
    def local_observation_space(self) -> Box:
        """Gymnasium Box for the local observation space."""
        return Box(
            low=np.array(self._obs_low, dtype=np.float32),
            high=np.array(self._obs_high, dtype=np.float32),
        )

    @property
    def local_action_space(self) -> Box:
        """Gymnasium Box for the local action space."""
        return Box(
            low=np.array(self._act_low, dtype=np.float32),
            high=np.array(self._act_high, dtype=np.float32),
        )

    @property
    def local_attribute_space(self) -> Box:
        """Gymnasium Box for the local attribute space."""
        return Box(
            low=np.array(self._attr_low, dtype=np.float32),
            high=np.array(self._attr_high, dtype=np.float32),
        )


# fmt: off
# Action bounds for the controllable schedules are imported from
# building2building.pipeline.actuators (the Bounds constants), so the
# morphological universe and the pipeline share a single source of truth for
# the static setpoint ranges and cannot drift apart.
# Exception: UNITARY_ZONE's fan-flow / supply-air-temp bounds are read
# per-building from design data in the pipeline (design fan flow, max SAT), so
# here they keep the static fallback envelope (DEFAULT_FAN_MAX_KGS /
# DEFAULT_SAT_MAX_C). Observation bounds still mirror
# building2building.simulator.observation_spaces by hand — not yet unified.
# Zone NodeTypes share the same 9-d attribute schema (ZONE_ATTRIBUTE_NAMES
# in building2building.geometry). All values are dimensionless and in [0,1]
# by construction of extract_zone_geometry(), so bounds = (0,)*9 / (1,)*9.
_ZONE_ATTR_LOW: tuple[float, ...] = (0.0,) * 9
_ZONE_ATTR_HIGH: tuple[float, ...] = (1.0,) * 9


WEATHER = NodeType("weather",
    _obs_low=(-30.0, 0.0),   _obs_high=(50.0, 100.0))      # outdoor_temp (°C), outdoor_humidity (%)
CALENDAR = NodeType("calendar",
    _obs_low=(1.0, 1.0, 1.0), _obs_high=(25.0, 7.0, 366.0)) # time_of_day, day_of_week, day_of_year
ENERGY = NodeType("energy",
    _obs_low=(0.0, 0.0),      _obs_high=(200.0, 200.0))      # electricity, gas (Wh/m²/timestep)
UNITARY_ZONE = NodeType("unitary_zone",
    _obs_low=(10.0,),  _obs_high=(45.0,),                     # zone_temp (°C)
    _act_low=(0.0, 5.0), _act_high=(15.0, 60.0),              # fan_flow (kg/s), supply_air_temp (°C)
    _attr_low=_ZONE_ATTR_LOW, _attr_high=_ZONE_ATTR_HIGH)     # zone geometry (see building2building.geometry)
VAV_ZONE = NodeType("vav_zone",                               # flow_frac, htg_sp (°C), clg_sp (°C)
    _obs_low=(10.0,),  _obs_high=(45.0,),                     # zone_temp (°C)
    _act_low=(FLOW_FRACTION.low, VAV_HEATING_SP_C.low, VAV_COOLING_SP_C.low),
    _act_high=(FLOW_FRACTION.high, VAV_HEATING_SP_C.high, VAV_COOLING_SP_C.high),
    _attr_low=_ZONE_ATTR_LOW, _attr_high=_ZONE_ATTR_HIGH)
VAV_ZONE_NO_COOLING = NodeType("vav_zone_no_cooling",         # flow_frac, htg_sp (°C) — clg_sp fixed
    _obs_low=(10.0,),  _obs_high=(45.0,),                     # zone_temp (°C)
    _act_low=(FLOW_FRACTION.low, VAV_HEATING_SP_C.low),
    _act_high=(FLOW_FRACTION.high, VAV_HEATING_SP_C.high),
    _attr_low=_ZONE_ATTR_LOW, _attr_high=_ZONE_ATTR_HIGH)
VAV_SUPPLY = NodeType("vav_supply",                           # supply_air_temp (°C), oa_mass_flow (kg/s)
    _act_low=(VAV_SUPPLY_TEMP_C.low, OA_MASS_FLOW_KGS.low),
    _act_high=(VAV_SUPPLY_TEMP_C.high, OA_MASS_FLOW_KGS.high))
HEATING_ZONE = NodeType("heating_zone",                       # htg_sp (°C)
    _obs_low=(10.0,),  _obs_high=(45.0,),                     # zone_temp (°C)
    _act_low=(HEATING_ONLY_SP_C.low,),  _act_high=(HEATING_ONLY_SP_C.high,),
    _attr_low=_ZONE_ATTR_LOW, _attr_high=_ZONE_ATTR_HIGH)
UNCONTROLLED_ZONE = NodeType("uncontrolled_zone",
    _obs_low=(10.0,),  _obs_high=(45.0,),                     # zone_temp (°C)
    _attr_low=_ZONE_ATTR_LOW, _attr_high=_ZONE_ATTR_HIGH)
# fmt: on

ALL_NODE_TYPES: tuple[NodeType, ...] = (
    WEATHER,
    CALENDAR,
    ENERGY,
    UNITARY_ZONE,
    VAV_ZONE,
    VAV_ZONE_NO_COOLING,
    VAV_SUPPLY,
    HEATING_ZONE,
    UNCONTROLLED_ZONE,
)

NODE_TYPE_BY_NAME: dict[str, NodeType] = {nt.name: nt for nt in ALL_NODE_TYPES}


# ---------------------------------------------------------------------------
# Common (morphology-wide) attribute schema.
#
# Per-node attributes live on :class:`MorphologyNode` and are typed by the
# node's :class:`NodeType`. Morphology-wide common attributes — things
# like total floor area, climate-zone index, or other building-level
# static features — share a single schema declared here. Currently empty;
# expand the two tuples (and update :attr:`Morphology.common_attributes`
# producers) when adding building-level attributes.
# ---------------------------------------------------------------------------

COMMON_ATTRIBUTE_LOW: tuple[float, ...] = ()
COMMON_ATTRIBUTE_HIGH: tuple[float, ...] = ()


def common_attribute_space() -> Box:
    """Gymnasium Box for the morphology-wide common attribute schema."""
    return Box(
        low=np.array(COMMON_ATTRIBUTE_LOW, dtype=np.float32),
        high=np.array(COMMON_ATTRIBUTE_HIGH, dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# Morphology graph
# ---------------------------------------------------------------------------

EdgeType = Literal["hvac_system", "thermal_adjacency", "controls"]


@dataclass(frozen=True)
class MorphologyNode:
    """A single node in a morphology graph.

    Attributes:
        node_id: Unique identifier (e.g. ``"weather"``, ``"zone:THERMAL ZONE 1"``).
        node_type: The node's type from the morphological universe.
        obs_indices: Indices into the flat global observation vector that
            belong to this node's local observation.
        action_indices: Indices into the flat global action vector that
            belong to this node's local action.
        attributes: Static per-node attribute values, conforming to the
            node type's :attr:`local_attribute_space`. Defaults to an
            empty array for types that declare no attributes. Excluded
            from ``__eq__``/``__hash__`` (see :func:`_empty_array`).
    """

    node_id: str
    node_type: NodeType
    obs_indices: tuple[int, ...]
    action_indices: tuple[int, ...]
    attributes: np.ndarray = field(default_factory=_empty_array, compare=False)


@dataclass(frozen=True)
class MorphologyEdge:
    """A directed edge in a morphology graph."""

    source: str
    target: str
    edge_type: EdgeType


@dataclass(frozen=True)
class Morphology:
    """Per-environment morphology graph with split / join operations.

    Attributes:
        nodes: All nodes in the graph.
        edges: All edges in the graph.
        unassigned_obs_indices: Observation indices not mapped to any node
            (e.g. task-specific occupancy or target temperature signals).
        unassigned_action_indices: Action indices not mapped to any node.
            Unlike unassigned observations (often benign task signals), an
            unassigned action means ``join_actions`` silently leaves that
            actuator at zero -- typically a sign the morphology is out of
            sync with the actuator set the pipeline emits.
        common_attributes: Static morphology-wide attribute values
            (e.g. building-level metadata shared across all nodes).
            Defaults to an empty array; the bounds schema is module-level
            (see :data:`COMMON_ATTRIBUTE_LOW` / :data:`COMMON_ATTRIBUTE_HIGH`).
            Excluded from ``__eq__``/``__hash__``.
    """

    nodes: tuple[MorphologyNode, ...]
    edges: tuple[MorphologyEdge, ...]
    unassigned_obs_indices: tuple[int, ...] = ()
    unassigned_action_indices: tuple[int, ...] = ()
    common_attributes: np.ndarray = field(
        default_factory=_empty_array,
        compare=False,
    )

    def node_ids(self) -> list[str]:
        """Return all node IDs in graph order."""
        return [n.node_id for n in self.nodes]

    def node_by_id(self, node_id: str) -> MorphologyNode:
        """Lookup a node by its ID."""
        for n in self.nodes:
            if n.node_id == node_id:
                return n
        raise KeyError(f"No node with id {node_id!r}")

    def nodes_by_type(self, node_type: NodeType) -> list[MorphologyNode]:
        """Return all nodes of a given type."""
        return [n for n in self.nodes if n.node_type is node_type]

    # split / join ---------------------------------------------------------

    def split_observation(self, obs: np.ndarray) -> dict[str, np.ndarray]:
        """Map a flat global observation to per-node local observations.

        Nodes with no observation indices are omitted from the result.

        Args:
            obs: Flat observation array from ``env.step()`` / ``env.reset()``.

        Returns:
            Dict mapping ``node_id`` to the node's local observation slice.
        """
        out: dict[str, np.ndarray] = {}
        for n in self.nodes:
            if n.obs_indices:
                out[n.node_id] = obs[np.array(n.obs_indices)]
        return out

    def join_actions(
        self,
        actions: dict[str, np.ndarray],
        action_dim: int | None = None,
    ) -> np.ndarray:
        """Assemble per-node local actions into a flat global action.

        Nodes with no action indices may be absent from *actions*.

        Args:
            actions: Dict mapping ``node_id`` to the node's local action.
            action_dim: Total global action dimension.  Inferred from the
                node metadata when ``None``.

        Returns:
            Flat action array ready for ``env.step()``.
        """
        if action_dim is None:
            all_idx = [i for n in self.nodes for i in n.action_indices]
            action_dim = max(all_idx) + 1 if all_idx else 0

        out = np.zeros(action_dim, dtype=np.float32)
        for n in self.nodes:
            if not n.action_indices:
                continue
            node_action = actions.get(n.node_id)
            if node_action is None:
                continue
            indices = np.array(n.action_indices)
            out[indices] = np.asarray(node_action, dtype=np.float32)
        return out

    def adjacency_list(self, edge_type: EdgeType | None = None) -> dict[str, list[str]]:
        """Return adjacency list, optionally filtered by edge type."""
        adj: dict[str, list[str]] = {n.node_id: [] for n in self.nodes}
        for e in self.edges:
            if edge_type is not None and e.edge_type != edge_type:
                continue
            adj.setdefault(e.source, []).append(e.target)
        return adj

    def type_counts(self) -> dict[str, int]:
        """Return a count of nodes per type name."""
        counts: dict[str, int] = {}
        for n in self.nodes:
            counts[n.node_type.name] = counts.get(n.node_type.name, 0) + 1
        return counts


# ---------------------------------------------------------------------------
# Construction helpers
# ---------------------------------------------------------------------------


def _find_obs_index(slot_names: list[str], key: str) -> int | None:
    """Find observation index by exact case-insensitive match."""
    key_l = key.strip().lower()
    for i, name in enumerate(slot_names):
        if name.strip().lower() == key_l:
            return i
    return None


def _find_zone_temp_index(slot_names: list[str], zone_name: str) -> int | None:
    """Find the observation index for a zone's air temperature."""
    target = f"zone air temperature {zone_name}".lower().strip()
    for i, name in enumerate(slot_names):
        if name.strip().lower() == target:
            return i
    return None


def _find_action_index_for_actuator(
    action_names: list[str],
    component_type: str,
    control_type: str,
    component_name: str,
) -> int | None:
    """Find the action index matching an actuator description triplet."""
    target = f"{component_type}::{control_type}::{component_name}".lower().strip()
    for i, name in enumerate(action_names):
        if name.strip().lower() == target:
            return i
    return None


def _ad_triple(ad: Any) -> tuple[str, str, str]:
    """Extract the (component_type, control_type, component_name) triple."""
    return ad.component_type, ad.control_type, ad.component_name


def build_morphology(
    hvac_equipment: Sequence[Equipment],
    observation_names: list[str],
    action_names: list[str],
    *,
    controlled_zones: list[str] | None = None,
    all_zone_names: list[str] | None = None,
    zone_geometry: dict[str, "ZoneGeometry"] | None = None,
) -> Morphology:
    """Build a morphology graph from equipment metadata and obs/action names.

    This is the main factory function.  It is called inside
    ``create_simulator`` and the result is stored in
    ``env.metadata["morphology"]``.

    Args:
        hvac_equipment: Sequence of :class:`Equipment` objects from the
            building's processed ``equipment.json``.
        observation_names: Flat observation slot names
            (``env.metadata["observation_names"]``).
        action_names: Flat action slot names
            (``env.metadata["action_names"]``).
        controlled_zones: Zone names served by HVAC equipment.
        all_zone_names: All thermal zone names in the building model.
            When provided, zones not in *controlled_zones* are added as
            ``uncontrolled_zone`` nodes.
        zone_geometry: Optional mapping ``zone_name -> ZoneGeometry`` from
            :func:`building2building.geometry.extract_zone_geometry`. When
            present, zone-typed nodes carry their geometric attributes;
            otherwise their ``attributes`` array stays empty.

    Returns:
        A fully-constructed :class:`Morphology`.
    """
    nodes: list[MorphologyNode] = []
    edges: list[MorphologyEdge] = []
    assigned_obs: set[int] = set()

    def _attrs_for(zone_name: str) -> np.ndarray:
        """Per-zone attribute array, looked up from `zone_geometry` if
        supplied; empty otherwise."""
        if zone_geometry is None:
            return np.empty(0, dtype=np.float32)
        zg = zone_geometry.get(zone_name)
        if zg is None:
            return np.empty(0, dtype=np.float32)
        return zg.to_array()

    # -- Global singleton nodes --------------------------------------------

    weather_obs: list[int] = []
    for key in ("outdoor_temperature", "outdoor_humidity"):
        idx = _find_obs_index(observation_names, key)
        if idx is not None:
            weather_obs.append(idx)
    if weather_obs:
        nodes.append(MorphologyNode("weather", WEATHER, tuple(weather_obs), ()))
        assigned_obs.update(weather_obs)

    calendar_obs: list[int] = []
    for key in ("time_of_day", "day_of_week", "day_of_year"):
        idx = _find_obs_index(observation_names, key)
        if idx is not None:
            calendar_obs.append(idx)
    if calendar_obs:
        nodes.append(MorphologyNode("calendar", CALENDAR, tuple(calendar_obs), ()))
        assigned_obs.update(calendar_obs)

    energy_obs: list[int] = []
    for key in ("energy_electricity", "energy_gas"):
        idx = _find_obs_index(observation_names, key)
        if idx is not None:
            energy_obs.append(idx)
    if energy_obs:
        nodes.append(MorphologyNode("energy", ENERGY, tuple(energy_obs), ()))
        assigned_obs.update(energy_obs)

    # -- Equipment-driven zone & supply nodes ------------------------------

    zones_with_nodes: set[str] = set()

    for eq in hvac_equipment:
        eq_type = getattr(eq, "equipment_type", None)

        if eq_type in ("unitarysystem", "heatpump"):
            zone = eq.zones()[0]
            temp_idx = _find_zone_temp_index(observation_names, zone)
            o_idx: list[int] = [temp_idx] if temp_idx is not None else []

            a_idx: list[int] = []
            for ad in eq.actuator_descriptions():
                ai = _find_action_index_for_actuator(
                    action_names,
                    ad.component_type,
                    ad.control_type,
                    ad.component_name,
                )
                if ai is not None:
                    a_idx.append(ai)

            node_id = f"zone:{zone}"
            nodes.append(
                MorphologyNode(
                    node_id,
                    UNITARY_ZONE,
                    tuple(o_idx),
                    tuple(a_idx),
                    attributes=_attrs_for(zone),
                )
            )
            assigned_obs.update(o_idx)
            zones_with_nodes.add(zone)

        elif eq_type == "vavsystem":
            supply_ad = eq.supply_temp_setpoint  # type: ignore[attr-defined]
            supply_ai = _find_action_index_for_actuator(
                action_names, *_ad_triple(supply_ad)
            )
            # Per-loop OA-mixer actuator shares the supply node (both control
            # the air loop, not a single zone). Order matches VAV_SUPPLY's
            # action schema: (supply_air_temp, oa_mass_flow).
            oa_ad = eq.oa_mass_flow  # type: ignore[attr-defined]
            oa_ai = _find_action_index_for_actuator(
                action_names, *_ad_triple(oa_ad)
            )
            supply_act = tuple(i for i in (supply_ai, oa_ai) if i is not None)
            supply_id = f"supply:{supply_ad.component_name}"
            nodes.append(MorphologyNode(supply_id, VAV_SUPPLY, (), supply_act))

            for terminal in eq.terminals:  # type: ignore[attr-defined]
                zone = terminal.zone
                temp_idx = _find_zone_temp_index(observation_names, zone)
                o_idx = [temp_idx] if temp_idx is not None else []

                # Check which actuators are in the agent action space.
                flow_ai = _find_action_index_for_actuator(
                    action_names, *_ad_triple(terminal.flow_fraction)
                )
                htg_ai = _find_action_index_for_actuator(
                    action_names, *_ad_triple(terminal.heating_setpoint)
                )
                clg_ai = _find_action_index_for_actuator(
                    action_names, *_ad_triple(terminal.cooling_setpoint)
                )

                if clg_ai is not None:
                    # All three actuators present.
                    a_idx = [i for i in (flow_ai, htg_ai, clg_ai) if i is not None]
                    nt = VAV_ZONE
                else:
                    # Cooling setpoint is fixed — use the reduced type.
                    a_idx = [i for i in (flow_ai, htg_ai) if i is not None]
                    nt = VAV_ZONE_NO_COOLING

                node_id = f"zone:{zone}"
                nodes.append(
                    MorphologyNode(
                        node_id,
                        nt,
                        tuple(o_idx),
                        tuple(a_idx),
                        attributes=_attrs_for(zone),
                    )
                )
                assigned_obs.update(o_idx)
                zones_with_nodes.add(zone)

                edges.append(MorphologyEdge(supply_id, node_id, "hvac_system"))
                edges.append(MorphologyEdge(supply_id, node_id, "controls"))

        elif eq_type == "heating_only":
            zone = eq.zones()[0]
            temp_idx = _find_zone_temp_index(observation_names, zone)
            o_idx = [temp_idx] if temp_idx is not None else []

            a_idx = []
            for ad in eq.actuator_descriptions():
                ai = _find_action_index_for_actuator(
                    action_names,
                    ad.component_type,
                    ad.control_type,
                    ad.component_name,
                )
                if ai is not None:
                    a_idx.append(ai)

            node_id = f"zone:{zone}"
            nodes.append(
                MorphologyNode(
                    node_id,
                    HEATING_ZONE,
                    tuple(o_idx),
                    tuple(a_idx),
                    attributes=_attrs_for(zone),
                )
            )
            assigned_obs.update(o_idx)
            zones_with_nodes.add(zone)

    # -- Uncontrolled zones (obs-only) -------------------------------------

    if all_zone_names is not None:
        for zone in all_zone_names:
            if zone in zones_with_nodes:
                continue
            temp_idx = _find_zone_temp_index(observation_names, zone)
            if temp_idx is None:
                continue
            node_id = f"zone:{zone}"
            nodes.append(
                MorphologyNode(
                    node_id,
                    UNCONTROLLED_ZONE,
                    (temp_idx,),
                    (),
                    attributes=_attrs_for(zone),
                )
            )
            assigned_obs.add(temp_idx)
            zones_with_nodes.add(zone)

    # -- Collect unassigned observations -----------------------------------

    unassigned = tuple(
        i for i in range(len(observation_names)) if i not in assigned_obs
    )
    if unassigned:
        unassigned_names = [observation_names[i] for i in unassigned]
        logger.warning(
            "Morphology: %d unassigned observation slots: %s",
            len(unassigned),
            unassigned_names,
        )

    # -- Collect unassigned actions ----------------------------------------
    # Every actuator the pipeline emits must land on some node, else the
    # slot silently defaults to zero in join_actions(). Unlike an unassigned
    # observation (often a benign task signal), an unassigned *action* means
    # the morphology cannot drive part of the building -- usually because the
    # actuator set in pipeline/actuators.py grew a type the NodeTypes here
    # don't account for. Warn loudly so that desync is visible.
    assigned_actions = {i for n in nodes for i in n.action_indices}
    unassigned_actions = tuple(
        i for i in range(len(action_names)) if i not in assigned_actions
    )
    if unassigned_actions:
        unassigned_action_names = [action_names[i] for i in unassigned_actions]
        logger.warning(
            "Morphology: %d action slot(s) not mapped to any node; "
            "join_actions() will leave them at zero: %s",
            len(unassigned_actions),
            unassigned_action_names,
        )

    return Morphology(
        nodes=tuple(nodes),
        edges=tuple(edges),
        unassigned_obs_indices=unassigned,
        unassigned_action_indices=unassigned_actions,
    )


def add_thermal_adjacency_edges(
    morphology: Morphology,
    zone_adjacency: dict[str, list[str]],
) -> Morphology:
    """Return a new Morphology with thermal adjacency edges added.

    Args:
        morphology: Base morphology (typically from :func:`build_morphology`).
        zone_adjacency: Mapping from zone name to adjacent zone names.
            Can be extracted from the epJSON ``BuildingSurface:Detailed``
            objects by finding zones that share interior wall surfaces.

    Returns:
        A new :class:`Morphology` with the additional edges.
    """
    existing_node_ids = {n.node_id for n in morphology.nodes}
    new_edges = list(morphology.edges)
    seen: set[tuple[str, str]] = {(e.source, e.target) for e in morphology.edges}

    for zone, neighbors in zone_adjacency.items():
        src_id = f"zone:{zone}"
        if src_id not in existing_node_ids:
            continue
        for neighbor in neighbors:
            tgt_id = f"zone:{neighbor}"
            if tgt_id not in existing_node_ids:
                continue
            pair = (src_id, tgt_id)
            if pair not in seen:
                new_edges.append(MorphologyEdge(src_id, tgt_id, "thermal_adjacency"))
                seen.add(pair)

    return Morphology(
        nodes=morphology.nodes,
        edges=tuple(new_edges),
        unassigned_obs_indices=morphology.unassigned_obs_indices,
    )
