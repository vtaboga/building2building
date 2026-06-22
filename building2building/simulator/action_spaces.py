"""Action space construction for EnergyPlus environments.

Translates :class:`~b2b.types.ActuatorDescription` lists into Gymnasium
``Box`` action spaces, splitting fixed actuators (e.g. VAV cooling
setpoints) from agent-controlled ones.
"""

import itertools
import logging
import urllib.parse
from dataclasses import dataclass
from typing import Any, Sequence

import minergym.ontology as ontology
import numpy as np
from gymnasium.spaces import Box, Dict
from minergym.simulation import ActuatorHole
from rdflib.term import Node

from building2building.types import ActuatorDescription

from .transform_utils import (
    Transform,
    TransformConcat,
    TransformDictSpace,
    TransformList,
    TransformListToArray,
    TransformListToArrayShift,
)

logger = logging.getLogger(__name__)


def hvac_actuators_transform(
    hvac_actuators: Sequence[ActuatorDescription],
) -> Transform[list, Box]:
    """Build an action transform from a list of actuator descriptions.

    Args:
        hvac_actuators: Ordered sequence of actuator descriptions.
            An empty sequence produces a valid but degenerate transform
            (a warning is logged).

    Returns:
        A ``TransformListToArray`` whose codomain ``Box`` has per-
        actuator bounds.
    """

    holes: list[ActuatorHole] = []
    lows: list[float] = []
    highs: list[float] = []

    for a in hvac_actuators:
        holes.append(ActuatorHole(a.component_type, a.control_type, a.component_name))
        lows.append(a.lower_bound)
        highs.append(a.upper_bound)

    if not holes:
        logger.warning(
            "hvac_actuators is empty; the environment will not make a lot of sense"
        )

    return TransformListToArray(
        holes,
        Box(low=np.asarray(lows, dtype=float), high=np.asarray(highs, dtype=float)),
    )


# ---------------------------------------------------------------------------
# Split action space: fixed cooling setpoints
# ---------------------------------------------------------------------------

FIXED_CLG_SP_VALUE = 40.0


def _is_fixed_actuator(a: ActuatorDescription) -> bool:
    """VAV cooling setpoints are fixed for simulation stability."""
    return "b2b vav clg setpoint" in a.component_name.lower()


@dataclass(slots=True)
class HvacActionSpace:
    """Full EnergyPlus action space + reduced space exposed to the agent.

    Cooling setpoints are removed from the agent-facing space and pinned
    to ``FIXED_CLG_SP_VALUE`` (40 °C), following the OfficeRL convention.

    Attributes:
        full_transform: Transform covering *all* EnergyPlus actuators.
        agent_transform: Transform covering only the actuators the
            agent is allowed to control.
        agent_actuators: Descriptions of agent-controlled actuators.
        fixed_indices: Positions in the full vector occupied by fixed
            actuators.
        fixed_values: Constant values assigned to fixed actuators.
    """

    full_transform: TransformListToArray
    agent_transform: TransformListToArray
    agent_actuators: list[ActuatorDescription]
    fixed_indices: list[int]
    fixed_values: list[float]

    def assemble_full_action(
        self, agent_action: np.ndarray | Sequence[float]
    ) -> list[float]:
        """Expand an agent action vector into the full EnergyPlus actuator vector.

        Args:
            agent_action: Action array of length
                ``len(self.agent_actuators)``.

        Returns:
            Full-length actuator list with fixed slots filled in.

        Raises:
            ValueError: If *agent_action* has the wrong length.
        """
        agent_list: list[float] = (
            agent_action.tolist()
            if hasattr(agent_action, "tolist")
            else [float(x) for x in agent_action]
        )
        if len(agent_list) != len(self.agent_actuators):
            raise ValueError(
                f"Expected agent_action of length {len(self.agent_actuators)}, "
                f"got {len(agent_list)}"
            )

        fixed_set = set(self.fixed_indices)
        n_full = len(self.fixed_indices) + len(self.agent_actuators)
        full: list[float] = [0.0] * n_full

        for idx, val in zip(self.fixed_indices, self.fixed_values):
            full[int(idx)] = float(val)

        j = 0
        for i in range(n_full):
            if i not in fixed_set:
                full[i] = agent_list[j]
                j += 1

        return full


FIXED_HEATING_ONLY_VALUE = 18.0


def hvac_action_space(
    hvac_actuators: Sequence[ActuatorDescription],
    *,
    fixed_heating_only_names: frozenset[str] = frozenset(),
    fixed_heating_only_value: float = FIXED_HEATING_ONLY_VALUE,
    additional_fixed: dict[str, float] | None = None,
) -> HvacActionSpace:
    """Build a split action space where some actuators are fixed.

    Returns an ``HvacActionSpace`` whose ``agent_transform`` exposes only
    non-fixed actuators to the agent, while ``full_transform`` covers all
    actuators for EnergyPlus.

    VAV cooling-setpoint actuators (identified by ``"b2b vav clg setpoint"``
    in the component name) are always fixed at 40 °C.  Heating-only zone
    actuators whose ``component_name`` is in *fixed_heating_only_names* are
    pinned at *fixed_heating_only_value*.

    Args:
        hvac_actuators: Complete ordered sequence of HVAC actuator
            descriptions.
        fixed_heating_only_names: Component names of heating-only
            actuators to pin (empty set means none are pinned).
        fixed_heating_only_value: Constant value for pinned
            heating-only actuators (default 18 °C).
        additional_fixed: Mapping from actuator ``component_name`` to
            the constant value at which it should be pinned.  Used by
            the action-space transfer benchmark to selectively remove
            actuators from the agent's action space.

    Returns:
        An ``HvacActionSpace`` with both full and agent transforms.
    """
    full_transform = hvac_actuators_transform(hvac_actuators)

    agent_actuators: list[ActuatorDescription] = []
    fixed_indices: list[int] = []
    fixed_values: list[float] = []

    for i, a in enumerate(hvac_actuators):
        if _is_fixed_actuator(a):
            fixed_indices.append(i)
            fixed_values.append(FIXED_CLG_SP_VALUE)
        elif a.component_name in fixed_heating_only_names:
            fixed_indices.append(i)
            fixed_values.append(fixed_heating_only_value)
        elif additional_fixed and a.component_name in additional_fixed:
            fixed_indices.append(i)
            fixed_values.append(additional_fixed[a.component_name])
        else:
            agent_actuators.append(a)

    agent_transform = hvac_actuators_transform(agent_actuators)

    return HvacActionSpace(
        full_transform=full_transform,
        agent_transform=agent_transform,
        agent_actuators=agent_actuators,
        fixed_indices=fixed_indices,
        fixed_values=fixed_values,
    )


def agent_action_dim(
    hvac_equipment: Sequence[Any],
    *,
    expose_heating_only_zones: bool = True,
    additional_fixed: dict[str, float] | None = None,
) -> int:
    """Return the agent-facing action dimension for a list of equipment.

    Single source of truth for the size of ``env.action_space`` constructed
    by :func:`building2building.simulator.create_simulator`.  Reuses the
    same :func:`hvac_action_space` filter so that
    :data:`BuildingInfo.action_dim` (sourced from ``metadata.parquet``)
    matches ``env.action_space.shape[0]`` by construction.

    Filters applied (in order):

    1. VAV cooling-setpoint actuators (``"b2b vav clg setpoint"`` in
       :attr:`ActuatorDescription.component_name`) are always pinned to
       :data:`FIXED_CLG_SP_VALUE` (40 °C) and removed from the agent space.
    2. If ``expose_heating_only_zones`` is ``False``, actuators on
       ``equipment_type == "heating_only"`` zones are pinned to
       :data:`FIXED_HEATING_ONLY_VALUE` (18 °C) and removed.
       Default ``True`` matches :class:`BuildingConfig` and
       :class:`EnvBuildConfig`.
    3. Any actuator whose ``component_name`` appears in
       ``additional_fixed`` is pinned and removed.

    Args:
        hvac_equipment: Sequence of equipment objects (e.g. ``VAVSystem``,
            ``UnitarySystem``, ``HeatingOnlyZone``) with an
            ``actuator_descriptions()`` method.
        expose_heating_only_zones: Match the corresponding flag on
            :class:`BuildingConfig`.  Defaults to ``True``.
        additional_fixed: Mapping of ``component_name -> pinned value`` for
            extra actuators removed from the agent space (used by the
            action-space transfer benchmark).

    Returns:
        ``len(HvacActionSpace.agent_actuators)`` for the equipment list.
    """
    actuators = list(
        itertools.chain.from_iterable(
            eq.actuator_descriptions() for eq in hvac_equipment
        )
    )

    if not expose_heating_only_zones:
        fixed_heating_only_names = frozenset(
            a.component_name
            for eq in hvac_equipment
            if getattr(eq, "equipment_type", None) == "heating_only"
            for a in eq.actuator_descriptions()
        )
    else:
        fixed_heating_only_names = frozenset()

    action_space_info = hvac_action_space(
        actuators,
        fixed_heating_only_names=fixed_heating_only_names,
        additional_fixed=additional_fixed,
    )
    return len(action_space_info.agent_actuators)
