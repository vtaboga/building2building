"""Named benchmark problems from the Building2Building paper.

Each class corresponds to one of the four generalization axes
described in Section 4 of the paper.
"""

from building2building.benchmarks.action_space_transfer import ActionSpaceTransfer
from building2building.benchmarks.cross_domain import CrossDomainGeneralization
from building2building.benchmarks.dynamics_adaptation import DynamicsAdaptation
from building2building.benchmarks.goal_adaptation import GoalAdaptation

__all__ = [
    "ActionSpaceTransfer",
    "CrossDomainGeneralization",
    "DynamicsAdaptation",
    "GoalAdaptation",
]
