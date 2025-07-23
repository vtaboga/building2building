"""In the morphology.py file, a pattern is repeated a bunch of times:

1. For a small piece of the action / observation space, we define a subtemplate
(or the minergym side) with its corresponding gymnasium space.

2. We define a way to transform data constructed from the template into the
shape compatible with the associated gymnasium space.

3. Once we have done that for a bunch of tiny pieces of the action/observation
space, we:

  1. glue all of the templates together.

  2. glue all of the spaces together.

  3. glue all of the transformation functions together.


This small module codifies this pattern through the `Transform` structure and
provides a couple of "glueing" functions.

"""

from dataclasses import dataclass
from typing import Any, Callable, Generic, TypeVar

from gymnasium.spaces import Dict, Space, Tuple

A = TypeVar("A")
B = TypeVar("B")


@dataclass(frozen=True)
class Transform(Generic[A, B]):
    """A structure containing a template, a gymnasium space codifying that template
    and an isomorphism between the two represetations.

            transform
    domain <---------> codomain

    """

    domain: A
    codomain: B
    transform: Callable[[Any], Any]
    detransform: Callable[[Any], Any]

    def inverse(self):
        return Transform(self.codomain, self.domain, self.detransform, self.transform)


def transform_dict(d: dict[str, Transform[Any, Space]]) -> Transform[Any, Space]:
    template: Any = {k: v.domain for k, v in d.items()}
    space: Space = Dict({k: v.codomain for k, v in d.items()})

    def transform(a):
        return {k: v.transform(a[k]) for k, v in d.items()}

    def detransform(b):
        return {k: v.detransform(b[k]) for k, v in d.items()}

    return Transform(template, space, transform, detransform)


def transform_list(l: list[Transform[Any, Space]]) -> Transform[Any, Space]:
    def transform(a):
        return tuple(s.transform(e) for s, e in zip(l, a))

    def detransform(b):
        return [s.detransform(e) for s, e in zip(l, b)]

    return Transform(
        [v.domain for v in l],
        Tuple([v.codomain for v in l]),
        transform,
        detransform,
    )
