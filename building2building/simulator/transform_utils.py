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

import cmath
import math
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar

import numpy as np
from gymnasium.spaces import Box, Dict, Space, Tuple

A = TypeVar("A", covariant=True)
B = TypeVar("B", covariant=True)


class Transform(Generic[A, B], Protocol):
    def domain(self) -> A: ...

    def codomain(self) -> B: ...

    # Forward call
    def __call__(self, obj) -> Any: ...
    # Reverse call
    def reverse(self, obj) -> Any: ...


@dataclass(slots=True, frozen=True)
class TransformDict(Transform[Any, Space]):
    fields: dict[str, Transform[Any, Space]]

    def domain(self) -> Any:
        return {k: v.domain() for k, v in self.fields.items()}

    def codomain(self) -> Space:
        return Dict({k: v.codomain() for k, v in self.fields.items()})

    def __call__(self, obj):
        return {k: v(obj[k]) for k, v in self.fields.items()}

    def reverse(self, obj) -> Any:
        return {k: v.reverse(obj[k]) for k, v in self.fields.items()}


x: Transform[Any, Space] = TransformDict({})


@dataclass(slots=True, frozen=True)
class TransformList(Transform[Any, Space]):
    fields: list[Transform[Any, Space]]

    def domain(self):
        return [field.domain() for field in self.fields]

    def codomain(self):
        return Tuple([field.codomain() for field in self.fields])

    def __call__(self, obj):
        return tuple(field(e) for field, e in zip(self.fields, obj))

    def reverse(self, obj):
        return [field.reverse(e) for field, e in zip(self.fields, obj)]


@dataclass(slots=True, frozen=True)
class TransformCyclical(Transform[Any, Box]):
    _domain: Any
    low: float
    high: float

    def domain(self):
        return self._domain

    def codomain(self):
        return Box(np.array([-1, -1]), np.array([1, 1]))

    def __call__(self, x: float) -> np.ndarray:
        rescaled = (x - self.low) / (self.high - self.low)
        c = cmath.exp(2 * math.pi * 1j * rescaled)
        a = np.array([c.real, c.imag])
        return a

    def reverse(self, obj):
        real, imag = obj
        c = real + imag * 1j
        phase = cmath.phase(c)
        if phase < 0:
            phase += 2 * math.pi

        rescaled = phase / (2 * math.pi)
        x = rescaled * (self.high - self.low) + self.low

        return x


@dataclass(slots=True, frozen=True)
class TransformInverse(Transform[A, B]):
    inner: Transform[B, A]

    def domain(self) -> A:
        return self.inner.codomain()

    def codomain(self) -> B:
        return self.inner.domain()

    def __call__(self, obj):
        return self.inner.reverse(obj)

    def reverse(self, obj):
        return self.inner(obj)


@dataclass(slots=True, frozen=True)
class TransformIdentity(Transform[A, B]):
    _domain: A
    _codomain: B

    def domain(self) -> A:
        return self._domain

    def codomain(self) -> B:
        return self._codomain

    def __call__(self, obj):
        return obj

    def reverse(self, obj):
        return obj
