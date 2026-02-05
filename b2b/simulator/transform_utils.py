"""When assembling action and observation spaces, a pattern is repeated a bunch
of times:

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
provides a couple of "gluing" functions.

"""

import cmath
import math
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Generic, Protocol, TypeVar

import gymnasium.spaces.utils
import numpy as np
from gymnasium.spaces import Box, Dict, Space, Tuple

A = TypeVar("A", covariant=True)
B = TypeVar("B", covariant=True)
C = Any


class Transform(Generic[A, B], Protocol):
    """A bidirectional transformation between two things.


    The `A` type variable denotes the type of the representation of the domain
    and `B` denotes the type of the representation of the codomain. A is often
    Any (when the domain is a minergym template with holes) and B is oftern a
    gymansium space.

    """

    def domain(self) -> A:
        """The domain in which we take the thing to transform."""
        ...

    def codomain(self) -> B:
        """The codomain into which we will transform things."""
        ...

    def __call__(self, obj, /) -> Any:
        """Transform obj into the codomain."""
        ...

    def reverse(self, obj, /) -> Any:
        """Transform obj back from the codomain to the domain."""
        ...


@dataclass(slots=True, frozen=True)
class TransformDict(Transform[dict, dict]):
    fields: dict[str, Transform[Any, Any]]

    def domain(self) -> dict:
        return {k: v.domain() for k, v in self.fields.items()}

    def codomain(self) -> dict:
        return {k: v.codomain() for k, v in self.fields.items()}

    def __call__(self, obj):
        return {k: v(obj[k]) for k, v in self.fields.items()}

    def reverse(self, obj) -> Any:
        return {k: v.reverse(obj[k]) for k, v in self.fields.items()}


@dataclass(slots=True, frozen=True)
class TransformDictSpace(Transform[dict, Dict]):
    fields: dict[str, Transform[Any, Any]]

    def domain(self) -> dict:
        return {k: v.domain() for k, v in self.fields.items()}

    def codomain(self) -> Dict:
        return Dict({k: v.codomain() for k, v in self.fields.items()})

    def __call__(self, obj):
        return {k: v(obj[k]) for k, v in self.fields.items()}

    def reverse(self, obj) -> Any:
        return {k: v.reverse(obj[k]) for k, v in self.fields.items()}


@dataclass(slots=True, frozen=True)
class TransformDictToList(Transform[dict, list]):
    the_dict: dict[str, Transform]

    def domain(self):
        return {k: v.domain() for k, v in self.the_dict.items()}

    def codomain(self):
        return [v.codomain() for v in self.the_dict.values()]

    def __call__(self, x):
        return [self.the_dict[k](x[k]) for k in self.the_dict.keys()]

    def reverse(self, y):
        o = {}

        keys = [k for k in self.the_dict.keys()]

        for k, v in zip(keys, y):
            o[k] = self.the_dict[k].reverse(v)

        return o


@dataclass(slots=True, frozen=True)
class TransformList(Transform[list, list]):
    fields: list[Transform[Any, Any]]

    def domain(self):
        return [field.domain() for field in self.fields]

    def codomain(self):
        return [field.codomain() for field in self.fields]

    def __call__(self, obj):
        return [field(e) for field, e in zip(self.fields, obj)]

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


@dataclass
class TransformListToArray(Transform[list, Box]):
    _domain: list
    _codomain: Box

    def domain(self):
        return self._domain

    def codomain(self):
        return self._codomain

    def __call__(self, obj):
        return np.array(obj)

    def reverse(self, obj):
        return obj.tolist()


@dataclass
class TransformScalarToArray(Transform[Any, Box]):
    _domain: Any
    low: float
    high: float

    def domain(self):
        return self._domain

    def codomain(self):
        return Box(np.array([self.low]), np.array([self.high]))

    def __call__(self, obj):
        return np.array([obj])

    def reverse(self, obj):
        return obj.tolist()[0]


@dataclass
class TransformListToArrayShift(Transform[list, Box]):
    _domain: Any
    _codomain: Any

    def domain(self):
        return self._domain

    def codomain(self):
        return self._codomain

    def __call__(self, t):
        return np.array([t[0], t[1] - t[0]])

    def reverse(self, obj):
        return [float(obj[0]), float(obj[0] + obj[1])]


@dataclass
class TransformConcat(Transform[list, Box]):
    subtransform: Transform[Any, list[Box]]

    def domain(self):
        return self.subtransform.domain()

    def codomain(self):
        boxes = self.subtransform.codomain()

        lows = np.concatenate([box.low for box in boxes])
        highs = np.concatenate([box.high for box in boxes])

        return Box(lows, highs)

    def __call__(self, t):
        subarrays = self.subtransform(t)
        return np.concatenate(subarrays)

    def reverse(self, obj):
        subarrays = []
        i = 0
        for box in self.subtransform.codomain():
            subarray_len = box.shape[0]
            subarrays.append(obj[i : i + subarray_len])
            i += subarray_len
        return self.subtransform.reverse(subarrays)


@dataclass
class TransformAppend(Transform[list, list]):
    subtransforms: list[Transform[list, Any]]

    def domain(self):
        return [subtransform.domain() for subtransform in self.subtransforms]

    def codomain(self):
        return sum(
            [subtransform.codomain() for subtransform in self.subtransforms], start=[]
        )

    def __call__(self, list_of_lists):
        out = []
        for i, list in enumerate(list_of_lists):
            out.extend(self.subtransforms[i](list))
        return out

    def reverse(self, obj):
        sublists = []
        i = 0
        for st in self.subtransforms:
            subarray_len = len(st.codomain())
            sublists.append(st.reverse(obj[i : i + subarray_len]))
            i += subarray_len
        return sublists


@dataclass(slots=True, frozen=True)
class TransformFlattenSpace(Transform[Space, Box]):
    space: Space

    def domain(self):
        return self.space

    @cached_property
    def compute_codomain(self) -> Box:
        b = gymnasium.spaces.utils.flatten_space(self.space)
        if isinstance(b, Box):
            return b
        else:
            # ???
            raise Exception("should't happen")

    def codomain(self):
        return self.compute_codomain

    def __call__(self, x):
        return gymnasium.spaces.utils.flatten(self.space, x)

    def reverse(self, y):
        return gymnasium.spaces.utils.unflatten(self.space, y)


@dataclass(slots=True, frozen=True)
class TransformCompose(Transform[A, B]):
    first: Transform[A, C]
    second: Transform[C, B]

    def domain(self):
        return self.first.domain()

    def codomain(self):
        return self.second.codomain()

    def __call__(self, x):
        return self.second(self.first(x))

    def reverse(self, y):
        return self.first.reverse(self.second.reverse(y))


@dataclass(slots=True, frozen=True)
class TransformMonoList(Transform[A, list]):
    subtransform: Transform[A, Any]

    def domain(self):
        return self.subtransform.domain()

    def codomain(self):
        return [self.subtransform.codomain()]

    def __call__(self, x):
        return [self.subtransform(x)]

    def reverse(self, y):
        return self.subtransform.reverse(y[0])


def transform_flatten(thing: Any) -> Transform[Any, list]:
    match thing:
        case dict():
            into_list: Transform[dict, Any] = TransformDictToList(
                {k: TransformIdentity(v, v) for k, v in thing.items()}
            )

            flattened = TransformAppend(
                [transform_flatten(part) for part in into_list.codomain()]
            )

            return TransformCompose(into_list, flattened)
        case list():
            return TransformAppend([transform_flatten(part) for part in thing])
        case _:
            return TransformMonoList(TransformIdentity(thing, thing))
