"""What a media pre-processing step is, and the registry that names each one.

A step is its own parameter model. Its fields are the step's parameters, and a
``step`` field holding its name is the discriminator that tells one step's JSON
from another's. Two methods say what the step does, both given the placement
before it: :meth:`MediaStep.place` returns the placement after it, and
:meth:`MediaStep.bind` builds the function the step applies to each frame. A
validated list of steps is therefore a list of step objects, and
:data:`MEDIA_STEPS` maps a step's name to its model class.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar, Final

import numpy as np
import numpy.typing as npt

from mosaic.core.media.preprocess.geometry import Placement
from mosaic.core.params import Params

__all__ = [
    "MEDIA_STEPS",
    "STEP_DESCRIPTION",
    "Frame",
    "FrameFn",
    "MediaStep",
    "register_media_step",
]

STEP_DESCRIPTION: Final = "The step's name, which selects this step in a list of steps."
"""The prose every step declares its ``step`` discriminator with."""

type Frame = npt.NDArray[np.uint8]
"""One frame: ``H x W x 3`` BGR, or ``H x W`` gray once a step has made it gray."""

type FrameFn = Callable[[Frame], Frame]
"""The function a step applies to each frame, bound to the placement before it."""


class MediaStep(Params):
    """Base for a media pre-processing step, whose fields are its parameters.

    A subclass declares a ``step: Literal["<name>"]`` field defaulting to its
    name, and the four class variables below. Every spatial parameter is in
    source-space pixels and every frame parameter is a source-space frame
    number, whatever the step's position in a list: each step converts through
    the placement it is given.
    """

    name: ClassVar[str]
    """The step's registered name, equal to its ``step`` field."""

    version: ClassVar[str]
    """The step's version. A change to what the step computes bumps it."""

    moves_pixels: ClassVar[bool]
    """Whether a position in the step's output is a different position in its input."""

    appearance: ClassVar[bool]
    """Whether the step changes every pixel by one rule, independent of position.

    ``mask`` is therefore not an appearance step: whether it blacks out a pixel
    depends on where the pixel is.
    """

    def place(self, placement: Placement) -> Placement:
        """The placement after this step, given *placement*, the one before it.

        Raises:
            NotImplementedError: Always, on the base. Every step overrides it.
        """
        raise NotImplementedError(f"{type(self).__name__} does not define place()")

    def bind(self, placement: Placement) -> FrameFn:
        """The per-frame function for this step at *placement*, the one before it.

        Called once per entry, so whatever the function needs that does not
        change from frame to frame is computed here rather than per frame.

        Raises:
            NotImplementedError: Always, on the base. Every step overrides it.
        """
        raise NotImplementedError(f"{type(self).__name__} does not define bind()")


MEDIA_STEPS: Final[dict[str, type[MediaStep]]] = {}
"""Every registered step's model class, keyed by its name."""


def register_media_step[StepT: type[MediaStep]](cls: StepT) -> StepT:
    """Class decorator: register *cls* under its ``name``.

    Raises:
        ValueError: If *cls* declares no non-empty ``name``, or another class is
            already registered under it.
        TypeError: If *cls* omits ``version``, ``moves_pixels`` or ``appearance``.
    """
    name: object = getattr(cls, "name", "")
    if not isinstance(name, str) or not name:
        raise ValueError(f"{cls.__name__} must declare a non-empty 'name'")
    for declaration in ("version", "moves_pixels", "appearance"):
        if not hasattr(cls, declaration):
            raise TypeError(f"{cls.__name__} declares no {declaration!r}")
    held = MEDIA_STEPS.get(name)
    if held is not None:
        raise ValueError(
            f"{cls.__module__}.{cls.__qualname__} cannot register as the media "
            f"step {name!r}: {held.__module__}.{held.__qualname__} already holds it"
        )
    MEDIA_STEPS[name] = cls
    return cls
