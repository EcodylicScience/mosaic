"""The parameters every tracker op shares: what to run over, and how to run it.

:class:`TrackerOpParams` declares the execution knobs, over the scope
:class:`~mosaic.core.pipeline.types.OpParams` declares. Each tracker's own
parameter model inherits both and adds its tool's settings.

:class:`PhasedTrackerOpParams` adds what a tool that runs in several subprocess
phases needs on top: every parameter its subclass declares names the phases that
consume it, checked when the subclass is created.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel

from mosaic.core.pipeline.markers import Phase
from mosaic.core.pipeline.types import OpParams
from mosaic.core.params import (
    HASH_EXCLUDE,
    Declared,
)

__all__ = ["PhasedTrackerOpParams", "TrackerOpParams", "refuse_unphased_fields"]

_CONVERT_TO_TRACKS_DESCRIPTION = (
    "Convert the tool's native output into a standardized tracks table once "
    "tracking finishes, instead of leaving the output where the tool wrote it."
)

_IDLE_TIMEOUT_DESCRIPTION = (
    "How long a phase may go without a line of output from the tool before it "
    "is killed. Silence is what tells a hung run from a slow one."
)

_MAX_RUNTIME_DESCRIPTION = (
    "Absolute wall-clock ceiling for one phase. Unset leaves the ceiling to "
    "whatever queue submitted the run."
)


class TrackerOpParams(OpParams):
    """Execution knobs shared by every tracker op.

    A tracker's own parameters -- its model reference, its thresholds, its
    tracker flavor -- are declared by the subclass. Pydantic allows a required
    field after defaulted ones, so ``model_path`` and its kin stay required.

    The execution knobs are all ``HASH_EXCLUDE``: they change how a run happens,
    not what it produces, so folding them in would move an identifier without
    moving the output, which is a cache miss costing a recompute for nothing.
    Placement knobs (which conda environment, which binary, which display) are
    deliberately *not* here at all -- they are properties of a machine, so they
    are read from ``MOSAIC_<TOOL>_*`` rather than carried in params a queued job
    would ship to a different one.
    """

    convert_to_tracks: Annotated[
        bool, HASH_EXCLUDE, Declared(_CONVERT_TO_TRACKS_DESCRIPTION)
    ] = True
    idle_timeout: Annotated[
        float, HASH_EXCLUDE, Declared(_IDLE_TIMEOUT_DESCRIPTION, unit="s")
    ] = 900
    max_runtime: Annotated[
        float | None, HASH_EXCLUDE, Declared(_MAX_RUNTIME_DESCRIPTION, unit="s")
    ] = None


def refuse_unphased_fields(model: type[BaseModel]) -> None:
    """Raise unless every field *model* introduces names the phases reading it.

    A field a base already declares is exempt: redeclaring ``idle_timeout`` to
    change its default states nothing about a tool's phases, and the field has
    no phase to name. What is checked is what *model* adds.

    Args:
        model: The parameter model to check.

    Raises:
        TypeError: A field *model* introduces declares no
            :class:`~mosaic.core.pipeline.markers.Phase`.
    """
    inherited = {
        name for base in model.__mro__[1:] for name in getattr(base, "model_fields", {})
    }
    unphased = sorted(
        name
        for name, info in model.model_fields.items()
        if name in model.__annotations__
        and name not in inherited
        and not any(isinstance(marker, Phase) for marker in info.metadata)
    )
    if unphased:
        refused = (
            f"{model.__name__} declares {unphased} without a Phase. Add "
            f"Phase(...) to each field's Annotated, naming the phases that "
            f"consume it."
        )
        raise TypeError(refused)


class PhasedTrackerOpParams(TrackerOpParams):
    """Base for a tracker whose phases each consume part of the parameters.

    Every field a subclass adds names the phases that consume it, and this class
    refuses the subclass at class creation otherwise.
    :func:`~mosaic.core.pipeline.markers.phase_fields` returns an empty tuple
    both for a phase no field names and for a model that declares no
    :class:`~mosaic.core.pipeline.markers.Phase` at all, so a model that lost
    its markers projects an empty settings dictionary and the tool runs on its
    own defaults.

    The scope and execution fields it inherits reach no phase of the tool and
    declare none, so :func:`refuse_unphased_fields` checks what a subclass adds
    and exempts what it inherits.
    """

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: object) -> None:
        """Refuse a subclass field that names no phase.

        ``model_fields`` is complete here, with its ``Annotated`` metadata
        intact. A plain ``__init_subclass__`` runs before either is true.
        """
        super().__pydantic_init_subclass__(**kwargs)
        refuse_unphased_fields(cls)


# The hook fires for subclasses only, so the base is checked here. A tool-facing
# field added to it would otherwise be the one field in the hierarchy that could
# name no phase.
refuse_unphased_fields(PhasedTrackerOpParams)
