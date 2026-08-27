"""The parameters every tracker op shares: what to run over, and how to run it.

Three op modules declare the same execution block before adding their tool's own
knobs, over the scope :class:`~mosaic.core.pipeline.types.OpParams` declares.

**Every field here is ``HASH_EXCLUDE``.** A tracker's ``run_id`` is minted from
its settings dictionary, which names no entry, so the scope selects which media
a run covers and never what any of its outputs is called. Folding a selector
into the identity would move an identifier while the output stays where it is,
which is a cache miss costing a recompute for nothing.

:class:`PhasedTrackerOpParams` adds what a tool that runs in several subprocess
phases needs on top: every parameter its subclass declares names the phases that
consume it, checked when the subclass is created.
"""

from __future__ import annotations

from typing import Annotated

from mosaic.core.pipeline.markers import Phase
from mosaic.core.pipeline.types import HASH_EXCLUDE, Declared, OpParams

__all__ = ["PhasedTrackerOpParams", "TrackerOpParams"]

_CONVERT_TO_TRACKS_DESCRIPTION = (
    "Convert the tool's native output into a standardized tracks table once "
    "tracking finishes, instead of leaving the output where the tool wrote it."
)

_IDLE_TIMEOUT_DESCRIPTION = (
    "Kill a phase after this many seconds without a line of output from the "
    "tool, which is how a hung run is told apart from a slow one."
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


class PhasedTrackerOpParams(TrackerOpParams):
    """Base for a tracker whose phases each consume part of the parameters.

    Every field a subclass adds names the phases that consume it, and this class
    refuses the subclass at class creation otherwise.
    :func:`~mosaic.core.pipeline.markers.phase_fields` returns an empty tuple
    both for a phase no field names and for a model that declares no
    :class:`~mosaic.core.pipeline.markers.Phase` at all, so a model that lost
    its markers projects an empty settings dictionary and the tool runs on its
    own defaults.

    The check reads ``cls.__annotations__``, which holds a subclass's own fields
    alone. The scope and execution fields inherited from
    :class:`~mosaic.core.pipeline.types.OpParams` and :class:`TrackerOpParams`
    reach no phase of the tool and declare none.
    """

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: object) -> None:
        """Refuse a subclass field that names no phase.

        ``model_fields`` is complete here, with its ``Annotated`` metadata
        intact. A plain ``__init_subclass__`` runs before either is true.
        """
        super().__pydantic_init_subclass__(**kwargs)
        unphased = sorted(
            name
            for name, info in cls.model_fields.items()
            if name in cls.__annotations__
            and not any(isinstance(marker, Phase) for marker in info.metadata)
        )
        if unphased:
            refused = (
                f"{cls.__name__} declares {unphased} without a Phase. Add "
                f"Phase(...) to each field's Annotated, naming the phases that "
                f"consume it."
            )
            raise TypeError(refused)
