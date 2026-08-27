"""The parameters every tracker op shares: what to run over, and how to run it.

Three op modules declare the same execution block before adding their tool's own
knobs, over the scope :class:`~mosaic.core.pipeline.types.OpParams` declares.

**Every field here is ``HASH_EXCLUDE``.** A tracker's ``run_id`` is minted from
its settings dictionary, which names no entry, so the scope selects which media
a run covers and never what any of its outputs is called. Folding a selector
into the identity would move an identifier while the output stays where it is,
which is a cache miss costing a recompute for nothing.
"""

from __future__ import annotations

from typing import Annotated

from mosaic.core.pipeline.types import HASH_EXCLUDE, Declared, OpParams

__all__ = ["TrackerOpParams"]

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
