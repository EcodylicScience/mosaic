"""Gate a read's measured facts against a declared target before injection.

Two independent axes exist, and each target consults only its own axis, never
the other and never "any reason at all":

* ``"analysis"`` raises when ``verdict.analysis_transcode == "required"``.
* ``"playback"`` raises when ``verdict.stream_transcode == "required"``.
* ``"raw"`` warns and proceeds when ``verdict.analysis_transcode ==
  "required"``.

``analysis_transcode`` has no non-null value other than ``"required"``. The
playback arm gates on ``stream_transcode == "required"`` and nothing weaker: a
verdict of ``"recommended"`` describes an ordinary healthy high-bitrate
original, not a defect, and raising on it would reject files that already play
back correctly. A ``"raw"`` read consults the analysis axis and never the
stream axis, because a raw read is a frame read: the analysis axis is what
says whether frame reads are reliable, while the stream axis is about browser
delivery, which a raw read is not doing.

A ``"raw"`` target is a declaration of intent, not an exemption from the gate:
it still derives a verdict and still reports when the file is not clean, only
as a warning rather than a raise. A deliberate look at an unprocessed original
is expected to warn; the same warning from an image store chunk read is a sign
that the store itself is anomalous.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from mosaic_media import (
    CHROME_149,
    HARD_STREAM_REASONS,
    MediaFacts,
    MediaProbeError,
    derive,
    probe_media,
)
from mosaic_media.transcode import Target

from mosaic.media_probe_config import media_thresholds

type ReadTarget = Target | Literal["raw"]

_UNFIT_ARTICLE: dict[Target, str] = {
    "analysis": "an analysis",
    "playback": "a playback",
}


def _unfit_message(path: Path, target: Target, reasons: Sequence[str]) -> str:
    """The raise message for a transcode target whose measured verdict is unfit."""
    article = _UNFIT_ARTICLE[target]
    listing = ", ".join(reasons)
    return (
        f"{path} is not fit for {article} read: its measured verdict requires "
        f"{article} transcode ({listing}); transcode it for {target} first"
    )


def _raw_warning_message(path: Path, reasons: Sequence[str]) -> str:
    """The warning message for a raw read whose measured verdict has an analysis
    defect."""
    listing = ", ".join(reasons)
    return (
        f"raw read of {path}: its measured verdict requires an analysis transcode "
        f"({listing}), so frames may be misindexed or misoriented. Expected for a "
        f"deliberate read of an unprocessed original; for an image store chunk it "
        f"means the store itself is anomalous."
    )


def verified_read_facts(
    paths: Path | str | Sequence[Path | str],
    facts: MediaFacts | Sequence[MediaFacts] | None,
    target: ReadTarget,
) -> list[MediaFacts]:
    """The MediaFacts a reader should be given for *paths*, gated for *target*.

    Returns *facts* verbatim wherever supplied; probes only a path whose facts
    are absent. Every returned element has passed the gate for *target*: a
    transcode target (``"analysis"`` or ``"playback"``) raises
    :class:`~mosaic_media.MediaProbeError` naming the transcode remedy when the
    corresponding axis of the derived verdict requires one; ``"raw"`` warns
    instead of raising.

    A :class:`~mosaic_media.MediaProbeError` raised by the probe itself (no
    video stream, or ffprobe failure) propagates unchanged for every target,
    including ``"raw"``: a reader could not decode such a file either, so
    swallowing the error would trade a clear failure for an obscure one.

    *paths* and, when not ``None``, *facts* are parallel sequences of equal
    length; a single path and a single :class:`~mosaic_media.MediaFacts` are
    each accepted directly. A length mismatch raises :class:`ValueError` before
    any probe runs, so a mismatched call cannot burn a full-file scan first.
    """
    if isinstance(paths, (str, Path)):
        resolved_paths = [Path(paths).expanduser().resolve()]
    else:
        resolved_paths = [Path(path).expanduser().resolve() for path in paths]

    if isinstance(facts, MediaFacts):
        facts_list: list[MediaFacts] | None = [facts]
    elif facts is not None:
        facts_list = list(facts)
    else:
        facts_list = None

    if facts_list is not None and len(facts_list) != len(resolved_paths):
        message = (
            f"facts length {len(facts_list)} does not match path count "
            f"{len(resolved_paths)}"
        )
        raise ValueError(message)

    verified: list[MediaFacts] = []
    for index, path in enumerate(resolved_paths):
        supplied = facts_list[index] if facts_list is not None else None
        measured = supplied if supplied is not None else probe_media(path)
        verdict = derive(measured, CHROME_149, media_thresholds())

        if target == "raw":
            # A raw read checks only the analysis axis: it is a frame read, not
            # a browser-delivery read, so a stream reason there is noise.
            if verdict.analysis_transcode == "required":
                reasons = sorted(verdict.analysis_reasons)
                message = _raw_warning_message(path, reasons)
                warnings.warn(message, stacklevel=2)
        elif target == "analysis":
            # analysis_transcode has no non-null value other than "required".
            if verdict.analysis_transcode == "required":
                reasons = sorted(verdict.analysis_reasons)
                message = _unfit_message(path, target, reasons)
                raise MediaProbeError(message)
        elif target == "playback":
            # Gate on "required" only: "recommended" describes an ordinary
            # healthy high-bitrate original, not a defect.
            if verdict.stream_transcode == "required":
                reasons = sorted(verdict.stream_reasons & HARD_STREAM_REASONS)
                message = _unfit_message(path, target, reasons)
                raise MediaProbeError(message)
        else:
            # An untyped caller can reach this with a target outside the three
            # literals; returning ungated facts here would be exactly the
            # silent-degrade shape this gate exists to remove.
            message = (
                f"unknown read target {target!r} for {path}; expected one of "
                f'"analysis", "playback", "raw"'
            )
            raise ValueError(message)

        verified.append(measured)

    return verified
