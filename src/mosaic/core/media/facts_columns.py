"""Mapping between MediaFacts + Verdict and the media index CSV row.

The media index persists the full MediaFacts as JSON so downstream opens inject
it verbatim instead of re-probing. Flat columns duplicate a few fields for the
untyped pandas readers and for routing, which must not parse JSON.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, TypedDict

from mosaic_media import MediaFacts, Verdict
from mosaic_media.transcode import Target

if TYPE_CHECKING:
    import pandas as pd

FLAT_FACTS_COLUMNS: list[str] = [
    "frame_count",
    "analysis_transcode",
    "stream_transcode",
    "analysis_derivative_path",
    "playback_derivative_path",
    "source_path",
]
FACTS_JSON_COLUMN = "media_facts"
FACTS_COLUMNS: list[str] = [*FLAT_FACTS_COLUMNS, FACTS_JSON_COLUMN]

MEDIA_INDEX_COLUMNS: list[str] = [
    "name",
    "group",
    "sequence",
    "group_safe",
    "sequence_safe",
    "abs_path",
    "size_bytes",
    "mtime_iso",
    "width",
    "height",
    "fps",
    "codec",
    "media_type",
    *FACTS_COLUMNS,
    "video_order",
]

# The analysis and playback transcode verdicts are independent: each target gets
# its own derivative and its own forward-link column, so a playback transcode can
# never overwrite the analysis routing (or vice versa).
_DERIVATIVE_COLUMN_BY_TARGET: dict[Target, str] = {
    "analysis": "analysis_derivative_path",
    "playback": "playback_derivative_path",
}


def derivative_column_for_target(target: Target) -> str:
    """Return the media-index forward-link column for a transcode *target*."""
    return _DERIVATIVE_COLUMN_BY_TARGET[target]


class MediaFactsRow(TypedDict):
    """The flat verdict cells plus the injectable MediaFacts JSON cell."""

    frame_count: int
    analysis_transcode: str
    stream_transcode: str
    analysis_derivative_path: str
    playback_derivative_path: str
    source_path: str
    media_facts: str


class ProbeMetadata(MediaFactsRow):
    """A full media-index probe row: display metadata plus the facts cells."""

    width: int
    height: int
    fps: float
    codec: str


def facts_to_row(facts: MediaFacts, verdict: Verdict) -> MediaFactsRow:
    """Flatten *facts* and *verdict* into the columns persisted in the media index."""
    return {
        "frame_count": facts.frame_count,
        "analysis_transcode": verdict.analysis_transcode or "",
        "stream_transcode": verdict.stream_transcode or "",
        "analysis_derivative_path": "",
        "playback_derivative_path": "",
        "source_path": "",
        "media_facts": json.dumps(dataclasses.asdict(facts)),
    }


def row_to_facts(row: Mapping[str, object]) -> MediaFacts:
    """Reconstruct the full :class:`MediaFacts` from a media index row."""
    payload = row[FACTS_JSON_COLUMN]
    if not isinstance(payload, str) or not payload:
        raise KeyError(FACTS_JSON_COLUMN)
    return MediaFacts(**json.loads(payload))


def row_facts_or_none(row: Mapping[str, object]) -> MediaFacts | None:
    """Reconstruct stored facts from a media index row, or ``None`` if absent.

    A row with no ``media_facts`` cell (or a NaN one from a CSV round-trip of an
    empty cell) has no stored facts; the caller's reader then probes the file.
    Callers holding a pandas ``Series`` should materialize it as a ``{str: value}`` mapping first
    (see :func:`row_mapping`), or call :func:`series_facts_or_none` directly.
    """
    try:
        return row_to_facts(row)
    except (KeyError, TypeError, ValueError):
        return None


def row_mapping(row: "pd.Series") -> dict[str, object]:
    """Convert a media-index Series row to a plain ``{str: value}`` mapping.

    :func:`row_to_facts` / :func:`row_facts_or_none` expect a
    ``Mapping[str, object]``; a pandas ``Series`` is keyed by an untyped index,
    so materialize it as a string-keyed dict first.
    """
    return {str(key): value for key, value in row.items()}


def series_facts_or_none(row: "pd.Series") -> MediaFacts | None:
    """Reconstruct stored facts directly from a media-index ``Series`` row.

    Combines :func:`row_mapping` and :func:`row_facts_or_none` for the common
    case of a caller holding a pandas row rather than an already-built mapping.
    """
    return row_facts_or_none(row_mapping(row))


def store_facts(
    width: int,
    height: int,
    fps: float,
    frame_count: int,
    codec: str,
    duration: float,
) -> MediaFacts:
    """Build a full :class:`MediaFacts` for an imgstore, whose reader needs no
    transcode negotiation: coded dimensions with no rotation, constant frame
    rate, single progressive video stream, no audio. Fields with no imgstore
    equivalent (declared_*, moov/gop layout, color/pixel-format metadata) are
    set to neutral values matching their declared type.
    """
    return MediaFacts(
        container="imgstore",
        codec_name=codec,
        pixel_format="",
        color_range="",
        color_primaries="",
        color_transfer="",
        width=width,
        height=height,
        rotation_degrees=0,
        square_pixels=True,
        progressive=True,
        has_audio=False,
        video_stream_count=1,
        duration=duration,
        fps=fps,
        frame_count=frame_count,
        start_time=0.0,
        constant_frame_rate=True,
        max_instantaneous_fps=None,
        declared_duration=duration,
        declared_fps=fps,
        declared_frame_count=frame_count,
        moov_at_start=None,
        max_keyframe_interval_frames=0,
        max_gop_bytes=0,
        timing_measured=True,
    )
