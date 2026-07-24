"""Mapping between MediaFacts + Verdict and the media index CSV row.

The media index persists the full MediaFacts as JSON so downstream opens inject
it verbatim instead of re-probing. Flat columns duplicate a few fields for the
untyped pandas readers and for routing, which must not parse JSON.
"""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Mapping
from pathlib import Path
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
    # video_uuid pins coded content and exact timing and is unique per file --
    # the value to compare for identity or name a path from. content_digest pins
    # coded content only and is the duplicate pathway's index key. Distinct from
    # the sync_uuid column, which is a Motif recording id deliberately SHARED
    # across a recording's cameras: sync_uuid is safe to collide, video_uuid is
    # not.
    "video_uuid",
    "content_digest",
    # source_video_uuid links a derivative row to its source. A transcode edge,
    # not a measured fact, so facts_to_row leaves it empty and
    # build_media_index_row overrides it from the TranscodeResult.
    "source_video_uuid",
    # recipe_hash records the recipe a derivative was produced under, and is
    # empty on an original, which has no recipe. A transcode edge like
    # source_video_uuid, so facts_to_row leaves it empty and
    # build_media_index_row overrides it from the transcode job.
    "recipe_hash",
]
FACTS_JSON_COLUMN = "media_facts"
FACTS_COLUMNS: list[str] = [*FLAT_FACTS_COLUMNS, FACTS_JSON_COLUMN]

MEDIA_INDEX_COLUMNS: list[str] = [
    "name",
    "group",
    "sequence",
    "group_safe",
    "sequence_safe",
    # camera : recording :: id : sequence -- a within-sequence axis, empty ("")
    # for single-camera media. sync_uuid is the recording id (Motif
    # synchronizationuuid) that groups a recording's cameras; both are text.
    "camera",
    "sync_uuid",
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


def read_link_cell(row: Mapping[str, object], column: str) -> str:
    """A media-index cell as a trimmed string, absent forms collapsed to ``""``.

    Empty, the string ``"nan"``, and a float NaN all mean absent -- the last is
    what pandas yields for an empty CSV cell. Every identity and derivative-link
    read reaches this function, a ``Series`` row through the ``_media_cell``
    adapter in ``dataset``, so ``"nan"`` can never be mistaken for a real value
    in one place while another treats it literally.
    """
    value = row.get(column, "")
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def media_row_uuid(row: Mapping[str, object]) -> str:
    """A row's ``video_uuid``, or ``""`` when it carries none.

    The match key for a media row, on both sides of a comparison: it survives a
    rename, where a path does not. There is no path fallback -- a row carrying
    no uuid matches nothing and carries no derivative link. After a re-probe the
    only rows without one are imgstores, which never hold a derivative link.
    """
    return read_link_cell(row, "video_uuid")


def media_row_path_key(row: Mapping[str, object]) -> tuple[str, str]:
    """A row's path key: the two leaf components of its stored ``abs_path``.

    The parent directory name and the filename, stable whether the stored path
    is absolute or root-relative. ``mosaic`` matches media rows by ``video_uuid``
    alone; this key is the control plane's disambiguator, telling apart two
    byte-identical files that share one uuid under different names so their
    distinct derivative links stay separate. It does not survive a rename.
    """
    path = Path(read_link_cell(row, "abs_path"))
    return (path.parent.name, path.name)


def derivative_cell(row: Mapping[str, object], target: Target) -> str:
    """The raw link cell for *target*, or ``""`` when unregistered."""
    return read_link_cell(row, derivative_column_for_target(target))


def derivative_path_for_target(
    row: Mapping[str, object], target: Target, media_root: Path
) -> Path | None:
    """The registered derivative path for *target*, or ``None`` when unregistered.

    Anchors the stored cell under *media_root*. Owns the present-check and the
    anchoring -- the two things three call sites re-derived -- and nothing else:
    it reads no verdict, decides no error, and never touches the filesystem.
    """
    cell = derivative_cell(row, target)
    return media_root / cell if cell else None


class MediaFactsRow(TypedDict):
    """The flat verdict cells plus the injectable MediaFacts JSON cell."""

    frame_count: int
    analysis_transcode: str
    stream_transcode: str
    analysis_derivative_path: str
    playback_derivative_path: str
    source_path: str
    video_uuid: str
    content_digest: str
    source_video_uuid: str
    recipe_hash: str
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
        "video_uuid": facts.video_uuid,
        "content_digest": facts.content_digest,
        "source_video_uuid": "",
        "recipe_hash": "",
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

    An imgstore is a directory of chunks with no elementary stream to hash, so
    it carries no identity, and -- since no ffprobe runs over it -- no identity
    scheme or prober version. Its frame rate is constant by construction, so
    timing_measured is True despite no prober having run.
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
        video_uuid="",
        content_digest="",
        identity_scheme="",
        prober_version="",
    )
