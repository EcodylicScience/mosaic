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
from typing import TYPE_CHECKING, Literal, TypedDict

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
    # encoder names the video encoder a derivative was produced by. Empty on an
    # original, which no encoder here wrote, and on a copy remux, which encodes
    # nothing. It is what tells a hardware encode from the CPU fallback the same
    # permission produces on a machine whose device cannot open the hardware
    # encoder: the codec column cannot, being a measured fact that reads "av1"
    # for both, and the recipe hash cannot, recording the recipe rather than the
    # machine. A transcode edge like the two above, so facts_to_row leaves it
    # empty and build_media_index_row overrides it from the TranscodeResult.
    "encoder",
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
    # How this row learned its (group, sequence) -- see AssignmentSource. Not a
    # measured fact and never hashed: it says how much to trust the identity the
    # other columns are keyed on, which is what lets the per-sequence composition
    # (item 4.4) refuse to compute a value for a guessed partition rather than
    # record a confident wrong one.
    "assignment_source",
    # Centimetres per pixel for this recording: how far apart two adjacent pixels
    # are in the world. A property of the camera and the rig, which is why it
    # lives beside the video rather than on a tracks table derived from it -- a
    # dataset may mix rigs, and a reconversion must not have to be told again.
    #
    # Deliberately NOT in MEDIA_NUMERIC_COLUMNS. It is text so that empty can
    # mean *uncalibrated*: as a number the empty cell would read as 0.0 or NaN,
    # and the first of those is a scale factor rather than an absence. Nothing
    # measures it -- no probe can -- so facts_to_row leaves it alone, exactly as
    # it does for source_video_uuid and recipe_hash.
    "cm_per_pixel",
]

AssignmentSource = Literal[
    "", "assigned", "scan-stem", "scan-keymap", "scan-imgstore-sync"
]
"""Where a media row's ``(group, sequence)`` came from.

- ``"assigned"`` -- the caller said so, through a ``MediaIndexScope``. The only
  cycle-free source: identity comes from outside and nothing derives it.
- ``"scan-stem"`` -- no track matched, so the file's own stem became its
  sequence. Stable only as long as the filename is.
- ``"scan-keymap"`` -- matched against a keymap built from ``tracks/index.csv``,
  which makes a *source* row's identity a function of a *derived* root. Item 4.7
  calls that backwards and it is why a composition over such a sequence is not
  well defined: converting more tracks would silently repartition media.
- ``"scan-imgstore-sync"`` -- a store's directory name, minus its camera-serial
  suffix, grouped with the other cameras sharing its ``sync_uuid``.
- ``""`` -- no claim. Either the row predates the column (the house idiom for an
  honest unknown) or it is a derivative under ``media/``, which takes its
  identity from the source row it was made from and has none of its own. Both
  are "do not draw a conclusion from this row's partition", which is the only
  thing a reader may do with an empty cell.
"""

# The analysis and playback transcode verdicts are independent: each target gets
# its own derivative and its own forward-link column, so a playback transcode can
# never overwrite the analysis routing (or vice versa).
_DERIVATIVE_COLUMN_BY_TARGET: dict[Target, str] = {
    "analysis": "analysis_derivative_path",
    "playback": "playback_derivative_path",
}


# The verdict column each target routes on, beside the link column above. Two
# tables rather than one, because the pair is not symmetric: a target has a
# verdict and a link, and their names share no rule -- ``analysis`` routes on
# ``analysis_transcode`` and ``playback`` on ``stream_transcode``. Written down
# here because it was spelled as a bare literal at the two call sites that read
# a verdict, so a third reader had to guess which one it wanted.
_VERDICT_COLUMN_BY_TARGET: dict[Target, str] = {
    "analysis": "analysis_transcode",
    "playback": "stream_transcode",
}

TRANSCODE_REQUIRED = "required"
"""The verdict value meaning this row cannot be read without a derivative."""


def derivative_column_for_target(target: Target) -> str:
    """Return the media-index forward-link column for a transcode *target*."""
    return _DERIVATIVE_COLUMN_BY_TARGET[target]


def verdict_column_for_target(target: Target) -> str:
    """Return the media-index verdict column a *target* routes on."""
    return _VERDICT_COLUMN_BY_TARGET[target]


def transcode_required(row: Mapping[str, object], target: Target) -> bool:
    """Does this row need a derivative before it can be read for *target*?"""
    return read_link_cell(row, verdict_column_for_target(target)) == TRANSCODE_REQUIRED


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
    encoder: str
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
        "encoder": "",
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
    video_uuid: str,
    identity_scheme: str,
) -> MediaFacts:
    """Build a full :class:`MediaFacts` for an imgstore, whose reader needs no
    transcode negotiation: coded dimensions with no rotation, constant frame
    rate, single progressive video stream, no audio. Fields with no imgstore
    equivalent (declared_*, moov/gop layout, color/pixel-format metadata) are
    set to neutral values matching their declared type.

    An imgstore is a directory of chunks with no elementary stream to hash, so
    it has no ``content_digest`` and -- since no ffprobe runs over it -- no
    prober version.

    The store supplies its own per-frame timestamps, so ``timing_source`` is
    ``"presentation"``. Stating the timing as supplied is also what keeps the
    ``variable_frame_rate`` verdict reachable: it fires only on a source whose
    timing the file supplied, so an ``"absent"`` store could never report uneven
    spacing however uneven it became.

    ``coded_reordering_depth`` and both delivery counts are zero on the
    reasoning that already zeroes ``max_keyframe_interval_frames`` and
    ``max_gop_bytes`` -- a store is a directory of chunks rather than one coded
    stream, so at store level there is no demultiplexer to flag a packet, no
    keyframe order for anything to precede, and no single reorder depth to
    state. A chunk read measures its own chunk and does not consult these.

    *video_uuid* and *identity_scheme* are **required** rather than defaulted,
    following this module's rule that every field is a claim the caller states
    rather than one it inherits. A store's uuid is a mint read from its
    metadata, not a measurement (open item O5), so the scheme is what tells a
    reader which kind of value it is holding -- and a defaulted pair would let a
    caller ship an unmarked mint by omission. ``""`` for both is the honest
    answer for a store whose metadata carries no uuid.
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
        discard_flagged_packets=0,
        leading_non_keyframe_frames=0,
        coded_reordering_depth=0,
        # One period, paired with the constant_frame_rate above: both take the
        # store's frame spacing as even. That is what the format intends rather
        # than what it guarantees -- a store's timestamps are wall-clock capture
        # times and can jitter -- so these two are the values to revisit when
        # store timing is measured rather than assumed.
        max_timestamp_gap_frame_periods=1.0,
        timing_source="presentation",
        video_uuid=video_uuid,
        content_digest="",
        identity_scheme=identity_scheme,
        prober_version="",
    )
