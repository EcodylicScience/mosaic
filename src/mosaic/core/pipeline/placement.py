"""Publish a table tracked on a media variant in its entry's source space.

A tracker run on a variant reports positions in the variant's pixels and frames
on the variant's frame axis. Every other table is in source space, the pixel grid
and frame axis of the entry media. :func:`to_source_space` maps the table there.
It shifts each position by the variant's offset, maps each frame through the
variant's frame map, and retimes the table on the entry's source timeline. A
column that the variant changed and that the table cannot recompute, such as a
speed or a border distance, is dropped and named.

Every column is classified by the quantity that it measures
(:func:`classify_column`). :func:`to_source_space` refuses a table with an
unclassified numeric column under any mapping but the identity. Such a column may
contain a variant pixel or frame, and published unmapped it reads as a plausible
source-space value. This follows
:func:`~mosaic.core.track_library.trex.unscale_to_pixels`, which refuses an
unclassified column when it has a unit to convert.

The retiming rule (:func:`retime`) also times a table tracked on the join of an
entry's clips (:func:`retime_joined_frame`), whose clips may have been recorded at
different rates. :class:`EntryAxis` chooses between the two for a tool's table,
and gives the length of the entry's frame axis that the table is meant to span.
The shared bridge applies it to every producer's table.

A divided recording numbers ``frame`` and ``time`` across the whole recording.
:func:`shift_to_recording_axis` moves a table from its sequence's axis onto the
recording's by the sequence's first frame, after every other mapping.
:class:`EntryAxis` applies it to a tool's table and :func:`place_converted` to a
converter's. A tracker or inference table is moved by
``ResolvedMedia.first_frame``: in a declared continuous group, the media frames of
the sequences before it, and 0 otherwise. A converter's table keeps the frame
numbers of its source, and :func:`place_converted` is passed 0.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from mosaic_media import MediaFacts

from mosaic.core.media.preprocess.geometry import Placement
from mosaic.core.media.timeline import ConcatenatedTimeline, concatenated_timeline
from mosaic.core.track_converter import EntryHints
from mosaic.core.track_library.helpers import column_array, column_names
from mosaic.core.track_library.trex import (
    PER_SECOND_FIELDS,
    PER_SECOND_SQUARED_FIELDS,
    base_field,
)

__all__ = [
    "EXTENT_DEPENDENT_BASES",
    "INVARIANT_BASES",
    "RATE_DEPENDENT_BASES",
    "ColumnKind",
    "EntryAxis",
    "MappedTable",
    "SourceMapping",
    "UnclassifiedColumnError",
    "classify_column",
    "place_converted",
    "retime",
    "retime_joined_frame",
    "shift_to_recording_axis",
    "to_source_space",
]

type ColumnKind = Literal[
    "x_position",
    "y_position",
    "frame",
    "retimed",
    "dropped_by_retiming",
    "rate_dependent",
    "extent_dependent",
    "invariant",
]
"""The quantity that a column measures, which decides a mapping's effect on it."""

_X_POSITION_BASES: Final = frozenset({"X", "bbox_x1", "bbox_x2", "midline_x"})
"""Base fields with an x position in the image that the tracker was given.

``midline_x`` is one, because TRex computes each midline point as the blob
position plus the midline offset.
"""

_Y_POSITION_BASES: Final = frozenset({"Y", "bbox_y1", "bbox_y2", "midline_y"})
"""Base fields with a y position in the image that the tracker was given."""

_FRAME_BASES: Final = frozenset({"frame", "frames", "tracklet_start"})
"""Base fields with a frame number on the tracker's frame axis.

``frames`` is TRex's name for the frame numbers of an export that is not the
per-individual table. When a table contains one, it names frames as ``frame``
does and is mapped the same way. ``tracklet_start`` is the first frame of the row's
TRex tracklet, and is NA on a row no tracklet covers.
"""

_RETIMED_BASES: Final = frozenset({"time", "frame_rate"})
"""Base fields that :func:`retime` recomputes from the source timeline."""

_DROPPED_BY_RETIMING_BASES: Final = frozenset({"timestamp"})
"""Base fields that :func:`retime` drops, minted from a frame index and one rate."""

_TIME_UNITS_PER_SECOND: Final[Mapping[str, float]] = MappingProxyType(
    {"time": 1.0, "timestamp": 1e6}
)
"""Base fields with a frame's time, by their units per second.

``time`` is in seconds and TRex's ``timestamp`` in microseconds. Each is on the
same axis as ``frame``. A shift of the frame axis shifts both.
"""

RATE_DEPENDENT_BASES: Final[frozenset[str]] = (
    PER_SECOND_FIELDS | PER_SECOND_SQUARED_FIELDS
)
"""Base fields that TRex computes per second, and therefore against one frame rate.

TRex differentiates over each frame's time, which it derives from the frame
index and the one rate that the file is labeled at. TRex's output annotations give
``SPEED``, ``SPEED_SMOOTH`` and ``SPEED_OLD`` in cm/s and ``ACCELERATION`` and
``ACCELERATION_SMOOTH`` in cm/s2, and ``ACCELERATION`` is the length of the
second derivative taken over that time. The TRex converter declares the two
sets, because it rescales them to the media's measured rate.

The set is matched on the base name, which covers every ``#`` variant.
``SPEED``, ``SPEED#wcentroid`` and ``SPEED#pcentroid`` are one quantity under
three estimators, each computed against the one labeled rate.

The list does not reuse the converter's ``DERIVED_COLUMNS``. That set also
contains ``ANGLE`` and the ``#wcentroid`` positions, which are an angle and a
coordinate. Neither depends on a rate, and ``X``/``Y`` are read from
``X#wcentroid``, which the mapping shifts and keeps.
"""

EXTENT_DEPENDENT_BASES: Final[frozenset[str]] = frozenset(
    {"BORDER_DISTANCE", "video_size"}
)
"""Base fields measured against the extent of the image that the tracker was given.

``BORDER_DISTANCE`` is the distance to that image's border, which TRex exports by
default, and ``video_size`` is that image's size. A crop changes both, and the
table cannot recompute either.
"""

INVARIANT_BASES: Final[frozenset[str]] = frozenset(
    {
        # Distances between two points of one frame.
        "NEIGHBOR_DISTANCE",
        "midline_segment_length",
        "segment_length",
        "midline_length",
        "midline_lengths",
        # Angles, and a midline expressed relative to the body.
        "ANGLE",
        "ORIENTATION",
        "MIDLINE_OFFSET",
        "normalized_midline",
        # Sizes and spreads of one blob, in pixels, which a crop does not rescale.
        "num_pixels",
        "outline_size",
        "variance",
        # Identities, classes, probabilities and flags.
        "id",
        "group",
        "sequence",
        "missing",
        "qr_id",
        "tracklet_id",
        "blobid",
        "blob_id",
        "detection_p",
        "visual_identification_p",
        "detect_type",
        "detect_format",
        "source_track_id",
        "det_conf",
        "det_cls",
        "confidence",
        "class_id",
        # The scale TRex applied, which a crop does not change.
        "cm_per_pixel",
    }
)
"""Base fields that a crop, trim, decimation or relabeled rate leaves unchanged.

Each is a measurement within one frame and does not contain a position, frame
number or rate.
"""

_BASE_KINDS: Final[tuple[tuple[frozenset[str], ColumnKind], ...]] = (
    (_X_POSITION_BASES, "x_position"),
    (_Y_POSITION_BASES, "y_position"),
    (_FRAME_BASES, "frame"),
    (_RETIMED_BASES, "retimed"),
    (_DROPPED_BY_RETIMING_BASES, "dropped_by_retiming"),
    (RATE_DEPENDENT_BASES, "rate_dependent"),
    (EXTENT_DEPENDENT_BASES, "extent_dependent"),
    (INVARIANT_BASES, "invariant"),
)

_PREFIX_KINDS: Final[tuple[tuple[str, ColumnKind], ...]] = (
    ("poseX", "x_position"),
    ("poseY", "y_position"),
    ("poseP", "invariant"),
)
"""Keypoint columns, which are named by a prefix and a keypoint index."""


class UnclassifiedColumnError(ValueError):
    """A table mapped to source space has an unclassified numeric column."""


@dataclass(frozen=True, slots=True)
class SourceMapping:
    """Pair a variant file's placement in its entry media with that media's timing.

    The two are one argument because neither maps a table alone. The placement
    gives the source frame of each file frame, and the timeline gives the time at
    which each source frame was recorded.

    Attributes:
        placement: The variant file's placement in the entry media.
        timeline: The entry media's clips as one frame axis and one time axis.
    """

    placement: Placement
    timeline: ConcatenatedTimeline

    def __post_init__(self) -> None:
        """Refuse a timeline that is not the one that the placement maps into.

        Raises:
            ValueError: If the timeline has a different number of frames from the
                source that the placement describes.
        """
        if self.placement.source_frame_count != self.timeline.total_frames:
            message = (
                f"the placement maps into a source of "
                f"{self.placement.source_frame_count} frames, and the timeline "
                f"has {self.timeline.total_frames}"
            )
            raise ValueError(message)

    @property
    def true_rate(self) -> float | None:
        """The rate that the file's frames were recorded at, or ``None`` if it varies.

        On a timeline whose clips share one rate, consecutive file frames are
        ``frames.step`` source frames apart. The true rate is that rate divided by
        the step.
        """
        if not self.timeline.uniform_rate:
            return None
        return self.timeline.segments[0].fps / self.placement.frames.step

    @property
    def labeled_rate_is_true(self) -> bool:
        """Whether the file is labeled at the rate that its frames were recorded at."""
        true_rate = self.true_rate
        return true_rate is not None and math.isclose(
            self.placement.fps, true_rate, rel_tol=1e-6
        )

    @property
    def is_identity(self) -> bool:
        """Whether a table tracked on the file is already in source space.

        The file must be the whole source, unmoved and every frame kept, labeled
        at its true rate, and the source must be a single clip. A table tracked on
        several clips joined into one file is timed at one rate, which the
        timeline corrects.
        """
        return (
            self.placement.is_identity
            and len(self.timeline.segments) == 1
            and self.labeled_rate_is_true
        )


@dataclass(frozen=True, slots=True)
class MappedTable:
    """Pair a mapped table with the columns that the mapping dropped.

    Attributes:
        frame: The mapped table.
        dropped: The names of the dropped columns, in the input table's column
            order.
    """

    frame: pd.DataFrame
    dropped: tuple[str, ...]


def classify_column(name: str) -> ColumnKind | None:
    """Return the kind of quantity that column *name* measures, or ``None``.

    ``None`` means that the column is unclassified. Names are matched after
    :func:`~mosaic.core.track_library.trex.base_field` strips a TRex ``#`` suffix
    and a flattening index. For example, ``X#wcentroid`` is an x position and
    ``video_size_0`` is extent-dependent. Keypoint columns are matched by prefix.
    """
    base = base_field(name)
    for prefix, kind in _PREFIX_KINDS:
        if base.startswith(prefix):
            return kind
    for bases, kind in _BASE_KINDS:
        if base in bases:
            return kind
    return None


def retime(
    df: pd.DataFrame, timeline: ConcatenatedTimeline, frames: npt.NDArray[np.int64]
) -> MappedTable:
    """Return *df* timed on *timeline*, without the columns that one rate made inexact.

    ``time`` becomes each row's time on the timeline and ``frame_rate``, where
    present, the rate in force at that row. ``timestamp`` is dropped, because a
    tracker mints it from its frame index and one rate instead of measuring it. On
    a timeline whose clips differ in rate, the rate-dependent columns are dropped
    as well, because they were computed against one rate and some clips were
    recorded at another. They are not rescaled, because rescaling by the ratio of
    rates is exact only for a plain first difference, and a tracker may smooth
    across a window that spans a clip boundary.

    Args:
        df: The table to retime.
        timeline: The source timeline.
        frames: The source frame of each row of *df*, in row order.

    Returns:
        A new table, and the names of the columns dropped, in table order.
    """
    names = column_names(df)
    out = df.copy()
    out["time"] = timeline.times(frames)
    if "frame_rate" in names:
        # Each row gets the rate in force at its frame. A per-file rate is incorrect
        # for the rows of every clip recorded at another rate.
        out["frame_rate"] = timeline.rates(frames)

    doomed: set[ColumnKind] = {"dropped_by_retiming"}
    if not timeline.uniform_rate:
        doomed.add("rate_dependent")
    dropped = tuple(name for name in names if classify_column(name) in doomed)
    return MappedTable(out.drop(columns=list(dropped)) if dropped else out, dropped)


def retime_joined_frame(
    df: pd.DataFrame, timeline: ConcatenatedTimeline
) -> MappedTable:
    """Return *df*, tracked on the join of *timeline*'s clips, timed by the clips.

    A tool reads a join, or a list of clips, at one frame rate. TRex takes it from
    the first clip and never compares it with the others, and its ``time`` is a
    frame index divided by that rate, because it loads no timestamps for a video
    file. One real session measures 30 fps, then 29.95, then 31 across seventeen
    clips. The error accumulates across the session, and every per-second
    quantity is computed against the wrong denominator. :func:`retime` puts
    ``time`` on the clips' own measured rates and drops what one rate spoiled.

    ``frame`` is left as the tool numbered it. On a join that is the entry's
    frame, because the join holds the clips' frames back to back. A tool that
    reads fewer frames than a clip holds numbers its rows early, and
    :meth:`ConcatenatedTimeline.times` then places them in the wrong clip. This
    function does not correct that. The bridge reports the shortfall, comparing
    the frames the tool read with :attr:`EntryAxis.media_frames`.

    Args:
        df: The table, with the join's ``frame``.
        timeline: The entry's clips, in the order they were joined.

    Returns:
        The retimed table and the names of the columns dropped from it. A
        single-segment timeline, or a table without ``frame``, returns *df*
        unchanged: one clip has one rate, and nothing was mistimed.
    """
    if len(timeline.segments) < 2 or "frame" not in column_names(df):
        return MappedTable(df, ())
    frames = column_array(df, "frame").astype(np.int64, copy=False)
    return retime(df, timeline, frames)


@dataclass(frozen=True, slots=True)
class EntryAxis:
    """Place a tool's table on the frame and time axes of its entry's media.

    A tool reads one file: the entry's one clip, the join of its clips, or a media
    variant's file. The localizer reads the clips themselves on the same axis as
    their join. The tool's table is on the axis of what it read, and
    :meth:`place` moves it onto the entry's. Build one with
    :meth:`of_entry_media` or :meth:`of_variant`.

    Attributes:
        clips: The facts of the entry's clips, in order, when the tool read the
            entry media. Empty when it read a variant, or when the facts are
            absent, which leaves the table as the tool timed it.
        mapping: The placement of the variant's file in the entry media, or
            ``None`` when the tool read the entry media.
        windowed: Whether the run read fewer than every frame of its file, under
            a frame window. A variant cannot be read under one.
        first_frame: The sequence's first frame on its recording's frame axis,
            which :meth:`place` adds last. :attr:`media_frames` stays local to
            the sequence.
        time_offset: The sequence's first frame on its recording's time axis,
            in seconds, which :meth:`place` adds to each time column. ``None``
            adds *first_frame* at the media's rate.
    """

    clips: tuple[MediaFacts, ...] = ()
    mapping: SourceMapping | None = None
    windowed: bool = False
    first_frame: int = 0
    time_offset: float | None = None

    def __post_init__(self) -> None:
        """Refuse an axis that is both the entry media and a variant.

        Raises:
            ValueError: If both *clips* and *mapping* are given, or if
                *first_frame* is negative.
        """
        if self.clips and self.mapping is not None:
            message = "an entry axis reads the entry media or a variant, not both"
            raise ValueError(message)
        _refuse_negative_first_frame(self.first_frame)

    @classmethod
    def of_entry_media(
        cls,
        clips: Sequence[MediaFacts],
        *,
        windowed: bool,
        first_frame: int = 0,
        time_offset: float | None = None,
    ) -> EntryAxis:
        """Return the axis of a tool that read the entry media: a clip, or a join."""
        return cls(
            clips=tuple(clips),
            windowed=windowed,
            first_frame=first_frame,
            time_offset=time_offset,
        )

    @classmethod
    def of_variant(
        cls,
        mapping: SourceMapping,
        *,
        first_frame: int = 0,
        time_offset: float | None = None,
    ) -> EntryAxis:
        """Return the axis of a tool that read a media variant's file."""
        return cls(mapping=mapping, first_frame=first_frame, time_offset=time_offset)

    @property
    def reads_every_frame(self) -> bool:
        """Whether the tool was meant to read every frame of the entry media.

        ``False`` under a frame window, and for a variant that trims or decimates
        its source. :attr:`media_frames` is known only where this holds.
        """
        if self.windowed:
            return False
        return self.mapping is None or self.mapping.placement.is_frame_identity

    @property
    def media_frames(self) -> int | None:
        """Return how many frames the tool should have read, or ``None``.

        The frames of the entry's clips, summed, when the tool read the entry media
        without a frame window, whether it read one clip, several or their join.
        The frames of the variant's source, when the tool read a variant that keeps
        every source frame. The bridge records it beside the frames the tool
        reported reading, and the two differ where the tool lost frames, as TRex
        does at the end of every file it opens.

        ``None`` when the tool read less on purpose (:attr:`reads_every_frame`).
        ``None`` too when a clip's frame count is unknown, because a partial sum
        would read as a measurement.
        """
        if not self.reads_every_frame:
            return None
        if self.mapping is not None:
            counts = [segment.frame_count for segment in self.mapping.timeline.segments]
        else:
            counts = [int(clip.frame_count) for clip in self.clips]
        if not counts or min(counts) <= 0:
            return None
        return sum(counts)

    @property
    def media_rate(self) -> float | None:
        """The frame rate of the media's first clip, ``None`` when not measured."""
        if self.mapping is not None:
            rate = self.mapping.timeline.segments[0].fps
        elif self.clips:
            rate = float(self.clips[0].fps)
        else:
            return None
        return rate if math.isfinite(rate) and rate > 0 else None

    def place(self, df: pd.DataFrame) -> MappedTable:
        """Return *df* on the entry's axes, and the columns dropped from it.

        A variant's table is mapped to source space (:func:`to_source_space`). A
        table from the entry's several clips is timed by the clips
        (:func:`retime_joined_frame`). A table from one clip is kept as it is.
        Last, the table is moved onto its recording's axis by
        :attr:`first_frame` and :attr:`time_offset`
        (:func:`shift_to_recording_axis`), at the rate of the media's first clip.

        Raises:
            UnclassifiedColumnError: As :func:`to_source_space` or
                :func:`shift_to_recording_axis` raises it.
            ValueError: As :func:`to_source_space` or
                :func:`shift_to_recording_axis` raises it, or when a clip reports
                no frame rate.
        """
        if self.mapping is not None:
            placed = to_source_space(df, self.mapping)
        elif len(self.clips) < 2:
            placed = MappedTable(df, ())
        else:
            placed = retime_joined_frame(df, concatenated_timeline(self.clips))
        if not self.first_frame and not self.time_offset:
            return placed
        shifted = shift_to_recording_axis(
            placed.frame,
            self.first_frame,
            self.media_rate,
            time_offset=self.time_offset,
        )
        return MappedTable(shifted, placed.dropped)


def to_source_space(df: pd.DataFrame, mapping: SourceMapping) -> MappedTable:
    """Return *df*, tracked on a variant file, as the same table in source space.

    - x and y positions are shifted by the placement's offset.
    - Each frame becomes its source frame, ``start + step * frame``. A frame at or
      past the file's frame count is mapped the same way and is not refused here,
      because a tool's frame count may differ from the file's by a frame or two.
    - The table is retimed on the source timeline (:func:`retime`) unless the
      file contains every source frame of a single clip at the true rate.
    - Rate-dependent columns are dropped when the file's labeled rate is not the
      rate that its frames were recorded at, which includes every timeline whose
      clips differ in rate.
    - Extent-dependent columns are dropped when the file is not the whole source
      image.
    - Every other classified column is unchanged, and so is every column that is
      not numeric: text, categories and flags.

    Args:
        df: The table as a tracker reported it on the variant file.
        mapping: The file's placement in the entry media, and the media's
            timeline.

    Returns:
        The mapped table and the names of the columns that it dropped, in table
        order. An identity mapping returns *df* itself and does not drop a column.

    Raises:
        UnclassifiedColumnError: If a numeric column is not classified. Every
            such column is named.
        ValueError: If a frame column contains a value that is not a finite whole
            number, other than NA in a nullable integer column such as
            ``tracklet_start``, or if the table must be retimed and has no
            ``frame`` column.
    """
    if mapping.is_identity:
        return MappedTable(df, ())

    names = column_names(df)
    kinds: dict[str, ColumnKind | None] = {
        name: classify_column(name) for name in names
    }
    _refuse_unclassified(
        df,
        kinds,
        "Cannot map this table to source space.",
        "Under a crop, trim or relabeled rate, such a column may contain a variant "
        "pixel or frame. Published unmapped, it puts a variant value in source "
        "space.",
    )

    placement = mapping.placement
    out = df.copy()
    source_frames: npt.NDArray[np.int64] | None = None
    for name in names:
        kind = kinds[name]
        if kind == "x_position":
            out[name] = _positions(df, name) + placement.offset_x
        elif kind == "y_position":
            out[name] = _positions(df, name) + placement.offset_y
        elif kind == "frame" and name == "frame":
            source_frames = placement.frames.source_frames(_frame_numbers(df, name))
            out[name] = source_frames
        elif kind == "frame":
            out[name] = _mapped_frame_column(df, name, placement.frames.source_frames)

    doomed: set[str] = set()
    # A relabeled rate retimes as well, because the tracker timed each frame by
    # the labeled rate and the frame was recorded at another time.
    if (
        not placement.is_frame_identity
        or len(mapping.timeline.segments) > 1
        or not mapping.labeled_rate_is_true
    ):
        if source_frames is None:
            message = (
                f"Cannot retime this table on the source timeline without a frame "
                f"column. Its columns are {names}"
            )
            raise ValueError(message)
        retimed = retime(out, mapping.timeline, source_frames)
        out = retimed.frame
        doomed.update(retimed.dropped)
    if not mapping.labeled_rate_is_true:
        doomed.update(name for name in names if kinds[name] == "rate_dependent")
    if not placement.is_spatial_identity:
        doomed.update(name for name in names if kinds[name] == "extent_dependent")

    dropped = tuple(name for name in names if name in doomed)
    remaining = [name for name in dropped if name in column_names(out)]
    return MappedTable(out.drop(columns=remaining) if remaining else out, dropped)


def shift_to_recording_axis(
    df: pd.DataFrame,
    first_frame: int,
    fps: float | None,
    *,
    time_offset: float | None = None,
) -> pd.DataFrame:
    """Return *df*, on its sequence's axes, on its recording's axes.

    Each frame column (``frame``, ``frames``, ``tracklet_start``, as
    :func:`classify_column` names them) gains *first_frame*, and each time column
    gains *time_offset* seconds, by default ``first_frame / fps``: ``time`` in
    seconds and TRex's ``timestamp`` in microseconds. NA in a nullable frame
    column stays NA. Every other column is unchanged. The offset may fall
    anywhere, inside a store chunk included.

    Args:
        df: The table on its sequence's frame axis, after any mapping to source
            space or retiming.
        first_frame: The sequence's first frame on the recording's axis.
        fps: The recording's frame rate, ``None`` when not measured.
        time_offset: The sequence's first frame on the recording's time axis,
            in seconds, which a division's rows record
            (``Dataset.division_facts``).
            ``None`` takes ``first_frame / fps``.

    Returns:
        The shifted table, or *df* itself when *first_frame* and *time_offset*
        are 0.

    Raises:
        UnclassifiedColumnError: If *first_frame* is not 0 and a numeric column
            is not classified, since it may contain a sequence frame. Every such
            column is named.
        ValueError: If *first_frame* is negative, if a frame column contains a
            value that is not a whole number, or if the table has a time column
            and neither *time_offset* nor *fps* is given.
    """
    _refuse_negative_first_frame(first_frame)
    if first_frame == 0 and not time_offset:
        return df
    names = column_names(df)
    kinds: dict[str, ColumnKind | None] = {
        name: classify_column(name) for name in names
    }
    _refuse_unclassified(
        df,
        kinds,
        "Cannot move this table onto its recording's frame axis.",
        "Such a column may contain a frame of the sequence, and left unshifted it "
        "names another moment of the recording.",
    )
    time_columns = {
        name: _TIME_UNITS_PER_SECOND[base_field(name)]
        for name in names
        if base_field(name) in _TIME_UNITS_PER_SECOND
    }
    if time_offset is None and fps is not None:
        time_offset = first_frame / fps
    if time_columns and time_offset is None:
        message = (
            f"Cannot move this table's time onto its recording's axis: the "
            f"sequence starts at frame {first_frame}, and the frame rate that "
            f"converts that frame to seconds is not measured."
        )
        raise ValueError(message)
    out = df.copy()
    for name in names:
        if kinds[name] == "frame":
            out[name] = _mapped_frame_column(
                df, name, lambda frames: frames + first_frame
            )
    if time_offset is not None:
        for name, units_per_second in time_columns.items():
            out[name] = (
                column_array(df, name).astype(np.float64)
                + time_offset * units_per_second
            )
    return out


def place_converted(
    df: pd.DataFrame, hints: EntryHints, media_frame_count: int | None
) -> pd.DataFrame:
    """Return a converter's table for *hints*' entry on its recording's axes.

    The table is on its sequence's frame axis, numbered from 0, and is moved by
    ``hints.first_frame`` and ``hints.time_offset``
    (:func:`shift_to_recording_axis`) at ``hints.fps``.

    Args:
        df: The converter's table.
        hints: The entry the table was converted for.
        media_frame_count: The frames of the entry's division, ``None`` when the
            entry has no division record. A table whose frames reach it is
            numbered on the recording's axis already, and shifting it would add
            the first frame twice.

    Returns:
        The placed table, or *df* itself when ``hints.first_frame`` is 0.

    Raises:
        ValueError: If a frame reaches *media_frame_count*, or as
            :func:`shift_to_recording_axis` raises it.
        UnclassifiedColumnError: As :func:`shift_to_recording_axis` raises it.
    """
    if media_frame_count is not None and "frame" in column_names(df) and len(df):
        last = int(np.nanmax(column_array(df, "frame").astype(np.float64)))
        if last >= media_frame_count:
            message = (
                f"the converted table of ({hints.group!r}, {hints.sequence!r}) "
                f"reaches frame {last}, and its division has {media_frame_count} "
                "frames. The table is numbered on the recording's frame axis "
                "already, and moving it by the division's first frame would count "
                "that frame twice. Convert the division's own frames, numbered "
                "from 0."
            )
            raise ValueError(message)
    return shift_to_recording_axis(
        df, hints.first_frame, hints.fps, time_offset=hints.time_offset
    )


def _refuse_negative_first_frame(first_frame: int) -> None:
    """Refuse a first frame before the recording's first frame, 0.

    Raises:
        ValueError: If *first_frame* is negative.
    """
    if first_frame < 0:
        message = (
            f"a sequence's first frame on its recording's axis is {first_frame}, "
            "before the recording's first frame, 0"
        )
        raise ValueError(message)


def _refuse_unclassified(
    df: pd.DataFrame, kinds: Mapping[str, ColumnKind | None], lead: str, why: str
) -> None:
    """Refuse *df* when a numeric column is not classified, naming every one.

    Raises:
        UnclassifiedColumnError: *lead*, the columns, and *why*.
    """
    unclassified = [
        name for name, kind in kinds.items() if kind is None and _is_numeric(df, name)
    ]
    if not unclassified:
        return
    verb = "are numeric columns" if len(unclassified) > 1 else "is a numeric column"
    message = (
        f"{lead} {unclassified} {verb} that the classification rules do not "
        "recognize as a position, a frame, a time or a value that a mapping "
        f"leaves unchanged. {why} Classify it in {__name__}."
    )
    raise UnclassifiedColumnError(message)


def _frame_numbers(frame: pd.DataFrame, name: str) -> npt.NDArray[np.int64]:
    """Return the frame column *name* as integers, refusing a value that is not whole.

    A tracker may store frame numbers as floats, and TRex pads a short array with
    NaN. Casting turns NaN or 0.5 into a plausible frame number. This function
    refuses a table with any value that is not a finite whole number.

    Raises:
        ValueError: If the column contains a NaN, an infinity or a fractional value.
    """
    values = column_array(frame, name)
    if not np.issubdtype(values.dtype, np.integer):
        numbers = values.astype(np.float64)
        whole = np.isfinite(numbers) & (numbers == np.trunc(numbers))
        count = int(np.count_nonzero(~whole))
        if count:
            message = (
                f"Cannot map the frame column {name!r} to source space. {count} of "
                f"its {len(numbers)} values are not whole frame numbers (NaN, "
                "infinite or fractional). A frame number cast from one names a "
                "source frame that the row was not tracked on."
            )
            raise ValueError(message)
    return values.astype(np.int64)


def _mapped_frame_column(
    frame: pd.DataFrame,
    name: str,
    to_frames: Callable[[npt.NDArray[np.int64]], npt.NDArray[np.int64]],
) -> npt.NDArray[np.int64] | pd.arrays.IntegerArray:
    """Return the frame column *name*, each frame mapped through *to_frames*.

    A nullable integer column maps its present values and keeps NA, which in such
    a column states that no frame applies. ``tracklet_start`` is one. Any other
    column must contain only whole frame numbers, as :func:`_frame_numbers`
    requires, because there a NaN may be TRex's padding.

    Raises:
        ValueError: If a column that is not a nullable integer contains a value
            that is not a finite whole number.
    """
    dtype = frame.dtypes[name]
    if isinstance(dtype, pd.api.extensions.ExtensionDtype) and (
        pd.api.types.is_integer_dtype(dtype)
    ):
        present = frame[name].notna().to_numpy(dtype=bool)
        values = frame[name].to_numpy(dtype=np.int64, na_value=0)
        return pd.arrays.IntegerArray(to_frames(values), ~present)
    return to_frames(_frame_numbers(frame, name))


def _positions(frame: pd.DataFrame, name: str) -> npt.NDArray[np.float64]:
    """Return the column *name* as float positions, keeping a missing one as NaN."""
    return column_array(frame, name).astype(np.float64)


def _is_numeric(frame: pd.DataFrame, name: str) -> bool:
    """Return whether column *name* is numeric. Text, categories and flags are not."""
    dtype = frame[name].dtype
    return pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(
        dtype
    )
