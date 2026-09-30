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

The retiming rule (:func:`retime`) is shared with the joined TRex conversion,
which times a session of clips recorded at different rates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from mosaic.core.media.preprocess.geometry import FrameMap, Placement
from mosaic.core.media.timeline import ConcatenatedTimeline
from mosaic.core.track_library.helpers import column_array, column_names
from mosaic.core.track_library.trex import base_field

__all__ = [
    "EXTENT_DEPENDENT_BASES",
    "INVARIANT_BASES",
    "RATE_DEPENDENT_BASES",
    "ColumnKind",
    "MappedTable",
    "SourceMapping",
    "UnclassifiedColumnError",
    "classify_column",
    "retime",
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

RATE_DEPENDENT_BASES: Final[frozenset[str]] = frozenset(
    {
        "VX",
        "VY",
        "AX",
        "AY",
        "SPEED",
        "SPEED_SMOOTH",
        "SPEED_OLD",
        "ACCELERATION",
        "ACCELERATION_SMOOTH",
        "ANGULAR_V",
        "ANGULAR_A",
    }
)
"""Base fields that TRex computes per second, and therefore against one frame rate.

TRex differentiates over each frame's time, which it derives from the frame
index and the one rate that the file is labeled at. TRex's output annotations give
``SPEED``, ``SPEED_SMOOTH`` and ``SPEED_OLD`` in cm/s and ``ACCELERATION`` and
``ACCELERATION_SMOOTH`` in cm/s2, and ``ACCELERATION`` is the length of the
second derivative taken over that time.

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
        "detection_id",
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
    kinds = {name: classify_column(name) for name in names}
    unclassified = [
        name for name in names if kinds[name] is None and _is_numeric(df, name)
    ]
    if unclassified:
        verb = "are numeric columns" if len(unclassified) > 1 else "is a numeric column"
        message = (
            f"Cannot map this table to source space. {unclassified} {verb} that "
            "the classification rules do not recognize as a position, a frame, a "
            "time or a value that a mapping leaves unchanged. Under a crop, trim or "
            "relabeled rate, such a column may contain a variant pixel or frame. "
            "Published unmapped, it puts a variant value in source space. Classify "
            f"it in {__name__}."
        )
        raise UnclassifiedColumnError(message)

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
            out[name] = _mapped_frame_column(df, name, placement.frames)

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
    frame: pd.DataFrame, name: str, frames: FrameMap
) -> npt.NDArray[np.int64] | pd.arrays.IntegerArray:
    """Return the frame column *name*, other than ``frame``, mapped to source frames.

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
        return pd.arrays.IntegerArray(frames.source_frames(values), ~present)
    return frames.source_frames(_frame_numbers(frame, name))


def _positions(frame: pd.DataFrame, name: str) -> npt.NDArray[np.float64]:
    """Return the column *name* as float positions, keeping a missing one as NaN."""
    return column_array(frame, name).astype(np.float64)


def _is_numeric(frame: pd.DataFrame, name: str) -> bool:
    """Return whether column *name* is numeric. Text, categories and flags are not."""
    dtype = frame[name].dtype
    return pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(
        dtype
    )
