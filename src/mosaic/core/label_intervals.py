"""Segment labels: the interval vocabulary for labels annotated as start and end times.

An annotation tool such as BORIS records a behaviour as a segment -- a start, an
end, a label -- and segments may overlap and nest: an animal can be *grooming*
inside a longer *resting* bout, and two annotators' segments can disagree. The
converted label file every reader opens holds one label per frame (``labels``),
and painting segments onto frames throws all of that away: an overlap is decided
by whichever segment was written last, a nested bout disappears into its parent,
and two back-to-back segments of one label become one bout.

This module keeps the segments. It defines a **sidecar convention** for a
converted label ``.npz``, the conversion from annotated seconds to frames, and the
one function that paints segments onto frames when a dense array is wanted, under
a policy the caller names.

**The sidecar.** A converted label file may carry, beside the arrays it already
holds:

========================  ==========  ==================================================
key                       dtype       meaning
========================  ==========  ==================================================
``seg_start_frame``       int64       first frame of the segment, inclusive
``seg_end_frame``         int64       frame after its last one, **exclusive**
``seg_label``             int32       the segment's label id, one of the file's
                                      ``label_ids``
``seg_weight``            float64     optional; a per-segment weight, such as an
                                      annotator's confidence
``seg_individual``        int64       optional; the track ``id`` the segment belongs to,
                                      ``-1`` for a scene-level segment
``seg_<anything else>``   any         optional; one value per segment, carried
                                      through unread
========================  ==========  ==================================================

Every ``seg_*`` array has one entry per segment, in the order the converter
listed them, and that order is kept: it is what breaks ties when segments are
painted (:func:`intervals_to_dense`). A half-open ``[start, end)`` interval is
what makes back-to-back segments share a boundary rather than a frame, so their
lengths add up.

**The sidecar is additive.** A file carrying it still carries the arrays every
existing reader indexes (``detect_label_format``,
``load_labels_for_feature_frames``): a dense ``labels`` array, painted from the
segments by the converter under a policy it chooses, or an event file's
``frames``, ``labels`` and ``individual_ids``. The keys were picked so that no
reader switching on a key sees a new one: those switch on ``label_format``,
``frames``, ``labels`` and ``individual_ids``, and no sidecar key is spelled like
any of them. A reader that does not know the sidecar reads the file exactly as
before; one that does (:func:`segments_from_npz`) reads the segments losslessly.

**Seconds become frames by the midpoint rule.** Frame ``i`` is the interval
``[i/fps, (i+1)/fps)``, and it belongs to the segment ``[start_s, end_s)`` that
holds its midpoint ``(i + 1/2)/fps``. Applied to both ends of every segment by
one function, that makes a shared boundary map to one frame boundary, so
back-to-back segments tile with neither a gap nor an overlap, and an exported
time rounded to the millisecond still lands on the frame boundary it was meant
for. See :func:`seconds_to_frames`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Final, Literal, get_args

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "OVERLAP_POLICIES",
    "SCENE_LEVEL",
    "SEGMENT_PREFIX",
    "SEG_END_FRAME",
    "SEG_INDIVIDUAL",
    "SEG_LABEL",
    "SEG_START_FRAME",
    "SEG_WEIGHT",
    "LabelSegments",
    "OverlapPolicy",
    "SegmentOverlapError",
    "intervals_to_dense",
    "label_vocabulary",
    "seconds_to_frames",
    "segments_from_npz",
]

SEGMENT_PREFIX: Final = "seg_"
"""Every sidecar key starts with this, and no key a label reader switches on does."""

SEG_START_FRAME: Final = "seg_start_frame"
SEG_END_FRAME: Final = "seg_end_frame"
SEG_LABEL: Final = "seg_label"
SEG_WEIGHT: Final = "seg_weight"
SEG_INDIVIDUAL: Final = "seg_individual"

SCENE_LEVEL: Final = -1
"""The ``seg_individual`` value of a segment about the scene rather than one track."""

_REQUIRED_KEYS: Final = (SEG_START_FRAME, SEG_END_FRAME, SEG_LABEL)
_NAMED_KEYS: Final = frozenset({*_REQUIRED_KEYS, SEG_WEIGHT, SEG_INDIVIDUAL})

OverlapPolicy = Literal["refuse", "shortest", "longest", "latest_start"]
"""How :func:`intervals_to_dense` decides a frame several segments cover."""

OVERLAP_POLICIES: Final[tuple[OverlapPolicy, ...]] = get_args(OverlapPolicy)

_INT32 = np.iinfo(np.int32)


class SegmentOverlapError(ValueError):
    """Two segments cover one frame and the policy was ``"refuse"``."""


def _as_vector(name: str, values: ArrayLike) -> NDArray[Any]:
    """*values* as a 1-D array, refusing anything of another rank."""
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    return array


_FRAME_HINT: Final = (
    " A frame number stored as a float has been rounded by someone, and the "
    "rule they used is not recorded; convert seconds with seconds_to_frames."
)


def _integer_vector(
    name: str, values: ArrayLike, *, hint: str = ""
) -> NDArray[np.int64]:
    """*values* as int64, refusing a float array rather than truncating it."""
    array = _as_vector(name, values)
    if array.size and not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must hold integers, got dtype {array.dtype}.{hint}")
    return array.astype(np.int64, copy=False)


def _validated_intervals(
    starts: ArrayLike, ends: ArrayLike, labels: ArrayLike
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int32]]:
    """The three segment arrays, checked for length, order and range."""
    start = _integer_vector("start", starts, hint=_FRAME_HINT)
    end = _integer_vector("end", ends, hint=_FRAME_HINT)
    label = _integer_vector("label", labels)
    if not start.size == end.size == label.size:
        raise ValueError(
            "segment arrays differ in length: "
            f"start {start.size}, end {end.size}, label {label.size}"
        )
    if start.size:
        if int(start.min()) < 0:
            first = int(np.flatnonzero(start < 0)[0])
            raise ValueError(
                f"segment {first} starts at frame {int(start[first])}; frames are "
                "numbered from 0"
            )
        backwards = np.flatnonzero(end < start)
        if backwards.size:
            first = int(backwards[0])
            raise ValueError(
                f"segment {first} ends (frame {int(end[first])}) before it starts "
                f"(frame {int(start[first])}); the end frame is exclusive and may "
                "equal the start, never precede it"
            )
        if int(label.min()) < _INT32.min or int(label.max()) > _INT32.max:
            raise ValueError("segment labels must fit in int32")
    return start, end, label.astype(np.int32)


def seconds_to_frames(
    start_s: ArrayLike,
    end_s: ArrayLike,
    fps: float,
    *,
    widen_empty: bool = False,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Convert annotated ``[start_s, end_s)`` segments to ``[start, end)`` frames.

    Frame ``i`` spans ``[i/fps, (i+1)/fps)`` and belongs to the segment holding its
    midpoint, so a segment covers frame ``i`` exactly when
    ``start_s <= (i + 1/2)/fps < end_s``. That is ``start = ceil(start_s*fps - 1/2)``
    and the same expression for ``end``. Because one expression maps both ends
    of every segment, a boundary two segments share becomes one frame boundary:
    back-to-back segments tile, with no frame in both and no frame in neither,
    whatever the frame rate. Rounding to the nearest boundary rather than
    flooring is what keeps an exported time written to the millisecond on the
    boundary it was meant for -- ``0.033`` s at 30 fps is frame boundary 1, not 0.

    A segment shorter than one frame can hold no midpoint and comes back empty
    (``start == end``). That is the tiling rule working, not an error: widening
    it would put its frame in a neighbour as well. Pass ``widen_empty=True`` to
    give each empty segment the one frame on screen at its start instead,
    ``floor(start_s*fps)`` -- what a point event in an annotation tool means --
    accepting that it may then overlap a neighbour.

    Args:
        start_s: Segment starts, in seconds from the first frame.
        end_s: Segment ends, in seconds, exclusive.
        fps: The video's frame rate.
        widen_empty: Give a segment that rounds to no frames the frame its start
            falls in.

    Returns:
        ``(start, end)`` as int64 arrays, ``end`` exclusive.

    Raises:
        ValueError: On a frame rate that is not a positive finite number, arrays
            of different lengths, a non-finite or negative time, or a segment
            that ends before it starts.
    """
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"fps must be a positive finite number, got {fps!r}")
    start = np.asarray(start_s, dtype=np.float64).ravel()
    end = np.asarray(end_s, dtype=np.float64).ravel()
    if start.shape != end.shape:
        raise ValueError(
            f"start_s and end_s differ in length: {start.size} vs {end.size}"
        )
    if not (np.isfinite(start).all() and np.isfinite(end).all()):
        raise ValueError("segment times must be finite")
    if start.size and float(start.min()) < 0.0:
        raise ValueError("segment times must not be negative")
    backwards = np.flatnonzero(end < start)
    if backwards.size:
        first = int(backwards[0])
        raise ValueError(
            f"segment {first} ends at {end[first]} s, before its start at "
            f"{start[first]} s"
        )
    start_frame = np.ceil(start * fps - 0.5).astype(np.int64)
    end_frame = np.ceil(end * fps - 0.5).astype(np.int64)
    # ceil(-0.5) is -0.0, which is 0 as an integer; a start of exactly 0 s is
    # therefore frame 0 without a special case.
    if widen_empty:
        empty = end_frame <= start_frame
        # The tolerance absorbs a time computed as k/fps landing a hair below k.
        on_screen = np.floor(start * fps + 1e-9).astype(np.int64)
        start_frame = np.where(empty, on_screen, start_frame)
        end_frame = np.where(empty, on_screen + 1, end_frame)
    return start_frame, end_frame


def _first_overlap(
    start: NDArray[np.int64], end: NDArray[np.int64]
) -> tuple[int, int] | None:
    """Positions of two segments covering one frame, or None. Empty ones excluded."""
    nonempty = np.flatnonzero(end > start)
    if nonempty.size < 2:
        return None
    order = nonempty[np.lexsort((nonempty, end[nonempty], start[nonempty]))]
    ends_sorted = end[order]
    reach = np.maximum.accumulate(ends_sorted)
    clash = np.flatnonzero(start[order][1:] < reach[:-1])
    if not clash.size:
        return None
    later = int(clash[0]) + 1
    # The earlier member is the one whose end the running maximum carries.
    earlier = int(np.argmax(ends_sorted[:later]))
    return int(order[earlier]), int(order[later])


def _priority_keys(
    policy: OverlapPolicy,
    position: NDArray[np.int64],
    start: NDArray[np.int64],
    length: NDArray[np.int64],
) -> tuple[NDArray[np.int64], ...]:
    """``np.lexsort`` keys ordering segments from lowest to highest priority.

    The last key is primary. Listing position is always the final tie-break, so
    the segment listed later wins a tie -- the rule the event reader already
    applies to two events on one frame ("last event wins").
    """
    if policy == "shortest":
        return (position, -length)
    if policy == "longest":
        return (position, length)
    if policy == "latest_start":
        return (position, start)
    return (position,)


def intervals_to_dense(
    starts: ArrayLike,
    ends: ArrayLike,
    labels: ArrayLike,
    n_frames: int,
    *,
    policy: OverlapPolicy,
    background: int = 0,
) -> NDArray[np.int32]:
    """Paint ``[start, end)`` segments onto a per-frame label array.

    Where segments overlap, *policy* decides each frame:

    - ``"refuse"`` raises :class:`SegmentOverlapError`, naming two segments
      that share a frame. For annotations that are mutually exclusive by
      design, where an overlap is a mistake. Only frames below ``n_frames``
      are checked: an overlap past the end of the output paints nothing.
    - ``"shortest"`` gives the frame to the shortest segment covering it. For
      nested annotations that is the innermost, most specific one: *grooming*
      inside *resting* shows as grooming, and resting resumes around it.
    - ``"longest"`` gives it to the longest covering segment.
    - ``"latest_start"`` gives it to the covering segment that started last,
      which reads a nested bout as interrupting its parent, as ``"shortest"``
      does, but also lets a later partial overlap take over.

    Every tie under every policy -- equal lengths, equal starts -- goes to the
    segment listed later, so the result is a function of the arrays alone.
    Lengths are measured before clipping, so a segment's priority does not
    depend on ``n_frames``.

    Args:
        starts: First frames, inclusive.
        ends: Frames after the last ones, exclusive.
        labels: One label id per segment.
        n_frames: Length of the output. Segment frames at or beyond it are
            dropped; a segment entirely beyond it contributes nothing.
        policy: How a frame covered twice is decided.
        background: The label of a frame no segment covers.

    Returns:
        An int32 array of shape ``(n_frames,)``.

    Raises:
        ValueError: On malformed segments (see :class:`LabelSegments`), a
            negative ``n_frames``, an unknown policy, or -- as
            :class:`SegmentOverlapError` -- an overlap under ``"refuse"``.
    """
    if policy not in OVERLAP_POLICIES:
        raise ValueError(f"policy must be one of {OVERLAP_POLICIES}, got {policy!r}")
    if n_frames < 0:
        raise ValueError(f"n_frames must not be negative, got {n_frames}")
    start, end, label = _validated_intervals(starts, ends, labels)
    out = np.full(n_frames, background, dtype=np.int32)
    clipped_end = np.minimum(end, n_frames)
    clipped_start = np.minimum(start, clipped_end)

    overlap = _first_overlap(clipped_start, clipped_end)
    if overlap is not None and policy == "refuse":
        a, b = overlap
        raise SegmentOverlapError(
            f"segments {a} (frames {int(start[a])}-{int(end[a])}, label "
            f"{int(label[a])}) and {b} (frames {int(start[b])}-{int(end[b])}, label "
            f"{int(label[b])}) cover the same frames; choose an overlap policy "
            "other than 'refuse', or correct the annotation"
        )

    span = clipped_end - clipped_start
    total = int(span.sum())
    if total == 0:
        return out
    segment_of = np.repeat(np.arange(start.size, dtype=np.int64), span)
    offset = np.arange(total, dtype=np.int64) - np.repeat(np.cumsum(span) - span, span)
    frame = clipped_start[segment_of] + offset

    if overlap is None:
        # Every frame is covered once, so a plain scatter is exact.
        out[frame] = label[segment_of]
        return out

    position = np.arange(start.size, dtype=np.int64)
    rank = np.empty(start.size, dtype=np.int64)
    rank[np.lexsort(_priority_keys(policy, position, start, end - start))] = position
    # Sort by frame, then by rank, and keep the last -- the highest-priority --
    # entry of each frame. A scatter with repeated indices would leave the
    # winner to numpy's unspecified write order.
    order = np.lexsort((rank[segment_of], frame))
    frame_sorted = frame[order]
    last = np.flatnonzero(np.append(frame_sorted[1:] != frame_sorted[:-1], True))
    out[frame_sorted[last]] = label[segment_of[order[last]]]
    return out


@dataclass(frozen=True, slots=True, eq=False)
class LabelSegments:
    """The segments of one converted label file, as the sidecar holds them.

    Built by :func:`segments_from_npz` from a file, or directly by a converter
    that wants :meth:`to_payload`. Construction validates and coerces: equal
    lengths, integer frames, ``0 <= start <= end``, and labels that fit int32.
    The listing order is kept exactly as given, because it breaks ties when the
    segments are painted.

    Attributes:
        start_frame: First frame of each segment, inclusive (int64).
        end_frame: Frame after each segment's last, exclusive (int64).
        label: Each segment's label id (int32).
        weight: Optional per-segment weight (float64), e.g. a confidence.
        individual: Optional track id each segment belongs to (int64),
            :data:`SCENE_LEVEL` for one about the scene.
        extra: Any further ``seg_*`` arrays, keyed by their full key, carried
            through unread.
    """

    start_frame: NDArray[np.int64]
    end_frame: NDArray[np.int64]
    label: NDArray[np.int32]
    weight: NDArray[np.float64] | None = None
    individual: NDArray[np.int64] | None = None
    extra: Mapping[str, NDArray[Any]] = field(default_factory=dict[str, NDArray[Any]])

    def __post_init__(self) -> None:
        start, end, label = _validated_intervals(
            self.start_frame, self.end_frame, self.label
        )
        object.__setattr__(self, "start_frame", start)
        object.__setattr__(self, "end_frame", end)
        object.__setattr__(self, "label", label)
        n = start.size
        if self.weight is not None:
            weight = _as_vector(SEG_WEIGHT, self.weight)
            if weight.size and not np.issubdtype(weight.dtype, np.number):
                raise ValueError(f"{SEG_WEIGHT} must be numeric, got {weight.dtype}")
            if weight.size != n:
                raise ValueError(
                    f"{SEG_WEIGHT} has {weight.size} entries for {n} segments"
                )
            object.__setattr__(self, "weight", weight.astype(np.float64))
        if self.individual is not None:
            individual = _integer_vector(SEG_INDIVIDUAL, self.individual)
            if individual.size != n:
                raise ValueError(
                    f"{SEG_INDIVIDUAL} has {individual.size} entries for {n} segments"
                )
            object.__setattr__(self, "individual", individual)
        extra: dict[str, NDArray[Any]] = {}
        for key, values in self.extra.items():
            if not key.startswith(SEGMENT_PREFIX) or key in _NAMED_KEYS:
                raise ValueError(
                    f"extra sidecar key {key!r} must start with {SEGMENT_PREFIX!r} "
                    "and not be one of the named keys"
                )
            array = np.asarray(values)
            if array.ndim < 1 or array.shape[0] != n:
                raise ValueError(
                    f"{key} must hold one value per segment ({n}), got shape "
                    f"{array.shape}"
                )
            extra[key] = array
        object.__setattr__(self, "extra", extra)

    def __len__(self) -> int:
        return int(self.start_frame.size)

    @property
    def lengths(self) -> NDArray[np.int64]:
        """Each segment's length in frames, ``end - start``."""
        return self.end_frame - self.start_frame

    def onset_order(self) -> NDArray[np.intp]:
        """Positions sorted by onset: start, then end, then listing order."""
        position = np.arange(len(self), dtype=np.int64)
        return np.lexsort((position, self.end_frame, self.start_frame))

    def to_dense(
        self, n_frames: int, *, policy: OverlapPolicy, background: int = 0
    ) -> NDArray[np.int32]:
        """These segments painted onto frames; see :func:`intervals_to_dense`."""
        return intervals_to_dense(
            self.start_frame,
            self.end_frame,
            self.label,
            n_frames,
            policy=policy,
            background=background,
        )

    def to_payload(self) -> dict[str, NDArray[Any]]:
        """The ``seg_*`` arrays, to be saved into a label ``.npz`` beside ``labels``."""
        payload: dict[str, NDArray[Any]] = {
            SEG_START_FRAME: self.start_frame,
            SEG_END_FRAME: self.end_frame,
            SEG_LABEL: self.label,
        }
        if self.weight is not None:
            payload[SEG_WEIGHT] = self.weight
        if self.individual is not None:
            payload[SEG_INDIVIDUAL] = self.individual
        payload.update(self.extra)
        return payload


def label_vocabulary(npz: Mapping[str, object]) -> dict[int, str]:
    """A label file's ``label_ids`` mapped to its ``label_names``.

    Empty when the file declares no ``label_ids``. A file declaring ids without
    names maps each id to its decimal spelling.

    Raises:
        ValueError: When ``label_names`` and ``label_ids`` differ in length, or
            an id is repeated.
    """
    if "label_ids" not in npz:
        return {}
    ids = _integer_vector("label_ids", np.asarray(npz["label_ids"]).ravel())
    if np.unique(ids).size != ids.size:
        raise ValueError("label_ids repeats an id")
    if "label_names" not in npz:
        return {int(i): str(int(i)) for i in ids}
    names = np.asarray(npz["label_names"], dtype=object).ravel()
    if names.size != ids.size:
        raise ValueError(
            f"label_names has {names.size} entries for {ids.size} label_ids"
        )
    return {int(i): str(name) for i, name in zip(ids.tolist(), names.tolist())}


def segments_from_npz(npz: Mapping[str, object]) -> LabelSegments | None:
    """Read the segment sidecar of a converted label file, or None if it has none.

    Takes the opened file (``np.load(path)``) or any mapping of its arrays.
    Only ``seg_*`` keys are read, plus ``label_ids`` to check the labels against.

    Raises:
        ValueError: When some ``seg_*`` key is present but a required one is
            missing, when the arrays are malformed (see :class:`LabelSegments`),
            or when a ``seg_label`` is not among the file's ``label_ids``.
    """
    keys = [key for key in npz if key.startswith(SEGMENT_PREFIX)]
    if not keys:
        return None
    missing = [key for key in _REQUIRED_KEYS if key not in npz]
    if missing:
        raise ValueError(
            f"label file carries segment keys {sorted(keys)} but not {missing}; "
            f"a segment sidecar needs all of {list(_REQUIRED_KEYS)}"
        )
    segments = LabelSegments(
        start_frame=np.asarray(npz[SEG_START_FRAME]),
        end_frame=np.asarray(npz[SEG_END_FRAME]),
        label=np.asarray(npz[SEG_LABEL]),
        weight=np.asarray(npz[SEG_WEIGHT]) if SEG_WEIGHT in npz else None,
        individual=np.asarray(npz[SEG_INDIVIDUAL]) if SEG_INDIVIDUAL in npz else None,
        extra={
            key: np.asarray(npz[key]) for key in sorted(keys) if key not in _NAMED_KEYS
        },
    )
    vocabulary = label_vocabulary(npz)
    if vocabulary and len(segments):
        unknown = sorted(set(np.unique(segments.label).tolist()) - set(vocabulary))
        if unknown:
            raise ValueError(
                f"{SEG_LABEL} holds {unknown}, which the file's label_ids "
                f"{sorted(vocabulary)} do not declare"
            )
    return segments
