"""The pose-keypoint column vocabulary, the arrays read through it, and the body center.

A leaf module by design: numpy, pandas and the standard library, and no import
from anywhere else in ``core``. Its callers are the track converters in
``core/track_library/``, which do not import ``core.pipeline``, the inference
bridge in ``tracking/``, and the features that read an individual's keypoints as
arrays. Leaving this beside the pipeline's data loading would put
``pipeline.index``, ``pipeline.types`` and ``core.scope`` on the import path of
every converter, for functions that need none of them.

``keypoint_centroid`` is the one answer to what ``X``/``Y`` mean for a producer
that measures keypoints and no centroid. It was written out longhand in five
converters and copied into two visualization features before it lived here, and
the copies had drifted: one used a plain ``mean`` where a single missing
keypoint poisons the whole row, one paired its X and Y column lists
independently so a ``poseX3`` without its ``poseY3`` silently misaligned the
stack, and only one guarded the all-NaN warning.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "KeypointArrays",
    "configured_pose_pairs",
    "frame_keypoint_centroid",
    "keypoint_arrays",
    "keypoint_centroid",
    "keypoint_suffixes",
    "pose_column_pairs",
    "reindex_contiguous",
    "resolve_keypoint_refs",
]


def _keypoint_sort_key(suffix: str) -> tuple[int, int, str]:
    """Order a pose suffix numerically where it is a number, lexically otherwise.

    Keypoint identity is positional: every caller that indexes into the returned
    list -- ``heading``'s ``front_idx`` / ``rear_idx``, the overlay's skeleton
    lines -- means "the Nth keypoint the converter emitted". A lexicographic sort
    breaks that silently from ten keypoints on, ordering ``poseX10`` between
    ``poseX1`` and ``poseX2``, so a 21-point midline is drawn and measured
    scrambled with nothing in the output to say so.

    Numeric suffixes sort first and among themselves by value; anything else
    keeps a stable lexicographic order after them, so a named keypoint set
    (``poseXhead``) is still ordered deterministically rather than raising.
    """
    if suffix.isdigit():
        return (0, int(suffix), "")
    return (1, 0, suffix)


def pose_column_pairs(columns: Iterable[str]) -> list[tuple[str, str]]:
    """Extract (poseX*, poseY*) column pairs, ordered by keypoint index.

    Args:
        columns: Column names to scan.

    Returns:
        The ``(poseX<k>, poseY<k>)`` pairs whose X and Y are both present, in
        keypoint order -- numerically for numeric suffixes. A ``poseX`` without
        its ``poseY`` is skipped rather than half-reported.
    """
    column_names = list(columns)
    present = set(column_names)
    return [
        (f"poseX{suffix}", f"poseY{suffix}")
        for suffix in keypoint_suffixes(column_names, "poseX")
        if f"poseY{suffix}" in present
    ]


def keypoint_suffixes(columns: Iterable[str], x_prefix: str) -> list[str]:
    """Return the suffix of every column named *x_prefix* plus a suffix, in keypoint order.

    Args:
        columns: Column names to scan.
        x_prefix: What an x column is called before its suffix, e.g. ``poseX``.

    Returns:
        The suffixes, the numeric ones by value and then the others in lexical
        order. Only the x column is looked for. A caller reading another axis
        names those columns, and a missing one raises there.
    """
    suffixes = [c[len(x_prefix) :] for c in columns if c.startswith(x_prefix)]
    return sorted(suffixes, key=_keypoint_sort_key)


@dataclass(frozen=True, eq=False)
class KeypointArrays:
    """One individual's keypoints as arrays, one row per frame.

    Attributes:
        frames: ``(T,)`` int64 frame numbers.
        position: ``(T, K, 2)`` float32 x and y of each keypoint, NaN in both
            coordinates where the point is missing.
        confidence: ``(T, K)`` float32 score of each point, 0 where the point is
            missing.
    """

    frames: np.ndarray[tuple[int], np.dtype[np.int64]]
    position: np.ndarray[tuple[int, int, int], np.dtype[np.float32]]
    confidence: np.ndarray[tuple[int, int], np.dtype[np.float32]]


def keypoint_arrays(
    rows: pd.DataFrame,
    *,
    x_prefix: str,
    y_prefix: str,
    confidence_prefix: str,
    suffixes: Sequence[str],
) -> KeypointArrays:
    """Read the keypoints that *suffixes* name from *rows*, row for row.

    Confidence comes from the ``confidence_prefix`` columns when every requested
    suffix has one, and is 1 for every point otherwise. Every keypoint is
    therefore scored by one rule. A position with a non-finite coordinate is NaN
    in both coordinates and scores 0, whatever its score column says. A NaN
    score at a finite position is kept for the caller to decide on.

    Args:
        rows: One individual's rows, with a ``frame`` column. Left unchanged.
        x_prefix: What an x column is called before its suffix.
        y_prefix: The same for y.
        confidence_prefix: The same for a score.
        suffixes: The keypoints to read, in array order.

    Returns:
        The arrays, in the row order of *rows*.

    Raises:
        KeyError: When ``frame`` or a requested x or y column is missing.
    """
    n_rows, n_keypoints = len(rows), len(suffixes)
    position = np.empty((n_rows, n_keypoints, 2), dtype=np.float32)
    position[:, :, 0] = rows[[f"{x_prefix}{s}" for s in suffixes]].to_numpy(
        dtype=np.float32
    )
    position[:, :, 1] = rows[[f"{y_prefix}{s}" for s in suffixes]].to_numpy(
        dtype=np.float32
    )
    confidence_columns = [f"{confidence_prefix}{s}" for s in suffixes]
    if all(column in rows.columns for column in confidence_columns):
        confidence = rows[confidence_columns].to_numpy(dtype=np.float32, copy=True)
    else:
        confidence = np.ones((n_rows, n_keypoints), dtype=np.float32)
    missing = ~np.isfinite(position).all(axis=2)
    position[missing] = np.nan
    confidence[missing] = 0.0
    return KeypointArrays(
        frames=rows["frame"].to_numpy(dtype=np.int64),
        position=position,
        confidence=confidence,
    )


def reindex_contiguous(arrays: KeypointArrays) -> KeypointArrays:
    """Place *arrays* on every frame from their first frame to their last.

    A frame without a row is a missing point for every keypoint: NaN position
    and confidence 0. The rows may arrive in any frame order.

    Args:
        arrays: One individual's keypoints, as :func:`keypoint_arrays` reads them.

    Returns:
        The arrays on the contiguous frame axis, or *arrays* when they are
        empty.

    Raises:
        ValueError: When a frame appears in more than one row.
    """
    if arrays.frames.size == 0:
        return arrays
    first = int(arrays.frames.min())
    offsets = arrays.frames - first
    rows_per_frame = np.bincount(offsets)
    repeated = np.flatnonzero(rows_per_frame > 1)
    if repeated.size:
        offset = int(repeated[0])
        msg = (
            f"frame {first + offset} appears in {int(rows_per_frame[offset])} rows; "
            "a contiguous frame axis has one row per frame"
        )
        raise ValueError(msg)
    n_frames = rows_per_frame.size
    n_keypoints = arrays.position.shape[1]
    position = np.full((n_frames, n_keypoints, 2), np.nan, dtype=np.float32)
    confidence = np.zeros((n_frames, n_keypoints), dtype=np.float32)
    position[offsets] = arrays.position
    confidence[offsets] = arrays.confidence
    return KeypointArrays(
        frames=np.arange(first, first + n_frames, dtype=np.int64),
        position=position,
        confidence=confidence,
    )


def resolve_keypoint_refs(
    refs: Sequence[int | str],
    names: Sequence[str] | None,
    n_keypoints: int,
    *,
    who: str,
) -> list[int]:
    """Resolve keypoint references, by index or by name, to indices.

    A parameter naming keypoints accepts either form. A recipe can say
    ``"left_wrist"`` where a name list is given and ``9`` where none is. Both
    forms are checked against the table's keypoint count, and an out-of-range
    index raises as an unknown name does.

    Args:
        refs: Keypoint indices or names, in the order wanted.
        names: The keypoint names, in index order, or ``None`` when none were
            given.
        n_keypoints: How many keypoints the table contains.
        who: The caller, for the message.

    Returns:
        The keypoint indices, in the order of *refs*.

    Raises:
        ValueError: On an unknown name, a name with no name list to resolve it
            in, or an index outside ``0..n_keypoints-1``.
    """
    lookup = {name: index for index, name in enumerate(names or [])}
    resolved: list[int] = []
    for ref in refs:
        if isinstance(ref, str):
            if not names:
                raise ValueError(
                    f"{who} names keypoint {ref!r} but was not given keypoint names; "
                    "pass keypoint_names or refer to keypoints by index"
                )
            if ref not in lookup:
                raise ValueError(
                    f"{who}: no keypoint named {ref!r}; the names are {list(names)}"
                )
            index = lookup[ref]
        else:
            index = int(ref)
        if not 0 <= index < n_keypoints:
            raise ValueError(
                f"{who}: keypoint {ref!r} is outside the table's {n_keypoints} "
                f"keypoints (0..{n_keypoints - 1})"
            )
        resolved.append(index)
    return resolved


def configured_pose_pairs(
    columns: Iterable[str],
    *,
    x_prefix: str,
    y_prefix: str,
    count: int,
) -> list[tuple[str, str]]:
    """The first *count* numbered pose pairs present, under the given prefixes.

    The bounded sibling of :func:`pose_column_pairs`, for a caller whose keypoint
    set is a declaration rather than a discovery -- the crop features, whose
    ``PoseConfig`` fixes both the prefixes and how many keypoints to read, and
    which must not silently widen to whatever a table happens to carry.

    Args:
        columns: Column names to scan.
        x_prefix: What an x column is called before its index, e.g. ``poseX``.
        y_prefix: The same for y.
        count: How many keypoint indices to consider, from ``0``.

    Returns:
        The ``(x_column, y_column)`` pairs, in index order, whose x *and* y are
        both present. Half a pair contributes no keypoint, so it is dropped
        whole: filtering the two sides independently is how one stack ends up
        shorter than the other and keypoint 3's x gets averaged against
        keypoint 4's y.
    """
    present = set(columns)
    return [
        (f"{x_prefix}{i}", f"{y_prefix}{i}")
        for i in range(count)
        if f"{x_prefix}{i}" in present and f"{y_prefix}{i}" in present
    ]


def keypoint_centroid(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The body centre of each row, as the mean of its detected keypoints.

    This is what ``mosaic_v1``'s ``X``/``Y`` mean for a producer that localizes
    landmarks and reports no centroid of its own: on a midline skeleton the mean
    of the keypoints *is* the body centre by construction. A producer that
    genuinely measures a centroid -- TRex's ``#wcentroid`` -- reports that one
    instead and never reaches here.

    Args:
        x: ``(n_rows, n_keypoints)`` of keypoint x coordinates.
        y: The same shape, of y coordinates.

    Returns:
        ``(cx, cy)``, each ``(n_rows,)`` of float64.

    A keypoint the producer did not detect is NaN and is left out of its row's
    mean, so a partly occluded animal still gets a centre from what was seen. A
    row with no detected keypoint at all, and a table with no keypoint columns at
    all, are NaN rather than an error: absent is a legitimate state that the
    schema records as a blank, and raising here would fail a whole conversion
    over one unseen frame.
    """
    if x.shape != y.shape:
        raise ValueError(
            f"keypoint x and y must be the same shape, got {x.shape} and {y.shape}"
        )
    n_rows = x.shape[0]
    if x.ndim != 2 or x.shape[1] == 0:
        empty = np.full(n_rows, np.nan, dtype=np.float64)
        return empty, empty.copy()
    # An all-NaN row makes `nanmean` warn and return NaN, and the warning says
    # nothing the NaN does not. Both guards are needed and neither substitutes
    # for the other: `errstate` covers the invalid-value floating point flag,
    # while "Mean of empty slice" is raised through `warnings` and survives it.
    # Four of the seven copies this replaces had one guard or neither, so a
    # conversion's stderr depended on which converter ran.
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        cx = np.nanmean(x.astype(np.float64, copy=False), axis=1)
        cy = np.nanmean(y.astype(np.float64, copy=False), axis=1)
    return cx, cy


def frame_keypoint_centroid(
    frame: pd.DataFrame,
    pairs: Sequence[tuple[str, str]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """:func:`keypoint_centroid` over a table's pose columns.

    Args:
        frame: The table to read. Never modified.
        pairs: Which ``(x_column, y_column)`` pairs to average, in keypoint
            order. Defaults to every pair :func:`pose_column_pairs` finds. A
            caller passes this when its keypoint set is narrower than what the
            table carries or is named under different prefixes -- the crop
            features, which are bounded by their ``PoseConfig``. Pass pairs, not
            two separate lists: an x whose y is missing has no keypoint to
            contribute, and filtering the two independently misaligns the stack.

    Returns:
        ``(cx, cy)``, each ``(len(frame),)`` of float64, NaN where the row had no
        detected keypoint and throughout when there are no pose columns.
    """
    chosen = pose_column_pairs(frame.columns.astype(str)) if pairs is None else pairs
    if not chosen:
        empty = np.full(len(frame), np.nan, dtype=np.float64)
        return empty, empty.copy()
    x = np.column_stack([frame[xc].to_numpy(dtype=np.float64) for xc, _ in chosen])
    y = np.column_stack([frame[yc].to_numpy(dtype=np.float64) for _, yc in chosen])
    return keypoint_centroid(x, y)
