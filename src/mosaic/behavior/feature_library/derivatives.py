"""Savitzky-Golay derivatives of keypoint tracks, shared by the keypoint features.

A derivative of noisy positions taken as a frame difference amplifies the noise
by the frame rate, and a second difference by its square, so an acceleration
from differences is mostly detector jitter. The fit estimates the derivative of
a local polynomial instead, which is one smoothing, applied the same way to
every keypoint.

**Method.** Each individual's rows are sorted by frame and placed on its
contiguous frame range, frames it lacks reading as missing. A keypoint is
missing on a row where any of its coordinates is not finite. Gaps of at most
``window`` frames are linearly interpolated, so a brief dropout does not cut
the filter short; a longer gap splits the track into stretches that are
filtered separately. Each coordinate is then differentiated once and twice by
a Savitzky-Golay fit with the time step ``1 / fps``, which near the ends of a
stretch evaluates the polynomial fitted to its first or last window, as
``scipy.signal.savgol_filter(mode="interp")`` does.

**What is NaN.** A keypoint's derivatives are NaN on every row where it was
missing in the input, interpolated or not: an interpolated position is there
to keep the filter running, and is not a measurement. They are also NaN on a
stretch shorter than ``window``, which has no window to fit. Rows are emitted
for exactly the input's ``(frame, id)`` rows; the frames the reindexing added
are dropped again.

``who`` in the functions below is the name of the feature that opens each error
message.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.ndimage import correlate1d
from scipy.signal import savgol_coeffs

from mosaic.core.pipeline.types import COLUMNS as C
from mosaic.core.pose_columns import pose_column_pairs, pose_column_triplets

__all__ = [
    "FPS_DESCRIPTION",
    "KEYPOINT_NAMES_DESCRIPTION",
    "WINDOW_DESCRIPTION",
    "BoolArray",
    "FloatArray",
    "FrameAxis",
    "IntArray",
    "KeypointColumns",
    "RowLayout",
    "SavgolKernel",
    "check_keypoint_names",
    "check_savgol_window",
    "frame_axis",
    "frame_rate",
    "keypoint_columns",
    "keypoint_positions",
    "row_norm",
    "savgol_derivatives",
]

type FloatArray = npt.NDArray[np.float64]
type BoolArray = npt.NDArray[np.bool_]
type IntArray = npt.NDArray[np.int64]

KEYPOINT_NAMES_DESCRIPTION = (
    "A name for each keypoint, in the table's keypoint order. Used in the output "
    "column names, and accepted wherever a parameter refers to a keypoint. Its "
    "length must equal the table's keypoint count. Unset, keypoint k is labeled "
    "kp{k} and is referred to by its index."
)

FPS_DESCRIPTION = (
    "The recording's frame rate, which sets the time step of every derivative. "
    "Unset, it is the reciprocal of the median time step per frame read off the "
    "time column, and a table whose time column cannot give one is refused."
)

WINDOW_DESCRIPTION = (
    "The Savitzky-Golay window length: odd, and larger than polyorder. It is also "
    "the longest gap in a keypoint's track that is linearly interpolated before "
    "filtering. A longer gap ends the stretch the filter runs over."
)


def check_savgol_window(window: int, polyorder: int, who: str) -> None:
    """Refuse a window no Savitzky-Golay fit could use.

    Args:
        window: The window length.
        polyorder: The order of the polynomial fitted in each window.
        who: The feature name that opens the error message.

    Raises:
        ValueError: If *window* is even, or no longer than *polyorder*.
    """
    if window % 2 == 0:
        raise ValueError(
            f"{who}: window={window} must be odd, so that each "
            "window is centered on the frame it estimates."
        )
    if window <= polyorder:
        raise ValueError(
            f"{who}: window={window} must exceed "
            f"polyorder={polyorder}; a window no longer than the "
            "polynomial's order fits it exactly and smooths nothing."
        )


def check_keypoint_names(names: list[str] | None, who: str) -> None:
    """Refuse keypoint names that would make output column names ambiguous.

    Args:
        names: The names, or ``None`` when the keypoints are unnamed.
        who: The feature name that opens the error message.

    Raises:
        ValueError: If a name repeats, is empty, or contains ``__``.
    """
    if names is None:
        return
    if len(set(names)) != len(names):
        raise ValueError(
            f"{who}: keypoint_names repeats a name, {names}; each "
            "name labels output columns of its own."
        )
    bad = [name for name in names if not name or "__" in name]
    if bad:
        raise ValueError(
            f"{who}: keypoint names {bad} are empty or contain "
            "'__', which separates the parts of an output column name."
        )


@dataclass(frozen=True)
class KeypointColumns:
    """A table's keypoints: their coordinate columns and their labels.

    Attributes:
        columns: The coordinate columns of every keypoint the table carries, in
            keypoint order: pairs on a 2D table, triplets on a 3D one.
        labels: Each keypoint's label in an output column name.
    """

    columns: tuple[tuple[str, ...], ...]
    labels: tuple[str, ...]

    @property
    def three_d(self) -> bool:
        return len(self.columns[0]) == 3


def keypoint_columns(
    df: pd.DataFrame,
    keypoint_names: list[str] | None,
    who: str,
    *,
    mixed_axes_hint: str = "",
    no_keypoints_hint: str = "",
) -> KeypointColumns:
    """Find *df*'s keypoint columns, in 2D or 3D as the table decides.

    A table carrying ``Z`` or ``poseZ*`` columns is measured on its keypoint
    triplets, and one carrying neither on its ``poseX*``/``poseY*`` pairs.

    Args:
        df: The tracks table.
        keypoint_names: A name per keypoint, or ``None`` to label keypoint k
            ``kp{k}``.
        who: The feature name that opens each error message.
        mixed_axes_hint: Appended after a space to the refusal of a Z axis with
            planar keypoints, when non-empty.
        no_keypoints_hint: Appended after a space to the refusal of a table
            without keypoints, when non-empty.

    Returns:
        The coordinate columns and labels of every keypoint.

    Raises:
        ValueError: If the table has a Z axis and some planar keypoints, no
            keypoints at all, or a different number of keypoints than
            *keypoint_names* has names.
    """
    names = [str(column) for column in df.columns]
    pairs = pose_column_pairs(names)
    triplets = pose_column_triplets(names)
    three_d = "Z" in names or any(name.startswith("poseZ") for name in names)
    if three_d and len(triplets) != len(pairs):
        message = (
            f"{who}: this table carries a Z axis, but "
            f"{len(pairs) - len(triplets)} of its {len(pairs)} keypoints have "
            "no poseZ column."
        )
        raise ValueError(f"{message} {mixed_axes_hint}" if mixed_axes_hint else message)
    columns: list[tuple[str, ...]] = [*triplets] if three_d else [*pairs]
    if not columns:
        message = (
            f"{who} needs pose keypoint columns (poseX*/poseY*, and "
            "poseZ* on a 3D table) and this table carries none."
        )
        raise ValueError(
            f"{message} {no_keypoints_hint}" if no_keypoints_hint else message
        )
    n_keypoints = len(columns)
    if keypoint_names is not None and len(keypoint_names) != n_keypoints:
        raise ValueError(
            f"{who}: keypoint_names has {len(keypoint_names)} names "
            f"and the table carries {n_keypoints} keypoints; a name is matched "
            "to a keypoint by position, so the two must agree."
        )
    labels = (
        tuple(keypoint_names)
        if keypoint_names is not None
        else tuple(f"kp{k}" for k in range(n_keypoints))
    )
    return KeypointColumns(columns=tuple(columns), labels=labels)


def keypoint_positions(
    df: pd.DataFrame, columns: tuple[str, ...], order: IntArray
) -> FloatArray:
    """Read one keypoint's coordinates in frame-axis row order.

    Args:
        df: The tracks table.
        columns: The keypoint's coordinate columns, a pair or a triplet.
        order: The row permutation of the table's frame axis.

    Returns:
        A new ``(n_rows, len(columns))`` float64 array with every non-finite
        value set to NaN.
    """
    values = df[list(columns)].to_numpy(dtype=np.float64)[order]
    values[~np.isfinite(values)] = np.nan
    return values


@dataclass(frozen=True)
class FrameAxis:
    """The table's rows in output order, and where each individual's rows lie.

    Attributes:
        order: The permutation of the input rows into output order: by group,
            sequence and id, then by frame.
        frames: The frame of each row, in that order.
        starts: The first row of each individual.
        ends: One past its last row.
        codes: The individual each row belongs to, ``0..n_individuals-1``.
    """

    order: IntArray
    frames: IntArray
    starts: IntArray
    ends: IntArray
    codes: IntArray


def frame_axis(df: pd.DataFrame, who: str) -> FrameAxis:
    """Order the rows by individual, then frame, and find each one's rows.

    Ordered through a permutation rather than a sort of the frame, which
    would copy every column of a table whose keypoints are read one at a
    time anyway.

    Args:
        df: The tracks table, with one row per ``(frame, id)``.
        who: The feature name that opens each error message.

    Returns:
        The row order and each individual's extent in it.

    Raises:
        ValueError: If a frame is not a whole number, or an individual has two
            rows at one frame.
    """
    keys = [c for c in (C.group_col, C.seq_col) if c in df.columns] + [C.id_col]
    raw: FloatArray = df[C.frame_col].to_numpy(dtype=np.float64)
    if not np.isfinite(raw).all() or (raw != np.floor(raw)).any():
        raise ValueError(
            f"{who}: the '{C.frame_col}' column holds values that are "
            "not whole frame numbers, so there is no frame axis to place rows "
            "on."
        )
    frames = raw.astype(np.int64)
    codes = df.groupby(keys, sort=True, dropna=False).ngroup().to_numpy(dtype=np.int64)
    order = np.lexsort((frames, codes)).astype(np.int64)
    frames, codes = frames[order], codes[order]
    same = np.diff(codes) == 0
    repeated = np.flatnonzero(same & (np.diff(frames) == 0))
    if repeated.size:
        first = int(order[repeated[0]])
        hint = (
            " This table has a camera column; a table holding one row per "
            "camera measures each camera's view apart, and this feature "
            "does not."
            if "camera" in df.columns
            else ""
        )
        raise ValueError(
            f"{who}: id {df[C.id_col].iloc[first]!r} has "
            f"{repeated.size} repeated frames, the first at frame "
            f"{df[C.frame_col].iloc[first]}. One position per frame is needed, "
            f"so one row per (frame, id).{hint}"
        )
    boundaries = np.flatnonzero(~same) + 1
    starts = np.concatenate(([0], boundaries)).astype(np.int64)
    ends = np.concatenate((boundaries, [frames.size])).astype(np.int64)
    return FrameAxis(
        order=order,
        frames=frames,
        starts=starts,
        ends=ends,
        codes=np.repeat(np.arange(starts.size, dtype=np.int64), ends - starts),
    )


def frame_rate(df: pd.DataFrame, axis: FrameAxis, fps: float | None, who: str) -> float:
    """The frame rate: *fps*, or else the time column's median step.

    Args:
        df: The tracks table.
        axis: The table's :func:`frame_axis`.
        fps: The frame rate the caller asked for, or ``None`` to read it.
        who: The feature name that opens each error message.

    Returns:
        Frames per second.

    Raises:
        ValueError: If *fps* is ``None`` and the time column is missing or
            gives no positive time step per frame.
    """
    if fps is not None:
        return fps
    if C.time_col not in df.columns:
        raise ValueError(
            f"{who} needs a frame rate: pass fps, or give the table a "
            f"'{C.time_col}' column to read it from."
        )
    times = df[C.time_col].to_numpy(dtype=np.float64)[axis.order]
    # A step across two individuals' rows is not a time step.
    within = np.ones(max(times.size - 1, 0), dtype=bool)
    within[axis.ends[:-1] - 1] = False
    frame_steps = np.diff(axis.frames)
    time_steps = np.diff(times)
    usable = within & (frame_steps > 0) & np.isfinite(time_steps)
    if usable.any():
        period = float(np.median(time_steps[usable] / frame_steps[usable]))
        if np.isfinite(period) and period > 0:
            return 1.0 / period
    raise ValueError(
        f"{who} could not read a frame rate off the '{C.time_col}' "
        "column: no two consecutive rows of an individual give a positive "
        "time step per frame. Pass fps."
    )


@dataclass(frozen=True)
class RowLayout:
    """Every individual's rows placed on one axis, which the filter runs along.

    Each individual is reindexed onto its contiguous frame range, and the ranges
    are laid end to end, so that one filter pass serves every individual: a
    table of tracklets holds tens of thousands of them, and a pass per
    individual and keypoint cost far more than the arithmetic. Where a gap
    between two rows is too long to bridge -- and between two individuals --
    a single *break* position stands in for it. A break is missing, so it ends
    the stretch on either side, and no gap holding one is interpolated across.
    That keeps the axis no longer than the rows plus the gaps short enough to
    bridge, however sparse the frame numbers are.

    Attributes:
        rows: The axis position of each row, in output order.
        size: The axis length.
        breaks: ``True`` at each break position.
    """

    rows: IntArray
    size: int
    breaks: BoolArray

    @classmethod
    def build(cls, axis: FrameAxis, longest_gap: int) -> RowLayout:
        """Lay *axis*'s individuals end to end, bridging gaps up to *longest_gap*."""
        frames = axis.frames
        steps = np.diff(frames)
        unbridged = (axis.codes[1:] != axis.codes[:-1]) | (steps - 1 > longest_gap)
        steps = np.where(unbridged, 2, steps)
        rows = np.concatenate(([0], np.cumsum(steps))).astype(np.int64)
        size = int(rows[-1]) + 1
        breaks = np.zeros(size, dtype=bool)
        breaks[rows[1:][unbridged] - 1] = True
        return cls(rows=rows, size=size, breaks=breaks)


@dataclass(frozen=True)
class SavgolKernel:
    """The Savitzky-Golay coefficients for one derivative, computed once per table.

    Attributes:
        center: Dotted with a window to estimate the derivative at its center.
        lead: One row per position among a run's first ``window // 2`` rows,
            dotted with the run's first ``window`` rows.
        trail: The same for its last ``window // 2`` rows and last window.
    """

    center: FloatArray
    lead: FloatArray
    trail: FloatArray

    @classmethod
    def build(
        cls, window: int, polyorder: int, deriv: int, delta: float
    ) -> SavgolKernel:
        """The coefficients of the *deriv*-th derivative at time step *delta*."""
        half = window // 2

        def at(pos: int | None) -> FloatArray:
            return np.asarray(
                savgol_coeffs(
                    window, polyorder, deriv=deriv, delta=delta, pos=pos, use="dot"
                ),
                dtype=np.float64,
            )

        return cls(
            center=at(None),
            lead=np.stack([at(j) for j in range(half)]),
            trail=np.stack([at(window - half + j) for j in range(half)]),
        )


def _runs(mask: BoolArray) -> tuple[IntArray, IntArray]:
    """The start and exclusive end of each run of ``True`` in *mask*."""
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1]).astype(np.int64)
    return edges[0::2], edges[1::2]


def _cover(starts: IntArray, ends: IntArray, n: int) -> BoolArray:
    """A length-*n* mask that is ``True`` on every ``[start, end)`` interval."""
    steps = np.zeros(n + 1, dtype=np.int64)
    np.add.at(steps, starts, 1)
    np.add.at(steps, ends, -1)
    return np.cumsum(steps[:n]) > 0


def _fill_short_gaps(
    positions: FloatArray, missing: BoolArray, longest: int, breaks: BoolArray
) -> tuple[FloatArray, BoolArray]:
    """Linearly interpolate every interior gap of at most *longest* rows.

    A gap touching either end of the axis, or holding a break, is left alone:
    at an end there is only one side to interpolate from, and a break stands for
    a gap too long to bridge or for the boundary between two individuals.

    Returns:
        The positions with those gaps filled, and what is still missing.
    """
    n = missing.size
    starts, ends = _runs(missing)
    broken = np.concatenate(([0], np.cumsum(breaks, dtype=np.int64)))
    short = (
        (starts > 0)
        & (ends < n)
        & (ends - starts <= longest)
        & (broken[ends] == broken[starts])
    )
    if not short.any():
        return positions, missing
    starts, ends = starts[short], ends[short]
    lengths = ends - starts
    # The rows of every gap, flattened, each paired with the known rows either
    # side of its gap: one interpolation for all gaps rather than a search.
    first_of_gap = np.repeat(np.cumsum(lengths) - lengths, lengths)
    rows = np.repeat(starts, lengths) + np.arange(int(lengths.sum())) - first_of_gap
    before = np.repeat(starts - 1, lengths)
    after = np.repeat(ends, lengths)
    fraction = ((rows - before) / (after - before))[:, None]
    filled = positions.copy()
    filled[rows] = positions[before] + fraction * (positions[after] - positions[before])
    still_missing = missing.copy()
    still_missing[rows] = False
    return filled, still_missing


def _savgol_on_runs(
    positions: FloatArray, valid: BoolArray, kernels: tuple[SavgolKernel, ...]
) -> list[FloatArray]:
    """``savgol_filter(mode="interp")`` applied to each run of valid rows.

    A run is filtered on its own, so no window reaches across a gap. The
    interior of a run is the centered filter; the first and last ``window // 2``
    rows of it are evaluated on the polynomial fitted to its first or last
    ``window`` rows, which is what ``mode="interp"`` does at an array's ends. A
    run shorter than the window has none to fit and stays NaN, as does every
    invalid row.

    Vectorized over runs, since a poorly tracked keypoint can break into tens of
    thousands of them, and one filter call per run would cost far more than the
    arithmetic does.

    Returns:
        One ``(n_rows, n_axes)`` array per kernel, in the kernels' order.
    """
    n, n_axes = positions.shape
    window = kernels[0].center.size
    half = window // 2
    starts, ends = _runs(valid)
    keep = ends - starts >= window
    starts, ends = starts[keep], ends[keep]
    if starts.size == 0:
        return [np.full((n, n_axes), np.nan) for _ in kernels]
    # Zero rather than NaN outside the runs: every window kept below lies wholly
    # inside one run, so the zeros never reach a result, while a NaN would make
    # the result depend on whether the arithmetic multiplies it by a zero
    # coefficient (the center coefficient of a first derivative is zero).
    filled = np.where(valid[:, None], positions, 0.0)
    outside = np.flatnonzero(~_cover(starts + half, ends - half, n))
    offsets = np.arange(window)
    edge = np.arange(half)
    head = filled[starts[:, None] + offsets]
    tail = filled[(ends - window)[:, None] + offsets]
    head_rows = starts[:, None] + edge
    tail_rows = (ends - half)[:, None] + edge
    outputs: list[FloatArray] = []
    for kernel in kernels:
        out = correlate1d(filled, kernel.center, axis=0, mode="constant", cval=0.0)
        out[outside] = np.nan
        out[head_rows] = np.einsum("jw,rwd->rjd", kernel.lead, head)
        out[tail_rows] = np.einsum("jw,rwd->rjd", kernel.trail, tail)
        outputs.append(out)
    return outputs


def savgol_derivatives(
    observed: FloatArray,
    layout: RowLayout,
    window: int,
    kernels: tuple[SavgolKernel, ...],
) -> tuple[list[FloatArray], BoolArray]:
    """One keypoint of every individual, differentiated along *layout*.

    Args:
        observed: ``(n_rows, n_axes)`` positions in output order, NaN where not
            finite.
        layout: Where each row sits on the axis the filter runs along.
        window: The longest gap to interpolate across.
        kernels: The derivatives to take.

    Returns:
        One ``(n_rows, n_axes)`` array per kernel on the input rows, and which
        input rows were missing the keypoint. Those rows are not yet masked.
    """
    absent = ~np.isfinite(observed).all(axis=1)
    contiguous = layout.size == observed.shape[0]
    if contiguous:
        dense, missing = observed, absent
    else:
        dense = np.full((layout.size, observed.shape[1]), np.nan)
        dense[layout.rows] = observed
        missing = np.ones(layout.size, dtype=bool)
        missing[layout.rows] = absent
    dense, missing = _fill_short_gaps(dense, missing, window, layout.breaks)
    derived = _savgol_on_runs(dense, ~missing, kernels)
    if not contiguous:
        derived = [values[layout.rows] for values in derived]
    return derived, absent


def row_norm(vectors: FloatArray) -> FloatArray:
    """The Euclidean length of each row."""
    return np.sqrt(np.einsum("ij,ij->i", vectors, vectors))
