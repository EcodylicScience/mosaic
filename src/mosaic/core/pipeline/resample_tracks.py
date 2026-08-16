"""Placing a tracks table on a uniform frame rate it was not recorded at.

A dataset whose recordings disagree on frame rate has a hidden problem that no
amount of care in the analysis code fixes: **every constant expressed in frames
means a different duration in each recording**. A ten-frame response lag is a
third of a second at 30 fps and 0.288 s at 34.679; a smoothing window of 31
frames is 1.03 s or 0.89 s; a bad-frame gate stated in cm/s is divided by the
frame rate to get cm/frame, so stating one number gives two thresholds. Making
each of those per-video is possible for the long windows and impossible for the
short ones -- five frames has no better neighbour than five -- and it makes every
lag a per-entry quantity that pooled analysis then has to carry.

Putting the tables on one grid instead makes all of it correct by construction,
and leaves every downstream parameter a single number again.

**The frame axis is a property of the tracks table, which is why this is not a
feature.** A feature adds columns at fixed row identity -- ``heading`` and
``scale-to-cm`` are features for exactly that reason. Re-gridding changes row
identity, and mosaic's loader merges a feature's inputs with an inner join on
``{frame, time, id}`` (``loading._merge_parquet_inputs``) with no check that two
inputs share an axis. A re-gridded table offered as a feature ``Result`` would
join to near-empty against anything reading the same tracks, and the entry would
then be *skipped without an error*. So this produces a tracks variant, and
``write_tracks_row`` records the new ``frame_min``/``frame_max`` it measures.

**Interpolate against ``frame / fps_native``, never against the ``time`` column.**
Some trackers embed wall-clock timestamps that jitter by milliseconds per frame
-- it is why ``speed-angvel`` offers an ``fps`` parameter that overrides ``time``
at all -- and resampling against them would write that jitter into positions.
The recorded rate names the axis; ``time`` is only consulted to *recover* the
rate when nothing else states it.

**Nothing here bridges a gap.** A target sample is NaN unless both of the native
samples bracketing it are finite, so a hole in the source stays a hole of the
same duration. Filling holes belongs to ``trajectory-smooth``, under limits it
declares and records; doing it here as a side effect of re-gridding would hide
the interpolation inside an operation named for something else.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from mosaic.core.pipeline.types import COLUMNS

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = [
    "AmbiguousFrameRateError",
    "MissingFrameRateError",
    "native_frame_rate",
    "resample_entry_table",
]

RATE_COLUMNS: tuple[str, ...] = ("frame_rate", "fps")
"""Columns that may state the rate a table was recorded at, in preference order.

``frame_rate`` is TRex's own name and survives conversion as a full column,
because the flattener broadcasts an NPZ scalar. ``fps`` is what three features
already look for (``pair-position``, ``pair-egocentric``, ``approach-avoidance``)
when they prefer a per-table rate over their parameter.
"""

_ANGULAR_FIELDS: frozenset[str] = frozenset(
    {
        COLUMNS.orientation_col,
        "ANGLE",
        "ORIENTATION",
        # Returns an angle despite the name -- see ``track_library.trex``, which
        # classifies it as dimensionless for the same underlying reason.
        "MIDLINE_OFFSET",
    }
)
"""Base field names holding an angle in radians, matched after stripping ``#``.

Interpolating one of these linearly across the +pi/-pi wrap invents a full sweep
of the body axis, which reads downstream as a real turn -- the same class of
artefact that made ``heading`` a feature with a declared method rather than
something a converter computed. They are interpolated on the unit circle.

Rates are deliberately absent: ``ANGULAR_V`` and ``ANGULAR_A`` are radians per
second, not positions on a circle, and interpolate linearly like any other
measurement.
"""

_IDENTITY_COLUMNS: frozenset[str] = frozenset(
    {COLUMNS.id_col, COLUMNS.group_col, COLUMNS.seq_col}
)
"""Columns that name the entry rather than measure it, carried unchanged."""


class MissingFrameRateError(ValueError):
    """A table does not say what rate it was recorded at, and none is recoverable.

    Fatal rather than defaulted. A resample against a guessed rate is a
    constant-factor error in every duration downstream, and every number it
    produces stays plausible -- which is the failure mode this whole module
    exists to remove, not one to reintroduce at its entrance.
    """


class AmbiguousFrameRateError(ValueError):
    """One table states two different frame rates.

    One entry is one recording on one time axis. Two rates for it is a
    contradiction, and averaging them or taking the first would silently pick
    one. A genuinely multi-rate session is several clips joined into one entry,
    and that is what ``core.media.timeline`` models -- a case this refuses rather
    than approximates.
    """


def _finite_pairs(
    frames: NDArray[np.float64], times: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The (frame, time) pairs where both are finite."""
    keep = np.isfinite(frames) & np.isfinite(times)
    return frames[keep], times[keep]


def _rate_from_column(frame: pd.DataFrame, source: str) -> float | None:
    """The rate a table states outright, or ``None`` when it states none."""
    for name in RATE_COLUMNS:
        if name not in frame.columns:
            continue
        values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
        stated = np.unique(values[np.isfinite(values) & (values > 0.0)])
        if stated.size == 0:
            continue
        if stated.size > 1:
            raise AmbiguousFrameRateError(
                f"{source or 'this table'} records {stated.size} different {name} "
                f"values ({sorted(float(v) for v in stated)}). One entry is one "
                "recording, so it has one frame rate; a session whose clips "
                "genuinely differ is several clips joined onto one timeline, "
                "which this cannot resample as a single rate."
            )
        return float(stated[0])
    return None


def native_frame_rate(frame: pd.DataFrame, *, source: str = "") -> float:
    """The rate *frame* was recorded at.

    A stated rate wins over a measured one. ``frame_rate`` is what the tracker
    wrote down; the endpoint estimate below only recovers it for a table that
    carries no such column.

    The estimator is deliberately ``(frame[-1] - frame[0]) / (time[-1] -
    time[0])`` over the whole span rather than a median of per-row differences.
    ``time`` is commonly ``float32``: three hours in, its resolution is about a
    millisecond, so consecutive differences snap to binary fractions and their
    median reads 30.117647 for a 30 fps recording -- 0.39 % high, and wrong in
    the same direction for every file, which is exactly the kind of error that
    survives a sanity check. Two endpoints two hours apart do not have that
    problem.

    Args:
        frame: One entry's table.
        source: What to name in an error message.

    Returns:
        Frames per second, strictly positive.

    Raises:
        AmbiguousFrameRateError: The table states two different rates.
        MissingFrameRateError: The table states none and none is recoverable.
    """
    stated = _rate_from_column(frame, source)
    if stated is not None:
        return stated

    where = source or "this table"
    if COLUMNS.frame_col not in frame.columns or COLUMNS.time_col not in frame.columns:
        raise MissingFrameRateError(
            f"{where} carries none of {list(RATE_COLUMNS)} and lacks "
            f"{COLUMNS.frame_col!r}/{COLUMNS.time_col!r}, so its frame rate can "
            "neither be read nor measured. Resampling against a default would be "
            "a constant-factor error in every duration derived from it."
        )

    frames, times = _finite_pairs(
        frame[COLUMNS.frame_col].to_numpy(dtype=float),
        frame[COLUMNS.time_col].to_numpy(dtype=float),
    )
    if frames.size < 2:
        raise MissingFrameRateError(
            f"{where} has fewer than two rows with both a frame and a time, so "
            "no rate can be measured from its span."
        )
    lo, hi = int(np.argmin(frames)), int(np.argmax(frames))
    span_frames = frames[hi] - frames[lo]
    span_seconds = times[hi] - times[lo]
    if span_frames <= 0.0 or span_seconds <= 0.0:
        raise MissingFrameRateError(
            f"{where} spans {span_frames} frames in {span_seconds} seconds, which "
            "names no rate. A table whose time axis does not advance with its "
            "frame axis cannot be placed on a uniform grid."
        )
    return float(span_frames / span_seconds)


def _bracket(
    native_times: NDArray[np.float64], target_times: NDArray[np.float64]
) -> tuple[NDArray[np.intp], NDArray[np.float64]]:
    """Left-bracket index and interpolation weight for each target time.

    Every target time is inside the native span by construction (see
    :func:`_target_frames`), so no clipping is needed beyond keeping the right
    bracket in range for the final sample.
    """
    left = np.searchsorted(native_times, target_times, side="right") - 1
    left = np.clip(left, 0, native_times.size - 2)
    step = native_times[left + 1] - native_times[left]
    weight = np.where(step > 0.0, (target_times - native_times[left]) / step, 0.0)
    return left, np.clip(weight, 0.0, 1.0)


def _both_ends_finite(
    values: NDArray[np.float64], left: NDArray[np.intp], weight: NDArray[np.float64]
) -> NDArray[np.bool_]:
    """Where an interpolated sample is defined.

    Both bracketing samples must be finite -- a gap is not bridged. The two
    exact-hit carve-outs are what make a resample onto the rate a table already
    carries a genuine identity: at weight 0 the answer is the left sample alone,
    so a missing *right* neighbour must not erase a value that was measured.
    """
    lo = np.isfinite(values[left])
    hi = np.isfinite(values[left + 1])
    return np.where(weight <= 0.0, lo, np.where(weight >= 1.0, hi, lo & hi))


def _interpolate_linear(
    values: NDArray[np.float64], left: NDArray[np.intp], weight: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Linear interpolation, NaN wherever the sample is not defined."""
    lo = np.where(np.isfinite(values[left]), values[left], 0.0)
    hi = np.where(np.isfinite(values[left + 1]), values[left + 1], 0.0)
    out = (1.0 - weight) * lo + weight * hi
    return np.where(_both_ends_finite(values, left, weight), out, np.nan)


def _interpolate_circular(
    values: NDArray[np.float64], left: NDArray[np.intp], weight: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Interpolation on the unit circle, wrapped back to (-pi, pi].

    Interpolating ``cos`` and ``sin`` separately and taking ``atan2`` takes the
    shorter way round every time, which is what a body axis does between two
    samples 30 ms apart. Doing it on the raw angle instead sweeps the long way
    across the +pi/-pi wrap and writes a full rotation into the table.
    """
    cos = _interpolate_linear(np.cos(values), left, weight)
    sin = _interpolate_linear(np.sin(values), left, weight)
    out = np.arctan2(sin, cos)
    return np.where(_both_ends_finite(values, left, weight), out, np.nan)


def _nearest(
    native_times: NDArray[np.float64],
    target_times: NDArray[np.float64],
    left: NDArray[np.intp],
    weight: NDArray[np.float64],
) -> NDArray[np.intp]:
    """The index of the native sample closest in time to each target."""
    del native_times, target_times
    return np.where(weight <= 0.5, left, left + 1)


def _target_frames(
    frame_min: float, frame_max: float, ratio: float
) -> NDArray[np.intp]:
    """The target frame numbers covering one series' own extent.

    The mapping is absolute -- target frame ``k`` is always time ``k /
    target_fps`` -- so every individual in a sequence lands on one shared grid
    even though each is interpolated over its own coverage. That is what keeps
    ``frame`` a key two individuals can be joined on, which is the whole reason
    the axis is global over the entry.

    Rounding inwards on both ends is what guarantees every target time sits
    inside the native span, so nothing is ever extrapolated.
    """
    low = math.ceil(frame_min * ratio)
    high = math.floor(frame_max * ratio)
    if high < low:
        return np.empty(0, dtype=np.intp)
    return np.arange(low, high + 1, dtype=np.intp)


def _prefiltered(
    values: dict[str, NDArray[np.float64]],
    native_times: NDArray[np.float64],
    prefilter: float,
) -> NDArray[np.bool_]:
    """Which native samples the displacement gate rejects.

    Expressed in table units per **second**, so one number is the same physical
    threshold at every recording rate -- which is the defect this module exists
    to fix, and would be silly to reintroduce in its own gate. It is the same
    quantity ``trajectory-smooth`` thresholds, applied one step earlier so that a
    mis-detection is removed before it is blended into its neighbours rather than
    after.

    The first sample has no predecessor and is never rejected: there is nothing
    to measure a displacement against, and rejecting it by default would trim a
    frame off every track.
    """
    x, y = values.get(COLUMNS.x_col), values.get(COLUMNS.y_col)
    if x is None or y is None or x.size < 2:
        return np.zeros(native_times.size, dtype=bool)
    step = np.diff(native_times)
    speed = np.full(native_times.size, np.nan, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        speed[1:] = np.hypot(np.diff(x), np.diff(y)) / np.where(
            step > 0.0, step, np.nan
        )
    return np.isfinite(speed) & (speed > prefilter)


def _column_plan(frame: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    """Split a table's columns into the three ways they are carried.

    Returns ``(interpolated, gathered, identity)``. Everything not a real float
    series is *gathered* from the nearest native sample rather than averaged:
    interpolating a tracklet id or a boolean produces a number that was never
    observed, and one that is not even of the right type.
    """
    interpolated: list[str] = []
    gathered: list[str] = []
    identity: list[str] = []
    for name in (str(column) for column in frame.columns):
        if name in _IDENTITY_COLUMNS:
            identity.append(name)
        elif name in (COLUMNS.frame_col, COLUMNS.time_col) or name in RATE_COLUMNS:
            continue  # recomputed from the target grid
        elif pd.api.types.is_bool_dtype(frame[name]) or pd.api.types.is_integer_dtype(
            frame[name]
        ):
            gathered.append(name)
        elif pd.api.types.is_float_dtype(frame[name]):
            interpolated.append(name)
        else:
            gathered.append(name)
    return interpolated, gathered, identity


def _base_field(name: str) -> str:
    """A TRex column name with its ``#`` qualifier removed."""
    return name.split("#", 1)[0]


def _resample_one_id(
    block: pd.DataFrame,
    native_fps: float,
    target_fps: float,
    prefilter: float | None,
    plan: tuple[list[str], list[str], list[str]],
    source: str,
) -> pd.DataFrame:
    """One individual's rows, re-gridded onto the target rate."""
    interpolated, gathered, identity = plan
    block = block.sort_values(COLUMNS.frame_col)
    native_frames = block[COLUMNS.frame_col].to_numpy(dtype=float)
    if native_frames.size < 2:
        return block.iloc[0:0]
    if np.any(np.diff(native_frames) <= 0.0):
        raise ValueError(
            f"{source or 'this table'} repeats a frame number for one individual, "
            "so its rows do not describe one time axis and no sample can be "
            "placed between them. Fix the source table rather than resampling it."
        )

    native_times = native_frames / native_fps
    target = _target_frames(
        float(native_frames[0]), float(native_frames[-1]), target_fps / native_fps
    )
    if target.size == 0:
        return block.iloc[0:0]
    target_times = target.astype(float) / target_fps
    left, weight = _bracket(native_times, target_times)
    nearest = _nearest(native_times, target_times, left, weight)

    series = {
        name: block[name].to_numpy(dtype=float, na_value=np.nan)
        if hasattr(block[name], "to_numpy")
        else np.asarray(block[name], dtype=float)
        for name in interpolated
    }
    reject = (
        _prefiltered(series, native_times, prefilter)
        if prefilter is not None
        else np.zeros(native_times.size, dtype=bool)
    )

    out: dict[str, object] = {}
    for name in block.columns:
        column = str(name)
        if column == COLUMNS.frame_col:
            out[column] = target
        elif column == COLUMNS.time_col:
            out[column] = target_times
        elif column in RATE_COLUMNS:
            out[column] = np.full(target.size, float(target_fps))
        elif column in identity:
            out[column] = np.repeat(block[column].to_numpy()[:1], target.size)
        elif column in gathered:
            out[column] = block[column].to_numpy()[nearest]
        else:
            values = np.where(reject, np.nan, series[column])
            # A column the flattener padded from a one-element array -- TRex
            # writes ``cm_per_pixel`` and ``video_size`` that way -- is a scalar
            # about the recording, not a series measured along it. It has no
            # bracket to interpolate between, and dropping it would take the
            # calibration with it, so it is carried in the same shape it arrived:
            # on the first row, NaN below.
            if int(np.count_nonzero(np.isfinite(values))) == 1:
                carried = np.full(target.size, np.nan)
                carried[0] = values[np.isfinite(values)][0]
                out[column] = carried
            elif _base_field(column) in _ANGULAR_FIELDS:
                out[column] = _interpolate_circular(values, left, weight)
            else:
                out[column] = _interpolate_linear(values, left, weight)

    resampled = pd.DataFrame(out, columns=list(block.columns))
    return resampled.astype(
        {name: block[name].dtype for name in gathered if name in resampled.columns}
    )


def resample_entry_table(
    frame: pd.DataFrame,
    target_fps: float,
    *,
    prefilter: float | None = None,
    source: str = "",
) -> pd.DataFrame:
    """One entry's tracks table, placed on a uniform *target_fps* grid.

    Each individual is interpolated over its own frame extent, onto the one
    absolute grid ``frame k <-> time k / target_fps``, so the entry keeps a
    single frame axis that two individuals can still be joined on.

    Args:
        frame: One ``(group, sequence)`` table, on whatever rate it was recorded
            at.
        target_fps: The rate to place it on.
        prefilter: Reject native samples whose displacement from their
            predecessor exceeds this many table units per second, before
            interpolating. ``None`` interpolates the samples as they are; see
            :func:`_prefiltered` for when each is appropriate.
        source: What to name in an error message.

    Returns:
        The re-gridded table, columns and dtypes preserved, individuals in the
        order they first appear. Empty in, empty out.

    Raises:
        AmbiguousFrameRateError: The table states two different rates.
        MissingFrameRateError: The table states no rate and none is recoverable.
        ValueError: *target_fps* is not positive, a required column is absent, or
            one individual repeats a frame number.
    """
    if not target_fps > 0.0:
        raise ValueError(f"target_fps={target_fps} is not a usable frame rate.")
    if frame.empty:
        return frame.copy()
    for required in (COLUMNS.frame_col, COLUMNS.id_col):
        if required not in frame.columns:
            raise ValueError(
                f"{source or 'this table'} has no {required!r} column, so it "
                "cannot be re-gridded: a resample is defined per individual "
                "along a frame axis."
            )

    native_fps = native_frame_rate(frame, source=source)
    plan = _column_plan(frame)
    blocks = [
        _resample_one_id(block, native_fps, target_fps, prefilter, plan, source)
        for _, block in frame.groupby(COLUMNS.id_col, sort=False)
    ]
    kept = [block for block in blocks if not block.empty]
    if not kept:
        return frame.iloc[0:0]
    return pd.concat(kept, ignore_index=True)
