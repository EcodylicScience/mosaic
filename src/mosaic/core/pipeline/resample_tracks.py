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

**Interpolate against the ``time`` column.** Every track schema carries ``time`` in
seconds, and a table that dropped frames is placed correctly only by it: a native
sample's position on the axis is when it was taken, not its frame number over a
mean rate. Target frame ``k`` is time ``k / target_fps``, so every individual
lands on one grid whatever its own coverage. A table whose ``time`` runs on a
jittery wall clock writes that jitter into positions, as every method that reads
``time`` does. ``frame_rate`` and ``fps`` columns the source carries are
rewritten at the target rate and never consulted.

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

from mosaic.core.individuals import iter_individuals, refuse_repeated_rows
from mosaic.core.pipeline.types import COLUMNS

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["resample_entry_table"]

RATE_COLUMNS: tuple[str, ...] = ("frame_rate", "fps")
"""Columns that may state the rate a table was recorded at.

``frame_rate`` is TRex's own name and survives conversion as a full column,
because the flattener broadcasts an NPZ scalar. The output states the target rate
in each that the source has.
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


_SNAP = 1e-3
"""The fraction of a step within which a target time is the native sample it meets."""


def _time_slack(native_times: NDArray[np.float64], is_float32: bool) -> float:
    """The seconds that a native time may differ from the clock it was rounded from.

    A ``float32`` time is good to its resolution at its magnitude, about a
    millisecond three hours into a recording and a few percent of a 30 fps step.
    Resampling a table onto the rate it already has would otherwise blend each
    sample with its neighbor and spread a missing sample to both of them. A
    ``float64`` time has no slack beyond :data:`_SNAP`.
    """
    if not is_float32 or native_times.size == 0:
        return 0.0
    return float(np.finfo(np.float32).eps) * float(np.max(np.abs(native_times)))


def _bracket(
    native_times: NDArray[np.float64],
    target_times: NDArray[np.float64],
    slack: float = 0.0,
) -> tuple[NDArray[np.intp], NDArray[np.float64]]:
    """Left-bracket index and interpolation weight for each target time.

    Every target time is inside the native span by construction (see
    :func:`_target_frames`), so no clipping is needed beyond keeping the right
    bracket in range for the final sample. A target within :data:`_SNAP` of a step,
    or within *slack* seconds, of a bracket's end is that end's.
    """
    left = np.searchsorted(native_times, target_times, side="right") - 1
    left = np.clip(left, 0, native_times.size - 2)
    step = native_times[left + 1] - native_times[left]
    weight = np.where(step > 0.0, (target_times - native_times[left]) / step, 0.0)
    weight = np.clip(weight, 0.0, 1.0)
    tolerance = np.where(
        step > 0.0, np.maximum(_SNAP, slack / np.where(step > 0.0, step, 1.0)), _SNAP
    )
    weight = np.where(weight < tolerance, 0.0, weight)
    return left, np.where(weight > 1.0 - tolerance, 1.0, weight)


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
    first_time: float, last_time: float, target_fps: float, slack: float = 0.0
) -> NDArray[np.intp]:
    """The target frame numbers covering one series' own span of time.

    The mapping is absolute -- target frame ``k`` is always time ``k /
    target_fps`` -- so every individual in a sequence lands on one shared grid
    even though each is interpolated over its own coverage. That is what keeps
    ``frame`` a key two individuals can be joined on, which is the whole reason
    the axis is global over the entry.

    Rounding inwards on both ends is what guarantees every target time sits
    inside the native span, so nothing is ever extrapolated.
    """
    low = math.ceil((first_time - slack) * target_fps - _SNAP)
    high = math.floor((last_time + slack) * target_fps + _SNAP)
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
    target_fps: float,
    prefilter: float | None,
    plan: tuple[list[str], list[str], list[str]],
    source: str,
) -> pd.DataFrame:
    """One individual's rows, given in frame order, re-gridded onto the target rate."""
    interpolated, gathered, identity = plan
    refuse_repeated_rows(block, source or "this table")
    native_times = block[COLUMNS.time_col].to_numpy(dtype=float)
    if native_times.size < 2:
        return block.iloc[0:0]
    if not (np.all(np.isfinite(native_times)) and np.all(np.diff(native_times) > 0.0)):
        raise ValueError(
            f"{source or 'this table'}: id {block[COLUMNS.id_col].iloc[0]!r} has a "
            f"{COLUMNS.time_col!r} that does not increase with {COLUMNS.frame_col!r}, "
            "so its rows have no order in time to interpolate along."
        )
    slack = _time_slack(native_times, str(block[COLUMNS.time_col].dtype) == "float32")
    target = _target_frames(
        float(native_times[0]), float(native_times[-1]), target_fps, slack
    )
    if target.size == 0:
        return block.iloc[0:0]
    target_times = target.astype(float) / target_fps
    left, weight = _bracket(native_times, target_times, slack)
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
        ValueError: *target_fps* is not positive, a required column is absent, one
            individual repeats a frame number, or an individual's ``time`` does not
            increase with its frame.
    """
    if not target_fps > 0.0:
        raise ValueError(f"target_fps={target_fps} is not a usable frame rate.")
    if frame.empty:
        return frame.copy()
    for required in (COLUMNS.frame_col, COLUMNS.time_col, COLUMNS.id_col):
        if required not in frame.columns:
            raise ValueError(
                f"{source or 'this table'} has no {required!r} column, so it "
                "cannot be re-gridded: a resample is defined per individual "
                "along a frame axis, placed by its time."
            )

    plan = _column_plan(frame)
    blocks = [
        _resample_one_id(block, target_fps, prefilter, plan, source)
        for _, block in iter_individuals(frame)
    ]
    kept = [block for block in blocks if not block.empty]
    if not kept:
        return frame.iloc[0:0]
    return pd.concat(kept, ignore_index=True)
