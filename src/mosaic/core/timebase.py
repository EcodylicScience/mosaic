"""Reading a tracks table's time axis: step seconds, a measured rate, irregularity.

Every track schema carries ``time``, in seconds, so no feature needs a frame
rate of its own to turn a frame step into a duration. A feature that
differences positions divides by the seconds between the rows
(:func:`step_seconds`). A method that reads its rows as evenly spaced samples
(a Savitzky-Golay derivative, a wavelet, a window given in seconds) needs one
rate for the whole table, which :func:`uniform_grid_rate` measures as the frame
span over the time span. It also says so when the steps are not even, because
that method then misplaces every sample after a gap. The remedy it names is
``resample-tracks``, which interpolates on ``time`` and returns a table the
method may read as it is.

The rate is measured over the whole span rather than as a median of per-row
steps. ``time`` is often ``float32``, whose resolution is about a millisecond
three hours in, so consecutive differences snap to binary fractions and their
median is biased in the same direction for every file.

This module imports numpy and the standard-library run-log module, so the label
converters, the resampling op and the features share it.
"""

from __future__ import annotations

import logging
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from mosaic.runlog import time_irregular_sentence

if TYPE_CHECKING:
    import pandas as pd
    from numpy.typing import NDArray

__all__ = [
    "IRREGULAR_STEP_RATIO",
    "IrregularNotice",
    "TimeAxis",
    "collect_irregular_notices",
    "last_frames_at_or_before",
    "local_periods",
    "measure_time_axis",
    "step_seconds",
    "uniform_grid_rate",
    "warn_if_irregular",
]

FRAME_COLUMN = "frame"
TIME_COLUMN = "time"

IRREGULAR_STEP_RATIO = 1.5
"""How many times the median step the largest step may be before a table is irregular.

One dropped frame doubles a step, so 1.5 flags it, and the millisecond jitter of a
wall clock does not.
"""

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TimeAxis:
    """What a table's frame and time columns say about its sampling.

    Attributes:
        rate: Frames per second, the frame span over the time span.
        median_step: The median seconds per frame step.
        largest_step: The longest step in seconds.
        largest_step_frames: The frame before and the frame after that step.
        ratio: The largest seconds-per-frame step over the median one, which is
            1 for an evenly sampled table.
    """

    rate: float
    median_step: float
    largest_step: float
    largest_step_frames: tuple[int, int]
    ratio: float

    @property
    def irregular(self) -> bool:
        """Whether a step runs longer than :data:`IRREGULAR_STEP_RATIO` medians."""
        return self.ratio > IRREGULAR_STEP_RATIO


def step_seconds(
    time: NDArray[np.float64],
    *,
    lag: int = 1,
    same: NDArray[np.bool_] | None = None,
) -> NDArray[np.float64]:
    """Seconds from each row back to the row *lag* before it.

    NaN where the step is undefined: the first *lag* rows, a row whose
    predecessor belongs to another individual (*same* False), a time that is not
    finite, and a step that is not positive.

    Args:
        time: The time of each row, ordered within each individual.
        lag: How many rows back the step reaches.
        same: Whether each row is the same individual as the row before it.
            Meaningful for a *lag* of 1 only. For a larger lag, pass one
            individual's rows at a time.

    Returns:
        An array shaped like *time*.

    Raises:
        ValueError: If *lag* is below 1, or *same* is given with a *lag* above 1.
    """
    if lag < 1:
        raise ValueError(f"lag={lag} must be at least 1.")
    if same is not None and lag != 1:
        raise ValueError("same applies to a lag of 1; pass one individual's rows.")
    steps = np.full(time.shape[0], np.nan, dtype=np.float64)
    if time.shape[0] > lag:
        steps[lag:] = time[lag:] - time[:-lag]
    usable = np.isfinite(steps) & (steps > 0.0)
    if same is not None:
        usable &= same
    return np.where(usable, steps, np.nan)


def measure_time_axis(df: pd.DataFrame, who: str) -> TimeAxis:
    """Measure the rate and the steps of *df*'s frame and time columns.

    Frames are taken once each, at the first time recorded for them, so a table
    of many individuals or of both perspectives of a pair is one axis. A frame
    that no individual has is a step that reaches across it. A table with no
    ``frame`` column is read as one row per distinct time, in time order.

    Args:
        df: The table, with a ``time`` column.
        who: The name that opens an error message.

    Raises:
        ValueError: If ``time`` is missing, or fewer than two frames carry a time,
            or the time does not advance with the frame.
    """
    if TIME_COLUMN not in df.columns:
        raise ValueError(
            f"{who} reads its rate off the {TIME_COLUMN!r} column, which the table "
            f"lacks: it has {sorted(map(str, df.columns))}."
        )
    times = df[TIME_COLUMN].to_numpy(dtype=np.float64)
    if FRAME_COLUMN in df.columns:
        frames = df[FRAME_COLUMN].to_numpy(dtype=np.float64)
    else:
        frames = np.unique(times[np.isfinite(times)], return_inverse=True)[1].astype(
            np.float64
        )
        frames = np.where(np.isfinite(times), frames, np.nan)
    keep = np.isfinite(frames) & np.isfinite(times)
    unique_frames, first = np.unique(frames[keep], return_index=True)
    unique_times = times[keep][first]
    if unique_frames.size < 2:
        raise ValueError(
            f"{who} cannot measure a frame rate: fewer than two frames carry "
            f"a {TIME_COLUMN!r}."
        )
    frame_span = float(unique_frames[-1] - unique_frames[0])
    time_span = float(unique_times[-1] - unique_times[0])
    if time_span <= 0.0:
        raise ValueError(
            f"{who} cannot measure a frame rate: {TIME_COLUMN!r} spans "
            f"{time_span:g} s over {frame_span:g} frames, so it does not advance "
            f"with {FRAME_COLUMN!r}."
        )
    seconds = np.diff(unique_times)
    frame_steps = np.diff(unique_frames)
    per_frame = seconds / frame_steps
    usable = per_frame > 0.0
    if not usable.any():
        raise ValueError(
            f"{who} cannot measure a frame rate: no two consecutive frames "
            f"advance in {TIME_COLUMN!r}."
        )
    largest = int(np.argmax(np.where(usable, per_frame, -np.inf)))
    median = float(np.median(per_frame[usable]))
    return TimeAxis(
        rate=frame_span / time_span,
        median_step=median,
        largest_step=float(seconds[largest]),
        largest_step_frames=(
            int(unique_frames[largest]),
            int(unique_frames[largest + 1]),
        ),
        ratio=float(per_frame[largest] / median),
    )


def local_periods(
    frames: NDArray[np.float64],
    times: NDArray[np.float64],
    fallback: float,
) -> NDArray[np.float64]:
    """Seconds per frame at each row of one individual's series.

    A row's period is its step to the next row, in seconds over frames, so a row
    before a dropped stretch lasts as long as the stretch, and a row before a
    missing frame is not stretched by it. The last row repeats the period before
    it. A row with no usable step, such as the only row of a series or one before
    a repeated time, takes *fallback*.

    Args:
        frames: The frame of each row, ascending and unique.
        times: The time of each row, in seconds.
        fallback: The period for a row with no usable step.

    Returns:
        An array shaped like *frames*.
    """
    periods = np.full(frames.shape[0], fallback, dtype=np.float64)
    if frames.shape[0] < 2:
        return periods
    with np.errstate(divide="ignore", invalid="ignore"):
        step = np.diff(times) / np.diff(frames)
    step = np.where(np.isfinite(step) & (step > 0.0), step, np.nan)
    periods[:-1] = step
    periods[-1] = step[-1]
    return np.where(np.isfinite(periods), periods, fallback)


@dataclass(frozen=True)
class IrregularNotice:
    """One uneven time axis that a method warned about.

    Attributes:
        who: The feature that read the table as evenly spaced.
        largest_step: The longest step in seconds.
        first_frame: The frame before that step.
        last_frame: The frame after it.
        median_step: The median seconds per frame step.
        rate: The frames per second the method used.
        message: The sentence the warning logs, naming the largest gap and
            ``resample-tracks``.
    """

    who: str
    largest_step: float
    first_frame: int
    last_frame: int
    median_step: float
    rate: float
    message: str


_COLLECTOR: ContextVar[list[IrregularNotice] | None] = ContextVar(
    "irregular_notices", default=None
)


@contextmanager
def collect_irregular_notices() -> Generator[list[IrregularNotice]]:
    """Collect the notices that :func:`warn_if_irregular` raises in this block.

    ``run_feature`` wraps each entry's ``apply`` in it and records what it collects
    as a run-log event, so a warning reaches a user whose process the queue started
    with its standard error discarded. The warning is still logged. The collector
    belongs to the calling thread or process, and a notice is kept once for each
    feature and gap.
    """
    notices: list[IrregularNotice] = []
    token = _COLLECTOR.set(notices)
    try:
        yield notices
    finally:
        _COLLECTOR.reset(token)


def warn_if_irregular(who: str, axis: TimeAxis) -> None:
    """Log one warning when *axis* steps unevenly, naming the largest gap.

    The message names ``resample-tracks``, the op that places a table on a
    uniform grid by interpolating on ``time``. A collector (see
    :func:`collect_irregular_notices`) also receives it.
    """
    if not axis.irregular:
        return
    first, last = axis.largest_step_frames
    message = time_irregular_sentence(
        axis.largest_step, first, last, axis.median_step, axis.rate
    )
    collector = _COLLECTOR.get()
    if collector is not None:
        notice = IrregularNotice(
            who=who,
            largest_step=axis.largest_step,
            first_frame=first,
            last_frame=last,
            median_step=axis.median_step,
            rate=axis.rate,
            message=message,
        )
        if notice not in collector:
            collector.append(notice)
    log.warning("[%s] %s", who, message)


def uniform_grid_rate(df: pd.DataFrame, who: str) -> float:
    """The frame rate of a method that reads *df*'s rows as evenly spaced.

    Measures the rate from ``time`` and warns when the steps are uneven.

    Args:
        df: The table, with a ``frame`` and a ``time`` column.
        who: The name that opens a message.

    Returns:
        Frames per second.

    Raises:
        ValueError: As :func:`measure_time_axis` raises it.
    """
    axis = measure_time_axis(df, who)
    warn_if_irregular(who, axis)
    return axis.rate


def last_frames_at_or_before(
    track_frames: NDArray[np.float64],
    track_times: NDArray[np.float64],
    seconds: NDArray[np.float64],
) -> NDArray[np.float64]:
    """The last track frame whose time is at or before each of *seconds*.

    The frame an event at a given second belongs to when the track is timed by
    ``time``: a floor, which on an evenly timed table is ``int(seconds * rate)`` and
    on one that dropped frames is the frame the clock had reached, not ``seconds *
    rate``.

    Args:
        track_frames: The frame of each track row.
        track_times: The time of each track row, in seconds.
        seconds: The event times.

    Returns:
        The frame for each event, NaN where the event is before the first track
        row or no track row has a finite time.
    """
    keep = np.isfinite(track_frames) & np.isfinite(track_times)
    order = np.argsort(track_times[keep], kind="stable")
    times = track_times[keep][order]
    frames = track_frames[keep][order]
    out = np.full(seconds.shape, np.nan)
    if times.size == 0:
        return out
    at = np.searchsorted(times, seconds, side="right") - 1
    inside = at >= 0
    out[inside] = frames[at[inside]]
    return out
