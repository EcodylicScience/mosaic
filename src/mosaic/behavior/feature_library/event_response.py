"""
EventResponse -- one row per individual and event, measuring a value around its onset.

Each event has an onset time. For every individual and event the feature takes
the mean of the value over ``baseline_s`` seconds before the onset, the minimum
and the mean over ``response_s`` seconds from it, and the latency until the
value recovers to ``recover_fraction`` times its baseline. It measures a
response that lowers the value, such as the freezing that a looming stimulus
causes in ``speed``.

Onsets come from one of three sources:

- ``onset_column``: each row where the column turns active after an inactive
  row, such as Tracktor's ``stimuli`` column (``onset_value="Y"``).
- ``labels`` with ``onset_classes``: each row where an individual's label turns
  to one of the classes, from a label file of the dataset.
- ``onset_times``: the listed times, the same for every individual.

The latency is anchored at the response window's minimum, the trough, so a
single low sample does not start it. A noisy value belongs to a low-pass filter
upstream: ``speed-angvel`` writes ``speed_smooth`` under ``smooth_window``, and
``trajectory-smooth`` smooths the positions before ``speed-angvel`` reads them.

Windows and latency are measured on the ``time`` column, in its own units
(seconds), and never from a frame count or a frame rate. A table that dropped
frames, or whose time steps differ, is measured the same way: a window holds the
rows whose time falls inside it.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Annotated, Any, ClassVar, Final, Self, final

import numpy as np
import pandas as pd
from pydantic import Field, field_validator, model_validator

from mosaic.core.params import Declared, Params
from mosaic.core.pipeline.column_contract import (
    Column,
    ParamColumn,
    Provides,
    Requirement,
    kept,
)
from mosaic.core.pipeline.loading import labels_for_rows
from mosaic.core.pipeline.types import (
    COLUMNS as C,
    DependencyLookup,
    EmitsLevel,
    GroundTruthLabelsSource,
    Inputs,
    InputStream,
    Result,
    TrackInput,
)
from mosaic.core.pipeline.types.feature import RelatesLevel
from mosaic.core.schema import ARENA_COLUMN, TRACK_SPACES, TrackSpace

from .helpers import ensure_columns, entry_identity
from .registry import register_feature

_ACTIVE: Final = "_event_response_active"
"""The working column that marks the rows where an onset source is active."""

_VALUE_COLUMN_DESCRIPTION = (
    "The column measured around each onset, such as the speed column of speed-angvel."
)

_ONSET_COLUMN_DESCRIPTION = (
    "A column whose rising edge is an onset: each row where it is active after a "
    "row where it is not. The first row of an individual has no row before it, so "
    "it is never an onset. A null is not active. Give exactly one of onset_column, "
    "labels and onset_times."
)

_ONSET_VALUE_DESCRIPTION = (
    "The value of onset_column that means active, such as Y for Tracktor's "
    "stimuli column. Unset, a boolean column is active where true and a numeric "
    "column where it is not 0. A column of text needs it."
)

_LABELS_DESCRIPTION = (
    "A label kind of the dataset whose runs of onset_classes are the events. An "
    "individual's label follows each of its rows, and its remap and mask apply "
    "before onset_classes. The first row of an individual is never an onset. Give "
    "exactly one of onset_column, labels and onset_times."
)

_ONSET_CLASSES_DESCRIPTION = (
    "The label classes that count as an event, after the labels' remap. Each run "
    "of rows labeled with one of them starts an event at its first row. Only with "
    "labels."
)

_ONSET_TIMES_DESCRIPTION = (
    "The onset times, in the units of the time column. Every individual has a row "
    "for every time, numbered in ascending order. Give exactly one of "
    "onset_column, labels and onset_times."
)

_BASELINE_DESCRIPTION = (
    "The length of the window before each onset whose mean is the baseline. It "
    "holds the rows whose time is at or after the onset minus this and before the "
    "onset."
)

_RESPONSE_DESCRIPTION = (
    "The length of the window from each onset over which the minimum, the mean and "
    "the recovery are measured. It holds the rows whose time is at or after the "
    "onset and before the onset plus this. A window that runs past the end of the "
    "table holds the rows there are."
)

_RECOVER_FRACTION_DESCRIPTION = (
    "The fraction of the baseline that the value must return to. The latency is "
    "the time from the onset to the first row after the response window's minimum "
    "whose value is at or above that level. A response whose minimum is at or "
    "above the level recovers at latency 0. One that does not return has a NaN "
    "latency and recovered false, and one with no baseline or no response rows has "
    "a NaN latency and a null recovered."
)


@final
@register_feature
class EventResponse:
    """
    Per-sequence feature measuring a value around each event, per individual.

    Output columns (one row per individual and event):
      - group, sequence: the entry, copied from the frame
      - id: the individual
      - arena: when the input has it
      - event: the event's number in time order, from 0
      - onset_time: the event's onset, in the time column's units
      - baseline_mean: the mean of the value over ``baseline_s`` before the onset
      - response_min, response_mean: the minimum and mean over ``response_s``
        from the onset
      - recovery_latency_s: the time from the onset until the value recovers to
        ``recover_fraction`` times the baseline, NaN when it does not
      - recovered: whether it did, a nullable boolean that is null when the
        baseline or the response has no value to measure
      - n_baseline, n_response: the rows with a value in each window

    Field documentation is on
    :class:`~mosaic.behavior.feature_library.event_response.EventResponse.Params`.

    Notes
    -----
    A window with no value gives NaN for its statistics. The latency is the time
    from the onset to the first value at or above the recovery level after the
    response window's minimum, the trough. It is 0 when the minimum is at or
    above the level, and NaN with ``recovered`` false when nothing after the
    trough reaches it. It is NaN with ``recovered`` null, not false, when there is
    nothing to measure: the baseline is NaN, or the response window holds no
    value. A column of nulls does not say the value failed to recover.

    An individual with no row inside an explicit onset's windows still has a row
    for it, with counts of 0. Under ``onset_column`` and ``labels``, ``event``
    counts an individual's own onsets, so the same stimulus can be event 1 of one
    individual and event 0 of another. ``onset_time`` is the key that joins one
    stimulus across individuals. Two events closer than ``response_s`` share rows
    of their windows.

    A single low sample sets the trough, so a noisy value gives a latency
    measured from the wrong sample. Filter it upstream: ``speed-angvel`` writes
    ``speed_smooth`` when ``smooth_window`` is set (name it as ``value_column``),
    and ``trajectory-smooth`` smooths the positions before ``speed-angvel``.

    The value is the column of an upstream input, so join the table that carries
    ``onset_column`` as a second input, such as the tracks beside a
    ``speed-angvel`` result.
    """

    category = "summary"
    name = "event-response"
    version = "0.1"
    parallelizable = True
    scope_dependent = False
    relates: RelatesLevel = "none"
    # One row per individual and event of the entry, which a neighbor's rows
    # would change.
    accepts_overlap = False
    consumed_roots: tuple[str, ...] = ()
    emits: EmitsLevel = "individual"
    follows_ids: bool = True
    track_spaces: ClassVar[frozenset[TrackSpace]] = frozenset(TRACK_SPACES)
    requires: ClassVar[tuple[Requirement, ...]] = (
        Column(C.id_col),
        Column(C.time_col),
        Column(C.group_col),
        Column(C.seq_col),
        ParamColumn("value_column"),
        ParamColumn("onset_column"),
    )
    provides: ClassVar[Provides] = Provides(
        adds=(
            Column(C.group_col),
            Column(C.seq_col),
            Column(C.id_col),
            Column("event"),
            Column("onset_time"),
            Column("baseline_mean"),
            Column("response_min"),
            Column("response_mean"),
            Column("recovery_latency_s"),
            Column("recovered"),
            Column("n_baseline"),
            Column("n_response"),
        ),
        conditional=kept(ARENA_COLUMN),
    )

    class Inputs(Inputs[TrackInput | Result]):
        pass

    class Params(Params):
        value_column: Annotated[str, Declared(_VALUE_COLUMN_DESCRIPTION)] = "speed"
        onset_column: Annotated[str | None, Declared(_ONSET_COLUMN_DESCRIPTION)] = None
        onset_value: Annotated[
            str | float | bool | None, Declared(_ONSET_VALUE_DESCRIPTION)
        ] = None
        labels: Annotated[
            GroundTruthLabelsSource | None, Declared(_LABELS_DESCRIPTION)
        ] = None
        onset_classes: Annotated[
            list[int] | None, Declared(_ONSET_CLASSES_DESCRIPTION)
        ] = Field(default=None, min_length=1)
        onset_times: Annotated[
            list[float] | None, Declared(_ONSET_TIMES_DESCRIPTION, unit="s")
        ] = Field(default=None, min_length=1)
        baseline_s: Annotated[float, Declared(_BASELINE_DESCRIPTION, unit="s")] = Field(
            default=10.0, gt=0.0
        )
        response_s: Annotated[float, Declared(_RESPONSE_DESCRIPTION, unit="s")] = Field(
            default=30.0, gt=0.0
        )
        recover_fraction: Annotated[float, Declared(_RECOVER_FRACTION_DESCRIPTION)] = (
            Field(default=0.9, gt=0.0)
        )

        @field_validator("onset_classes")
        @classmethod
        def _canonical_classes(cls, value: list[int] | None) -> list[int] | None:
            """Sort the classes without repeats, so one set is one identity."""
            return sorted(set(value)) if value else value

        @field_validator("onset_times")
        @classmethod
        def _canonical_times(cls, value: list[float] | None) -> list[float] | None:
            """Refuse a time that is not finite, and sort without repeats."""
            if value is None:
                return None
            if not all(math.isfinite(time) for time in value):
                msg = "onset_times must be finite"
                raise ValueError(msg)
            return sorted(set(value))

        @model_validator(mode="after")
        def _one_source_of_onsets(self) -> Self:
            """Require exactly one onset source, with the settings it reads."""
            sources = [
                self.onset_column is not None,
                self.labels is not None,
                self.onset_times is not None,
            ]
            if sum(sources) != 1:
                msg = (
                    "give exactly one of onset_column, labels and onset_times, "
                    f"not {sum(sources)}"
                )
                raise ValueError(msg)
            if (self.labels is not None) != (self.onset_classes is not None):
                msg = "onset_classes goes with labels, and labels needs onset_classes"
                raise ValueError(msg)
            if self.onset_value is not None and self.onset_column is None:
                msg = "onset_value applies only with onset_column"
                raise ValueError(msg)
            return self

    def __init__(
        self,
        inputs: EventResponse.Inputs = Inputs(("tracks",)),
        params: dict[str, object] | None = None,
    ):
        self.inputs = inputs
        self.params = self.Params.from_overrides(params)
        self._labels_lookup: DependencyLookup = {}

    # --- State protocol (stateless) ---

    def load_state(
        self,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_lookups: dict[str, DependencyLookup],
    ) -> bool:
        self._labels_lookup = dependency_lookups.get("labels", {})
        return True

    def fit(self, inputs: InputStream) -> None:
        pass

    def save_state(self, run_root: Path) -> None:
        pass

    # --- Apply ---

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()
        params = self.params
        ensure_columns(df, [C.id_col, C.time_col, params.value_column])
        if params.onset_column is not None:
            ensure_columns(df, [params.onset_column])
        group, sequence = entry_identity(df)
        frame = df.assign(**{_ACTIVE: self._active_rows(df, group, sequence)})
        keys = [C.id_col] + ([ARENA_COLUMN] if ARENA_COLUMN in df.columns else [])
        rows: list[dict[str, object]] = []
        for key, rows_of_one in frame.groupby(keys, sort=True, dropna=True):
            timed = rows_of_one.dropna(subset=[C.time_col]).sort_values(
                C.time_col, kind="stable"
            )
            time = timed[C.time_col].to_numpy(dtype=float)
            value = timed[params.value_column].to_numpy(dtype=float)
            for event, onset in enumerate(
                self._onsets(time, timed[_ACTIVE].to_numpy(dtype=bool))
            ):
                measured = _measure(time, value, onset, params)
                rows.append(
                    {
                        C.group_col: group,
                        C.seq_col: sequence,
                        **dict(zip(keys, key, strict=True)),
                        "event": event,
                        "onset_time": onset,
                        **measured,
                    }
                )
        out = pd.DataFrame(rows, columns=[*_key_columns(keys), *_MEASURE_COLUMNS])
        out["recovered"] = out["recovered"].astype("boolean")
        return out

    def _onsets(self, time: np.ndarray, active: np.ndarray) -> list[float]:
        """The onset times of one individual, ascending.

        *time* and *active* are the individual's rows in time order.
        """
        if self.params.onset_times is not None:
            return list(self.params.onset_times)
        return [float(t) for t in time[_rising_edges(active)]]

    def _active_rows(self, df: pd.DataFrame, group: str, sequence: str) -> np.ndarray:
        """Whether each row of *df* is active under the onset source.

        An ``onset_times`` source has no active rows.
        """
        params = self.params
        if params.onset_column is not None:
            return _column_active(df[params.onset_column], params.onset_value)
        if params.labels is not None and params.onset_classes is not None:
            path = self._labels_lookup.get((group, sequence))
            if path is None:
                msg = (
                    f"[event-response] entry ({group!r}, {sequence!r}) has no label "
                    f"file of kind {params.labels.kind!r}."
                )
                raise ValueError(msg)
            labels = labels_for_rows(path, df, params.labels)
            return np.isin(labels, params.onset_classes)
        return np.zeros(len(df), dtype=bool)


_MEASURE_COLUMNS: Final = (
    "baseline_mean",
    "response_min",
    "response_mean",
    "recovery_latency_s",
    "recovered",
    "n_baseline",
    "n_response",
)


def _key_columns(keys: list[str]) -> list[str]:
    """The columns that name a row: the entry, the individual, and its event."""
    return [C.group_col, C.seq_col, *keys, "event", "onset_time"]


def _rising_edges(active: np.ndarray) -> np.ndarray:
    """The positions that are active after an inactive position."""
    return np.flatnonzero(active[1:] & ~active[:-1]) + 1


def _column_active(
    column: pd.Series[Any], value: str | float | bool | None
) -> np.ndarray:
    """Whether each row of *column* is active, a null never."""
    if value is not None:
        return (column == value).to_numpy(dtype=bool)
    if column.dtype == bool:
        return column.to_numpy(dtype=bool)
    if pd.api.types.is_numeric_dtype(column):
        return (column.notna() & (column != 0)).to_numpy(dtype=bool)
    msg = (
        f"[event-response] column {column.name!r} holds text, so onset_value must "
        "name the value that means active."
    )
    raise ValueError(msg)


def _measure(
    time: np.ndarray, value: np.ndarray, onset: float, params: EventResponse.Params
) -> dict[str, object]:
    """The baseline, response and recovery of one individual around one onset.

    *time* ascends, and *value* is the value at each time.
    """
    known = ~np.isnan(value)
    before = known & (time >= onset - params.baseline_s) & (time < onset)
    after = known & (time >= onset) & (time < onset + params.response_s)
    baseline = float(value[before].mean()) if before.any() else math.nan
    response_time, response = time[after], value[after]
    measured: dict[str, object] = {
        "baseline_mean": baseline,
        "response_min": float(response.min()) if response.size else math.nan,
        "response_mean": float(response.mean()) if response.size else math.nan,
        "n_baseline": int(before.sum()),
        "n_response": int(after.sum()),
    }
    latency, recovered = _recovery(
        response_time, response, onset, baseline * params.recover_fraction
    )
    measured["recovery_latency_s"] = latency
    measured["recovered"] = recovered
    return {name: measured[name] for name in _MEASURE_COLUMNS}


def _recovery(
    time: np.ndarray, value: np.ndarray, onset: float, level: float
) -> tuple[float, bool | None]:
    """The latency from *onset* to *level* after the trough, and whether it came.

    The trough is the first minimum of *value*. The latency is the time of the
    first value after it at or above *level*, less *onset*. It is ``0.0`` when
    the minimum is at or above *level*, and NaN when nothing after the trough
    reaches it. When *level* is NaN or *value* is empty there is nothing to
    measure, and ``recovered`` is ``None``.
    """
    if math.isnan(level) or value.size == 0:
        return math.nan, None
    trough = int(np.argmin(value))
    if value[trough] >= level:
        return 0.0, True
    back = np.flatnonzero(value[trough + 1 :] >= level)
    if back.size == 0:
        return math.nan, False
    return float(time[trough + 1 + back[0]] - onset), True
