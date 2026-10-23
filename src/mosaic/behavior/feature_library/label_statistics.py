"""LabelStatistics -- a behavioural fingerprint per individual from a label sequence.

Reduces a categorical sequence to one row per ``(group, sequence, id)``: how much
of the time each state occupies, how often it starts, how long its bouts last,
and what tends to follow it. That row is what two individuals are compared by --
a fingerprint -- and it means the same thing whatever produced the sequence:
annotated ground truth, an ARHMM's syllables, a k-means cluster per frame.

It reads labels, not coordinates, so it computes in every track space.
"""

from __future__ import annotations

import numbers
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, ClassVar, Final, Self, cast, final

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import Field, model_validator

# Its argument is a bare ``dict``, so its signature is partly Unknown.
from mosaic.core.helpers import detect_label_format  # pyright: ignore[reportUnknownVariableType]
from mosaic.core.label_intervals import (
    SCENE_LEVEL,
    LabelSegments,
    label_vocabulary,
    segments_from_npz,
)
from mosaic.core.params import Declared, Params
from mosaic.core.pipeline.column_contract import (
    Column,
    Prefix,
    Provides,
    Requirement,
    ParamIs,
    When,
)
from mosaic.core.pipeline.parquet_reader import read_parquet_column_names
from mosaic.core.pipeline.types import (
    COLUMNS as C,
)
from mosaic.core.pipeline.types import (
    DependencyLookup,
    EmitsLevel,
    GroundTruthLabelsSource,
    Inputs,
    InputStream,
    LabelsSourceSpec,
    Result,
    ResultColumn,
    TrackInput,
)
from mosaic.core.pipeline.writers import read_parquet_table_columns

from .helpers import ensure_columns
from .registry import register_feature

__all__ = ["LabelStatistics", "state_suffix"]

_LABELS_DESCRIPTION = (
    "Where the label sequence comes from. A GroundTruthLabelsSource reads "
    "each entry's converted label file from labels/<kind>/, through its "
    "segment sidecar when the file carries one. A ResultColumn reads a "
    "categorical column of a feature's output: from that feature's run when "
    "it names a feature, from this feature's own input when its feature is "
    "left empty."
)

_STATES_DESCRIPTION = (
    "The fixed vocabulary of states, which gives every entry the same "
    "columns -- a state an entry never shows reads as zero usage and zero "
    "bouts rather than as a missing column. An integer names a label id or "
    "a numeric column value, a string a label name or a text column value. "
    "Required for a ResultColumn source. Unset for ground truth, it is the "
    "label file's label_names in file order, less those in ignore, so it can "
    "differ between entries whose files were converted with different "
    "vocabularies; a file that declares no label_ids has no vocabulary to "
    "default to and is refused. A label or value in neither states nor "
    "ignore is refused."
)

_FPS_DESCRIPTION = (
    "The frame rate that converts bout lengths to seconds and bout counts "
    "to rates. Unset, it is read from the input's time column as the median "
    "time step per frame, and an input without a usable time column is "
    "refused."
)

_MIN_BOUT_FRAMES_DESCRIPTION = (
    "The shortest bout kept. A shorter run or segment is dropped before "
    "any statistic is taken: it adds to no usage and no coverage, is no "
    "bout, and takes part in no transition. A bout's length is its rows "
    "times the entry's frame step, so on a table holding every k-th frame a "
    "bout of r rows has length r*k."
)

_IGNORE_DESCRIPTION = (
    "Labels or values excluded from every statistic, such as a background "
    "or an unsure class. Their frames count toward n_frames only, so they "
    "lower coverage. Matched as states are, and refused when also listed in "
    "states."
)

_UNLABELLED: Final = -1
"""The code of a frame no state names: unlabelled, ignored, or outside the labels."""

_NO_LABEL: Final = int(np.iinfo(np.int64).min)
"""A label file's "no label here", outside every int32 label id.

Not ``-1``: a file may declare ``-1`` as a label id of its own, and a sentinel
it can collide with would read that label's frames as unlabelled.
"""

_EVENT_FORMAT: Final = "individual_pair_v1"

_SUFFIX_INVALID = re.compile(r"[^0-9A-Za-z]+")


def state_suffix(state: str | int) -> str:
    """The column suffix a state is spelled with: ``usage__<suffix>``.

    A string keeps its ASCII letters and digits; every run of anything else,
    underscores included, becomes one underscore, and leading and trailing ones
    are dropped -- ``"rear up"`` and ``"rear__up"`` both become ``rear_up``.
    Collapsing underscores is what keeps ``p__<a>__<b>`` readable back into its
    two states: no suffix can contain the ``__`` that separates them. An
    integer is its decimal spelling, with ``neg`` for a minus sign.

    Raises:
        ValueError: When nothing identifier-safe is left of a string.
    """
    if isinstance(state, int):
        return f"neg{-state}" if state < 0 else str(state)
    suffix = _SUFFIX_INVALID.sub("_", state).strip("_")
    if not suffix:
        raise ValueError(
            f"state {state!r} has no letters or digits to name a column with"
        )
    return suffix


def _suffixes(states: Sequence[str | int]) -> tuple[str, ...]:
    """Each state's suffix, refusing two states that would share a column."""
    suffixes = tuple(state_suffix(state) for state in states)
    seen: dict[str, str | int] = {}
    for state, suffix in zip(states, suffixes):
        if suffix in seen:
            raise ValueError(
                f"states {seen[suffix]!r} and {state!r} would both be spelled "
                f"{suffix!r} in a column name; rename one"
            )
        seen[suffix] = state
    return suffixes


type _Key = tuple[int | None, str | None]
"""How a label or a value can be named: by integer, by string, or by both."""


def _matches(ref: str | int, key: _Key) -> bool:
    """Whether the state or ignore entry *ref* names the label or value *key*."""
    as_int, as_text = key
    if isinstance(ref, int):
        return as_int is not None and ref == as_int
    return as_text is not None and ref == as_text


def _code_table(
    keys: Sequence[_Key],
    states: Sequence[str | int],
    ignore: Sequence[str | int],
    shown: Sequence[object],
) -> NDArray[np.int32]:
    """Each key's state index, or ``_UNLABELLED`` for an ignored one.

    One pass over the distinct labels or values, never over frames.
    """
    table = np.empty(len(keys), dtype=np.int32)
    for position, key in enumerate(keys):
        hits = [index for index, ref in enumerate(states) if _matches(ref, key)]
        if len(hits) > 1:
            raise ValueError(
                f"{LabelStatistics.name}: states {[states[i] for i in hits]} all "
                f"name {shown[position]!r}; name each label once"
            )
        if hits:
            table[position] = hits[0]
        elif any(_matches(ref, key) for ref in ignore):
            table[position] = _UNLABELLED
        else:
            raise ValueError(
                f"{LabelStatistics.name}: {shown[position]!r} is in neither states "
                f"{list(states)} nor ignore {list(ignore)}. Add it to one, so the "
                "statistics say what was done with it."
            )
    return table


def _value_key(value: object) -> _Key:
    """A column value's key: an integral number by integer, a string by text.

    A fractional number gets neither, so no state can name it and it is refused
    as unmatched rather than truncated onto a neighbouring state.
    """
    if isinstance(value, str):
        return (None, value)
    if isinstance(value, numbers.Real):
        number = float(value)
        if number.is_integer():
            return (int(number), None)
    return (None, None)


@dataclass(frozen=True, slots=True)
class _PerFrame:
    """A state code per input row, ``_UNLABELLED`` where no state applies."""

    codes: NDArray[np.int32]


@dataclass(frozen=True, slots=True)
class _EventFile:
    """An event file's arrays as read: one frame, label id and actor per event."""

    frame: NDArray[np.int64]
    label: NDArray[np.int64]
    actor: NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class _Events:
    """An event file's per-frame events, mapped to state codes, in file order.

    ``actor`` is the first of each event's ``individual_ids``: the individual the
    event is about, :data:`SCENE_LEVEL` for one about the scene.
    """

    frame: NDArray[np.int64]
    code: NDArray[np.int32]
    actor: NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class _Intervals:
    """Annotated segments, already mapped to state codes; ignored ones removed."""

    start: NDArray[np.int64]
    end: NDArray[np.int64]
    code: NDArray[np.int32]
    individual: NDArray[np.int64] | None


@dataclass(frozen=True, slots=True)
class _Bouts:
    """One individual's retained bouts in onset order, and the rows they cover."""

    state: NDArray[np.int64]
    length: NDArray[np.int64]
    covered: NDArray[np.int64]
    covered_any: int


def _frame_step(frame: NDArray[np.int64]) -> int:
    """The entry's frame step: the smallest gap between two of its distinct frames.

    1 for a table with a row on every frame. ``k`` for one tracked on a media
    variant decimated by ``k``, which is published on source frames and so holds
    every ``k``-th. Measured over every individual's rows together, because it
    describes the table's sampling: an individual missing every other frame of a
    table that has them all was missing, not sampled more coarsely.
    """
    distinct = np.unique(frame)
    if distinct.size < 2:
        return 1
    return int(np.diff(distinct).min())


def _min_rows(min_bout_frames: int, step: int) -> int:
    """The fewest rows a kept bout has: ``rows * step >= min_bout_frames``."""
    return -(-min_bout_frames // step)


def _runs(
    frames: NDArray[np.int64],
    codes: NDArray[np.int32],
    n_states: int,
    min_rows: int,
    step: int,
) -> _Bouts:
    """Bouts as maximal runs of one code over consecutive rows, lengths in rows.

    Two rows are consecutive when they are less than one and a half steps apart:
    the next sample of a regular axis is one step on, and a missed one two. A
    longer gap ends a run as a change of state does -- an individual lost for a
    stretch was not observed in any state during it.
    """
    n = frames.size
    begins = np.ones(n, dtype=bool)
    # 2 * gap < 3 * step is gap < 1.5 * step, in integers.
    begins[1:] = (codes[1:] != codes[:-1]) | (
        2 * (frames[1:] - frames[:-1]) >= 3 * step
    )
    first = np.flatnonzero(begins)
    length = np.diff(np.append(first, n)).astype(np.int64)
    state = codes[first].astype(np.int64)
    keep = (state != _UNLABELLED) & (length >= min_rows)
    state = state[keep]
    length = length[keep]
    covered = np.bincount(state, weights=length, minlength=n_states)
    return _Bouts(
        state=state,
        length=length,
        covered=np.rint(covered).astype(np.int64),
        covered_any=int(length.sum()),
    )


def _union_count(
    frames: NDArray[np.int64], start: NDArray[np.int64], end: NDArray[np.int64]
) -> int:
    """How many of *frames* (sorted, unique) lie in the union of the intervals."""
    if start.size == 0:
        return 0
    order = np.lexsort((end, start))
    start = start[order]
    reach = np.maximum.accumulate(end[order])
    begins = np.ones(start.size, dtype=bool)
    begins[1:] = start[1:] > reach[:-1]
    first = np.flatnonzero(begins)
    last = np.append(first[1:] - 1, start.size - 1)
    inside = np.searchsorted(frames, reach[last]) - np.searchsorted(
        frames, start[first]
    )
    return int(inside.sum())


def _event_codes(
    frames: NDArray[np.int64], events: _Events, mask: NDArray[np.bool_]
) -> NDArray[np.int32]:
    """A state code per frame of *frames* (sorted, unique) from the masked events.

    Where several events fall on one frame the one listed last wins, the rule
    every reader of an event file applies.
    """
    out = np.full(frames.size, _UNLABELLED, dtype=np.int32)
    frame = events.frame[mask]
    if frame.size == 0:
        return out
    code = events.code[mask]
    # Sorted by frame and, within a frame, by listing order, so the last entry
    # of each frame is the event listed last.
    order = np.lexsort((np.arange(frame.size, dtype=np.int64), frame))
    frame, code = frame[order], code[order]
    last = np.flatnonzero(np.append(frame[1:] != frame[:-1], True))
    event_frame, event_code = frame[last], code[last]
    at = np.searchsorted(event_frame, frames)
    hit = at < event_frame.size
    hit[hit] = event_frame[at[hit]] == frames[hit]
    out[hit] = event_code[at[hit]]
    return out


def _segment_bouts(
    frames: NDArray[np.int64],
    intervals: _Intervals,
    mask: NDArray[np.bool_],
    n_states: int,
    min_rows: int,
) -> _Bouts:
    """Bouts as the segments themselves, overlaps and nesting preserved.

    A segment's length is the number of this individual's rows it covers, so
    a segment reaching past the frames the input holds counts what it covers
    there. Usage is the union per state, so two overlapping segments of one
    state do not count their shared frames twice.
    """
    start = intervals.start[mask]
    end = intervals.end[mask]
    code = intervals.code[mask].astype(np.int64)
    length = np.searchsorted(frames, end) - np.searchsorted(frames, start)
    keep = length >= min_rows
    start, end, code, length = start[keep], end[keep], code[keep], length[keep]
    position = np.arange(start.size, dtype=np.int64)
    order = np.lexsort((position, end, start))
    covered = np.array(
        [
            _union_count(frames, start[code == state], end[code == state])
            for state in range(n_states)
        ],
        dtype=np.int64,
    )
    return _Bouts(
        state=code[order],
        length=length[order].astype(np.int64),
        covered=covered,
        covered_any=_union_count(frames, start, end),
    )


def _entropy_bits(weights: NDArray[np.float64]) -> float:
    """Shannon entropy in bits of *weights* normalised to sum 1; NaN if all zero."""
    total = float(weights.sum())
    if total <= 0.0:
        return float("nan")
    p = weights[weights > 0] / total
    # + 0.0 turns the -0.0 of a single certain state into 0.0.
    return float(-(p * np.log2(p)).sum()) + 0.0


@final
@register_feature
class LabelStatistics:
    """
    Per-individual behavioural statistics of a label sequence: a fingerprint.

    One row per ``(group, sequence, id)``. The sequence is either ground-truth
    labels (``GroundTruthLabelsSource``) or a categorical column of a feature's
    output (``ResultColumn``), such as ``arhmm``'s ``syllable`` or a cluster id.

    Output columns, with ``s`` each state's suffix (:func:`state_suffix`) and
    ``fps`` the frame rate:

      - ``group``, ``sequence``, ``id``
      - ``n_frames``: the distinct frames this individual has rows on, which is
        the axis every fraction is over
      - ``coverage``: fraction of ``n_frames`` covered by any retained bout
      - ``n_bouts``: retained bouts, all states
      - ``entropy``: Shannon entropy, bits, of the usages normalised to sum 1
      - ``transition_entropy``: bits; the entropy of the next bout's state given
        the current one's, over the off-diagonal transition matrix, each row
        weighted by how often its state starts a transition
      - ``usage__s``: fraction of ``n_frames`` covered by ``s``
      - ``bouts__s``: number of bouts of ``s``
      - ``rate_per_min__s``: bouts of ``s`` per minute the individual was present
      - ``bout_mean_s__s`` / ``bout_median_s__s``: bout duration in seconds
      - ``p__a__b`` for every ordered pair of distinct states: the probability
        that a bout of ``a`` is followed by one of ``b``

    Counts are int64 and the other measurements float32. A state that never
    occurs has zero usage, bouts and rate. What is undefined is NaN, never zero:
    the bout duration of a state with no bouts, the ``p__a__*`` row of a state
    no transition leaves, and either entropy with nothing to take it over.

    **The frame step.** A table need not hold every frame: one tracked on a
    media variant decimated by ``k`` is published on source frames and holds
    every ``k``-th. The entry's step is the smallest gap between its distinct
    frames, over all individuals -- 1 for a row on every frame, ``k`` there --
    and each row stands for one step. A bout's length, the time an individual
    was present and ``min_bout_frames`` are all counted that way, so a bout of
    ``r`` rows lasts ``r * step / fps`` seconds.

    **Bouts.** From a per-frame sequence, a bout is a maximal run of one state
    over consecutive rows of the individual. Rows less than one and a half
    steps apart are consecutive, so a regular axis's next sample continues a
    run and a missed sample ends it: an individual lost for a stretch was not
    observed in any state during it. From the segment sidecar of a label file
    (:mod:`mosaic.core.label_intervals`), a bout is a segment, and segments may
    overlap or nest -- so usages may sum past 1 and a nested bout is a bout of
    its own rather than painted over. A segment's length is the number of the
    individual's rows it covers. ``min_bout_frames`` and ``ignore`` remove bouts
    before anything is counted. Transitions are between consecutive retained
    bouts in onset order (start, then end, then listing order), whatever gap or
    removed bout lies between them; two consecutive bouts of one state are not a
    transition.

    **Unlabelled frames** -- beyond a dense array, without an event in an event
    file, outside every segment, or null in a column -- count toward
    ``n_frames`` and nothing else.

    **Whose labels.** A column read from a feature's output is joined on
    ``(frame, id)`` when that output has an ``id`` and on ``frame`` alone when it
    does not. A dense label file is scene-level: every individual in the entry
    is given the same sequence over the frames it is present in. An event file
    (``individual_pair_v1``) names an individual per event, the first of its
    ``individual_ids`` -- the subject of a BORIS event, the actor of a directed
    pair such as CalMS21's ``[resident, intruder]`` -- and the event counts for
    that individual alone; an event whose first id is ``-1`` is scene-level and
    counts for every individual. Where several events fall on one of an
    individual's frames, the one listed last wins, as in every other reader of
    an event file. A segment sidecar names an individual per segment through
    ``seg_individual``; a segment of ``-1``, or a sidecar without that array,
    is scene-level and counts for every individual. An individual of an event
    or a segment that is not among the input's ids is in no row.

    The input supplies the entries, the frame axis and the individuals. The
    default is ``tracks``; a feature's output works as well, and is how a
    ``ResultColumn`` with an empty ``feature`` reaches its column. Every row
    must carry ``frame`` and ``id``, and the input must carry ``group`` and
    ``sequence``, which say whose entry the rows are -- an output without them
    (``arhmm``'s is one) is read by naming it in the ``ResultColumn`` over a
    ``tracks`` input instead.

    Field documentation is on
    :class:`~mosaic.behavior.feature_library.label_statistics.LabelStatistics.Params`.
    """

    category = "summary"
    name = "label-statistics"
    version = "0.1"
    parallelizable = True
    scope_dependent = False
    # A summary has no frame axis for the overlap trim to work on, and an entry's
    # labels are looked up by the one (group, sequence) its frame carries.
    accepts_overlap = False
    consumed_roots: tuple[str, ...] = ()
    emits: EmitsLevel = "individual"
    requires: ClassVar[tuple[Requirement, ...]] = (
        Column(C.frame_col),
        Column(C.id_col),
        Column(C.group_col),
        Column(C.seq_col),
        When(Column(C.time_col), (ParamIs("fps", negated=True),)),
    )
    provides: ClassVar[Provides] = Provides(
        adds=(
            Column(C.group_col),
            Column(C.seq_col),
            Column(C.id_col),
            Column("n_frames"),
            Column("coverage"),
            Column("n_bouts"),
            Column("entropy"),
            Column("transition_entropy"),
            Prefix("usage__"),
            Prefix("bouts__"),
            Prefix("rate_per_min__"),
            Prefix("bout_mean_s__"),
            Prefix("bout_median_s__"),
            Prefix("p__"),
        ),
    )

    class Inputs(Inputs[TrackInput | Result]):
        pass

    class Params(Params):
        # A plain default rather than a default_factory: from_overrides merges a
        # dict override into a factory's default, and a ResultColumn's keys
        # merged into a GroundTruthLabelsSource's validate as neither -- which
        # is the dict every process worker and every recipe hands in.
        labels: Annotated[LabelsSourceSpec, Declared(_LABELS_DESCRIPTION)] = (
            GroundTruthLabelsSource()
        )
        states: Annotated[list[str | int] | None, Declared(_STATES_DESCRIPTION)] = None
        # allow_inf_nan: an infinite rate passes gt=0 and makes every duration
        # zero and every presence zero minutes long.
        fps: Annotated[
            float | None,
            Field(gt=0.0, allow_inf_nan=False),
            Declared(_FPS_DESCRIPTION, unit="fps"),
        ] = None
        min_bout_frames: Annotated[
            int, Field(ge=1), Declared(_MIN_BOUT_FRAMES_DESCRIPTION, unit="frames")
        ] = 1
        ignore: Annotated[list[str | int], Declared(_IGNORE_DESCRIPTION)] = Field(
            default_factory=list[str | int]
        )

        @model_validator(mode="after")
        def _check_vocabulary(self) -> Self:
            """Refuse at construction, where a raise is visible.

            ``run_feature`` records an exception out of ``apply`` as one failed
            entry and carries on, so a vocabulary that can never work would
            otherwise fail every entry one at a time.
            """
            if isinstance(self.labels, ResultColumn) and self.states is None:
                raise ValueError(
                    "label-statistics needs states for a ResultColumn source: a "
                    "column carries no vocabulary of its own, and without one "
                    "two entries showing different states get different columns"
                )
            if self.states is not None:
                if not self.states:
                    raise ValueError("states must name at least one state")
                repeated = sorted(
                    {repr(s) for s in self.states if self.states.count(s) > 1}
                )
                if repeated:
                    raise ValueError(f"states repeats {repeated}")
                both = [s for s in self.states if s in self.ignore]
                if both:
                    raise ValueError(f"{both} are listed in both states and ignore")
                _ = _suffixes(self.states)
            return self

    def __init__(
        self,
        inputs: LabelStatistics.Inputs = Inputs(("tracks",)),
        params: dict[str, object] | None = None,
    ):
        self.inputs = inputs
        self.params = self.Params.from_overrides(params)
        self._labels_lookup: DependencyLookup | None = None

    # --- State protocol (stateless per-entry summary) ---

    def load_state(
        self,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_lookups: dict[str, DependencyLookup],
    ) -> bool:
        # Present for ground truth, and for a ResultColumn naming a feature;
        # absent for one that reads its column off the input.
        self._labels_lookup = dependency_lookups.get("labels")
        return True

    def fit(self, inputs: InputStream) -> None:
        pass

    def save_state(self, run_root: Path) -> None:
        pass

    # --- Apply ---

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()
        ensure_columns(df, [C.frame_col, C.id_col])
        group = self._single_text(df, C.group_col)
        sequence = self._single_text(df, C.seq_col)
        fps = self._frame_rate(df)
        frame = self._frame_numbers(df)
        id_code, id_values = self._individuals(df)

        labels = self.params.labels
        if isinstance(labels, ResultColumn):
            states: tuple[str | int, ...] = tuple(self.params.states or ())
            source: _PerFrame | _Events | _Intervals = self._column_source(
                df, labels, (group, sequence), states
            )
        else:
            states, source = self._ground_truth_source(frame, labels, (group, sequence))
        suffixes = _suffixes(states)
        n_states = len(states)
        # Every length below is counted in rows, and each row stands for one
        # step of the frame axis.
        step = _frame_step(frame)
        min_rows = _min_rows(self.params.min_bout_frames, step)
        seconds_per_row = step / fps

        order = np.lexsort((frame, id_code))
        bounds = np.searchsorted(id_code[order], np.arange(len(id_values) + 1))
        rows: list[dict[str, object]] = []
        for k, id_value in enumerate(id_values):
            at = order[bounds[k] : bounds[k + 1]]
            frames = frame[at]
            first = np.ones(frames.size, dtype=bool)
            first[1:] = frames[1:] != frames[:-1]
            if isinstance(source, _PerFrame):
                codes = source.codes[at]
                clash = ~first[1:] & (codes[1:] != codes[:-1])
                if clash.any():
                    raise ValueError(
                        f"{self.name}: id {id_value!r} has two rows on frame "
                        f"{int(frames[1:][clash][0])} carrying different labels"
                    )
                bouts = _runs(frames[first], codes[first], n_states, min_rows, step)
            elif isinstance(source, _Events):
                mask = self._owned_by(
                    source.actor, source.frame.size, id_value, "events"
                )
                codes = _event_codes(frames[first], source, mask)
                bouts = _runs(frames[first], codes, n_states, min_rows, step)
            else:
                mask = self._owned_by(
                    source.individual, source.start.size, id_value, "segments"
                )
                bouts = _segment_bouts(frames[first], source, mask, n_states, min_rows)
            rows.append(
                {
                    C.group_col: group,
                    C.seq_col: sequence,
                    C.id_col: id_value,
                    **self._statistics(
                        bouts, int(first.sum()), seconds_per_row, suffixes
                    ),
                }
            )
        return self._frame_of(rows, suffixes)

    # --- Entry identity and axes ---

    def _single_text(self, df: pd.DataFrame, column: str) -> str:
        """The one value of *column* in this entry, refusing an input without it.

        Not defaulted to ``""``: ``apply`` is handed rows and no entry, and a
        guessed entry would put the right statistics under the wrong
        ``(group, sequence)`` -- or look up another entry's label file.
        """
        if column not in df.columns:
            raise ValueError(
                f"{self.name}: the input has no {column!r} column, so the entry "
                "its rows belong to is unknown. Give it an input that carries "
                f"{C.group_col!r} and {C.seq_col!r}, such as tracks; to read a "
                "column of an output that lacks them, name that feature in the "
                "ResultColumn instead."
            )
        values = pd.unique(df[column])
        if len(values) != 1:
            raise ValueError(
                f"{self.name}: expected one {column} per entry, got {len(values)}"
            )
        return str(cast("object", values[0]))

    def _frame_numbers(self, df: pd.DataFrame) -> NDArray[np.int64]:
        """The frame column as int64, refusing a missing or fractional frame."""
        raw = df[C.frame_col].to_numpy(dtype=np.float64)
        if not np.isfinite(raw).all() or not (raw == np.floor(raw)).all():
            raise ValueError(
                f"{self.name}: the {C.frame_col} column holds a missing or "
                "fractional value; bouts are counted on whole frames"
            )
        return raw.astype(np.int64)

    def _individuals(self, df: pd.DataFrame) -> tuple[NDArray[np.int64], list[Any]]:
        """A dense code per row for its id, and the ids in sorted order."""
        codes, uniques = pd.factorize(df[C.id_col], sort=True)
        code_array = np.asarray(codes, dtype=np.int64)
        if (code_array < 0).any():
            raise ValueError(
                f"{self.name}: some rows carry no {C.id_col}; every statistic is "
                "per individual"
            )
        return code_array, cast("list[Any]", np.asarray(uniques, dtype=object).tolist())

    def _frame_rate(self, df: pd.DataFrame) -> float:
        """``fps`` when set, else the median time step per frame of the input."""
        if self.params.fps is not None:
            return self.params.fps
        refusal = (
            f"{self.name} needs a frame rate to turn bout lengths into seconds "
            "and bout counts into rates. Set fps, or give it an input whose "
            f"{C.time_col} column advances with {C.frame_col}"
        )
        if C.time_col not in df.columns:
            raise ValueError(f"{refusal}; this input has no {C.time_col} column.")
        frames = df[C.frame_col].to_numpy(dtype=np.float64)
        times = df[C.time_col].to_numpy(dtype=np.float64)
        unique_frames, first = np.unique(frames, return_index=True)
        frame_step = np.diff(unique_frames)
        time_step = np.diff(times[first])
        usable = (frame_step > 0) & np.isfinite(time_step) & (time_step > 0)
        if not usable.any():
            raise ValueError(f"{refusal}; its {C.time_col} column never does.")
        seconds_per_frame = float(np.median(time_step[usable] / frame_step[usable]))
        return 1.0 / seconds_per_frame

    # --- Label sources ---

    def _entry_label_path(self, entry: tuple[str, str], what: str) -> Path:
        """The entry's file in the labels lookup, refusing an entry with none."""
        path = (self._labels_lookup or {}).get(entry)
        if path is None:
            raise ValueError(
                f"{self.name}: no {what} for entry {entry}. Narrow the scope to "
                "the entries that have them."
            )
        return path

    def _column_source(
        self,
        df: pd.DataFrame,
        labels: ResultColumn,
        entry: tuple[str, str],
        states: Sequence[str | int],
    ) -> _PerFrame:
        """A state code per input row from a categorical column."""
        column = labels.column
        if not labels.feature:
            ensure_columns(df, [column])
            values = df[column]
        else:
            path = self._entry_label_path(entry, f"{labels.feature!r} output")
            names = read_parquet_column_names(path) or ()
            if column not in names or C.frame_col not in names:
                raise ValueError(
                    f"{self.name}: {labels.feature!r} output for {entry} has no "
                    f"{column!r} or no {C.frame_col!r} column"
                )
            keys = [C.frame_col, C.id_col] if C.id_col in names else [C.frame_col]
            upstream = read_parquet_table_columns(path, [*keys, column])
            if upstream.duplicated(keys).any():
                raise ValueError(
                    f"{self.name}: {labels.feature!r} output for {entry} has more "
                    f"than one row per {keys}, so its {column!r} cannot be read "
                    "as one value per frame"
                )
            # The join keys take the input's dtypes, so an int frame meets an
            # int frame however either parquet stored it.
            for key in keys:
                upstream[key] = upstream[key].astype(df[key].dtype)
            joined = df[keys].merge(upstream, on=keys, how="left")
            values = joined[column]
        factor_codes, uniques = pd.factorize(values)
        distinct = cast("list[object]", np.asarray(uniques, dtype=object).tolist())
        table = _code_table(
            [_value_key(value) for value in distinct],
            states,
            self.params.ignore,
            distinct,
        )
        row_codes = np.asarray(factor_codes, dtype=np.int64)
        codes = np.full(row_codes.size, _UNLABELLED, dtype=np.int32)
        present = row_codes >= 0
        codes[present] = table[row_codes[present]]
        return _PerFrame(codes=codes)

    def _ground_truth_source(
        self,
        frame: NDArray[np.int64],
        labels: GroundTruthLabelsSource,
        entry: tuple[str, str],
    ) -> tuple[tuple[str | int, ...], _PerFrame | _Events | _Intervals]:
        """The entry's states and its labels: segments if the file has them."""
        path = self._entry_label_path(entry, f"labels/{labels.kind} file")
        vocabulary, content = self._read_label_file(path, entry)
        states = self._ground_truth_states(vocabulary, entry)

        if isinstance(content, LabelSegments):
            distinct, inverse = np.unique(
                content.label.astype(np.int64), return_inverse=True
            )
            code = self._label_table(vocabulary, distinct, states)[inverse]
            keep = code != _UNLABELLED
            return states, _Intervals(
                start=content.start_frame[keep],
                end=content.end_frame[keep],
                code=code[keep],
                individual=None
                if content.individual is None
                else content.individual[keep],
            )

        if isinstance(content, _EventFile):
            distinct, inverse = np.unique(content.label, return_inverse=True)
            table = self._label_table(vocabulary, distinct, states)
            return states, _Events(
                frame=content.frame, code=table[inverse], actor=content.actor
            )

        # Looked up once per distinct frame, not once per row: every individual
        # on a frame shares that frame's label.
        unique_frames, frame_inverse = np.unique(frame, return_inverse=True)
        per_frame = np.full(unique_frames.size, _NO_LABEL, dtype=np.int64)
        inside = (unique_frames >= 0) & (unique_frames < content.size)
        per_frame[inside] = content[unique_frames[inside]]
        distinct, inverse = np.unique(per_frame, return_inverse=True)
        labelled = distinct != _NO_LABEL
        table = np.full(distinct.size, _UNLABELLED, dtype=np.int32)
        table[labelled] = self._label_table(vocabulary, distinct[labelled], states)
        return states, _PerFrame(codes=table[inverse][frame_inverse])

    def _read_label_file(
        self, path: Path, entry: tuple[str, str]
    ) -> tuple[dict[int, str], LabelSegments | _EventFile | NDArray[np.int64]]:
        """The file's vocabulary, and its segments, its events or its dense array.

        Without a sidecar the file is told apart the way
        ``load_labels_for_feature_frames`` tells it apart -- an event file by
        ``detect_label_format``, otherwise the dense ``labels`` array -- but read
        here rather than through it. Its symmetric deduplication keeps only the
        events whose first id is at most the second, which drops every
        individual-level ``[subject, -1]`` event a BORIS converter writes, and a
        per-individual statistic needs those most.
        """
        with np.load(path, allow_pickle=True) as npz:
            archive: Mapping[str, object] = npz
            vocabulary = label_vocabulary(archive)
            segments = segments_from_npz(archive)
            if segments is not None:
                return vocabulary, segments
            if detect_label_format(npz) == _EVENT_FORMAT:
                pairs = np.asarray(npz["individual_ids"], dtype=np.int64)
                events = _EventFile(
                    frame=np.asarray(npz["frames"], dtype=np.int64).ravel(),
                    label=np.asarray(npz["labels"], dtype=np.int64).ravel(),
                    actor=pairs.reshape(-1, 2)[:, 0],
                )
                if not events.frame.size == events.label.size == events.actor.size:
                    raise ValueError(
                        f"{self.name}: the event file for {entry} holds "
                        f"{events.frame.size} frames, {events.label.size} labels "
                        f"and {events.actor.size} individual_ids"
                    )
                return vocabulary, events
            if "labels" in archive:
                return vocabulary, np.asarray(npz["labels"], dtype=np.int64).ravel()
        raise ValueError(
            f"{self.name}: the label file for {entry} holds no segment sidecar, "
            "no events and no labels array"
        )

    def _ground_truth_states(
        self, vocabulary: dict[int, str], entry: tuple[str, str]
    ) -> tuple[str | int, ...]:
        """``states`` when set, else the file's label names less those ignored.

        A file declaring no ``label_ids`` is refused rather than given the
        labels it happens to hold: those differ between entries, and the
        columns would differ with them -- a state one entry never shows would
        be missing there rather than zero.
        """
        if self.params.states is not None:
            return tuple(self.params.states)
        if not vocabulary:
            raise ValueError(
                f"{self.name}: the label file for {entry} declares no "
                "label_ids, so it has no vocabulary to default states to. Set "
                "states, so every entry gets the same columns."
            )
        ignore = self.params.ignore
        states = tuple(
            name
            for label_id, name in vocabulary.items()
            if not any(_matches(ref, (label_id, name)) for ref in ignore)
        )
        if not states:
            raise ValueError(
                f"{self.name}: ignore {list(ignore)} removes every label the "
                f"file for {entry} declares, {list(vocabulary.values())}, so "
                "there is no state left to describe"
            )
        return states

    def _label_table(
        self,
        vocabulary: dict[int, str],
        label_ids: NDArray[np.int64],
        states: Sequence[str | int],
    ) -> NDArray[np.int32]:
        """Each label id's state index, matched by id and by name."""
        ids = [int(label_id) for label_id in label_ids.tolist()]
        shown: list[object] = [vocabulary.get(label_id, label_id) for label_id in ids]
        return _code_table(
            [(label_id, vocabulary.get(label_id)) for label_id in ids],
            states,
            self.params.ignore,
            shown,
        )

    def _owned_by(
        self,
        owner: NDArray[np.int64] | None,
        count: int,
        id_value: object,
        what: str,
    ) -> NDArray[np.bool_]:
        """Which segments or events count for *id_value*: its own and the scene's.

        *owner* is each one's individual (``seg_individual``, or an event's
        actor), :data:`SCENE_LEVEL` for the scene; None means all are the scene's.
        """
        if owner is None:
            return np.ones(count, dtype=bool)
        scene = owner == SCENE_LEVEL
        as_int, _ = _value_key(id_value)
        if as_int is None:
            if not scene.all():
                raise ValueError(
                    f"{self.name}: id {id_value!r} is not an integer, so the "
                    f"{what}' individual ids cannot name it"
                )
            return scene
        return scene | (owner == as_int)

    # --- Statistics ---

    @staticmethod
    def _statistics(
        bouts: _Bouts,
        n_frames: int,
        seconds_per_row: float,
        suffixes: tuple[str, ...],
    ) -> dict[str, object]:
        """One individual's row, from its retained bouts; lengths are in rows."""
        n_states = len(suffixes)
        usage = bouts.covered.astype(np.float64) / n_frames
        bout_count = np.bincount(bouts.state, minlength=n_states)
        minutes = n_frames * seconds_per_row / 60.0

        source = bouts.state[:-1]
        target = bouts.state[1:]
        moves = source != target
        counts = np.bincount(
            source[moves] * n_states + target[moves], minlength=n_states * n_states
        ).reshape(n_states, n_states)
        leaving = counts.sum(axis=1)
        n_transitions = int(leaving.sum())
        transition_entropy = (
            float(
                sum(
                    leaving[a]
                    / n_transitions
                    * _entropy_bits(counts[a].astype(np.float64))
                    for a in range(n_states)
                    if leaving[a] > 0
                )
            )
            if n_transitions
            else float("nan")
        )

        row: dict[str, object] = {
            "n_frames": n_frames,
            "coverage": bouts.covered_any / n_frames,
            "n_bouts": int(bouts.state.size),
            "entropy": _entropy_bits(usage),
            "transition_entropy": transition_entropy,
        }
        for index, suffix in enumerate(suffixes):
            row[f"usage__{suffix}"] = float(usage[index])
        for index, suffix in enumerate(suffixes):
            row[f"bouts__{suffix}"] = int(bout_count[index])
        for index, suffix in enumerate(suffixes):
            row[f"rate_per_min__{suffix}"] = float(bout_count[index]) / minutes
        per_state = [bouts.length[bouts.state == index] for index in range(n_states)]
        for lengths, suffix in zip(per_state, suffixes):
            row[f"bout_mean_s__{suffix}"] = (
                float(lengths.mean()) * seconds_per_row
                if lengths.size
                else float("nan")
            )
        for lengths, suffix in zip(per_state, suffixes):
            row[f"bout_median_s__{suffix}"] = (
                float(np.median(lengths)) * seconds_per_row
                if lengths.size
                else float("nan")
            )
        for a, from_suffix in enumerate(suffixes):
            for b, to_suffix in enumerate(suffixes):
                if a == b:
                    continue
                row[f"p__{from_suffix}__{to_suffix}"] = (
                    float(counts[a, b]) / float(leaving[a])
                    if leaving[a]
                    else float("nan")
                )
        return row

    @staticmethod
    def _frame_of(
        rows: list[dict[str, object]], suffixes: tuple[str, ...]
    ) -> pd.DataFrame:
        """The rows as a frame, counts int64 and measurements float32."""
        out = pd.DataFrame(rows)
        counts = {"n_frames", "n_bouts", *(f"bouts__{s}" for s in suffixes)}
        identity = {C.group_col, C.seq_col, C.id_col}
        for column in out.columns:
            name = str(column)
            if name in identity:
                continue
            out[name] = out[name].astype(np.int64 if name in counts else np.float32)
        return out
