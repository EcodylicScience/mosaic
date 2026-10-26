"""Measures of a track's quality, the events they flag, and the frames chosen from them.

numpy is imported at module scope, and pandas only inside functions and under
``TYPE_CHECKING``. A module free of pandas stays free of it after importing
this one. Nothing here imports from ``core.pipeline``.

A tracks table has ``frame``, ``id``, ``X`` and ``Y``, and ``poseX<k>``,
``poseY<k>`` and ``poseP<k>`` where it has keypoints and their scores. A row is
present when its ``X`` and ``Y`` are finite. :func:`row_flags` flags rows,
:func:`frame_flags` flags frames, :func:`events` folds runs of flagged frames
into events, and :func:`select` chooses at most one frame per event. Every
distance threshold is a multiple of the entry's body length (:func:`body_length`).

**The jump rule here differs from ``trajectory_smooth``'s.** The smoother
differences consecutive rows. A step across a frame gap therefore counts, and
its threshold is a speed in units per frame or per second. The rules here
measure consecutive frames only, in body lengths. The two are not logically
equivalent, and neither is written in terms of the other.
"""

from __future__ import annotations

import bisect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt
    import pandas as pd

__all__ = [
    "EVENT_COLUMNS",
    "FRAME_CRITERIA",
    "ROW_CRITERIA",
    "Criterion",
    "QualityRules",
    "body_length",
    "consecutive_runs",
    "events",
    "frame_flags",
    "instance_body_lengths",
    "presence",
    "row_flags",
    "select",
]

Criterion = Literal[
    "jump", "low_score", "missing_keypoints", "duplicate", "fewer_than_expected"
]
"""A rule that flags a row or a frame, and the cause of the events it forms."""

ROW_CRITERIA: tuple[Criterion, ...] = ("jump", "low_score", "missing_keypoints")
"""The criteria that flag a row, each a column of :func:`row_flags` when it applies."""

FRAME_CRITERIA: tuple[Criterion, ...] = ("duplicate", "fewer_than_expected")
"""The criteria that flag a frame, each a column of :func:`frame_flags` when it applies."""

EVENT_COLUMNS: tuple[str, ...] = (
    "frame",
    "cause",
    "event_id",
    "id",
    "x",
    "y",
    "score",
    "event_frames",
)
"""The columns of :func:`events` and :func:`select`, in order."""

_EVENT_TYPES: dict[str, type] = {
    "frame": np.int64,
    "cause": str,
    "event_id": str,
    "id": str,
    "x": np.float64,
    "y": np.float64,
    "score": np.float64,
    "event_frames": np.int64,
}


@dataclass(frozen=True)
class QualityRules:
    """The thresholds of the track-quality criteria.

    A criterion whose field is None is off. ``low_score`` is on when one of its
    two fields is set.

    Attributes:
        jump_body_lengths: A step of one id between consecutive frames longer
            than this many body lengths is a ``jump``.
        low_score_threshold: A row whose mean finite confidence is below this
            has a ``low_score``.
        low_score_quantile: A row whose mean finite confidence is below this
            quantile of the table's means has a ``low_score``.
        missing_keypoints: The keypoint indices a present row must have finite,
            or ``()`` for every keypoint.
        duplicate_body_lengths: Two present rows of one frame whose centers lie
            closer than this many body lengths are a ``duplicate`` pair.
        n_expected: A frame with fewer detected rows than this is
            ``fewer_than_expected``.
        head: The head keypoint index of the body axis.
        tail: The tail keypoint index of the body axis. With ``head``, it vetoes
            a duplicate pair with heads or tails apart.

    Raises:
        ValueError: Both ``low_score`` fields are set.
    """

    jump_body_lengths: float | None = None
    low_score_threshold: float | None = None
    low_score_quantile: float | None = None
    missing_keypoints: tuple[int, ...] | None = None
    duplicate_body_lengths: float | None = None
    n_expected: int | None = None
    head: int | None = None
    tail: int | None = None

    def __post_init__(self) -> None:
        """Refuse a ``low_score`` threshold given together with a quantile."""
        if self.low_score_threshold is not None and self.low_score_quantile is not None:
            msg = "low_score takes a threshold or a quantile, and both are given"
            raise ValueError(msg)


def instance_body_lengths(
    xy: np.ndarray, head: int | None, tail: int | None
) -> np.ndarray:
    """Return one body length per instance.

    The length is the head-to-tail distance when both keypoints are in range
    and finite, and otherwise the diagonal of the tight box around the finite
    keypoints.

    Args:
        xy: Keypoints of shape ``(N, K, 2)``.
        head: Index of the head keypoint, or None.
        tail: Index of the tail keypoint, or None.

    Returns:
        Float array of shape ``(N,)``, NaN for an instance with no finite
        keypoint.
    """
    points = np.asarray(xy, dtype=float)
    finite = np.isfinite(points).all(axis=-1)
    low = np.where(finite[..., None], points, np.inf).min(axis=1)
    high = np.where(finite[..., None], points, -np.inf).max(axis=1)
    lengths = np.hypot(*(high - low).T)
    lengths = np.where(finite.any(axis=1), lengths, np.nan)
    count = points.shape[1]
    if (
        head is not None
        and tail is not None
        and 0 <= head < count
        and 0 <= tail < count
    ):
        both = finite[:, head] & finite[:, tail]
        direct = np.linalg.norm(points[:, head] - points[:, tail], axis=-1)
        lengths = np.where(both, direct, lengths)
    return lengths


def consecutive_runs(
    frames: npt.ArrayLike, keep: npt.ArrayLike
) -> list[tuple[int, int]]:
    """Return the runs of consecutive frames that *keep* marks, as row positions.

    A run ends at a gap in *frames* and at a row that *keep* leaves out.

    Args:
        frames: The frame number of each row of one series, in increasing order.
        keep: One flag per row. A row that is ``False`` belongs to no run.

    Returns:
        The ``(start, stop)`` positions of each run, in order, half-open as a
        slice reads them.
    """
    frame = np.asarray(frames)
    kept = np.asarray(keep, dtype=bool)
    if frame.size == 0:
        return []
    follows = np.zeros(frame.size, dtype=bool)
    follows[1:] = kept[1:] & kept[:-1] & (np.diff(frame) == 1)
    starts = np.flatnonzero(kept & ~follows)
    ends = np.zeros(frame.size, dtype=bool)
    ends[:-1] = ~follows[1:]
    ends[-1] = True
    stops = np.flatnonzero(kept & ends) + 1
    return list(zip(starts.tolist(), stops.tolist(), strict=True))


def _floats(table: pd.DataFrame, column: str) -> np.ndarray:
    """Return *column* of *table* as float64, NaN for a missing value."""
    return table[column].to_numpy(dtype=np.float64, na_value=np.nan)


def _pose_pairs(table: pd.DataFrame) -> list[tuple[str, str]]:
    """Return the ``(poseX<k>, poseY<k>)`` column pairs of *table*, in keypoint order."""
    from mosaic.core.pose_columns import pose_column_pairs

    return pose_column_pairs(str(name) for name in table.columns)


def _keypoints(table: pd.DataFrame) -> np.ndarray:
    """Return the keypoints of *table* as ``(N, K, 2)``, with ``K`` zero without any."""
    pairs = _pose_pairs(table)
    points = np.full((len(table), len(pairs), 2), np.nan)
    for k, (x_column, y_column) in enumerate(pairs):
        points[:, k, 0] = _floats(table, x_column)
        points[:, k, 1] = _floats(table, y_column)
    return points


def _mean_scores(table: pd.DataFrame) -> np.ndarray | None:
    """Return each row's mean finite ``poseP`` score, or None without score columns."""
    from mosaic.core.pose_columns import keypoint_suffixes

    names = [str(name) for name in table.columns]
    columns = [f"poseP{suffix}" for suffix in keypoint_suffixes(names, "poseP")]
    if not columns:
        return None
    scores = table[columns].to_numpy(dtype=np.float64, na_value=np.nan)
    finite = np.isfinite(scores)
    count = finite.sum(axis=1)
    total = np.where(finite, scores, 0.0).sum(axis=1)
    return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def _refuse_past(field: str, indices: Sequence[int], count: int) -> None:
    """Refuse a keypoint index of *field* outside ``0 .. count - 1``."""
    past = [index for index in indices if not 0 <= index < count]
    if past:
        msg = f"{field} names keypoint {past[0]}, and the table has {count} keypoints"
        raise ValueError(msg)


def _refuse_axis(head: int | None, tail: int | None, count: int) -> None:
    """Refuse a *head* or *tail* outside ``0 .. count - 1``."""
    for field, index in (("head", head), ("tail", tail)):
        if index is not None:
            _refuse_past(field, [index], count)


def _known(body: float | None, criterion: Criterion) -> float:
    """Return *body*, or refuse a *criterion* in body lengths when it is unknown."""
    if body is None:
        msg = (
            f"{criterion} is measured in body lengths, and the table's keypoints do "
            "not measure a body length. Set body_length_px to give one."
        )
        raise ValueError(msg)
    return body


def presence(table: pd.DataFrame) -> np.ndarray:
    """Return whether each row of *table* is present.

    Args:
        table: A tracks table.

    Returns:
        A boolean array, one value per row, true where ``X`` and ``Y`` are both
        finite.
    """
    return np.isfinite(_floats(table, "X")) & np.isfinite(_floats(table, "Y"))


def body_length(
    table: pd.DataFrame, head: int | None, tail: int | None
) -> float | None:
    """Return the body length of *table*, measured over its present rows.

    The length is the median of :func:`instance_body_lengths` over the present
    rows that measure one. An instance with one finite keypoint has a zero box
    and measures none.

    Args:
        table: A tracks table.
        head: Index of the head keypoint, or None.
        tail: Index of the tail keypoint, or None.

    Returns:
        The length in the table's units, or None when the table lacks keypoint
        columns or none of its present rows measures one.

    Raises:
        ValueError: *head* or *tail* names a keypoint past the table's keypoints.
    """
    points = _keypoints(table)[presence(table)]
    _refuse_axis(head, tail, points.shape[1])
    if points.shape[1] == 0:
        return None
    lengths = instance_body_lengths(points, head, tail)
    measured = lengths[np.isfinite(lengths) & (lengths > 0)]
    if measured.size == 0:
        return None
    return float(np.median(measured))


def _jumps(
    frame: np.ndarray,
    codes: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    present: np.ndarray,
    limit: float,
) -> np.ndarray:
    """Flag each row that steps farther than *limit* from its id's row at the frame before.

    Both rows must be present.
    """
    order = np.lexsort((frame, codes))
    both = present[order][1:] & present[order][:-1]
    step = np.hypot(np.diff(x[order]), np.diff(y[order]))
    consecutive = (np.diff(codes[order]) == 0) & (np.diff(frame[order]) == 1)
    hit = np.zeros(frame.size, dtype=bool)
    hit[1:] = both & consecutive & (step > limit)
    flags = np.zeros(frame.size, dtype=bool)
    flags[order] = hit
    return flags


def _score_floor(scores: np.ndarray, rules: QualityRules) -> float | None:
    """Return the score below which a row has a ``low_score``, None when none can be."""
    if rules.low_score_threshold is not None:
        return rules.low_score_threshold
    finite = scores[np.isfinite(scores)]
    if rules.low_score_quantile is None or finite.size == 0:
        return None
    return float(np.quantile(finite, rules.low_score_quantile))


def row_flags(
    table: pd.DataFrame, rules: QualityRules, body_length: float | None
) -> pd.DataFrame:
    """Return the row criteria of *rules* evaluated on each row of *table*.

    Every criterion judges present rows only.

    - ``jump``: the row's id is present at the frame before, and the step
      between the two centers is longer than ``jump_body_lengths`` body lengths.
    - ``low_score``: the row's mean finite ``poseP`` score is below the
      threshold, or below the quantile of the present rows' finite means. A row
      without a finite score has none. A table without ``poseP`` columns skips
      the criterion.
    - ``missing_keypoints``: one of the chosen keypoints is not finite.

    Args:
        table: A tracks table.
        rules: The criteria and their thresholds.
        body_length: The scale of the criteria in body lengths, in the table's
            units, or None when unknown.

    Returns:
        A frame on *table*'s index with ``frame``, ``id`` as text, ``x``, ``y``,
        ``score`` (the mean finite ``poseP``, NaN without one) and ``present``,
        and one boolean column for each row criterion that *rules* turn on and
        that applies to *table*.

    Raises:
        ValueError: ``jump`` is on and *body_length* is None, or
            ``missing_keypoints`` names a keypoint past the table's keypoints.
    """
    import pandas as pd

    present = presence(table)
    frame = table["frame"].to_numpy(dtype=np.int64)
    ids = np.array([str(value) for value in table["id"].tolist()], dtype=object)
    x = _floats(table, "X")
    y = _floats(table, "Y")
    scores = _mean_scores(table)
    columns: dict[str, np.ndarray] = {
        "frame": frame,
        "id": ids,
        "x": x,
        "y": y,
        "score": np.full(len(table), np.nan) if scores is None else scores,
        "present": present,
    }
    if rules.jump_body_lengths is not None:
        limit = rules.jump_body_lengths * _known(body_length, "jump")
        _, codes = np.unique(ids, return_inverse=True)
        columns["jump"] = _jumps(frame, codes, x, y, present, limit)
    if scores is not None and (
        rules.low_score_threshold is not None or rules.low_score_quantile is not None
    ):
        floor = _score_floor(scores[present], rules)
        low = np.zeros(len(table), dtype=bool) if floor is None else scores < floor
        columns["low_score"] = present & low
    if rules.missing_keypoints is not None:
        points = _keypoints(table)
        count = points.shape[1]
        chosen = list(rules.missing_keypoints) or list(range(count))
        _refuse_past("missing_keypoints", chosen, count)
        finite = np.isfinite(points[:, chosen]).all(axis=(1, 2))
        columns["missing_keypoints"] = present & ~finite
    return pd.DataFrame(columns, index=table.index)


def _joined(
    table: pd.DataFrame,
    row: pd.DataFrame,
    rules: QualityRules,
    radius: float,
    span: int,
) -> np.ndarray:
    """Return, per frame, its present rows less the groups that duplicate pairs form.

    A row in no duplicate pair forms a group of one. The centers of present rows
    are finite. Only a head or tail distance can be NaN, and a NaN distance does
    not veto a pair. A ``head`` and ``tail`` given index the table's keypoints.
    """
    present = np.flatnonzero(row["present"].to_numpy(dtype=bool))
    frame_of = row["frame"].to_numpy(dtype=np.int64)[present]
    order = np.argsort(frame_of, kind="stable")
    rows = present[order]
    frame = frame_of[order]
    points = [np.column_stack([_floats(table, "X"), _floats(table, "Y")])[rows]]
    if rules.head is not None and rules.tail is not None:
        keypoints = _keypoints(table)[rows]
        points += [keypoints[:, rules.head], keypoints[:, rules.tail]]
    widest = int(np.bincount(frame).max()) if frame.size else 0
    earlier: list[np.ndarray] = []
    later: list[np.ndarray] = []
    for offset in range(1, widest):
        pair = frame[offset:] == frame[:-offset]
        for point in points:
            apart = np.hypot(*(point[offset:] - point[:-offset]).T)
            pair &= ~(apart >= radius)
        hits = np.flatnonzero(pair)
        earlier.append(hits)
        later.append(hits + offset)
    parent = np.arange(rows.size)

    def root(position: int) -> int:
        while parent[position] != position:
            parent[position] = parent[parent[position]]
            position = int(parent[position])
        return position

    joins = np.zeros(span, dtype=np.int64)
    if earlier:
        for a, b in zip(
            np.concatenate(earlier).tolist(),
            np.concatenate(later).tolist(),
            strict=True,
        ):
            low, high = root(a), root(b)
            if low != high:
                parent[high] = low
                joins[frame[b]] += 1
    return joins


def frame_flags(
    table: pd.DataFrame,
    span: int,
    rules: QualityRules,
    body_length: float | None,
    row: pd.DataFrame,
) -> pd.DataFrame:
    """Return the frame criteria of *rules* evaluated on every frame of the span.

    - ``duplicate``: two present rows of the frame have centers closer than
      ``duplicate_body_lengths`` body lengths, and, when ``head`` and ``tail``
      are given, heads and tails closer too. A non-finite distance does not
      veto.
    - ``fewer_than_expected``: ``n_detected`` is below ``n_expected``.

    ``n_detected`` counts the frame's present rows, and counts as one row each
    set of rows that a chain of duplicate pairs links. Three mutually close
    rows therefore count 1. A frame without rows counts 0.

    Args:
        table: A tracks table.
        span: The frame count of the entry's media. The frames are
            ``0 .. span - 1``.
        rules: The criteria and their thresholds.
        body_length: The scale of the criteria in body lengths, in the table's
            units, or None when unknown.
        row: The :func:`row_flags` of *table*.

    Returns:
        One row per frame of the span, in order, with ``frame``, ``n_present``
        and ``n_detected``, and one boolean column for each frame criterion that
        *rules* turn on.

    Raises:
        ValueError: A row's frame lies outside the span, ``duplicate`` is on
            and *body_length* is None, or ``duplicate`` is on and ``head`` or
            ``tail`` names a keypoint past the table's keypoints.
    """
    import pandas as pd

    frame = row["frame"].to_numpy(dtype=np.int64)
    if frame.size and (frame.min() < 0 or frame.max() >= span):
        msg = (
            f"the table's frames {frame.min()}..{frame.max()} lie outside the "
            f"span 0..{span - 1}"
        )
        raise ValueError(msg)
    present = row["present"].to_numpy(dtype=bool)
    n_present = np.bincount(frame[present], minlength=span)
    columns: dict[str, np.ndarray] = {
        "frame": np.arange(span, dtype=np.int64),
        "n_present": n_present,
    }
    joins = np.zeros(span, dtype=np.int64)
    if rules.duplicate_body_lengths is not None:
        radius = rules.duplicate_body_lengths * _known(body_length, "duplicate")
        _refuse_axis(rules.head, rules.tail, len(_pose_pairs(table)))
        joins = _joined(table, row, rules, radius, span)
        columns["duplicate"] = joins > 0
    n_detected = n_present - joins
    columns["n_detected"] = n_detected
    if rules.n_expected is not None:
        columns["fewer_than_expected"] = n_detected < rules.n_expected
    return pd.DataFrame(columns)


def _runs(frames: np.ndarray) -> list[tuple[int, int]]:
    """Return the runs of consecutive values of sorted unique *frames*."""
    return consecutive_runs(frames, np.ones(frames.size, dtype=bool))


def _row_events(
    row: pd.DataFrame, cause: Criterion, *, by_id: bool
) -> Mapping[str, Sequence[object]]:
    """Return one event per run of *cause*'s flagged frames, per id or per frame.

    A second row of one id at one frame does not extend the id's run.
    """
    flagged = np.flatnonzero(row[cause].to_numpy(dtype=bool))
    frame = row["frame"].to_numpy(dtype=np.int64)[flagged]
    ids = row["id"].to_numpy(dtype=object)[flagged]
    _, codes = np.unique(ids, return_inverse=True)
    onsets: list[int] = []
    lengths: list[int] = []
    if by_id:
        order = np.lexsort((frame, codes))
        repeat = np.zeros(order.size, dtype=bool)
        repeat[1:] = (np.diff(codes[order]) == 0) & (np.diff(frame[order]) == 0)
        order = order[~repeat]
        for code in np.unique(codes).tolist():
            members = order[codes[order] == code]
            for start, stop in _runs(frame[members]):
                onsets.append(int(members[start]))
                lengths.append(stop - start)
    else:
        order = np.lexsort((codes, frame))
        unique, first = np.unique(frame[order], return_index=True)
        for start, stop in _runs(unique):
            onsets.append(int(order[first[start]]))
            lengths.append(stop - start)
    at = np.array(onsets, dtype=np.intp)
    picked = flagged[at]
    onset_ids = [str(value) for value in ids[at].tolist()]
    onset_frames = [int(value) for value in frame[at].tolist()]
    return {
        "frame": onset_frames,
        "cause": [cause] * len(onsets),
        "event_id": [
            f"{cause}:{id_}:{onset}"
            for id_, onset in zip(onset_ids, onset_frames, strict=True)
        ],
        "id": onset_ids,
        "x": _floats(row, "x")[picked].tolist(),
        "y": _floats(row, "y")[picked].tolist(),
        "score": _floats(row, "score")[picked].tolist(),
        "event_frames": lengths,
    }


def _frame_events(
    frame: pd.DataFrame, cause: Criterion
) -> Mapping[str, Sequence[object]]:
    """Return one event per run of *cause*'s flagged frames."""
    frames = frame["frame"].to_numpy(dtype=np.int64)
    runs = consecutive_runs(frames, frame[cause].to_numpy(dtype=bool))
    onsets = [int(frames[start]) for start, _ in runs]
    return {
        "frame": onsets,
        "cause": [cause] * len(runs),
        "event_id": [f"{cause}:*:{onset}" for onset in onsets],
        "id": [""] * len(runs),
        "x": [np.nan] * len(runs),
        "y": [np.nan] * len(runs),
        "score": [np.nan] * len(runs),
        "event_frames": [stop - start for start, stop in runs],
    }


def _event_table(parts: Sequence[Mapping[str, Sequence[object]]]) -> pd.DataFrame:
    """Return *parts* as one table of :data:`EVENT_COLUMNS`, sorted by onset, cause and id."""
    import pandas as pd

    merged = {
        column: [value for part in parts for value in part[column]]
        for column in EVENT_COLUMNS
    }
    table = pd.DataFrame(merged, columns=list(EVENT_COLUMNS)).astype(_EVENT_TYPES)
    ordered = table.sort_values(["frame", "cause", "id"], kind="mergesort")
    return ordered.reset_index(drop=True)


def events(
    row: pd.DataFrame, frame: pd.DataFrame, *, by_id: bool, min_event_frames: int
) -> pd.DataFrame:
    """Return the events of the flagged rows and frames.

    A row criterion's event is a maximal run of consecutive flagged frames of
    one id, or, without ``by_id``, of any id. A frame criterion's event is a
    maximal run of consecutive flagged frames. The event's candidate frame is
    its onset, the run's first frame.

    Args:
        row: The :func:`row_flags` of a table.
        frame: The :func:`frame_flags` of the same table.
        by_id: Whether a row criterion's runs follow one id. Without it, a run
            follows the frames that have a flagged row. Pass False for a producer
            whose ids number each frame's detections.
        min_event_frames: Runs shorter than this are not events.

    Returns:
        One row per event with :data:`EVENT_COLUMNS`, sorted by onset, cause and
        id. ``frame`` is the onset and ``event_frames`` the run's length.
        ``event_id`` is ``<cause>:<id>:<onset>``. A row criterion's event takes
        the id, center and score of a flagged row at its onset, the first in id
        order. A frame criterion's event has the id ``""``, ``*`` in its
        ``event_id``, and NaN center and score.
    """
    parts = [
        _row_events(row, cause, by_id=by_id)
        for cause in ROW_CRITERIA
        if cause in row.columns
    ]
    parts += [
        _frame_events(frame, cause)
        for cause in FRAME_CRITERIA
        if cause in frame.columns
    ]
    table = _event_table(parts)
    long_enough = table["event_frames"].to_numpy(dtype=np.int64) >= min_event_frames
    return table.iloc[np.flatnonzero(long_enough)].reset_index(drop=True)


def _blocked(chosen: list[int], frame: int, min_separation: int) -> bool:
    """Return whether *frame* equals a *chosen* frame or lies within the separation."""
    after = bisect.bisect_left(chosen, frame)
    if after < len(chosen) and (
        chosen[after] == frame or chosen[after] - frame < min_separation
    ):
        return True
    return after > 0 and frame - chosen[after - 1] < min_separation


def select(
    events: pd.DataFrame,
    *,
    n_frames: int,
    min_separation: int,
    exclude: Sequence[int],
    seed: int,
) -> pd.DataFrame:
    """Choose up to *n_frames* event onsets at least *min_separation* frames apart.

    The candidates are sorted by onset, cause and id, then permuted by a
    generator seeded with *seed*. The chosen set starts with *exclude*. A
    candidate is taken when every chosen frame differs from it and lies at least
    *min_separation* frames from it.

    Args:
        events: The :func:`events` of an entry.
        n_frames: The most frames chosen.
        min_separation: No two chosen frames, *exclude* included, lie closer
            than this many frames.
        exclude: Frames already annotated. They seed the chosen set and are not
            returned.
        seed: The seed of the permutation.

    Returns:
        The chosen events, with :data:`EVENT_COLUMNS`, sorted by frame.
    """
    ordered = events.sort_values(
        ["frame", "cause", "id"], kind="mergesort"
    ).reset_index(drop=True)
    frames = ordered["frame"].to_numpy(dtype=np.int64)
    chosen = sorted({int(frame) for frame in exclude})
    taken: list[int] = []
    for position in np.random.default_rng(seed).permutation(frames.size).tolist():
        if len(taken) >= n_frames:
            break
        frame = int(frames[position])
        if _blocked(chosen, frame, min_separation):
            continue
        bisect.insort(chosen, frame)
        taken.append(position)
    return ordered.iloc[sorted(taken)].reset_index(drop=True)
