"""A multi-input merge must refuse rather than invent rows.

``_merge_parquet_inputs`` picked join keys as the intersection of a hardcoded
``_ALIGN_COLS = {frame, time, id, id1, id2}`` with both frames' columns, raised only
when that intersection was **empty**, and merged inner with no ``validate=``.

``group`` and ``sequence`` are not in that set, so joining an individual-level
feature (``speed-angvel``: frame, time, id, group, sequence) with a pair-level one
(``pair-egocentric``: frame, perspective, id1, id2) intersects to ``{frame}`` alone
-- a per-frame cartesian product of every individual against every pair. Silently.
The documented getting-started example does exactly this, and feeds the result to
``extract-templates`` and ``global-scaler``, so a t-SNE was fitted on invented rows.

A second silent mode: joining a canonicalised pair feature (``approach-avoidance``,
``id1 < id2``) with a both-perspectives one merges on ``{frame, id1, id2}`` and drops
roughly half the rows, because only one perspective matches.

The rule is exported as a predicate rather than buried in the raise, so a caller
composing a chain ahead of running it uses the same rule the merge enforces. There
is no such caller in-tree yet; the seam is the point.
"""

from __future__ import annotations

import pandas as pd
import pytest

from mosaic.core.pipeline import alignment_verdict, entity_level_of
from mosaic.core.pipeline.loading import (
    MultiInputAlignmentError,
    _merge_parquet_inputs,
)

_INDIVIDUAL = ["frame", "time", "id", "group", "sequence", "speed"]
_PAIR = ["frame", "perspective", "id1", "id2", "ego_x"]


def _frame(columns: list[str], rows: int = 4) -> pd.DataFrame:
    data: dict[str, object] = {}
    for name in columns:
        if name in {"frame", "id", "id1", "id2", "perspective"}:
            data[name] = list(range(rows))
        elif name in {"group", "sequence"}:
            data[name] = [""] * rows
        else:
            data[name] = [float(i) for i in range(rows)]
    return pd.DataFrame(data)


def test_an_individual_level_input_and_a_pair_level_input_refuse_to_merge() -> None:
    """The exact pairing docs/getting-started.md recommends."""
    with pytest.raises(MultiInputAlignmentError, match="individual"):
        _ = _merge_parquet_inputs(iter([(0, _frame(_INDIVIDUAL)), (1, _frame(_PAIR))]))


def test_a_merge_that_would_multiply_rows_refuses() -> None:
    """Many-to-many fans out; one-to-many is a legitimate broadcast and stays.

    A per-frame table joined against a per-frame-per-id one is exactly how a global
    feature's output reaches individual rows, so only *neither* side being unique is
    the refusal.
    """
    keyed = pd.DataFrame({"frame": [0, 0, 1, 1], "id": [0, 1, 0, 1], "a": [1.0] * 4})
    same = pd.DataFrame({"frame": [0, 0, 1, 1], "id": [0, 1, 0, 1], "b": [2.0] * 4})
    merged = _merge_parquet_inputs(iter([(0, keyed), (1, same)]))
    assert merged is not None and len(merged) == 4

    per_frame = pd.DataFrame({"frame": [0, 1], "id": [0, 0], "b": [2.0, 3.0]})
    broadcast = _merge_parquet_inputs(iter([(0, keyed), (1, per_frame)]))
    assert broadcast is not None and len(broadcast) == 2

    dup_a = pd.DataFrame({"frame": [0, 0], "id": [0, 0], "a": [1.0, 2.0]})
    dup_b = pd.DataFrame({"frame": [0, 0], "id": [0, 0], "b": [3.0, 4.0]})
    with pytest.raises(MultiInputAlignmentError, match="multiply"):
        _ = _merge_parquet_inputs(iter([(0, dup_a), (1, dup_b)]))


def test_two_inputs_at_the_same_level_still_merge() -> None:
    """The counter-test: the legitimate case must stay legitimate.

    Two pair-level features -- the shipped notebook chain merges two
    ``pair-wavelet`` runs -- share frame, id1 and id2, which is a real key.
    """
    left = _frame(_PAIR)
    right = _frame(["frame", "perspective", "id1", "id2", "wave"])
    merged = _merge_parquet_inputs(iter([(0, left), (1, right)]))

    assert merged is not None
    assert len(merged) == len(left)


def test_the_level_of_a_column_set_is_a_callable_rule() -> None:
    """One rule, exported, so a submit-time check cannot drift from the merge."""
    assert entity_level_of(_INDIVIDUAL) == "individual"
    assert entity_level_of(_PAIR) == "pair"
    assert entity_level_of(["frame", "time", "value"]) == "global"

    verdict = alignment_verdict([_INDIVIDUAL, _PAIR])
    assert verdict.compatible is False
    assert "individual" in verdict.reason and "pair" in verdict.reason

    assert alignment_verdict([_PAIR, _PAIR]).compatible is True


def test_a_collided_column_is_suffixed_with_the_input_that_declared_it() -> None:
    """The suffix counted surviving inputs, not declared ones.

    With three inputs and an empty middle one, input 2's column was renamed
    ``__1``, and ``_find_merged_column`` -- asked for ``__2``, missing, falling
    through to the bare name -- returned input 0's column under input 2's name.
    """
    first = pd.DataFrame({"frame": [0, 1], "id": [0, 0], "v": [1.0, 1.0]})
    third = pd.DataFrame({"frame": [0, 1], "id": [0, 0], "v": [3.0, 3.0]})

    merged = _merge_parquet_inputs(iter([(0, first), (2, third)]))

    assert merged is not None
    assert "v__2" in merged.columns
    assert list(merged["v__2"]) == [3.0, 3.0]
