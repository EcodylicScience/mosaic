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

A third, the sharpest of them, is why ``perspective`` is an alignment column. A pair
row is one *ordered* pair -- ``id1`` the focal, ``id2`` the other -- so two rows of
one frame differ in ``perspective``, and a producer that wrote the ids unswapped made
them differ in nothing else. Merging that against a producer that swapped them keyed
on ``{frame, id1, id2}``: one side was unique on it, so the multiplicity guard
allowed the join as a broadcast, and every row of the non-unique side was bound to
the *wrong* perspective of the other while half of the second input matched nothing
and vanished. The collided ``perspective`` was then renamed ``perspective__1``, which
matched no exclusion list and was read downstream as a measurement.

The rule is exported as a predicate rather than buried in the raise, so a caller
composing a chain ahead of running it uses the same rule the merge enforces. There
is no such caller in-tree yet; the seam is the point.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mosaic.core.pipeline import alignment_verdict, entity_level_of
from mosaic.core.pipeline.types import LoadSpec, ParquetLoadSpec
from mosaic.core.pipeline.loading import (
    MultiInputAlignmentError,
    PairRowsMissing,
    _merge_parquet_inputs,
    load_entry_data,
    merge_naming,
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
    """The exact pairing docs/guides/analysis/write-your-own-feature.md warns about."""
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


def _ordered_pair(
    payload: str, values: tuple[float, float], *, swap: bool
) -> pd.DataFrame:
    """Three frames of one pair, both perspectives, in one of the two conventions.

    *swap* writes the focal into ``id1`` -- the rule. Without it both rows carry
    ``(0, 1)`` and only ``perspective`` separates them, which is what
    ``pair-posedistance-pca`` wrote.
    """
    frames = [0, 1, 2]
    low, high = values
    return pd.concat(
        [
            pd.DataFrame(
                {"frame": frames, "id1": 0, "id2": 1, "perspective": 0, payload: low}
            ),
            pd.DataFrame(
                {
                    "frame": frames,
                    "id1": 1 if swap else 0,
                    "id2": 0 if swap else 1,
                    "perspective": 1,
                    payload: high,
                }
            ),
        ],
        ignore_index=True,
    )


def test_two_pair_inputs_bind_perspective_to_perspective() -> None:
    """Every row keeps its own perspective on both sides, and none is dropped.

    Keyed on ``{frame, id1, id2}`` this returned six rows in which the three
    perspective-1 rows of the left input carried the perspective-0 values of the
    right, and the right input's perspective-1 rows were absent entirely.
    """
    left = _ordered_pair("social", (10.0, 20.0), swap=True)
    right = _ordered_pair("ego", (100.0, 200.0), swap=True)

    merged = _merge_parquet_inputs(iter([(0, left), (1, right)]))

    assert merged is not None
    assert len(merged) == 6
    assert set(zip(merged["social"], merged["ego"])) == {
        (10.0, 100.0),
        (20.0, 200.0),
    }


def test_a_collided_identity_column_is_dropped_rather_than_numbered() -> None:
    """``perspective__1`` was a 0/1 identity flag read downstream as data.

    Numbering a collided identity column put it in the feature space, where
    ``feature_columns`` -- which excludes only the bare name -- handed it to the
    scaler and the embedding. ``group`` and ``sequence`` are the same case, and
    produced ``group__1`` / ``sequence__1`` beside it.
    """
    left = _ordered_pair("social", (10.0, 20.0), swap=True)
    right = _ordered_pair("ego", (100.0, 200.0), swap=True)
    for frame in (left, right):
        frame["group"] = "g"
        frame["sequence"] = "s"

    merged = _merge_parquet_inputs(iter([(0, left), (1, right)]))

    assert merged is not None
    assert [c for c in merged.columns if "__" in c] == []
    assert {"perspective", "group", "sequence"} <= set(merged.columns)


def test_a_stray_perspective_does_not_license_a_cross_level_join() -> None:
    """``perspective`` is an alignment key but names no individual.

    It joined ``ALIGN_COLS`` so two pair inputs bind correctly. The level check
    must keep reading ``ID_COLS``: an individual-level frame that happens to carry
    a ``perspective`` shares an alignment column with a pair frame and still shares
    no identity, so the join is still a per-frame cartesian product.
    """
    individual = ["frame", "time", "id", "group", "sequence", "perspective", "speed"]

    verdict = alignment_verdict([individual, _PAIR])

    assert verdict.compatible is False
    assert "individual" in verdict.reason and "pair" in verdict.reason


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


# --- widening individual inputs onto pair rows ---------------------------------


def _pairs(frames: list[int]) -> pd.DataFrame:
    """Both perspectives of the pair of individuals 0 and 1, per frame."""
    rows = [
        {"frame": f, "id1": a, "id2": b, "perspective": p, "group": "g", "dist": 1.0}
        for f in frames
        for p, (a, b) in enumerate(((0, 1), (1, 0)))
    ]
    return pd.DataFrame(rows)


def _individuals(cells: list[tuple[int, int]]) -> pd.DataFrame:
    """One row per ``(frame, id)``, with ``speed`` naming the cell and bookkeeping."""
    return pd.DataFrame(
        {
            "frame": [f for f, _ in cells],
            "time": [f / 30 for f, _ in cells],
            "id": [i for _, i in cells],
            "group": "g",
            "fps": 30.0,
            "speed": [10.0 * f + i for f, i in cells],
        }
    )


def test_a_widening_merge_gives_each_pair_row_both_individuals() -> None:
    """Each pair row carries its focal's payload as ``__id1`` and the other's as ``__id2``."""
    individuals = _individuals([(0, 0), (0, 1), (1, 0), (1, 1)])

    merged = _merge_parquet_inputs(
        iter([(0, _pairs([0, 1])), (1, individuals)]), expands_individuals=True
    )

    assert merged is not None
    assert len(merged) == 4
    assert set(merged.columns) == {
        "frame",
        "id1",
        "id2",
        "perspective",
        "group",
        "dist",
        "speed__id1",
        "speed__id2",
    }
    frames = merged["frame"].to_numpy()
    for slot in ("id1", "id2"):
        expected = 10.0 * frames + merged[slot].to_numpy()
        assert list(merged[f"speed__{slot}"]) == list(expected)


def test_a_pair_row_without_both_individuals_is_dropped() -> None:
    """An inner join: frame 1 lacks individual 1, so both of its pair rows go."""
    individuals = _individuals([(0, 0), (0, 1), (1, 0)])

    merged = _merge_parquet_inputs(
        iter([(0, _pairs([0, 1])), (1, individuals)]), expands_individuals=True
    )

    assert merged is not None
    assert sorted(merged["frame"]) == [0, 0]


def test_the_widening_merge_is_refused_without_the_flag() -> None:
    with pytest.raises(MultiInputAlignmentError, match="pair every row"):
        _ = _merge_parquet_inputs(
            iter([(0, _pairs([0])), (1, _individuals([(0, 0), (0, 1)]))])
        )


def test_an_individual_input_ahead_of_the_pair_input_is_refused() -> None:
    """The reverse direction needs a choice among partners, so it stays refused."""
    with pytest.raises(MultiInputAlignmentError, match="pair-level input first"):
        _ = _merge_parquet_inputs(
            iter([(0, _individuals([(0, 0), (0, 1)])), (1, _pairs([0]))]),
            expands_individuals=True,
        )


def test_a_repeated_individual_row_is_refused() -> None:
    with pytest.raises(MultiInputAlignmentError, match="repeats"):
        _ = _merge_parquet_inputs(
            iter([(0, _pairs([0])), (1, _individuals([(0, 0), (0, 0), (0, 1)]))]),
            expands_individuals=True,
        )


def test_the_widening_merge_joins_on_the_shared_order_column() -> None:
    """With no shared ``frame``, the rows align on ``time``."""
    pairs = _pairs([0, 1]).rename(columns={"frame": "time"})
    individuals = _individuals([(0, 0), (0, 1), (1, 0), (1, 1)]).drop(columns=["time"])
    individuals = individuals.rename(columns={"frame": "time"})

    merged = _merge_parquet_inputs(
        iter([(0, pairs), (1, individuals)]), expands_individuals=True
    )

    assert merged is not None
    assert len(merged) == 4
    assert "frame" not in merged.columns


def test_a_collided_widened_name_is_numbered_by_its_input() -> None:
    cells = [(0, 0), (0, 1)]

    merged = _merge_parquet_inputs(
        iter([(0, _pairs([0])), (1, _individuals(cells)), (2, _individuals(cells))]),
        expands_individuals=True,
    )

    assert merged is not None
    assert {"speed__id1", "speed__id2", "speed__2__id1", "speed__2__id2"} <= set(
        merged.columns
    )


def test_a_global_input_still_broadcasts_under_the_flag() -> None:
    per_frame = pd.DataFrame({"frame": [0, 1], "order": [0.5, 0.7]})

    merged = _merge_parquet_inputs(
        iter([(0, _pairs([0, 1])), (1, per_frame)]), expands_individuals=True
    )

    assert merged is not None
    assert len(merged) == 4
    assert "order" in merged.columns


def _specs(tmp_path: Path, tables: list[pd.DataFrame]) -> list[tuple[Path, LoadSpec]]:
    specs: list[tuple[Path, LoadSpec]] = []
    for position, table in enumerate(tables):
        path = tmp_path / f"{position}.parquet"
        table.to_parquet(path)
        specs.append((path, ParquetLoadSpec()))
    return specs


def test_the_widening_naming_rule_is_what_the_merge_does(tmp_path: Path) -> None:
    cells = [(0, 0), (0, 1)]
    tables = [_pairs([0]), _individuals(cells), _individuals(cells)]

    merged = load_entry_data(_specs(tmp_path, tables), expands_individuals=True)
    assert merged is not None

    columns: frozenset[str] = frozenset(tables[0].columns)
    for position, table in ((1, tables[1]), (2, tables[2])):
        columns = merge_naming(
            columns, list(table.columns), position, widen=True
        ).columns
    assert columns == set(merged.columns)


def test_an_empty_pair_input_beside_individual_rows_is_refused(tmp_path: Path) -> None:
    """Without it the entry would hand individual rows to a pair-level consumer."""
    tables = [_pairs([0]).iloc[0:0], _individuals([(0, 0), (0, 1)])]

    with pytest.raises(PairRowsMissing, match="input 0"):
        _ = load_entry_data(_specs(tmp_path, tables), expands_individuals=True)

    assert load_entry_data(_specs(tmp_path, tables)) is not None
