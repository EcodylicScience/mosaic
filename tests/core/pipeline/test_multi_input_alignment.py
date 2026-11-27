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

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest

from mosaic.core.pipeline import alignment_verdict, entity_level_of
from mosaic.core.pipeline.types import LoadSpec, ParquetLoadSpec
from mosaic.core.pipeline.loading import (
    EntryRowsMissing,
    MultiInputAlignmentError,
    PairRowsMissing,
    _merge_parquet_inputs,
    entry_summary_join,
    load_entry_data,
    merge_naming,
    merge_step,
    summary_join_cell,
)
from tests.helpers import (
    UPDATE_GOLDEN_ENV,
    read_golden,
    regenerate_command,
    updating_golden,
    write_golden,
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

    with pytest.raises(PairRowsMissing, match="input 0") as raised:
        _ = load_entry_data(_specs(tmp_path, tables), expands_individuals=True)

    assert isinstance(raised.value, EntryRowsMissing)

    assert load_entry_data(_specs(tmp_path, tables)) is not None


# --- the merge where no summary key is shared ------------------------------------

_MERGE_GOLDEN = "merge_without_summary_keys.json"


@dataclass(frozen=True)
class _MergeCase:
    """The loaded inputs of one entry, as ``(declared position, frame)``, and the flags."""

    inputs: tuple[tuple[int, pd.DataFrame], ...]
    cross_join: bool = False
    expands_individuals: bool = False


def _values(name: str, frames: list[int], ids: list[int], salt: float) -> list[float]:
    """One value per row, set by the column name, the row's frame and id, and *salt*."""
    scale = float(sum(ord(letter) for letter in name))
    return [
        scale + salt + 10.0 * frame + identity for frame, identity in zip(frames, ids)
    ]


def _per_id(
    payload: list[str],
    *,
    time: bool = True,
    salt: float = 1.0,
    ids: tuple[int, ...] = (0, 1),
    frames: tuple[int, ...] = (0, 1, 2),
    bookkeeping: bool = False,
    perspective: bool = False,
) -> pd.DataFrame:
    """Individual rows: every ``(frame, id)``, with the group and sequence columns."""
    rows = [(frame, identity) for frame in frames for identity in ids]
    row_frames = [frame for frame, _ in rows]
    row_ids = [identity for _, identity in rows]
    data: dict[str, list[object]] = {"frame": list(row_frames)}
    if time:
        data["time"] = [frame / 30 for frame in row_frames]
    data["id"] = list(row_ids)
    data["group"] = ["g"] * len(rows)
    data["sequence"] = ["s"] * len(rows)
    if perspective:
        data["perspective"] = [identity % 2 for identity in row_ids]
    if bookkeeping:
        data["fps"] = [30.0] * len(rows)
        data["nn_id"] = [
            float(1 - identity) if frame else float("nan") for frame, identity in rows
        ]
    for name in payload:
        data[name] = list(_values(name, row_frames, row_ids, salt))
    return pd.DataFrame(data)


def _per_pair(
    payload: list[str],
    *,
    time: bool = False,
    salt: float = 1.0,
    frames: tuple[int, ...] = (0, 1, 2),
    swap: bool = True,
    bookkeeping: bool = False,
) -> pd.DataFrame:
    """Pair rows: both perspectives of the pair of individuals 0 and 1 per frame."""
    rows = [(frame, perspective) for frame in frames for perspective in (0, 1)]
    row_frames = [frame for frame, _ in rows]
    perspectives = [perspective for _, perspective in rows]
    data: dict[str, list[object]] = {"frame": list(row_frames)}
    if time:
        data["time"] = [frame / 30 for frame in row_frames]
    data["id1"] = [perspective if swap else 0 for perspective in perspectives]
    data["id2"] = [1 - perspective if swap else 1 for perspective in perspectives]
    data["perspective"] = list(perspectives)
    data["group"] = ["g"] * len(rows)
    data["sequence"] = ["s"] * len(rows)
    if bookkeeping:
        data["fps"] = [30.0] * len(rows)
    for name in payload:
        data[name] = list(_values(name, row_frames, perspectives, salt))
    return pd.DataFrame(data)


def _per_frame(
    payload: list[str],
    *,
    frame: bool = True,
    time: bool = True,
    salt: float = 1.0,
    frames: tuple[int, ...] = (0, 1, 2),
    bookkeeping: bool = False,
    extra: dict[str, list[object]] | None = None,
) -> pd.DataFrame:
    """Collective rows: one per frame, naming nobody."""
    data: dict[str, list[object]] = {}
    if frame:
        data["frame"] = list(frames)
    if time:
        data["time"] = [step / 30 for step in frames]
    data["group"] = ["g"] * len(frames)
    data["sequence"] = ["s"] * len(frames)
    if bookkeeping:
        data["fps"] = [30.0] * len(frames)
    for name in payload:
        data[name] = list(_values(name, list(frames), [0] * len(frames), salt))
    data.update(extra or {})
    return pd.DataFrame(data)


def _merge_corpus() -> dict[str, _MergeCase]:
    """Pairs and triples of inputs that share no summary key, each a fresh set of frames."""
    cases: dict[str, _MergeCase] = {}

    def add(
        name: str,
        *tables: pd.DataFrame,
        positions: tuple[int, ...] | None = None,
        cross_join: bool = False,
        expands_individuals: bool = False,
    ) -> None:
        declared = positions if positions is not None else tuple(range(len(tables)))
        cases[name] = _MergeCase(
            tuple(zip(declared, tables)), cross_join, expands_individuals
        )

    add(
        "individual_and_global_on_frame_and_time",
        _per_id(["speed"]),
        _per_frame(["order"]),
    )
    add(
        "individual_without_time_and_global_with_time",
        _per_id(["speed"], time=False),
        _per_frame(["order"]),
    )
    add(
        "individual_and_global_sharing_no_order_column",
        _per_id(["speed"], time=False),
        _per_frame(["order"], frame=False),
    )
    add(
        "pair_and_global_on_frame",
        _per_pair(["ego"]),
        _per_frame(["order"], time=False),
    )
    add(
        "pair_and_global_on_frame_and_time",
        _per_pair(["ego"], time=True),
        _per_frame(["order"]),
    )
    add("pair_and_pair_swapped", _per_pair(["a"]), _per_pair(["b"], salt=2.0))
    add(
        "pair_and_pair_unswapped",
        _per_pair(["a"], swap=False),
        _per_pair(["b"], swap=False),
    )
    add(
        "pair_swapped_and_pair_unswapped",
        _per_pair(["a"]),
        _per_pair(["b"], swap=False),
    )
    add(
        "individual_and_individual_on_id",
        _per_id(["speed"]),
        _per_id(["turn"], salt=2.0),
    )
    add(
        "individual_and_individual_collided_payload",
        _per_id(["speed", "turn"]),
        _per_id(["speed"], salt=2.0),
    )
    add(
        "individual_and_individual_collided_bookkeeping",
        _per_id(["speed"], bookkeeping=True),
        _per_id(["turn"], salt=2.0, bookkeeping=True),
    )
    add(
        "individual_and_individual_with_perspective",
        _per_id(["speed"], perspective=True),
        _per_id(["turn"], salt=2.0, perspective=True),
    )
    add(
        "global_and_global_collided_payload",
        _per_frame(["order", "spread"], bookkeeping=True),
        _per_frame(["order"], salt=2.0, bookkeeping=True),
    )
    add(
        "global_on_frame_and_global_on_time",
        _per_frame(["order"], time=False),
        _per_frame(["spread"], frame=False),
    )
    add("individual_and_pair_levels", _per_id(["speed"]), _per_pair(["ego"]))
    add(
        "individual_with_perspective_and_pair",
        _per_id(["speed"], perspective=True),
        _per_pair(["ego"]),
    )
    add(
        "individual_and_pair_with_the_time_column",
        _per_id(["speed"]),
        _per_pair(["ego"], time=True),
    )
    add(
        "individual_global_global",
        _per_id(["speed"]),
        _per_frame(["order"]),
        _per_frame(["spread"], salt=2.0),
    )
    add(
        "individual_global_global_collided",
        _per_id(["speed"]),
        _per_frame(["order"]),
        _per_frame(["order"], salt=2.0),
    )
    add(
        "global_individual_individual",
        _per_frame(["order"]),
        _per_id(["speed"]),
        _per_id(["turn"], salt=2.0),
    )
    add(
        "pair_global_pair",
        _per_pair(["ego"]),
        _per_frame(["order"], time=False),
        _per_pair(["social"], salt=2.0),
    )
    add(
        "global_then_individual_then_pair",
        _per_frame(["order"]),
        _per_id(["speed"]),
        _per_pair(["ego"]),
    )
    add(
        "an_empty_middle_input_numbers_by_declared_position",
        _per_id(["v"]),
        _per_id(["v"], salt=3.0),
        positions=(0, 2),
    )
    add(
        "an_empty_first_input_is_left_out",
        _per_id(["v"]),
        _per_frame(["order"]),
        positions=(1, 2),
    )
    add(
        "an_empty_middle_input_beside_a_widening_consumer",
        _per_pair(["ego"]),
        _per_id(["speed"]),
        positions=(0, 2),
        expands_individuals=True,
    )
    add(
        "widening_onto_pair_rows",
        _per_pair(["ego"]),
        _per_id(["speed"]),
        expands_individuals=True,
    )
    add(
        "widening_two_individual_inputs",
        _per_pair(["ego"], bookkeeping=True),
        _per_id(["speed"], bookkeeping=True),
        _per_id(["speed"], salt=2.0, bookkeeping=True),
        expands_individuals=True,
    )
    add(
        "widening_on_time",
        _per_pair(["ego"], time=True).drop(columns=["frame"]),
        _per_id(["speed"]).drop(columns=["frame"]),
        expands_individuals=True,
    )
    add(
        "widening_on_frame_when_both_have_time",
        _per_pair(["ego"], time=True),
        _per_id(["speed"]),
        expands_individuals=True,
    )
    add(
        "widening_with_no_shared_order_column",
        _per_pair(["ego"]),
        _per_id(["speed"], time=False).rename(columns={"frame": "step"}),
        expands_individuals=True,
    )
    add(
        "widening_with_the_individual_input_first",
        _per_id(["speed"]),
        _per_pair(["ego"]),
        expands_individuals=True,
    )
    add(
        "widening_a_repeated_individual_row",
        _per_pair(["ego"]),
        pd.concat([_per_id(["speed"]), _per_id(["speed"]).iloc[:1]], ignore_index=True),
        expands_individuals=True,
    )
    add(
        "widening_leaves_a_global_input_to_broadcast",
        _per_pair(["ego"], time=True),
        _per_frame(["order"]),
        _per_id(["speed"]),
        expands_individuals=True,
    )
    add(
        "widening_leaves_a_pair_input_to_merge",
        _per_pair(["ego"]),
        _per_pair(["social"], salt=2.0),
        _per_id(["speed"]),
        expands_individuals=True,
    )
    add(
        "a_cross_join_consumer_joins_on_frame_alone",
        _per_id(["speed"], time=False),
        _per_pair(["ego"]),
        cross_join=True,
    )
    add(
        "a_cross_join_consumer_numbers_a_collided_payload",
        _per_id(["v"], time=False),
        _per_pair(["v"], salt=2.0),
        cross_join=True,
    )
    add(
        "a_cross_join_consumer_with_no_shared_key",
        _per_id(["speed"], time=False),
        _per_frame(["order"], frame=False),
        cross_join=True,
    )
    add(
        "a_cross_join_consumer_beside_a_global_input",
        _per_pair(["ego"]),
        _per_frame(["order"], time=False),
        _per_id(["speed"], time=False),
        cross_join=True,
    )
    duplicated = pd.concat(
        [_per_frame(["a"]), _per_frame(["a"], salt=5.0)], ignore_index=True
    )
    add(
        "a_key_set_duplicated_on_both_sides",
        duplicated,
        duplicated.rename(columns={"a": "b"}),
    )
    add(
        "a_key_set_duplicated_on_both_sides_under_a_cross_join",
        duplicated,
        duplicated.rename(columns={"a": "b"}),
        cross_join=True,
    )
    add(
        "a_key_set_duplicated_on_one_side",
        duplicated,
        _per_frame(["b"], salt=2.0),
    )
    add(
        "an_arena_on_the_individual_side_alone",
        _per_id(["speed"]).assign(arena=[0, 0, 1, 1, 0, 0]),
        _per_frame(["order"]),
    )
    add(
        "an_arena_on_the_collective_side_alone",
        _per_id(["speed"]),
        _per_frame(["order"], extra={"arena": [0, 1, 0]}),
    )
    add(
        "bin_columns_on_the_collective_side_alone",
        _per_id(["speed"]),
        _per_frame(
            ["order"], extra={"bin_start": [0.0, 0.0, 1.0], "bin_end": [1.0, 1.0, 2.0]}
        ),
    )
    add(
        "an_arena_on_one_pair_input_alone",
        _per_pair(["ego"]).assign(arena=[0, 0, 0, 0, 0, 0]),
        _per_pair(["social"], salt=2.0),
    )
    return cases


def _merge_record(case: _MergeCase) -> str | dict[str, object]:
    """``"refused"``, or the merged frame's column names, dtypes and values digest.

    The digest reads the values and the index alone, so the names and the dtypes
    are recorded beside it.
    """
    try:
        merged = _merge_parquet_inputs(
            iter(case.inputs),
            cross_join=case.cross_join,
            expands_individuals=case.expands_individuals,
        )
    except MultiInputAlignmentError:
        return "refused"
    assert merged is not None
    hashed = pd.util.hash_pandas_object(merged, index=True)
    digest = hashlib.blake2b(hashed.to_numpy().tobytes(), digest_size=16)
    return {
        "columns": [str(column) for column in merged.columns],
        "dtypes": [str(dtype) for dtype in merged.dtypes],
        "digest": digest.hexdigest(),
    }


def test_the_merge_is_today_s_where_no_summary_key_is_shared() -> None:
    """Where two inputs share no summary key, the merge is the one before summary keys.

    ``merge_without_summary_keys.json`` records, per case, ``refused`` or the
    merged frame as it was before a shared summary key joined: its column names
    in order, its dtypes, and a digest of its values and index. A case recorded
    as a refusal may now merge, since a later step joins what it refused.
    """
    corpus = _merge_corpus()
    if updating_golden():
        write_golden(
            _MERGE_GOLDEN, {name: _merge_record(case) for name, case in corpus.items()}
        )
        return
    recorded = read_golden(_MERGE_GOLDEN)
    assert not recorded.keys() - corpus.keys(), (
        "the golden file has a case the corpus lacks"
    )
    for name, case in corpus.items():
        assert name in recorded, (
            f"no recorded merge for {name!r}; run `{regenerate_command(__name__)}`"
        )
        if recorded[name] == "refused":
            continue
        assert _merge_record(case) == recorded[name], (
            f"the merge of {name!r} differs from the recorded one; "
            f"{UPDATE_GOLDEN_ENV}=1 rewrites it only for a deliberate change"
        )


# --- shared summary keys ---------------------------------------------------------


def _binned_per_id(payload: str, *, bins: int = 2, salt: float = 0.0) -> pd.DataFrame:
    """A per-id summary: one row per ``(id, bin)``, with its bin's start and end."""
    rows = [(identity, bin_index) for identity in (0, 1) for bin_index in range(bins)]
    return pd.DataFrame(
        {
            "id": [identity for identity, _ in rows],
            "group": "g",
            "sequence": "s",
            "bin_start": [10.0 * bin_index for _, bin_index in rows],
            "bin_end": [10.0 * (bin_index + 1) for _, bin_index in rows],
            payload: [
                salt + 100.0 * identity + bin_index for identity, bin_index in rows
            ],
        }
    )


def test_two_binned_per_id_summaries_join_on_the_id_and_the_bin() -> None:
    """The bin is a key beside ``id``, so no copy of it is numbered."""
    left = _binned_per_id("speed")
    right = _binned_per_id("turn", salt=0.5)

    merged = _merge_parquet_inputs(iter([(0, left), (1, right)]))

    assert merged is not None
    assert len(merged) == 4
    assert [c for c in merged.columns if "__" in c] == []
    assert list(merged["turn"]) == [value + 0.5 for value in merged["speed"]]


def test_binned_rows_whose_bins_differ_between_the_sides_do_not_match() -> None:
    left = _binned_per_id("speed")
    right = _binned_per_id("turn", salt=0.5)
    right.loc[right["bin_start"] == 10.0, ["bin_start", "bin_end"]] = [5.0, 15.0]

    merged = _merge_parquet_inputs(iter([(0, left), (1, right)]))

    assert merged is not None
    assert list(merged["bin_start"]) == [0.0, 0.0]


def test_per_frame_rows_with_arena_join_per_frame_per_arena_rows_on_the_arena() -> None:
    """``arena`` joins beside ``frame`` and ``time``, and is not numbered ``arena__1``."""
    individuals = _per_id(["speed"]).assign(arena=[0, 1] * 3)
    per_arena = pd.DataFrame(
        {
            "frame": [0, 0, 1, 1, 2, 2],
            "time": [0.0, 0.0, 1 / 30, 1 / 30, 2 / 30, 2 / 30],
            "arena": [0, 1] * 3,
            "group": "g",
            "sequence": "s",
            "count": [10.0, 20.0, 11.0, 21.0, 12.0, 22.0],
        }
    )

    merged = _merge_parquet_inputs(iter([(0, individuals), (1, per_arena)]))

    assert merged is not None
    assert "arena__1" not in merged.columns
    assert len(merged) == 6
    assert list(merged["count"]) == [10.0, 20.0, 11.0, 21.0, 12.0, 22.0]


def test_a_per_frame_collective_joins_individual_rows_on_frame_and_time() -> None:
    """A per-frame collective names nobody and has no summary key: the keys are today's."""
    individuals = _per_id(["speed"])
    collective = _per_frame(["order"])

    merged = _merge_parquet_inputs(iter([(0, individuals), (1, collective)]))

    assert merged is not None
    assert len(merged) == len(individuals)
    assert merge_naming(individuals.columns, collective.columns, 1).keys == (
        "frame",
        "time",
    )
    assert list(merged["order"]) == [
        float(sum(ord(letter) for letter in "order")) + 1.0 + 10.0 * frame
        for frame in individuals["frame"]
    ]


def test_a_nan_summary_key_refuses_before_the_join() -> None:
    """pandas matches ``NaN`` to ``NaN``, so a blank bin would join a blank bin."""
    left = _binned_per_id("speed")
    right = _binned_per_id("turn", salt=0.5)
    left["bin_start"] = left["bin_start"].astype(float)
    left.loc[left.index[0], "bin_start"] = float("nan")

    with pytest.raises(
        MultiInputAlignmentError,
        match=(
            "cannot merge input 1: bin_start is missing in 1 rows of the inputs "
            "before it, and a summary key names a bin or an arena"
        ),
    ):
        _ = _merge_parquet_inputs(iter([(0, left), (1, right)]))

    with pytest.raises(
        MultiInputAlignmentError,
        match=(
            "cannot merge input 1: bin_end is missing in 2 rows of input 1, and a "
            "summary key names a bin or an arena"
        ),
    ):
        right.loc[right.index[:2], "bin_end"] = float("nan")
        _ = _merge_parquet_inputs(iter([(0, _binned_per_id("speed")), (1, right)]))


def test_a_cross_join_consumer_keeps_the_numbered_arena() -> None:
    """A consumer in ``CROSS_JOIN_FEATURES`` keeps today's keys, without summary keys."""
    individuals = _per_id(["speed"], time=False).assign(arena=[0, 1] * 3)
    pairs = _per_pair(["ego"]).assign(arena=[1] * 6)

    merged = _merge_parquet_inputs(
        iter([(0, individuals), (1, pairs)]), cross_join=True
    )

    assert merged is not None
    assert {"arena", "arena__1"} <= set(merged.columns)


def test_a_shared_summary_key_join_keeps_the_merged_sides_row_order() -> None:
    left = _binned_per_id("speed").iloc[[3, 0, 2, 1]].reset_index(drop=True)
    right = _binned_per_id("turn", salt=0.5)

    merged = _merge_parquet_inputs(iter([(0, left), (1, right)]))

    assert merged is not None
    assert list(merged["speed"]) == list(left["speed"])


# --- the column-level decision of a merge position -------------------------------

_BINNED = ["id", "group", "sequence", "bin_start", "bin_end", "speed"]


def test_merge_step_joins_the_summary_keys_that_both_sides_have() -> None:
    step = merge_step(
        _BINNED,
        [*_BINNED[:-1], "turn"],
        1,
        cross_join=False,
        expands_individuals=False,
    )

    assert step.kind == "shared"
    assert step.keys == ("bin_end", "bin_start", "id")
    assert step.summary_keys == ("bin_end", "bin_start")
    assert step.refusal == ""
    assert dict(step.naming.renamed) == {}
    assert set(step.naming.dropped) == {"group", "sequence"}
    assert step.naming.columns == frozenset([*_BINNED, "turn"])


def test_merge_step_keeps_a_collided_payload_column_numbered() -> None:
    step = merge_step(_BINNED, _BINNED, 2, cross_join=False, expands_individuals=False)

    assert dict(step.naming.renamed) == {"speed": "speed__2"}


def test_merge_step_joins_no_summary_key_where_only_one_side_has_it() -> None:
    step = merge_step(
        [*_INDIVIDUAL, "arena"],
        ["frame", "time", "order"],
        1,
        cross_join=False,
        expands_individuals=False,
    )

    assert step.keys == ("frame", "time")
    assert step.summary_keys == ()
    assert "arena" in step.naming.columns


def test_merge_step_for_a_cross_join_consumer_keeps_todays_keys() -> None:
    columns = ["frame", "id", "arena", "speed"]
    step = merge_step(
        columns,
        ["frame", "arena", "count"],
        1,
        cross_join=True,
        expands_individuals=False,
    )

    assert step.keys == ("frame",)
    assert step.summary_keys == ()
    assert dict(step.naming.renamed) == {"arena": "arena__1"}


def test_merge_step_widens_on_the_shared_order_column_without_summary_keys() -> None:
    pair_summary = [*_PAIR, "arena", "bin_start", "bin_end"]
    individual = [*_INDIVIDUAL, "arena", "bin_start", "bin_end"]

    step = merge_step(
        pair_summary, individual, 1, cross_join=False, expands_individuals=True
    )

    assert step.kind == "widen"
    assert step.keys == ("frame",)
    assert step.summary_keys == ()
    assert step.refusal == ""
    assert "arena__id1" in step.naming.columns


def test_merge_step_refuses_a_widening_that_shares_no_order_column() -> None:
    step = merge_step(
        _PAIR,
        ["step", "id", "speed"],
        3,
        cross_join=False,
        expands_individuals=True,
    )

    assert step.kind == "widen"
    assert step.refusal == (
        "cannot merge input 3: it shares no order column (frame or time) with "
        "the pair rows"
    )


def test_merge_step_refuses_levels_with_the_widening_hint() -> None:
    plain = merge_step(
        _INDIVIDUAL, _PAIR, 1, cross_join=False, expands_individuals=False
    )
    widening = merge_step(
        _INDIVIDUAL, _PAIR, 1, cross_join=False, expands_individuals=True
    )

    assert plain.refusal.startswith("cannot merge input 1: inputs are at different")
    assert not plain.refusal.endswith("pair-level input first")
    assert widening.refusal == (
        f"{plain.refusal}; widening individual rows onto pair rows needs the "
        "pair-level input first"
    )


def test_merge_step_refuses_inputs_that_share_no_alignment_column() -> None:
    columns = ["bin_start", "bin_end", "speed"]
    for cross_join in (False, True):
        step = merge_step(
            columns,
            ["bin_start", "bin_end", "turn"],
            1,
            cross_join=cross_join,
            expands_individuals=False,
        )
        assert step.refusal == ("cannot merge input 1: no shared alignment columns")
        assert step.summary_keys == ()


def test_an_alignment_verdict_names_its_cause() -> None:
    assert alignment_verdict([_INDIVIDUAL, _PAIR]).cause == "levels"
    assert alignment_verdict([["frame", "a"], ["time", "b"]]).cause == "no_keys"
    assert alignment_verdict([_PAIR, _PAIR]).cause == ""


def test_a_collided_column_among_the_given_keys_is_neither_dropped_nor_renamed() -> (
    None
):
    naming = merge_naming(
        ["frame", "arena", "group", "speed"],
        ["frame", "arena", "group", "count"],
        1,
        keys=["frame", "arena"],
    )

    assert naming.keys == ("arena", "frame")
    assert naming.dropped == ("group",)
    assert dict(naming.renamed) == {}
    assert naming.columns == frozenset(["frame", "arena", "group", "speed", "count"])


# --- the summary keys a step joins ------------------------------------------------

_ARENA_FRAMES = ["frame", "id", "arena", "speed"]


def test_the_cell_names_the_summary_keys_that_both_inputs_have() -> None:
    cell = summary_join_cell(
        [_ARENA_FRAMES, ["frame", "id", "arena", "count"]],
        cross_join=False,
        expands_individuals=False,
    )

    assert cell == "arena"


def test_the_cell_is_the_sorted_union_over_the_positions() -> None:
    binned = ["id", "bin_start", "bin_end", "mean"]
    cell = summary_join_cell(
        [
            [*binned, "arena"],
            [*binned[:3], "arena", "max"],
            [*binned[:3], "min"],
        ],
        cross_join=False,
        expands_individuals=False,
    )

    assert cell == "arena,bin_end,bin_start"


def test_the_cell_is_blank_where_no_summary_key_is_shared() -> None:
    assert (
        summary_join_cell(
            [_ARENA_FRAMES, ["frame", "id", "count"]],
            cross_join=False,
            expands_individuals=False,
        )
        == ""
    )


def test_a_cross_join_consumer_joins_no_summary_key() -> None:
    assert (
        summary_join_cell(
            [_ARENA_FRAMES, ["frame", "id", "arena", "count"]],
            cross_join=True,
            expands_individuals=False,
        )
        == ""
    )


def test_a_widening_position_joins_no_summary_key() -> None:
    pairs = ["frame", "perspective", "id1", "id2", "arena", "ego_x"]

    cell = summary_join_cell(
        [pairs, ["frame", "id", "arena", "speed"]],
        cross_join=False,
        expands_individuals=True,
    )

    assert cell == ""


def test_a_refused_position_ends_the_fold() -> None:
    cell = summary_join_cell(
        [_ARENA_FRAMES, ["bin_start", "total"], ["frame", "id", "arena", "count"]],
        cross_join=False,
        expands_individuals=False,
    )

    assert cell == ""


def test_an_input_that_is_left_out_takes_no_part_in_the_fold() -> None:
    cell = summary_join_cell(
        [_ARENA_FRAMES, None, ["frame", "id", "arena", "count"]],
        cross_join=False,
        expands_individuals=False,
    )

    assert cell == "arena"


def test_the_cell_of_one_input_is_blank() -> None:
    assert (
        summary_join_cell([_ARENA_FRAMES], cross_join=False, expands_individuals=False)
        == ""
    )


def _arena_table(payload: str, rows: int = 3) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "frame": list(range(rows)),
            "id": [0] * rows,
            "arena": ["a"] * rows,
            payload: [float(i) for i in range(rows)],
        }
    )


def test_the_entry_cell_is_read_from_the_footers_of_the_files(tmp_path: Path) -> None:
    specs = _specs(tmp_path, [_arena_table("speed"), _arena_table("count")])

    cell = entry_summary_join(specs, cross_join=False, expands_individuals=False)

    assert cell == "arena"


def test_an_input_whose_footer_records_no_rows_is_left_out(tmp_path: Path) -> None:
    empty = _arena_table("count").iloc[0:0]
    specs = _specs(tmp_path, [_arena_table("speed"), empty])

    cell = entry_summary_join(specs, cross_join=False, expands_individuals=False)

    assert cell == ""
    assert load_entry_data(specs) is not None


def test_an_empty_input_leaves_the_others_to_join(tmp_path: Path) -> None:
    empty = _arena_table("count").iloc[0:0]
    specs = _specs(tmp_path, [_arena_table("speed"), empty, _arena_table("share")])

    cell = entry_summary_join(specs, cross_join=False, expands_individuals=False)

    assert cell == "arena"


def test_an_unreadable_input_is_left_out(tmp_path: Path) -> None:
    specs = _specs(tmp_path, [_arena_table("speed"), _arena_table("count")])
    _ = specs[1][0].write_bytes(b"not a parquet")

    cell = entry_summary_join(specs, cross_join=False, expands_individuals=False)

    assert cell == ""
