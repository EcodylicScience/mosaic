"""``merge_linked_rows`` merges the index rows of a dataset and of its links.

The merge is a pure function over frames: the first frame is the reading
dataset's own, and a row's key keeps the first frame that holds it. A linked
row names its files relative to its own dataset, so the merge anchors them
there, and an own row stays as stored.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mosaic.core.pipeline.linked_rows import merge_linked_rows

KEYS = ("group", "sequence")
"""The key columns of the frames below."""


def _frame(rows: list[tuple[str, str, str]], column: str = "abs_path") -> pd.DataFrame:
    """A frame of ``(group, sequence, path)`` rows, the path in *column*."""
    return pd.DataFrame(rows, columns=["group", "sequence", column])


def test_the_first_frames_row_wins_per_key(tmp_path: Path) -> None:
    own = _frame([("g", "a", "own_a.parquet")])
    linked = _frame([("g", "a", "link_a.parquet"), ("g", "b", "link_b.parquet")])

    merged = merge_linked_rows(
        [("", tmp_path / "own", own), ("src", tmp_path / "src", linked)],
        KEYS,
        ("abs_path",),
    )

    assert merged["sequence"].tolist() == ["a", "b"]
    assert merged["link_id"].tolist() == ["", "src"]
    assert merged["abs_path"].tolist()[0] == "own_a.parquet"


def test_a_later_link_never_replaces_an_earlier_one(tmp_path: Path) -> None:
    own = _frame([])
    first = _frame([("g", "a", "first.parquet")])
    second = _frame([("g", "a", "second.parquet")])

    merged = merge_linked_rows(
        [
            ("", tmp_path / "own", own),
            ("one", tmp_path / "one", first),
            ("two", tmp_path / "two", second),
        ],
        KEYS,
        ("abs_path",),
    )

    assert merged["link_id"].tolist() == ["one"]
    assert merged["abs_path"].tolist() == [str(tmp_path / "one" / "first.parquet")]


def test_own_rows_that_share_a_key_all_survive_a_link(tmp_path: Path) -> None:
    own = _frame([("g", "a", "first.parquet"), ("g", "a", "second.parquet")])
    linked = _frame([("g", "b", "link_b.parquet")])

    merged = merge_linked_rows(
        [("", tmp_path / "own", own), ("src", tmp_path / "src", linked)],
        KEYS,
        ("abs_path",),
    )

    assert merged["abs_path"].tolist() == [
        "first.parquet",
        "second.parquet",
        str(tmp_path / "src" / "link_b.parquet"),
    ]


def test_a_linked_row_is_dropped_when_an_earlier_frame_holds_its_key(
    tmp_path: Path,
) -> None:
    own = _frame([("g", "a", "own_a.parquet"), ("g", "a", "own_a2.parquet")])
    linked = _frame([("g", "a", "link_a.parquet"), ("g", "b", "link_b.parquet")])

    merged = merge_linked_rows(
        [("", tmp_path / "own", own), ("src", tmp_path / "src", linked)],
        KEYS,
        ("abs_path",),
    )

    assert merged["link_id"].tolist() == ["", "", "src"]
    assert merged["sequence"].tolist() == ["a", "a", "b"]


def test_a_linked_relative_path_is_anchored_at_its_base_directory(
    tmp_path: Path,
) -> None:
    linked = _frame(
        [("g", "a", "tracks/a.parquet"), ("g", "b", "/elsewhere/b.parquet")]
    )

    merged = merge_linked_rows(
        [("", tmp_path / "own", _frame([])), ("src", tmp_path / "src", linked)],
        KEYS,
        ("abs_path",),
    )

    assert merged["abs_path"].tolist() == [
        str(tmp_path / "src" / "tracks" / "a.parquet"),
        "/elsewhere/b.parquet",
    ]


def test_an_own_cell_stays_as_stored(tmp_path: Path) -> None:
    own = _frame([("g", "a", "tracks/a.parquet"), ("g", "b", "/abs/b.parquet")])

    merged = merge_linked_rows([("", tmp_path / "own", own)], KEYS, ("abs_path",))

    assert merged["abs_path"].tolist() == ["tracks/a.parquet", "/abs/b.parquet"]


def test_every_linked_frame_empty_returns_the_own_columns_and_the_link_id(
    tmp_path: Path,
) -> None:
    own = pd.DataFrame(
        {"group": ["g"], "sequence": ["a"], "abs_path": ["a.parquet"], "frames": [10]}
    )
    empty = pd.DataFrame(columns=["group", "sequence", "abs_path", "other"])

    merged = merge_linked_rows(
        [("", tmp_path / "own", own), ("src", tmp_path / "src", empty)],
        KEYS,
        ("abs_path",),
    )

    assert merged.columns.tolist() == [*own.columns, "link_id"]
    assert merged["frames"].dtype == own["frames"].dtype
    assert merged["link_id"].tolist() == [""]


def test_an_empty_own_frame_with_no_links_is_returned_with_the_link_id(
    tmp_path: Path,
) -> None:
    own = _frame([])

    merged = merge_linked_rows([("", tmp_path / "own", own)], KEYS, ("abs_path",))

    assert merged.empty
    assert merged.columns.tolist() == [*own.columns, "link_id"]


def test_the_columns_are_the_union_and_a_missing_cell_is_empty(tmp_path: Path) -> None:
    own = pd.DataFrame({"group": ["g"], "sequence": ["a"], "own_only": ["x"]})
    linked = pd.DataFrame({"group": ["g"], "sequence": ["b"], "link_only": ["y"]})

    merged = merge_linked_rows(
        [("", tmp_path / "own", own), ("src", tmp_path / "src", linked)], KEYS, ()
    )

    assert merged.columns.tolist() == [
        "group",
        "sequence",
        "own_only",
        "link_only",
        "link_id",
    ]
    assert merged["own_only"].tolist() == ["x", ""]
    assert merged["link_only"].tolist() == ["", "y"]


def test_the_inputs_are_left_untouched(tmp_path: Path) -> None:
    own = _frame([("g", "a", "a.parquet")])
    linked = _frame([("g", "b", "b.parquet")])
    own_before, linked_before = own.copy(), linked.copy()

    merge_linked_rows(
        [("", tmp_path / "own", own), ("src", tmp_path / "src", linked)],
        KEYS,
        ("abs_path",),
    )

    pd.testing.assert_frame_equal(own, own_before)
    pd.testing.assert_frame_equal(linked, linked_before)


def test_every_listed_path_column_of_a_linked_row_is_anchored(tmp_path: Path) -> None:
    own = pd.DataFrame(
        {
            "group": ["g"],
            "sequence": ["a"],
            "abs_path": ["a.parquet"],
            "source_abs_path": ["raw/a.csv"],
        }
    )
    linked = pd.DataFrame(
        {
            "group": ["g", "g"],
            "sequence": ["b", "c"],
            "abs_path": ["b.parquet", "c.parquet"],
            "source_abs_path": ["raw/b.csv", "/outside/c.csv"],
        }
    )
    base = tmp_path / "src"

    merged = merge_linked_rows(
        [("", tmp_path / "own", own), ("src", base, linked)],
        KEYS,
        ("abs_path", "source_abs_path"),
    )

    assert merged["source_abs_path"].tolist() == [
        "raw/a.csv",
        str(base / "raw" / "b.csv"),
        "/outside/c.csv",
    ]
    assert merged["abs_path"].tolist()[1:] == [
        str(base / "b.parquet"),
        str(base / "c.parquet"),
    ]


def test_a_column_that_is_not_listed_is_left_as_stored(tmp_path: Path) -> None:
    linked = pd.DataFrame(
        {
            "group": ["g"],
            "sequence": ["b"],
            "abs_path": ["b.parquet"],
            "source_abs_path": ["raw/b.csv"],
        }
    )

    merged = merge_linked_rows(
        [("", tmp_path / "own", linked.iloc[0:0]), ("src", tmp_path / "src", linked)],
        KEYS,
        ("abs_path",),
    )

    assert merged["source_abs_path"].tolist() == ["raw/b.csv"]


def test_a_listed_column_that_a_frame_lacks_is_skipped(tmp_path: Path) -> None:
    linked = _frame([("g", "b", "b.parquet")])

    merged = merge_linked_rows(
        [("", tmp_path / "own", _frame([])), ("src", tmp_path / "src", linked)],
        KEYS,
        ("abs_path", "source_abs_path"),
    )

    assert merged["abs_path"].tolist() == [str(tmp_path / "src" / "b.parquet")]
    assert "source_abs_path" not in merged.columns


def test_an_empty_or_array_cell_of_a_linked_row_is_kept_as_stored(
    tmp_path: Path,
) -> None:
    linked = pd.DataFrame(
        {
            "group": ["g", "g"],
            "sequence": ["a", "b"],
            "abs_path": ["", '["clip_0.mp4", "clip_1.mp4"]'],
        }
    )

    merged = merge_linked_rows(
        [("", tmp_path / "own", linked.iloc[0:0]), ("src", tmp_path / "src", linked)],
        KEYS,
        ("abs_path",),
    )

    assert merged["abs_path"].tolist() == ["", '["clip_0.mp4", "clip_1.mp4"]']
