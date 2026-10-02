"""The tracks-index query methods of `Dataset`.

``list_groups``, ``list_sequences``, ``query_sequences`` and
``get_sequence_metadata`` read the tracks index. None of them had any test at
all, so a regression in three of them was invisible.
"""

from __future__ import annotations

from pathlib import Path

from tests.helpers import add_track_sequences, make_dataset


def test_query_methods_on_an_unconverted_dataset_are_empty(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path)
    add_track_sequences(ds, ("g", "s1"), ("g", "s2"), n_rows=12)
    (ds.get_root("tracks") / "index.csv").unlink()

    assert ds.list_groups() == []
    assert ds.list_sequences() == []
    assert ds.query_sequences(sequence_contains="s") == []
    assert len(ds.get_sequence_metadata()) == 0


def test_query_methods_on_a_populated_dataset(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path)
    add_track_sequences(ds, ("g", "s1"), ("g", "s2"), n_rows=12)

    assert ds.list_groups() == ["g"]
    assert ds.list_sequences() == ["s1", "s2"]
    assert ds.query_sequences(sequence_contains="s1") == [("g", "s1")]
    meta = ds.get_sequence_metadata()
    assert len(meta) == 2
    # The safe-name columns this method documents are re-derived, not stored.
    assert list(meta["sequence_safe"]) == ["s1", "s2"]
