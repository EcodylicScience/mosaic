"""Tables written by a worker and published by the process that owns the index.

A worker writes a table next to its final name under a hidden name that no reader or
scan takes for a table. The owner moves it into place and writes its row in one step,
so a killed run leaves tables with rows or files that the next run removes, and two
files that name one entry are settled by the order of the raw index.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.entry import EntryKeyCollisionError
from mosaic.core.helpers import make_entry_key
from mosaic.core.pipeline import staged_tables
from mosaic.core.pipeline.entry_pool import EntryFailure
from mosaic.core.pipeline.staged_tables import (
    StagedTrackTable,
    TableCommitter,
    stage_table,
    sweep_staged,
)
from mosaic.core.pipeline.tracks_index import (
    TracksRowDraft,
    read_tracks_index,
    tracks_index_path,
)
from tests.helpers import make_dataset

VARIANT = "convert-trex_npz.0.2-aaaaaaaaaa"


def _staged(
    ds: Dataset, sequence: str, value: int, group: str = ""
) -> StagedTrackTable:
    final = (
        ds.get_root("tracks") / VARIANT / f"{make_entry_key(group, sequence)}.parquet"
    )
    staged = stage_table(pd.DataFrame({"frame": [0, 1, 2], "X": [value] * 3}), final)
    return StagedTrackTable(
        staged=staged,
        row=TracksRowDraft(
            run_id=VARIANT,
            group=group,
            sequence=sequence,
            out_path=final,
            producer="convert-trex_npz",
            std_format="trex_v2",
            n_rows=3,
            carries_tracklets=None,
            source_md5=str(value),
        ),
    )


def _commit(
    committer: TableCommitter, tables: list[StagedTrackTable]
) -> list[EntryFailure]:
    """Add *tables* and publish the whole batch now."""
    return [*committer.add(tables), *committer.flush()]


def test_a_staged_table_is_hidden_and_is_not_a_parquet_file(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds")

    table = _staged(ds, "s1", 1)

    assert table.staged.parent == table.row.out_path.parent
    assert table.staged.name.startswith(".")
    assert table.staged.suffix != ".parquet"
    assert not list(table.staged.parent.glob("*.parquet"))


def test_committing_moves_the_table_into_place_and_writes_its_row(
    tmp_path: Path,
) -> None:
    ds = make_dataset(tmp_path / "ds")
    table = _staged(ds, "s1", 1)

    _commit(TableCommitter(ds, overwrite=False), [table])

    assert not table.staged.exists()
    assert pd.read_parquet(table.row.out_path)["X"].tolist() == [1, 1, 1]
    assert list(read_tracks_index(ds)["sequence"]) == ["s1"]


def test_the_first_table_of_an_entry_wins_and_the_later_one_is_discarded(
    tmp_path: Path,
) -> None:
    ds = make_dataset(tmp_path / "ds")
    first, second = _staged(ds, "s1", 1), _staged(ds, "s1", 2)
    committer = TableCommitter(ds, overwrite=False)

    _ = _commit(committer, [first])
    _ = _commit(committer, [second])

    assert not second.staged.exists()
    assert pd.read_parquet(first.row.out_path)["X"].tolist() == [1, 1, 1]
    index = read_tracks_index(ds)
    assert list(index["source_md5"]) == ["1"]


def test_the_later_table_of_an_entry_wins_when_overwriting(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds")
    first, second = _staged(ds, "s1", 1), _staged(ds, "s1", 2)
    committer = TableCommitter(ds, overwrite=True)

    _ = _commit(committer, [first])
    _ = _commit(committer, [second])

    assert pd.read_parquet(first.row.out_path)["X"].tolist() == [2, 2, 2]
    assert list(read_tracks_index(ds)["source_md5"]) == ["2"]


def test_sweeping_removes_the_staged_files_of_a_variant_and_nothing_else(
    tmp_path: Path,
) -> None:
    ds = make_dataset(tmp_path / "ds")
    left = _staged(ds, "s1", 1)
    _commit(TableCommitter(ds, overwrite=False), [_staged(ds, "s2", 2)])
    other = _staged(ds, "s3", 3)
    elsewhere = left.staged.parent.parent / "other-variant"
    elsewhere.mkdir()
    kept = elsewhere / left.staged.name
    kept.write_bytes(left.staged.read_bytes())

    removed = sweep_staged(left.staged.parent)

    assert removed == 2
    assert not left.staged.exists() and not other.staged.exists()
    assert kept.exists()
    assert (left.staged.parent / "s2.parquet").exists()


def test_sweeping_a_variant_that_does_not_exist_does_nothing(tmp_path: Path) -> None:
    assert sweep_staged(tmp_path / "absent") == 0


def test_a_staged_table_that_is_gone_is_a_failure_and_the_others_are_published(
    tmp_path: Path,
) -> None:
    ds = make_dataset(tmp_path / "ds")
    gone, kept = _staged(ds, "s1", 1), _staged(ds, "s2", 2)
    gone.staged.unlink()

    failures = _commit(TableCommitter(ds, overwrite=False), [gone, kept])

    [failure] = failures
    assert failure.key == "s1"
    assert "concurrent run" in str(failure.error)
    assert not gone.row.out_path.exists()
    assert list(read_tracks_index(ds)["sequence"]) == ["s2"]


def _sequences(ds: Dataset) -> list[str]:
    return list(read_tracks_index(ds)["sequence"])


class _Clock:
    """A clock that moves only when a test says so."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_added_tables_wait_for_a_flush_and_are_then_published_together(
    tmp_path: Path,
) -> None:
    ds = make_dataset(tmp_path / "ds")
    tables = [_staged(ds, f"s{i}", i) for i in range(3)]
    committer = TableCommitter(ds, overwrite=False)

    assert committer.add(tables) == []

    assert not tracks_index_path(ds).exists()
    assert all(table.staged.exists() for table in tables)
    assert committer.flush() == []
    assert _sequences(ds) == ["s0", "s1", "s2"]
    assert all(table.row.out_path.exists() for table in tables)
    assert not any(table.staged.exists() for table in tables)


def test_a_flush_of_a_batch_writes_the_index_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = make_dataset(tmp_path / "ds")
    batches: list[int] = []
    original = staged_tables.write_tracks_rows

    def counting(target: Dataset, drafts: Sequence[TracksRowDraft]) -> None:
        batches.append(len(drafts))
        original(target, drafts)

    monkeypatch.setattr(staged_tables, "write_tracks_rows", counting)
    committer = TableCommitter(ds, overwrite=False)

    _ = committer.add([_staged(ds, f"s{i}", i) for i in range(5)])
    _ = committer.flush()
    _ = committer.flush()

    assert batches == [5]


def test_a_batch_is_flushed_when_it_reaches_its_size(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds")
    committer = TableCommitter(ds, overwrite=False, max_batch=3)

    _ = committer.add([_staged(ds, "s0", 0), _staged(ds, "s1", 1)])
    assert not tracks_index_path(ds).exists()
    _ = committer.add([_staged(ds, "s2", 2)])

    assert _sequences(ds) == ["s0", "s1", "s2"]


def test_a_batch_is_flushed_when_its_oldest_table_has_waited_the_interval(
    tmp_path: Path,
) -> None:
    ds = make_dataset(tmp_path / "ds")
    clock = _Clock()
    committer = TableCommitter(ds, overwrite=False, interval=10.0, clock=clock)

    _ = committer.add([_staged(ds, "s0", 0)])
    clock.now += 9.0
    _ = committer.add([_staged(ds, "s1", 1)])
    assert not tracks_index_path(ds).exists()
    clock.now += 1.0
    _ = committer.add([_staged(ds, "s2", 2)])

    assert _sequences(ds) == ["s0", "s1", "s2"]


def test_the_interval_stretches_with_what_the_last_write_cost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = make_dataset(tmp_path / "ds")
    clock = _Clock()
    original = staged_tables.write_tracks_rows

    def slow(target: Dataset, drafts: Sequence[TracksRowDraft]) -> None:
        original(target, drafts)
        clock.now += 8.0

    monkeypatch.setattr(staged_tables, "write_tracks_rows", slow)
    committer = TableCommitter(ds, overwrite=False, interval=10.0, clock=clock)
    _ = committer.add([_staged(ds, "s0", 0)])
    _ = committer.flush()

    _ = committer.add([_staged(ds, "s1", 1)])
    clock.now += 31.0
    _ = committer.add([_staged(ds, "s2", 2)])
    assert _sequences(ds) == ["s0"]
    clock.now += 1.0
    _ = committer.add([_staged(ds, "s3", 3)])

    assert _sequences(ds) == ["s0", "s1", "s2", "s3"]


def test_the_first_table_of_an_entry_wins_inside_a_batch_and_across_batches(
    tmp_path: Path,
) -> None:
    ds = make_dataset(tmp_path / "ds")
    committer = TableCommitter(ds, overwrite=False)
    inside_first, inside_second = _staged(ds, "s1", 1), _staged(ds, "s1", 2)
    later = _staged(ds, "s1", 3)

    _ = committer.add([inside_first, inside_second])
    _ = committer.flush()
    _ = committer.add([later])
    _ = committer.flush()

    assert pd.read_parquet(inside_first.row.out_path)["X"].tolist() == [1, 1, 1]
    assert not inside_second.staged.exists() and not later.staged.exists()
    assert list(read_tracks_index(ds)["source_md5"]) == ["1"]


def test_the_last_table_of_an_entry_wins_inside_a_batch_and_across_batches_when_overwriting(
    tmp_path: Path,
) -> None:
    ds = make_dataset(tmp_path / "ds")
    committer = TableCommitter(ds, overwrite=True)
    first, second, later = (
        _staged(ds, "s1", 1),
        _staged(ds, "s1", 2),
        _staged(ds, "s1", 3),
    )

    _ = committer.add([first, second])
    _ = committer.flush()
    assert pd.read_parquet(first.row.out_path)["X"].tolist() == [2, 2, 2]
    _ = committer.add([later])
    _ = committer.flush()

    assert pd.read_parquet(first.row.out_path)["X"].tolist() == [3, 3, 3]
    assert list(read_tracks_index(ds)["source_md5"]) == ["3"]


def test_a_batch_publishes_the_tables_that_remain_when_one_is_gone(
    tmp_path: Path,
) -> None:
    ds = make_dataset(tmp_path / "ds")
    first, gone, last = _staged(ds, "s1", 1), _staged(ds, "s2", 2), _staged(ds, "s3", 3)
    gone.staged.unlink()
    committer = TableCommitter(ds, overwrite=False)

    _ = committer.add([first, gone, last])
    failures = committer.flush()

    assert [failure.key for failure in failures] == ["s2"]
    assert _sequences(ds) == ["s1", "s3"]


def test_a_table_that_is_gone_does_not_claim_its_entry(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds")
    gone, replacement = _staged(ds, "s1", 1), _staged(ds, "s1", 2)
    gone.staged.unlink()
    committer = TableCommitter(ds, overwrite=False)

    _ = committer.add([gone, replacement])
    _ = committer.flush()

    assert list(read_tracks_index(ds)["source_md5"]) == ["2"]


def test_a_batch_naming_a_colliding_entry_is_refused_before_any_table_moves(
    tmp_path: Path,
) -> None:
    ds = make_dataset(tmp_path / "ds")
    _ = _commit(TableCommitter(ds, overwrite=False), [_staged(ds, "x", 0, "a__b")])
    fresh, repeated = _staged(ds, "fresh", 1), _staged(ds, "fresh", 3)
    # Joined with its group, this entry names the key of ("a__b", "x").
    colliding = _staged(ds, "b__x", 2, "a")
    committer = TableCommitter(ds, overwrite=False)
    _ = committer.add([fresh, repeated])
    _ = committer.add([colliding])

    with pytest.raises(EntryKeyCollisionError):
        _ = committer.flush()

    held = [fresh.staged, repeated.staged, colliding.staged]
    assert all(path.exists() for path in held)
    assert not fresh.row.out_path.exists()
    assert _sequences(ds) == ["x"]
