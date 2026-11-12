"""Tables written by a worker and published by the process that owns the index.

A worker writes a table next to its final name under a hidden name that no reader or
scan takes for a table. The owner moves it into place and writes its row in one step,
so a killed run leaves tables with rows or files that the next run removes, and two
files that name one entry are settled by the order of the raw index.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.staged_tables import (
    StagedTrackTable,
    TableCommitter,
    stage_table,
    sweep_staged,
)
from mosaic.core.pipeline.tracks_index import TracksRowDraft, read_tracks_index
from tests.helpers import make_dataset

VARIANT = "convert-trex_npz.0.2-aaaaaaaaaa"


def _staged(ds: Dataset, sequence: str, value: int) -> StagedTrackTable:
    final = ds.get_root("tracks") / VARIANT / f"{sequence}.parquet"
    staged = stage_table(pd.DataFrame({"frame": [0, 1, 2], "X": [value] * 3}), final)
    return StagedTrackTable(
        staged=staged,
        row=TracksRowDraft(
            run_id=VARIANT,
            group="",
            sequence=sequence,
            out_path=final,
            producer="convert-trex_npz",
            std_format="trex_v2",
            n_rows=3,
            carries_tracklets=None,
            source_md5=str(value),
        ),
    )


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

    TableCommitter(ds, overwrite=False).commit([table])

    assert not table.staged.exists()
    assert pd.read_parquet(table.row.out_path)["X"].tolist() == [1, 1, 1]
    assert list(read_tracks_index(ds)["sequence"]) == ["s1"]


def test_the_first_table_of_an_entry_wins_and_the_later_one_is_discarded(
    tmp_path: Path,
) -> None:
    ds = make_dataset(tmp_path / "ds")
    first, second = _staged(ds, "s1", 1), _staged(ds, "s1", 2)
    committer = TableCommitter(ds, overwrite=False)

    committer.commit([first])
    committer.commit([second])

    assert not second.staged.exists()
    assert pd.read_parquet(first.row.out_path)["X"].tolist() == [1, 1, 1]
    index = read_tracks_index(ds)
    assert list(index["source_md5"]) == ["1"]


def test_the_later_table_of_an_entry_wins_when_overwriting(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds")
    first, second = _staged(ds, "s1", 1), _staged(ds, "s1", 2)
    committer = TableCommitter(ds, overwrite=True)

    committer.commit([first])
    committer.commit([second])

    assert pd.read_parquet(first.row.out_path)["X"].tolist() == [2, 2, 2]
    assert list(read_tracks_index(ds)["source_md5"]) == ["2"]


def test_sweeping_removes_the_staged_files_of_a_variant_and_nothing_else(
    tmp_path: Path,
) -> None:
    ds = make_dataset(tmp_path / "ds")
    left = _staged(ds, "s1", 1)
    TableCommitter(ds, overwrite=False).commit([_staged(ds, "s2", 2)])
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

    failures = TableCommitter(ds, overwrite=False).commit([gone, kept])

    [failure] = failures
    assert failure.key == "s1"
    assert "concurrent run" in str(failure.error)
    assert not gone.row.out_path.exists()
    assert list(read_tracks_index(ds)["sequence"]) == ["s2"]
