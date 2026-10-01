"""A backfill pass reads what its rows share once, and locks the index only to write.

``mosaic measure-tracks`` rewrites every row of the tracks index to what its rule
gives. A pass reads the media index and the sequence projection, where its rule
needs them, once for all its rows. It asks the rules outside the index lock,
because over a large index they take long enough that a tracker publishing
meanwhile would time out waiting for the lock.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import pytest

import mosaic.tracking.trex.dataset_runs as trex_runs
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.index_lock import index_lock
from mosaic.core.pipeline.tracks_index import (
    backfill_frames_read,
    backfill_known_tail_loss,
    backfill_media_frames,
    read_frames_read,
    read_tracks_index,
    tracks_index_path,
    write_tracks_row,
)
from mosaic.tracking.trex.params import TrexParams
from mosaic.tracking.trex.pv import PvHeader
from tests.helpers import (
    MediaClip,
    count_index_reads,
    install_fake_trex,
    make_dataset,
    set_tracks_cell,
    write_media_index,
)

_SEQUENCES = ("a", "b", "c")


def _tracked(
    tmp_path: Path, sequences: Sequence[str], monkeypatch: pytest.MonkeyPatch
) -> Dataset:
    """Run the fake TREx over one clip for each of *sequences*."""
    ds = make_dataset(tmp_path / "ds")
    write_media_index(
        ds,
        [
            MediaClip(sequence=sequence, filename=f"{sequence}.mp4")
            for sequence in sequences
        ],
    )
    _ = ds.rebuild_sequence_index("media_raw")
    _ = install_fake_trex(monkeypatch)
    _ = trex_runs.run_trex(ds, TrexParams())
    return ds


def test_a_pass_reads_the_media_records_once_for_all_its_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = _tracked(tmp_path, _SEQUENCES, monkeypatch)
    set_tracks_cell(ds, "media_frames", "")
    reads = count_index_reads(monkeypatch)

    media = backfill_media_frames(ds)
    tail = backfill_known_tail_loss(ds)

    assert sorted(media.written["sequence"]) == list(_SEQUENCES)
    assert tail.unregistered.empty
    assert reads.media_scopes == 1, "the media pass resolves every entry at once"
    assert reads.compositions == 2, "each pass reads the compositions once"


def _another_thread_takes_the_lock(path: Path) -> bool:
    """Whether another thread takes the lock on *path* within a second."""
    taken: list[bool] = []

    def take() -> None:
        with index_lock(path, timeout=1.0):
            taken.append(True)

    worker = threading.Thread(target=take, daemon=True)
    worker.start()
    worker.join(timeout=1.0)
    return taken == [True]


def test_a_writer_does_not_wait_while_a_pass_judges_its_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The rules read media and the runs' files, the slow part of a pass."""
    ds = _tracked(tmp_path, _SEQUENCES, monkeypatch)
    read_header = trex_runs.read_pv_header
    taken: list[bool] = []

    def judged(path: Path) -> PvHeader | None:
        taken.append(_another_thread_takes_the_lock(tracks_index_path(ds)))
        return read_header(path)

    monkeypatch.setattr(trex_runs, "read_pv_header", judged)

    _ = backfill_frames_read(ds)

    assert taken == [True] * len(_SEQUENCES)


def test_a_row_published_while_a_pass_judges_keeps_what_it_was_published_with(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A row republished and a row added after the pass read the index.

    Each holds what its producer wrote after the rule was asked, so the pass
    leaves it, and writes only the row it judged as it still is.
    """
    ds = _tracked(tmp_path, ("a", "b"), monkeypatch)
    set_tracks_cell(ds, "frames_read", "")
    (republished,) = [
        row for _, row in read_tracks_index(ds).iterrows() if row["sequence"] == "a"
    ]
    table = ds.resolve_path(str(republished["abs_path"]))
    added = table.with_name("c.parquet")
    pd.DataFrame({"frame": [0, 1], "id": [0, 0]}).to_parquet(added)
    read_header = trex_runs.read_pv_header
    published: list[str] = []

    def publish(sequence: str, out_path: Path, frames_read: int) -> None:
        write_tracks_row(
            ds,
            run_id=str(republished["run_id"]),
            group="",
            sequence=sequence,
            out_path=out_path,
            producer="trex",
            std_format=str(republished["std_format"]),
            n_rows=2,
            frames_read=frames_read,
        )
        published.append(sequence)

    def judged(path: Path) -> PvHeader | None:
        if not published:
            publish("a", table, 77)
            publish("c", added, 88)
        return read_header(path)

    monkeypatch.setattr(trex_runs, "read_pv_header", judged)

    done = backfill_frames_read(ds)

    assert done.written["sequence"].tolist() == ["b"]
    held = {
        str(row["sequence"]): read_frames_read(row)
        for _, row in read_tracks_index(ds).iterrows()
    }
    assert held == {"a": 77, "b": 4, "c": 88}
