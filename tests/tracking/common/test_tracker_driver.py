"""The progress and losses that ``run_tracker`` records on an attempt's run-log."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.entry import make_entry_key
from mosaic.core.pipeline.job import Cancelled, CancelToken
from mosaic.core.pipeline.markers import new_inflight, write_inflight
from mosaic.core.pipeline.run_log import read_runs, run_log_dir, run_log_path
from mosaic.runlog import read_entry_error_keys
from mosaic.tracking.common.driver import EntryJob, run_tracker
from mosaic.tracking.common.index import tracker_index, tracker_index_path
from mosaic.tracking.common.mint import MintedRun
from mosaic.tracking.common.scope import TrackerWorkItem, WorkItems
from mosaic.tracking.trex.dataset_runs import TRexIndexRow
from tests.helpers import make_dataset


def _never_run(job: EntryJob) -> TRexIndexRow | None:
    raise AssertionError(f"{job.item.key} is claimed by another execution")


def test_a_run_that_skips_every_entry_ends_at_its_total(tmp_path: Path) -> None:
    """Another execution claims each entry, and progress still ends at 3 of 3.

    A skipped entry also counts on the context's heartbeat, whose first call is
    written.
    """
    ds = make_dataset(tmp_path)
    run_id = "trex.0.1-0123456789"
    minted = MintedRun(
        run_id=run_id,
        run_root=ds.get_root("trex") / run_id,
        params_hash="0123456789",
        tracks_variant=run_id,
    )
    items: list[TrackerWorkItem] = []
    for sequence in ("s1", "s2", "s3"):
        key = make_entry_key("g", sequence)
        items.append(
            TrackerWorkItem(
                group="g",
                sequence=sequence,
                key=key,
                video_paths=(ds.base_dir / f"{sequence}.mp4",),
                fps=30.0,
            )
        )
        claimed = minted.run_root / key
        claimed.mkdir(parents=True)
        write_inflight(
            claimed,
            new_inflight(
                execution_id="EXEC-THEIRS",
                host="h",
                pid=1,
                phase="track",
                idle_seconds=600.0,
            ),
        )

    _ = run_tracker(
        ds,
        kind="trex",
        target="trex",
        minted=minted,
        work_items=WorkItems(items=tuple(items)),
        index=tracker_index(tracker_index_path(ds, "trex"), TRexIndexRow),
        run_entry=_never_run,
    )

    (snap,) = read_runs(run_log_dir(ds.base_dir))
    assert snap["status"] == "finished"
    assert (snap["progress_done"], snap["progress_total"]) == (3, 3)
    path = run_log_path(ds.base_dir, snap["execution_id"])
    records = [json.loads(line) for line in path.read_text().splitlines()]
    heartbeats = [(r["done"], r["total"]) for r in records if r["ev"] == "heartbeat"]
    assert heartbeats == [(1, 3)]


def _run_two_entries(
    ds: Dataset,
    run_entry: Callable[[EntryJob], TRexIndexRow | None],
    *,
    cancel_token: CancelToken | None = None,
) -> None:
    """Run a TRex attempt over entries ``g/s1`` and ``g/s2``."""
    run_id = "trex.0.1-0123456789"
    minted = MintedRun(
        run_id=run_id,
        run_root=ds.get_root("trex") / run_id,
        params_hash="0123456789",
        tracks_variant=run_id,
    )
    items = tuple(
        TrackerWorkItem(
            group="g",
            sequence=sequence,
            key=make_entry_key("g", sequence),
            video_paths=(ds.base_dir / f"{sequence}.mp4",),
            fps=30.0,
        )
        for sequence in ("s1", "s2")
    )
    _ = run_tracker(
        ds,
        kind="trex",
        target="trex",
        minted=minted,
        work_items=WorkItems(items=items),
        index=tracker_index(tracker_index_path(ds, "trex"), TRexIndexRow),
        run_entry=run_entry,
        cancel_token=cancel_token,
    )


_SECOND_KEY = make_entry_key("g", "s2")


def test_an_entry_whose_export_raises_is_lost_and_not_written(tmp_path: Path) -> None:
    """The first entry finishes, and the second raises after its inference.

    The run fails with the export's error, the run-log records the second entry
    as lost, and ``entries_written`` counts only the first.
    """

    def export_fails_on_the_second(job: EntryJob) -> TRexIndexRow | None:
        if job.item.key == _SECOND_KEY:
            raise RuntimeError("analysis export failed after inference")
        return None

    ds = make_dataset(tmp_path)
    with pytest.raises(RuntimeError, match="analysis export failed"):
        _run_two_entries(ds, export_fails_on_the_second)

    log_dir = run_log_dir(ds.base_dir)
    (snap,) = read_runs(log_dir)
    assert snap["status"] == "failed"
    assert snap["entries_written"] == 1
    assert snap["entries_failed"] == 1
    assert read_entry_error_keys(log_dir, snap["execution_id"]) == [_SECOND_KEY]


def test_an_entry_failing_under_a_cancel_is_neither_written_nor_lost(
    tmp_path: Path,
) -> None:
    """The second entry's tool fails once a cancel is requested.

    The attempt ends cancelled, and ``entries_written`` counts only the first.
    """
    token = CancelToken()

    def cancelled_on_the_second(job: EntryJob) -> TRexIndexRow | None:
        if job.item.key == _SECOND_KEY:
            token.cancel()
            raise RuntimeError("tool killed by the cancel")
        return None

    ds = make_dataset(tmp_path)
    with pytest.raises(Cancelled):
        _run_two_entries(ds, cancelled_on_the_second, cancel_token=token)

    (snap,) = read_runs(run_log_dir(ds.base_dir))
    assert snap["status"] == "cancelled"
    assert snap["entries_written"] == 1
    assert snap["entries_failed"] == 0


def test_an_interrupted_entry_is_neither_written_nor_lost(tmp_path: Path) -> None:
    """Ctrl+C during the second entry leaves ``entries_written`` at the first."""

    def interrupted_on_the_second(job: EntryJob) -> TRexIndexRow | None:
        if job.item.key == _SECOND_KEY:
            raise KeyboardInterrupt
        return None

    ds = make_dataset(tmp_path)
    with pytest.raises(KeyboardInterrupt):
        _run_two_entries(ds, interrupted_on_the_second)

    (snap,) = read_runs(run_log_dir(ds.base_dir))
    assert snap["entries_written"] == 1
    assert snap["entries_failed"] == 0
