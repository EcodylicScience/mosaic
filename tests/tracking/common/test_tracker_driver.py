"""The progress that ``run_tracker`` records on an attempt's run-log."""

from __future__ import annotations

from pathlib import Path

from mosaic.core.entry import make_entry_key
from mosaic.core.pipeline.markers import new_inflight, write_inflight
from mosaic.core.pipeline.run_log import read_runs, run_log_dir
from mosaic.tracking.common.driver import EntryJob, run_tracker
from mosaic.tracking.common.index import tracker_index, tracker_index_path
from mosaic.tracking.common.mint import MintedRun
from mosaic.tracking.common.scope import TrackerWorkItem, WorkItems
from mosaic.tracking.trex.dataset_runs import TRexIndexRow
from tests.helpers import make_dataset


def _never_run(job: EntryJob) -> TRexIndexRow | None:
    raise AssertionError(f"{job.item.key} is claimed by another execution")


def test_a_run_that_skips_every_entry_ends_at_its_total(tmp_path: Path) -> None:
    """Another execution claims each entry, and progress still ends at 3 of 3."""
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
                video_paths=(tmp_path / f"{sequence}.mp4",),
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
