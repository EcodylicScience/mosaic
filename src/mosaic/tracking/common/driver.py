"""The loop every tracker run is, with the tool-specific part left out of it.

What a tracker run does per entry -- which phases, which subprocess, which
converter -- is genuinely its own. What surrounds that is not: open or adopt a
Job Contract, walk the work items reporting progress, skip an entry another
execution holds, translate a killed subprocess into a cancelled attempt, append
the index rows whatever happened, and say what was done. Three copies of that,
differing in a tool name and a row class.

**The tool-specific part stays a closure in the tracker's own module.** It is
passed in rather than declared, so the subprocess call keeps happening at a
module-level name in ``<tool>/dataset_runs.py`` -- which is what the marker
suites patch, and more importantly what makes a tracker readable as one function
rather than as a set of callbacks scattered across a class.

**The index rows are appended in a ``finally``.** A run killed halfway has still
done the entries it finished, and their rows are what a later run reads to know
that. Losing them would mean the work happened and nothing recorded it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

from mosaic.core.pipeline.index_csv import IndexCSV
from mosaic.core.pipeline.job import Cancelled, CancelToken, JobContext, job_context
from mosaic.core.pipeline.subprocess_util import ProcessCancelled
from mosaic.tracking.common.entry import open_entry, release_entry
from mosaic.tracking.common.index import TrackerRunRowBase
from mosaic.tracking.common.mint import MintedRun
from mosaic.tracking.common.scope import TrackerWorkItem

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset
    from mosaic.core.pipeline.progress import ProgressCallback

__all__ = ["EntryJob", "run_tracker"]

RowT = TypeVar("RowT", bound=TrackerRunRowBase)


@dataclass(frozen=True, slots=True)
class EntryJob:
    """Everything one entry's tool-specific work needs, and nothing more.

    ``work_dir`` is already claimed and, when ``overwrite`` was asked for,
    already cleared: by the time this reaches a tracker the directory is its own
    to write, and releasing it is the driver's job rather than the tracker's.
    """

    ds: Dataset
    ctx: JobContext
    item: TrackerWorkItem
    work_dir: Path
    minted: MintedRun
    overwrite: bool


def run_tracker(
    ds: Dataset,
    *,
    kind: str,
    target: str,
    minted: MintedRun,
    work_items: Sequence[TrackerWorkItem],
    index: IndexCSV[RowT],
    run_entry: Callable[[EntryJob], RowT | None],
    overwrite: bool = False,
    execution_id: str | None = None,
    owner: str = "",
    track: bool = True,
    progress_callback: ProgressCallback | None = None,
    cancel_token: CancelToken | None = None,
    ctx: JobContext | None = None,
) -> str:
    """Drive one tracker run over its work items, and return its ``run_id``.

    Args:
        ds: The dataset.
        kind: The tracker's op kind, used for the Job Contract and messages.
        target: What this run is doing, recorded on the attempt.
        minted: What :func:`~mosaic.tracking.common.mint.mint_tracker_run`
            returned. Minted by the caller, because an unresolvable model must
            abort before any of it is written.
        work_items: One per entry, from
            :func:`~mosaic.tracking.common.scope.build_work_items`.
        index: This tracker's run index, already ensured.
        run_entry: The tool-specific work for one entry, returning the row
            that records it -- or ``None`` for an entry worth no row. ``None`` is
            not the channel for failure: an exception propagates, ending the run
            and leaving the rows for the entries that did finish.
        overwrite: Clear each entry's working directory before running it.
        ctx: An already-open Job Contract to run inside. The ``mosaic run
            --kind <tool>`` path hands its context here so a tracker rides the
            standard runner without double-wrapping the contract; the standalone
            path leaves it ``None`` and one is opened from the arguments above.
    """
    index.ensure()
    rows: list[RowT] = []
    skipped: list[str] = []

    managed: AbstractContextManager[JobContext] = (
        nullcontext(ctx)
        if ctx is not None
        else job_context(
            ds,
            kind=kind,
            target=target,
            execution_id=execution_id,
            owner=owner,
            track=track,
            progress_callback=progress_callback,
            cancel_token=cancel_token,
        )
    )
    with managed as job:
        job.set_run_id(minted.run_id)
        job.set_total(len(work_items))

        try:
            for i, item in enumerate(work_items):
                job.check_cancel()
                job.progress.on_entry_start(i, len(work_items), item.key)

                opened = open_entry(
                    ds,
                    job,
                    minted.run_root,
                    item.key,
                    kind=kind,
                    overwrite=overwrite,
                )
                if opened is None:
                    skipped.append(item.key)
                    job.progress.on_entry_end(i + 1, len(work_items), item.key)
                    continue
                work_dir, _held = opened

                try:
                    row = run_entry(
                        EntryJob(
                            ds=ds,
                            ctx=job,
                            item=item,
                            work_dir=work_dir,
                            minted=minted,
                            overwrite=overwrite,
                        )
                    )
                finally:
                    release_entry(work_dir, job.execution_id)

                if row is not None:
                    rows.append(row)
                job.progress.on_entry_end(i + 1, len(work_items), item.key)
                job.heartbeat(i + 1)
        except ProcessCancelled as exc:
            # A killed subprocess is a cancelled attempt, not a failed one.
            raise Cancelled() from exc
        finally:
            if rows:
                index.append(rows)
                index.mark_finished(minted.run_id)

    held = f", {len(skipped)} held by another execution" if skipped else ""
    print(
        f"[{kind}] completed run_id={minted.run_id} "
        f"({len(rows)}/{len(work_items)} sequences{held}) -> {minted.run_root}"
    )
    return minted.run_id
