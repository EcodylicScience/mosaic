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
from mosaic.core.pipeline.run import AllEntriesFailed
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
    # Entries this run actually opened. Not `work_items`: one held by another
    # execution was never this run's to lose, and counting it would let a
    # contended run declare itself a total failure.
    attempted: set[str] = set()

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
                attempted.add(item.key)

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
            # Counted against this run's own entries rather than `failed_keys`
            # wholesale: the op path may hand in a JobContext that already carries
            # failures from an earlier stage of the same attempt.
            lost = {key for key in job.failed_keys if key in attempted}
            # Attempted-minus-lost, deliberately **not** `len(rows)`. An entry
            # whose bridge failed still gets an index row -- the tool output is
            # real and adoptable, which is what lets a re-run redo only the
            # conversion -- so `rows` counts tracking done, while
            # `entries_written` means tracks tables published. The two differ on
            # exactly the partial run where the number matters.
            #
            # In the `finally` and inside the `with`, for two further reasons. A
            # run killed halfway has still published what it finished, and that
            # is what an operator resubmits on. And once the context exits,
            # `job_context` has emitted its terminal event and closed the log --
            # `_emit` then returns silently, so a count written after the block
            # would be dropped with no error, and one written after `finished`
            # would never reach the ledger anyway.
            job.entries_written(len(attempted) - len(lost))

        # Losing every entry means the run produced nothing, and reporting that
        # as finished is the defect AllEntriesFailed exists to close -- the CLI
        # exits 0 and mosaic-queue maps exit 0 to a `finished` ledger row. The
        # shape a tracker reaches it by is a bridge that failed on every entry:
        # one systematic converter fault, nine sessions of GPU time, and no
        # table published. Raised after the `finally` above, so the rows for the
        # tracking that *did* happen are already durable and a re-run adopts the
        # finished directories rather than recomputing them.
        if attempted and lost == attempted:
            raise AllEntriesFailed(
                f"[{kind}] every one of {len(attempted)} attempted entries failed "
                f"to publish, so run_id={minted.run_id} produced no tracks: "
                f"{', '.join(sorted(lost))}. The tool output is kept under "
                f"{minted.run_root}, so fixing the cause and re-running will "
                f"adopt it and retry only the conversion. The per-entry errors "
                f"are in this attempt's run-log."
            )

    held = f", {len(skipped)} held by another execution" if skipped else ""
    print(
        f"[{kind}] completed run_id={minted.run_id} "
        f"({len(rows)}/{len(work_items)} sequences{held}) -> {minted.run_root}"
    )
    return minted.run_id
