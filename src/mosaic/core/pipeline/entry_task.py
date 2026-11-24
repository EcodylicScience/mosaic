"""The work of one entry of a feature run, in every mode and at every worker count.

One path computes an entry wherever it runs, in the run's own thread, a pool thread
or a worker process: check the input's columns, refuse an arena-blind input that
spans arenas, apply, trim under overlap, and write the output. The task writes the
entry's output and nothing else. The run's process alone writes the index and the
run-log, from the report the task returns.

An entry error (the column check, the arena refusal, ``apply``) raises, and the run
records it as the entry's loss. An output failure (the trim, the write) is returned
as an :class:`~mosaic.core.pipeline.entry_pool.EntryFailure`, and the run raises it
as its own failure.
"""

from __future__ import annotations

import pickle
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Final

import pandas as pd

from mosaic.core.helpers import make_entry_key
from mosaic.core.timebase import IrregularNotice, collect_irregular_notices

from ._utils import FeatureMeta
from .admission import WorkerMeasure
from .arena_guard import refuse_arena_blind
from .column_contract import check_input_columns
from .entry_pool import EntryFailure
from .memory_watch import (
    NonreclaimablePeak,
    max_rss_bytes,
    peak_nonreclaimable_while,
    process_nonreclaimable_bytes,
    process_rss_bytes,
)
from .pacing import PacedCollector
from .parquet_reader import release_arrow_memory
from .writers import FeatureOutput, trim_feature_output, write_output

if TYPE_CHECKING:
    from .manifest import CoreSelector
    from .types import Feature

__all__ = [
    "EntryReport",
    "EntryTask",
    "NotedOutput",
    "apply_noting_irregular_time",
    "run_entry",
    "run_entry_in_process",
]


type NotedOutput = tuple[FeatureOutput, tuple[IrregularNotice, ...]]
"""One entry's output with the uneven-time notices its ``apply`` raised."""


def apply_noting_irregular_time(
    apply: Callable[[pd.DataFrame], FeatureOutput], df: pd.DataFrame
) -> NotedOutput:
    """Run *apply* on *df* and return its output with the notices it raised.

    A method that reads its rows as evenly spaced warns when a table steps
    unevenly (:func:`~mosaic.core.timebase.warn_if_irregular`). The notices are
    collected where the entry runs, a worker process included, and returned beside
    the output so the calling process records them in the run-log.
    """
    with collect_irregular_notices() as notices:
        output = apply(df)
    return output, tuple(notices)


@dataclass(frozen=True, eq=False)
class EntryTask:
    """One entry to compute, as the run hands it to a worker.

    A task holds a frame, so it compares by identity.

    Attributes:
        meta: The entry and its output path.
        table: The entry's input, loaded by the run.
        selector: The rows of the output that belong to the entry, under overlap,
            ``None`` otherwise.
        arena_count: The arenas of the entry's recording, as
            :func:`~mosaic.core.pipeline.arena_guard.entry_arena_count` counts
            them, for an arena-blind feature. ``None`` for any other feature.
    """

    meta: FeatureMeta
    table: pd.DataFrame
    selector: CoreSelector | None = None
    arena_count: int | None = None

    @property
    def key(self) -> str:
        """The entry's key."""
        return make_entry_key(self.meta.group, self.meta.sequence)


@dataclass(frozen=True)
class EntryReport:
    """What one entry's task wrote, returned once its output is in place.

    Attributes:
        rows: The rows of the output written.
        output_bytes: The memory of the output frame, its index included, 0 for
            no output.
        notices: The uneven-time notices that ``apply`` raised.
        seconds_after_apply: The seconds from the end of ``apply`` to the
            task's return, the write included.
        worker: The memory of the worker process that ran the task, ``None``
            where the task ran in the run's own process or the measure failed.
    """

    rows: int
    output_bytes: int
    notices: tuple[IrregularNotice, ...]
    seconds_after_apply: float
    worker: WorkerMeasure | None = None


def run_entry(
    feature: Feature,
    task: EntryTask,
    after_apply: Callable[[], None] | None,
) -> EntryReport | EntryFailure:
    """Compute one entry and write its output.

    The output is written without creating its directory, so a run root that a
    removal deleted is not made again and the write fails instead.

    Nothing after the rename may raise. An exception there would record an
    ``entry_error`` for an output on disk without a row, which the run's
    completeness check still counts. The report is therefore built from values
    computed before the write, and a failed release of pyarrow's memory is
    ignored.

    Args:
        feature: The feature to apply.
        task: The entry.
        after_apply: Called once after ``apply`` returns and before the output is
            written.

    Returns:
        The report of the output written, or the failure of the trim or the
        write, with its error as raised.

    Raises:
        MissingInputColumns: The input lacks a column the feature requires.
        ArenaBlindFeature: The feature is arena-blind and the input spans arenas.
        Exception: Whatever ``apply`` raises, or the error of a notice that does
            not pickle.
    """
    table = task.table
    check_input_columns(
        feature, table.columns, numeric=table.select_dtypes(include="number").columns
    )
    if task.arena_count is not None:
        refuse_arena_blind(feature.name, table, task.arena_count)
    output, notices = apply_noting_irregular_time(feature.apply, table)
    applied = time.monotonic()
    if after_apply is not None:
        after_apply()
    # A notice that does not pickle fails here, as the entry's error, and not
    # after the output is written.
    _ = pickle.dumps(notices)
    try:
        if task.selector is not None:
            output = trim_feature_output(output, task.selector)
        rows = 0 if output is None else len(output)
        output_bytes = (
            0 if output is None else int(output.memory_usage(index=True).sum())
        )
        _ = write_output(task.meta, output, create_parent=False)
    except Exception as error:
        return EntryFailure.capture(task.key, error, portable=False)
    del output
    with suppress(Exception):
        release_arrow_memory()
    return EntryReport(
        rows=rows,
        output_bytes=output_bytes,
        notices=notices,
        seconds_after_apply=time.monotonic() - applied,
    )


_COLLECTOR: Final = PacedCollector()
"""The collector of this worker process, fed the bytes of each output it wrote."""


def run_entry_in_process(
    feature: Feature, task: EntryTask
) -> EntryReport | EntryFailure:
    """Run :func:`run_entry` in a worker process, and measure the worker's memory.

    A failure is returned in a form that crosses the process boundary
    (:meth:`~mosaic.core.pipeline.entry_pool.EntryFailure.portable`).

    The measure, the garbage collection and the report after the write each catch
    an exception, for the reason :func:`run_entry` gives: a failed measure leaves
    the report's ``worker`` ``None``, and a failed collection is ignored.

    Returns:
        The report of :func:`run_entry`, with the worker's own memory: its
        resident memory after ``load_state`` and before ``apply`` less the table
        it was handed, its high-water mark at that moment, and its high-water mark
        after its output was written. The same for its anonymous and shared
        memory, where ``/proc`` gives it: the figure before ``apply`` less the
        table, and the largest figure read from before ``apply`` to after its
        output was written. ``seconds_after_apply`` runs to this function's
        return.
    """
    # The table arrives as an argument, so it is resident before ``apply`` and
    # stays referenced by the pool's call item. It is the entry's memory, not the
    # worker's, and a worker handed no table measures without it
    # (``process_worker_base``).
    handed = int(task.table.memory_usage(deep=True).sum())
    base = max(0, process_rss_bytes() - handed)
    resident = process_nonreclaimable_bytes()
    base_nonreclaimable = None if resident is None else max(0, resident - handed)
    peak_before = max_rss_bytes()
    outcome: EntryReport | EntryFailure | None = None
    sampled: NonreclaimablePeak | None = None
    returned = 0.0
    try:
        with peak_nonreclaimable_while() as sampled:
            outcome = run_entry(feature, task, None)
            returned = time.monotonic()
    except Exception:
        # Without an outcome the entry raised, and its error is the entry's.
        # With one, the sampler's exit raised after the write, and the peak is
        # unknown.
        if outcome is None:
            raise
        sampled = None
    if isinstance(outcome, EntryFailure):
        return outcome.portable()
    worker: WorkerMeasure | None = None
    if sampled is not None:
        with suppress(Exception):
            worker = WorkerMeasure(
                base_rss_bytes=base,
                peak_before_rss_bytes=peak_before,
                peak_rss_bytes=max_rss_bytes(),
                base_nonreclaimable_bytes=base_nonreclaimable,
                peak_nonreclaimable_bytes=sampled.peak,
            )
    with suppress(Exception):
        _ = _COLLECTOR.finished(outcome.output_bytes)
    return replace(
        outcome,
        worker=worker,
        seconds_after_apply=outcome.seconds_after_apply + time.monotonic() - returned,
    )
