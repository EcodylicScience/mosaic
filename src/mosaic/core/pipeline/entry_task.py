"""The work of one entry of a feature run, in every mode and at every worker count.

One path computes an entry wherever it runs, in the run's own thread, a pool thread
or a worker process: load the input unless the run handed it over, check the input's
columns, refuse an arena-blind input that spans arenas, apply, trim under overlap,
and write the output. The task reads the entry's inputs and writes the entry's
output, and nothing else. The run's process alone writes the index and the run-log,
from the report the task returns.

An entry error (missing pair rows, the column check, the arena refusal, ``apply``)
raises, and the run records it as the entry's loss. An input failure (any other
error of the load) and an output failure (the trim, the write) are returned as an
:class:`~mosaic.core.pipeline.entry_pool.EntryFailure`, and the run raises it as its
own failure.
"""

from __future__ import annotations

import pickle
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Final

import pandas as pd

from mosaic.core.filter_bounds import TimeWindow
from mosaic.core.helpers import make_entry_key
from mosaic.core.timebase import IrregularNotice, collect_irregular_notices

from ._loaders import LoadSpec
from ._utils import FeatureMeta
from .admission import WorkerMeasure
from .arena_guard import refuse_arena_blind
from .column_contract import check_input_columns
from .entry_pool import EntryFailure
from .loading import (
    ColumnProjection,
    PairRowsMissing,
    entry_bounds_filters,
    load_entry_data,
)
from .memory_watch import (
    NonreclaimablePeak,
    max_rss_bytes,
    peak_nonreclaimable_while,
    process_nonreclaimable_bytes,
    process_rss_bytes,
)
from .pacing import PacedCollector
from .parquet_reader import (
    FooterRecord,
    ParquetFooterMemo,
    memoized_footers,
    release_arrow_memory,
)
from .writers import FeatureOutput, trim_feature_output, write_output

if TYPE_CHECKING:
    from .manifest import CoreSelector
    from .types import Feature

__all__ = [
    "EntryLoad",
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


@dataclass(frozen=True)
class EntryLoad:
    """The load of one entry's input, as the run hands it to the worker that loads it.

    Attributes:
        entry_key: The entry's manifest key, which the time window's error names.
        file_specs: The entry's input files and how each is read, in declared
            order.
        projection: The columns read from each input, ``None`` for every column.
        frame_start: The first frame kept, ``None`` for no lower bound.
        frame_end: The frame after the last kept, ``None`` for no upper bound.
        time_window: The window of ``time`` kept.
        footers: The footers of the inputs that the run read to size the entry,
            each keyed by the path spelled as in *file_specs*. The load parses
            only the footers of inputs that are missing here or moved since.
    """

    entry_key: str
    file_specs: tuple[tuple[Path, LoadSpec], ...]
    projection: ColumnProjection | None
    frame_start: int | None
    frame_end: int | None
    time_window: TimeWindow
    footers: tuple[FooterRecord, ...]


@dataclass(frozen=True, eq=False)
class EntryTask:
    """One entry to compute, as the run hands it to a worker.

    A task may hold a frame, so it compares by identity.

    Attributes:
        meta: The entry and its output path.
        source: The entry's input, either loaded by the run or the load that the
            task runs.
        selector: The rows of the output that belong to the entry, under overlap,
            ``None`` otherwise. Set only with a loaded table.
        arena_count: The arenas of the entry's recording, as
            :func:`~mosaic.core.pipeline.arena_guard.entry_arena_count` counts
            them, for an arena-blind feature. ``None`` for any other feature.
    """

    meta: FeatureMeta
    source: pd.DataFrame | EntryLoad
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
        written: Whether the task wrote an output. A load of no rows writes none,
            and its other figures are 0.
        rows: The rows of the output written.
        output_bytes: The memory of the output frame, its index included, 0 for
            no output.
        input_bytes: The memory of the table the task loaded, its index included,
            0 for a table the run handed over, which the run counted.
        notices: The uneven-time notices that ``apply`` raised.
        seconds_after_apply: The seconds from the end of ``apply`` to the
            task's return, the write included.
        worker: The memory of the worker process that ran the task, ``None``
            where the task ran in the run's own process or the measure failed.
        build_seconds: The seconds the worker spent building the feature before
            this task, 0 where it found the feature built or the task ran in the
            run's own process.
    """

    written: bool
    rows: int
    output_bytes: int
    input_bytes: int
    notices: tuple[IrregularNotice, ...]
    seconds_after_apply: float
    worker: WorkerMeasure | None = None
    build_seconds: float = 0.0


def _loaded(load: EntryLoad) -> pd.DataFrame | None:
    """The entry's merged and filtered input, ``None`` for one of no rows.

    The footers that *load* carries are known to the load's reads, so a footer
    whose file did not move is not parsed again.
    """
    filters = entry_bounds_filters(
        load.entry_key, load.frame_start, load.frame_end, load.time_window
    )
    with memoized_footers(ParquetFooterMemo.from_records(load.footers)):
        return load_entry_data(
            list(load.file_specs), filters=filters, projection=load.projection
        )


def run_entry(
    feature: Feature,
    task: EntryTask,
    after_apply: Callable[[], None] | None,
) -> EntryReport | EntryFailure:
    """Compute one entry and write its output.

    A task that holds a table uses it as it is. A task that holds a load reads,
    merges and filters the entry's input first. A load of no rows writes nothing,
    skips *after_apply*, and returns a report that is not ``written``.

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
        The report of the output written or of a load of no rows, or the
        failure of the load, the trim or the write, with its error as raised.
        A load's failure is any error of the load other than missing pair rows:
        an unreadable or corrupt input, or a time window over rows without
        ``time``.

    Raises:
        PairRowsMissing: The load found no pair rows to widen onto.
        MissingInputColumns: The input lacks a column the feature requires.
        ArenaBlindFeature: The feature is arena-blind and the input spans arenas.
        Exception: Whatever ``apply`` raises, or the error of a notice that does
            not pickle.
    """
    table: pd.DataFrame | None
    if isinstance(task.source, EntryLoad):
        try:
            table = _loaded(task.source)
        except PairRowsMissing:
            raise
        except Exception as error:
            return EntryFailure.capture(task.key, error, portable=False)
        if table is None:
            return EntryReport(
                written=False,
                rows=0,
                output_bytes=0,
                input_bytes=0,
                notices=(),
                seconds_after_apply=0.0,
            )
        input_bytes = int(table.memory_usage(index=True).sum())
    else:
        table = task.source
        input_bytes = 0
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
    del output, table
    with suppress(Exception):
        release_arrow_memory()
    return EntryReport(
        written=True,
        rows=rows,
        output_bytes=output_bytes,
        input_bytes=input_bytes,
        notices=notices,
        seconds_after_apply=time.monotonic() - applied,
    )


_COLLECTOR: Final = PacedCollector()
"""The collector of this worker process, fed the bytes each task loaded and wrote."""


def run_entry_in_process(
    feature: Feature, task: EntryTask
) -> EntryReport | EntryFailure:
    """Run :func:`run_entry` in a worker process, and measure the worker's memory.

    A failure is returned in a form that crosses the process boundary
    (:meth:`~mosaic.core.pipeline.entry_pool.EntryFailure.portable`).

    The measure, the garbage collection and the report after the write each catch
    an exception, for the reason :func:`run_entry` gives: a failed measure leaves
    the report's ``worker`` ``None``, and a failed collection is ignored.

    The worker's collector is fed the bytes of the input the task loaded and of
    the output it wrote.

    Returns:
        The report of :func:`run_entry`, with the worker's own memory: its
        resident memory after ``load_state`` and before the load or ``apply``,
        less the table it was handed, if any; its high-water mark at that
        moment; and its high-water mark after its output was written. The same
        for its anonymous and shared memory, where ``/proc`` gives it: the figure
        before the load or ``apply`` less the handed table, and the largest
        figure read from then to after its output was written. A task that
        loads its input is handed no table, so nothing is subtracted, and the
        peak contains the load. ``seconds_after_apply`` runs to this function's
        return.
    """
    # A handed table arrives as an argument, so it is resident before ``apply``
    # and stays referenced by the pool's call item. It is the entry's memory, not
    # the worker's, and a worker handed no table measures without it
    # (``process_worker_base``). A task that loads reads its base before the load.
    # A worker that built the feature for this task built it before this call, so
    # the base contains the feature's loaded state.
    source = task.source
    handed = (
        int(source.memory_usage(deep=True).sum())
        if isinstance(source, pd.DataFrame)
        else 0
    )
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
        _ = _COLLECTOR.finished(outcome.input_bytes + outcome.output_bytes)
    return replace(
        outcome,
        worker=worker,
        seconds_after_apply=outcome.seconds_after_apply + time.monotonic() - returned,
    )
