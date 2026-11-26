"""How a feature run ends while entries are in flight.

Every exit of the apply loop flushes the rows recorded so far, waits for the entries
in flight and records each output they write, flushes again, and raises the
exception that ended the loop. A rerun then serves every output on disk and computes
only the rest.

The parallel runs below hold ``seq_b`` and ``seq_c`` in their workers until the
exit's settle releases them, and end the loop while they are held: once the dispatch
of ``seq_c`` returns, from the wait for a worker before ``seq_d``'s dispatch, from
the load of ``seq_d`` in its task, or from ``apply`` itself. A serial run holds
nothing, and ends the loop once ``seq_c`` is computed. Under overlap the run loads
``seq_d`` in its own process, and an input that fails there ends the loop after
``seq_c``'s dispatch, with ``seq_a`` to ``seq_c`` recorded. A worker that dies
ends the loop at the dispatch of ``seq_d`` or ends the settle, and a second
interrupt abandons the settle and kills the process workers. A worker that dies
while another is idle and a third holds an entry ends the run's process too. A
worker killed in its write leaves its temporary file, and one killed after its
rename an output without a row. A cancel that arrives while the last entries are in
flight ends the run cancelled with their rows. A SIGTERM or SIGINT sent to the
process group of a child running the run ends the loop once the dispatch of
``seq_c`` returns, the process workers finish the entries they hold, and a SIGKILL
of the child alone ends its process workers with it.

The run's own threads alone write the index and the run-log, and a pool's threads
do not.
"""

from __future__ import annotations

import errno
import json
from contextlib import suppress
import os
import signal
import sys
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pandas as pd
import psutil
import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.entry import parse_entry_key
from mosaic.core.pipeline import entry_task as entry_task_module
from mosaic.core.pipeline import manifest as manifest_module
from mosaic.core.pipeline import run as run_module
from mosaic.core.pipeline._loaders import LoadSpec
from mosaic.core.pipeline._utils import FeatureMeta
from mosaic.core.pipeline.admission import RowCostEstimate
from mosaic.core.pipeline.entry_pool import EntryFailure, EntryPool, ResourceReporter
from mosaic.core.pipeline.entry_task import EntryReport
from mosaic.core.pipeline.index import (
    FeatureIndexRow,
    feature_index,
    feature_index_path,
)
from mosaic.core.pipeline.index_csv import IndexCSV
from mosaic.core.pipeline.index_lock import IndexLockTimeout
from mosaic.core.pipeline.job import Cancelled, CancelToken, install_signal_handler
from mosaic.core.pipeline.markers import ATOMIC_TEMP_PREFIX, ATOMIC_TEMP_SUFFIX
from mosaic.core.pipeline.loading import ColumnProjection
from mosaic.core.pipeline.manifest import CoreSelector
from mosaic.core.pipeline.memory_watch import MemoryWatch
from mosaic.core.pipeline.parallelism import (
    EntryOverMemoryLimit,
    MemoryFloorReached,
    ParallelMode,
    WorkerProcessDied,
)
from mosaic.core.pipeline.refusal import ToolUnavailable
from mosaic.core.pipeline.run import run_feature
from mosaic.core.pipeline.types import Result
from mosaic.core.pipeline.writers import FeatureOutput, write_parquet_atomic
from mosaic.runlog import JsonlRunLog, read_run, run_log_dir
from tests.helpers import (
    CHILD_TIMEOUT_SECONDS,
    HOLD_DIRECTORY_VARIABLE,
    HoldsBC,
    HoldsUntilReleased,
    SignalledGroup,
    add_tracks_variant,
    kill_running,
    logged_events,
    make_dataset,
    observed,
    process_alive,
    run_child_to_files,
    signal_group_when,
    wait_for,
    wrapped_write,
)

VARIANT = "convert-trex.0.2-1111111111"
_SEQUENCES = ("seq_a", "seq_b", "seq_c", "seq_d")
_HELD = ("seq_b", "seq_c")
_COMPUTED_BEFORE_THE_EXIT = frozenset({"seq_a", "seq_b", "seq_c"})
_STORAGE = f"{HoldsUntilReleased.name}__from__tracks"

type RunOutcome = EntryReport | EntryFailure
"""What a feature run's pool hands back for each entry it computed."""

RunPool = EntryPool[FeatureMeta, RunOutcome]
"""The pool of a feature run."""

type OnResult = Callable[[FeatureMeta, RunOutcome], None]
"""The callback a feature run's pool hands each result to."""


class HoldsBCWithOverlap(HoldsBC):
    """A :class:`HoldsBC` that accepts the overlap of its neighbors."""

    accepts_overlap = True


class ReleasedWithOverlap(HoldsUntilReleased):
    """The rerun of :class:`HoldsBCWithOverlap`, which holds nothing."""

    accepts_overlap = True


class HoldsBCOnAFullDisk(HoldsBC):
    """A :class:`HoldsBC` whose process workers fail the write of ``seq_a``."""

    fault = "full-disk"
    fault_on = ("seq_a",)


class HoldsBCWithOverlapUntrimmable(HoldsBCWithOverlap):
    """A :class:`HoldsBCWithOverlap` whose process workers fail to trim ``seq_a``."""

    fault = "trim"
    fault_on = ("seq_a",)


class HoldsBCUnreadableSeqD(HoldsBC):
    """A :class:`HoldsBC` whose process workers fail the load of ``seq_d``."""

    fault = "unreadable-input"
    fault_on = ("seq_d",)


class HoldsBCWritingNoneOfThem(HoldsBC):
    """A :class:`HoldsBC` whose process workers fail the writes of the held entries."""

    fault = "full-disk"
    fault_on = _HELD


class RefusesSeqA(HoldsBC):
    """Finds its tool missing on ``seq_a`` and holds ``seq_b`` and ``seq_c``."""

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if not df.empty and str(df["sequence"].iloc[0]) == "seq_a":
            raise ToolUnavailable("fake-tool", "environment", "fake-tool is not set up")
        return super().apply(df)


class LosesSeqB(HoldsUntilReleased):
    """Loses ``seq_b`` to an ordinary error and holds nothing."""

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if not df.empty and str(df["sequence"].iloc[0]) == "seq_b":
            message = "a lost entry"
            raise RuntimeError(message)
        return super().apply(df)


type Mode = Literal["serial", "thread", "process"]

_PARALLEL: tuple[Mode, ...] = ("thread", "process")
_EVERY_MODE: tuple[Mode, ...] = ("serial", "thread", "process")

type Site = Literal["worker", "parent"]
"""Where ``seq_d``'s input is loaded: in its task, or in the run's own process under
overlap."""

_SITES: tuple[Site, ...] = ("worker", "parent")


def _holding(mode: Mode) -> HoldsUntilReleased:
    """The feature that holds ``seq_b`` and ``seq_c`` in a parallel run.

    A serial run computes each entry before it takes the next, so a held entry
    would never let it reach ``seq_d``.
    """
    return HoldsUntilReleased() if mode == "serial" else HoldsBC()


@dataclass(frozen=True)
class RunOptions:
    """The options of a run beyond its mode, which a rerun repeats or leaves out."""

    workers: int = 3
    overlap_frames: int = 0
    time_window: tuple[float, float] | None = None
    min_free_memory_fraction: float = 0.0
    admission_margin: float = 0.0


_EVERY_ROW = (-1.0e9, 1.0e9)
"""A time window that covers every row."""


@dataclass
class Exits:
    """The dataset a test runs, and what the exit's settle did.

    Attributes:
        ds: Four sequences, ``seq_a`` to ``seq_d``.
        marks: The hold directory of :class:`HoldsUntilReleased`.
        token: The cancel token the next run is handed.
        settles: The ``results_only`` of each settle of a pool, in order.
        settle_error: Raised by each settle after it returns, when given.
        releases: Whether a settle releases the held entries before it waits.
    """

    ds: Dataset
    marks: Path
    token: CancelToken = field(default_factory=CancelToken)
    settles: list[bool] = field(default_factory=list[bool])
    settle_error: Exception | None = None
    releases: bool = True

    def held(self) -> bool:
        return all((self.marks / sequence).exists() for sequence in _HELD)

    def applied(self) -> set[str]:
        prefix = "applied-"
        return {
            path.name.removeprefix(prefix) for path in self.marks.glob(f"{prefix}*")
        }

    def run(
        self,
        feature: HoldsUntilReleased,
        mode: Mode,
        execution_id: str,
        options: RunOptions = RunOptions(),
    ) -> Result:
        window = options.time_window
        return run_feature(
            self.ds,
            feature,
            parallel_workers=1 if mode == "serial" else options.workers,
            parallel_mode="thread" if mode == "serial" else mode,
            overlap_frames=options.overlap_frames,
            filter_start_time=None if window is None else window[0],
            filter_end_time=None if window is None else window[1],
            min_free_memory_fraction=options.min_free_memory_fraction,
            admission_margin=options.admission_margin,
            cancel_token=self.token,
            execution_id=execution_id,
        )


def _record_settles(monkeypatch: pytest.MonkeyPatch, harness: Exits) -> None:
    """Have each settle of a pool record itself in *harness* and release the held entries.

    The release is left out when ``harness.releases`` is false, and the settle
    raises ``harness.settle_error`` after it returns when one is given.
    """
    real_settle = RunPool.settle

    def settle(
        self: RunPool,
        *,
        results_only: bool = False,
    ) -> list[Exception] | None:
        harness.settles.append(results_only)
        if harness.releases:
            (harness.marks / "release").touch()
        settled = real_settle(self, results_only=results_only)
        if harness.settle_error is not None:
            raise harness.settle_error
        return settled

    monkeypatch.setattr(EntryPool, "settle", settle)


@pytest.fixture
def exits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Exits]:
    """A dataset whose pools record their settles, and release the held entries.

    The release is touched again on the way out, so a run that never settles
    leaves no worker holding.
    """
    ds = make_dataset(tmp_path / "ds")
    add_tracks_variant(ds, VARIANT, *_SEQUENCES, std_format="mosaic_v1")
    marks = tmp_path / "marks"
    marks.mkdir()
    monkeypatch.setenv(HOLD_DIRECTORY_VARIABLE, str(marks))
    harness = Exits(ds=ds, marks=marks)
    _record_settles(monkeypatch, harness)
    try:
        yield harness
    finally:
        (marks / "release").touch()


type Load = Callable[[], pd.DataFrame | None]
"""The real load of one entry's table."""

type Fault = Callable[[], None]
"""What a test does once the dispatch of ``seq_c`` has returned."""


def _once_seq_c_is_dispatched(
    monkeypatch: pytest.MonkeyPatch,
    exits: Exits,
    mode: Mode,
    fault: Fault,
    *,
    holding: bool = True,
) -> None:
    """Call *fault* once ``seq_c``'s dispatch returns, ``seq_b`` and ``seq_c`` held.

    It is the last step of the run's process before the turn of ``seq_d``, whose
    first step is the check for a cancel. A serial run, or a call that is not
    *holding*, holds nothing, and its fault fires at once.
    """
    real = RunPool.submit

    def submit(
        self: RunPool,
        tag: FeatureMeta,
        fn: Callable[..., RunOutcome],
        *args: object,
    ) -> None:
        real(self, tag, fn, *args)
        if tag.sequence != "seq_c":
            return
        if holding and mode != "serial":
            wait_for(exits.held, CHILD_TIMEOUT_SECONDS, "seq_b and seq_c never held")
        fault()

    monkeypatch.setattr(EntryPool, "submit", submit)


def _on_load_of_seq_d(
    monkeypatch: pytest.MonkeyPatch,
    exits: Exits,
    mode: Mode,
    fault: Callable[[Load], pd.DataFrame | None],
    *,
    site: Site = "parent",
) -> None:
    """Hand the load of ``seq_d``'s input at *site* to *fault*.

    *fault* receives the real load. At the parent site the run's process loads
    ``seq_d``, and a parallel run's fault fires once ``seq_b`` and ``seq_c`` are
    held. At the worker site ``seq_d``'s task loads it, and the fault fires at
    once. The patch reaches a serial or thread task and not a process worker.
    """
    module = manifest_module if site == "parent" else entry_task_module
    real = module.load_entry_data

    def load(
        file_specs: list[tuple[Path, LoadSpec]],
        filters: Iterable[Callable[[pd.DataFrame], pd.DataFrame]] = (),
        *,
        cross_join: bool = False,
        expands_individuals: bool = False,
        projection: ColumnProjection | None = None,
    ) -> pd.DataFrame | None:
        def load_now() -> pd.DataFrame | None:
            return real(
                file_specs,
                filters,
                cross_join=cross_join,
                expands_individuals=expands_individuals,
                projection=projection,
            )

        if not any(path.stem == "seq_d" for path, _ in file_specs):
            return load_now()
        if site == "parent" and mode != "serial":
            wait_for(exits.held, CHILD_TIMEOUT_SECONDS, "seq_b and seq_c never held")
        return fault(load_now)

    monkeypatch.setattr(module, "load_entry_data", load)


def _cancel(exits: Exits) -> Fault:
    def fault() -> None:
        exits.token.cancel()

    return fault


def _interrupt() -> None:
    raise KeyboardInterrupt


def _nothing() -> None:
    pass


def _unreadable(load_now: Load) -> pd.DataFrame | None:
    raise OSError(errno.EIO, "input/output error")


def _load(load_now: Load) -> pd.DataFrame | None:
    return load_now()


def _fail_writes(
    monkeypatch: pytest.MonkeyPatch, fails: Callable[[str], bool], error: Exception
) -> None:
    """Make the write of each sequence for which *fails* is true raise *error*.

    The write runs in the entry's worker, so the fault reaches a thread worker and
    not a process worker.
    """

    def before(meta: FeatureMeta) -> None:
        if fails(meta.sequence):
            raise error

    monkeypatch.setattr(entry_task_module, "write_output", wrapped_write(before))


def _indexed(ds: Dataset, run_id: str) -> pd.DataFrame:
    """The index rows of *run_id*, none when the index holds none."""
    try:
        return feature_index(feature_index_path(ds, _STORAGE)).read(run_id=run_id)
    except FileNotFoundError:
        return pd.DataFrame({"sequence": [], "abs_path": []})


def _check_exit(
    exits: Exits,
    execution_id: str,
    *,
    status: str,
    failed: set[str],
    rows: frozenset[str],
    mode: Mode,
    unrecorded: frozenset[str] = frozenset(),
    temporary: frozenset[str] = frozenset(),
    rerun: HoldsUntilReleased | None = None,
    options: RunOptions = RunOptions(),
) -> None:
    """Check what a run left once an exception ended it, then rerun it.

    The run-log holds *status*, the entry errors of *failed* and one count of the
    entries written, equal to the rows of the run. :func:`_check_disk_and_rerun`
    checks the outputs, the rows and the rerun.
    """
    snapshot = read_run(run_log_dir(exits.ds.base_dir), execution_id)
    assert snapshot is not None
    assert snapshot["status"] == status
    errors = logged_events(exits.ds.base_dir, execution_id, "entry_error")
    keys = (str(error["key"]) for error in errors)
    assert sorted(parse_entry_key(key)[1] for key in keys) == sorted(failed)
    written = logged_events(exits.ds.base_dir, execution_id, "entries_written")
    assert [event["entries_written"] for event in written] == [len(rows)]
    _check_disk_and_rerun(
        exits,
        execution_id,
        snapshot["run_id"],
        rows=rows,
        mode=mode,
        unrecorded=unrecorded,
        temporary=temporary,
        rerun=rerun,
        options=options,
    )


def _temporary_sequence(path: Path) -> str:
    """The sequence whose output the temporary file *path* of ``atomic_write`` was for.

    The file is named ``.<entry key>-<random>.tmp``.
    """
    name = path.name.removeprefix(ATOMIC_TEMP_PREFIX).removesuffix(ATOMIC_TEMP_SUFFIX)
    return parse_entry_key(name.rsplit("-", 1)[0])[1]


def _check_disk_and_rerun(
    exits: Exits,
    execution_id: str,
    run_id: str,
    *,
    rows: frozenset[str],
    mode: Mode,
    unrecorded: frozenset[str] = frozenset(),
    temporary: frozenset[str] = frozenset(),
    rerun: HoldsUntilReleased | None = None,
    options: RunOptions = RunOptions(),
) -> None:
    """Check the outputs and rows that the run *execution_id* left, then rerun it.

    The index holds the rows of *rows*. Every output under the run root has a row
    but those of *unrecorded*, whose rows the run could not append, and every row's
    output exists. Each sequence of *temporary*, whose worker died in its write,
    left exactly one hidden temporary file, which no ``*.parquet`` filter takes, and
    no other temporary file is left. A rerun of *rerun*, a feature that holds
    nothing, with *options* and the same run id computes exactly the sequences
    without an output.
    """
    assert run_id
    index = _indexed(exits.ds, run_id)
    assert sorted(index["sequence"]) == sorted(rows)
    recorded = {
        exits.ds.resolve_path(str(stored)).resolve() for stored in index["abs_path"]
    }
    assert all(path.exists() for path in recorded)
    run_root = exits.ds.get_root("features") / _STORAGE / run_id
    on_disk = {path.resolve() for path in run_root.glob("*.parquet")}
    assert {path for path in on_disk if path.stem not in unrecorded} == recorded
    assert {path.stem for path in on_disk} >= unrecorded
    left = list(run_root.rglob(".*.tmp"))
    assert sorted(map(_temporary_sequence, left)) == sorted(temporary)
    assert not any(path.match("*.parquet") for path in left)

    for mark in exits.marks.glob("applied-*"):
        mark.unlink()
    exits.token = CancelToken()
    again = exits.run(
        HoldsUntilReleased() if rerun is None else rerun,
        mode,
        f"{execution_id}AGAIN",
        options,
    )
    assert again.run_id == run_id
    assert exits.applied() == set(_SEQUENCES) - {path.stem for path in on_disk}
    finished = read_run(run_log_dir(exits.ds.base_dir), f"{execution_id}AGAIN")
    assert finished is not None
    assert finished["status"] == "finished"


# --- Cancel and interrupt -------------------------------------------------------


class HoldsD(HoldsUntilReleased):
    """Holds ``seq_d`` until the test releases it."""

    hold_on = ("seq_d",)


def test_a_cancel_during_the_final_drain_ends_the_run_cancelled(
    exits: Exits, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Every entry is dispatched before the cancel, so only the check after the
    # last entries are drained sees it.
    _wrap_pools(monkeypatch, exits.marks)

    def cancel_then_release() -> None:
        wait_for(
            lambda: (
                (exits.marks / "seq_d").exists()
                and _computed(exits.marks, "seq_a", "seq_b", "seq_c")
            ),
            CHILD_TIMEOUT_SECONDS,
            "seq_d was never held with seq_a to seq_c computed",
        )
        exits.token.cancel()
        (exits.marks / "release").touch()

    canceller = threading.Thread(target=cancel_then_release, daemon=True)
    canceller.start()
    with pytest.raises(Cancelled):
        _ = exits.run(HoldsD(), "thread", "TESTEXITFINALDRAIN", RunOptions(workers=2))
    canceller.join(CHILD_TIMEOUT_SECONDS)

    snapshot = read_run(run_log_dir(exits.ds.base_dir), "TESTEXITFINALDRAIN")
    assert snapshot is not None
    assert snapshot["status"] == "cancelled"
    index = _indexed(exits.ds, snapshot["run_id"])
    assert sorted(index["sequence"]) == list(_SEQUENCES)
    assert set(index["finished_at"]) == {""}


@pytest.mark.parametrize("mode", _EVERY_MODE)
def test_a_cancel_records_the_outputs_of_the_entries_in_flight(
    exits: Exits, capsys: pytest.CaptureFixture[str], mode: Mode
) -> None:
    with pytest.MonkeyPatch.context() as faults:
        _once_seq_c_is_dispatched(faults, exits, mode, _cancel(exits))
        with pytest.raises(Cancelled):
            _ = exits.run(_holding(mode), mode, "TESTEXITCANCEL")

    _check_exit(
        exits,
        "TESTEXITCANCEL",
        status="cancelled",
        failed=set(),
        rows=_COMPUTED_BEFORE_THE_EXIT,
        mode=mode,
    )
    assert exits.settles == [True]
    waits = "waits for the entries in flight" in capsys.readouterr().err
    assert waits == (mode != "serial")


@pytest.mark.parametrize("mode", _EVERY_MODE)
def test_an_interrupt_records_the_outputs_of_the_entries_in_flight(
    exits: Exits, mode: Mode
) -> None:
    with pytest.MonkeyPatch.context() as faults:
        _once_seq_c_is_dispatched(faults, exits, mode, _interrupt)
        with pytest.raises(KeyboardInterrupt):
            _ = exits.run(_holding(mode), mode, "TESTEXITINTERRUPT")

    _check_exit(
        exits,
        "TESTEXITINTERRUPT",
        status="cancelled",
        failed=set(),
        rows=_COMPUTED_BEFORE_THE_EXIT,
        mode=mode,
    )
    assert exits.settles == [True]


# --- Input and output failures --------------------------------------------------


def _tracks_of_seq_d(exits: Exits) -> Path:
    return exits.ds.get_root("tracks") / VARIANT / "seq_d.parquet"


@dataclass(frozen=True)
class InputSite:
    """How an input-failure test runs at one site.

    Attributes:
        feature: The feature of the run.
        options: The options of the run and of its rerun.
        rerun: The feature of the rerun, ``None`` for the default.
    """

    feature: HoldsUntilReleased
    options: RunOptions
    rerun: HoldsUntilReleased | None


def _input_site(
    site: Site,
    mode: Mode,
    *,
    time_window: tuple[float, float] | None = None,
    unreadable: bool = False,
) -> InputSite:
    """The run of an input-failure test at *site*.

    At the worker site ``seq_d``'s task fails its load, and a process run's
    *unreadable* input comes from :class:`HoldsBCUnreadableSeqD`. At the parent
    site the run loads ``seq_d`` in its own process, because overlap is asked, and
    the load fails there. Either way the exit settles ``seq_b`` and ``seq_c``.
    """
    if site == "worker":
        return InputSite(
            feature=HoldsBCUnreadableSeqD()
            if unreadable and mode == "process"
            else _holding(mode),
            options=RunOptions(time_window=time_window),
            rerun=None,
        )
    return InputSite(
        feature=ReleasedWithOverlap() if mode == "serial" else HoldsBCWithOverlap(),
        options=RunOptions(overlap_frames=1, time_window=time_window),
        rerun=ReleasedWithOverlap(),
    )


def _fail_at(
    monkeypatch: pytest.MonkeyPatch,
    exits: Exits,
    site: Site,
    mode: Mode,
    fault: Callable[[Load], pd.DataFrame | None],
) -> None:
    """Have the first load of ``seq_d`` at *site* go through *fault*.

    At the worker site the run waits after the dispatch of ``seq_c`` until
    ``seq_b`` and ``seq_c`` are held, and at the parent site the load waits. The
    fault is patched into a serial or thread
    task, and a process worker's fault comes from its feature.
    """
    if site == "parent":
        _on_load_of_seq_d(monkeypatch, exits, mode, fault)
        return
    _once_seq_c_is_dispatched(monkeypatch, exits, mode, _nothing)
    if mode != "process":
        _on_load_of_seq_d(monkeypatch, exits, mode, fault, site="worker")


@pytest.mark.parametrize("site", _SITES)
@pytest.mark.parametrize("mode", _EVERY_MODE)
def test_an_input_error_ends_the_run_with_the_outputs_in_flight_recorded(
    exits: Exits, mode: Mode, site: Site
) -> None:
    run = _input_site(site, mode, unreadable=True)

    with pytest.MonkeyPatch.context() as faults:
        _fail_at(faults, exits, site, mode, _unreadable)
        with pytest.raises(OSError, match="input/output error") as raised:
            _ = exits.run(run.feature, mode, "TESTEXITEIO", run.options)

    assert raised.value.errno == errno.EIO
    _check_exit(
        exits,
        "TESTEXITEIO",
        status="failed",
        failed=set(),
        rows=_COMPUTED_BEFORE_THE_EXIT,
        mode=mode,
        rerun=run.rerun,
        options=run.options,
    )
    assert exits.settles == [True]


@pytest.mark.parametrize("site", _SITES)
@pytest.mark.parametrize("mode", _EVERY_MODE)
def test_a_corrupt_input_ends_the_run_with_the_outputs_in_flight_recorded(
    exits: Exits, mode: Mode, site: Site
) -> None:
    tracks = _tracks_of_seq_d(exits)
    table = tracks.read_bytes()
    _ = tracks.write_bytes(b"not a parquet file")
    run = _input_site(site, mode)

    with pytest.MonkeyPatch.context() as faults:
        _fail_at(faults, exits, site, mode, _load)
        with pytest.raises(ValueError, match="magic bytes not found") as raised:
            _ = exits.run(run.feature, mode, "TESTEXITCORRUPT", run.options)

    _ = tracks.write_bytes(table)
    snapshot = read_run(run_log_dir(exits.ds.base_dir), "TESTEXITCORRUPT")
    assert snapshot is not None
    assert json.loads(snapshot["error_json"])["type"] == type(raised.value).__name__
    _check_exit(
        exits,
        "TESTEXITCORRUPT",
        status="failed",
        failed=set(),
        rows=_COMPUTED_BEFORE_THE_EXIT,
        mode=mode,
        rerun=run.rerun,
        options=run.options,
    )
    assert exits.settles == [True]


@pytest.mark.parametrize("mode", _EVERY_MODE)
def test_a_time_window_over_rows_without_time_ends_the_run_with_the_outputs_in_flight_recorded(
    exits: Exits, mode: Mode
) -> None:
    # A time window and overlap exclude each other, so the window's error arises
    # only where the task loads.
    tracks = _tracks_of_seq_d(exits)
    table = tracks.read_bytes()
    _ = write_parquet_atomic(pd.read_parquet(tracks).drop(columns=["time"]), tracks)
    run = _input_site("worker", mode, time_window=_EVERY_ROW)

    with pytest.MonkeyPatch.context() as faults:
        _fail_at(faults, exits, "worker", mode, _load)
        with pytest.raises(ValueError, match="a time filter reads"):
            _ = exits.run(run.feature, mode, "TESTEXITWINDOW", run.options)

    _ = tracks.write_bytes(table)
    _check_exit(
        exits,
        "TESTEXITWINDOW",
        status="failed",
        failed=set(),
        rows=_COMPUTED_BEFORE_THE_EXIT,
        mode=mode,
        rerun=run.rerun,
        options=run.options,
    )
    assert exits.settles == [True]


@pytest.mark.parametrize("mode", _PARALLEL)
def test_an_output_error_ends_the_run_with_the_outputs_in_flight_recorded(
    exits: Exits, mode: Mode
) -> None:
    feature = HoldsBC() if mode == "thread" else HoldsBCOnAFullDisk()

    with pytest.MonkeyPatch.context() as faults:
        if mode == "thread":
            _fail_writes(
                faults,
                lambda sequence: sequence == "seq_a",
                OSError("no space left on device"),
            )
        with pytest.raises(OSError, match="no space left"):
            _ = exits.run(feature, mode, "TESTEXITWRITE")

    _check_exit(
        exits,
        "TESTEXITWRITE",
        status="failed",
        failed=set(),
        rows=frozenset(_HELD),
        mode=mode,
    )
    assert exits.settles == [True]


@pytest.mark.parametrize("mode", _PARALLEL)
def test_a_trim_error_ends_the_run_with_the_outputs_in_flight_recorded(
    exits: Exits, mode: Mode
) -> None:
    real_trim = entry_task_module.trim_feature_output
    options = RunOptions(overlap_frames=1)
    feature = (
        HoldsBCWithOverlap() if mode == "thread" else HoldsBCWithOverlapUntrimmable()
    )

    def trim(df: FeatureOutput, selector: CoreSelector) -> FeatureOutput:
        if parse_entry_key(selector.entry_key)[1] == "seq_a":
            message = "the overlap could not be trimmed"
            raise ValueError(message)
        return real_trim(df, selector)

    with pytest.MonkeyPatch.context() as faults:
        if mode == "thread":
            faults.setattr(entry_task_module, "trim_feature_output", trim)
        with pytest.raises(ValueError, match="could not be trimmed"):
            _ = exits.run(feature, mode, "TESTEXITTRIM", options)

    _check_exit(
        exits,
        "TESTEXITTRIM",
        status="failed",
        failed=set(),
        rows=frozenset(_HELD),
        mode=mode,
        rerun=ReleasedWithOverlap(),
        options=options,
    )
    assert exits.settles == [True]


@pytest.mark.parametrize("mode", _PARALLEL)
def test_a_refusal_from_apply_records_the_outputs_of_the_entries_in_flight(
    exits: Exits, mode: Mode
) -> None:
    # The refusal cancels the calls not yet started, so the run waits after the
    # dispatch of seq_c until seq_b and seq_c are held, before the wait for a
    # worker for seq_d drains the refusal.
    with pytest.MonkeyPatch.context() as faults:
        _once_seq_c_is_dispatched(faults, exits, mode, _nothing)
        with pytest.raises(ToolUnavailable):
            _ = exits.run(RefusesSeqA(), mode, "TESTEXITREFUSED")

    _check_exit(
        exits,
        "TESTEXITREFUSED",
        status="failed",
        failed={"seq_a"},
        rows=frozenset(_HELD),
        mode=mode,
    )
    assert exits.settles == [True]


# --- Failures on the way out ----------------------------------------------------


def _fail_appends(monkeypatch: pytest.MonkeyPatch, failures: int | None) -> None:
    """Make the first *failures* appends of index rows fail, every one when ``None``.

    The flush pacer is never due, so the only appends are the exit's two flushes.
    """
    monkeypatch.setattr(run_module, "IDX_FLUSH_SECONDS", 1.0e9)
    real_append = IndexCSV[FeatureIndexRow].append
    calls: list[int] = []

    def append(self: IndexCSV[FeatureIndexRow], rows: list[FeatureIndexRow]) -> None:
        calls.append(len(rows))
        if failures is None or len(calls) <= failures:
            message = "index.csv is locked"
            raise IndexLockTimeout(message)
        real_append(self, rows)

    monkeypatch.setattr(IndexCSV, "append", append)


def _notes(error: BaseException, text: str) -> list[str]:
    """The notes on *error* that contain *text*."""
    return [note for note in getattr(error, "__notes__", ()) if text in note]


@pytest.mark.parametrize(
    ("mode", "failures", "flushes", "rows"),
    [
        ("serial", 1, ["first flush"], _COMPUTED_BEFORE_THE_EXIT),
        ("serial", None, ["first flush", "second flush"], frozenset[str]()),
        ("thread", None, ["second flush"], frozenset[str]()),
        ("process", None, ["second flush"], frozenset[str]()),
    ],
    ids=["serial-once", "serial-always", "thread-always", "process-always"],
)
def test_a_failed_flush_is_a_note_on_the_exception_that_ended_the_run(
    exits: Exits,
    mode: Mode,
    failures: int | None,
    flushes: list[str],
    rows: frozenset[str],
) -> None:
    with pytest.MonkeyPatch.context() as faults:
        _once_seq_c_is_dispatched(faults, exits, mode, _cancel(exits))
        _fail_appends(faults, failures)
        with pytest.raises(Cancelled) as raised:
            _ = exits.run(_holding(mode), mode, "TESTEXITFLUSH")

    notes = _notes(raised.value, "index.csv is locked")
    assert len(notes) == len(flushes)
    for note, flush in zip(notes, flushes, strict=True):
        assert flush in note
    _check_exit(
        exits,
        "TESTEXITFLUSH",
        status="cancelled",
        failed=set(),
        rows=rows,
        unrecorded=_COMPUTED_BEFORE_THE_EXIT - rows,
        mode=mode,
    )


@pytest.mark.parametrize("mode", _EVERY_MODE)
def test_a_settle_that_raises_is_a_note_on_the_exception_that_ended_the_run(
    exits: Exits, mode: Mode
) -> None:
    exits.settle_error = RuntimeError("the pool broke")

    with pytest.MonkeyPatch.context() as faults:
        _once_seq_c_is_dispatched(faults, exits, mode, _cancel(exits))
        with pytest.raises(Cancelled) as raised:
            _ = exits.run(_holding(mode), mode, "TESTEXITSETTLE")

    exits.settle_error = None
    (note,) = _notes(raised.value, "the pool broke")
    assert "settle" in note
    _check_exit(
        exits,
        "TESTEXITSETTLE",
        status="cancelled",
        failed=set(),
        rows=_COMPUTED_BEFORE_THE_EXIT,
        mode=mode,
    )


@pytest.mark.parametrize("mode", _PARALLEL)
def test_the_writes_a_settle_drops_are_one_note_on_the_exception_that_ended_the_run(
    exits: Exits, mode: Mode
) -> None:
    # A process worker writes a held entry only once the settle releases it.
    feature = HoldsBC() if mode == "thread" else HoldsBCWritingNoneOfThem()

    with pytest.MonkeyPatch.context() as faults:
        _once_seq_c_is_dispatched(faults, exits, mode, _cancel(exits))
        if mode == "thread":
            _fail_writes(
                faults,
                lambda sequence: bool(exits.settles) and sequence in _HELD,
                OSError("no space left on device"),
            )
        with pytest.raises(Cancelled) as raised:
            _ = exits.run(feature, mode, "TESTEXITDROPPED")

    (note,) = _notes(raised.value, "no space left on device")
    assert note.startswith("2 of the outputs of the entries in flight raised")
    assert "the first: OSError: no space left on device" in note
    _check_exit(
        exits,
        "TESTEXITDROPPED",
        status="cancelled",
        failed=set(),
        rows=frozenset({"seq_a"}),
        mode=mode,
    )


# --- Worker deaths and a second interrupt --------------------------------------


def _computed(marks: Path, *sequences: str) -> bool:
    return all((marks / f"computed-{sequence}").exists() for sequence in sequences)


class DiesOnSeqC(HoldsUntilReleased):
    """Ends its worker with SIGKILL on ``seq_c`` once ``seq_a`` and ``seq_b`` are computed.

    It first holds as :class:`HoldsUntilReleased` does, and touches ``killed`` in the
    hold directory just before the kill.
    """

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        output = super().apply(df)
        if not df.empty and str(df["sequence"].iloc[0]) == "seq_c":
            marks = Path(os.environ[HOLD_DIRECTORY_VARIABLE])
            wait_for(
                lambda: _computed(marks, "seq_a", "seq_b"),
                CHILD_TIMEOUT_SECONDS,
                "seq_a and seq_b were never computed",
            )
            (marks / "killed").touch()
            os.kill(os.getpid(), signal.SIGKILL)
        return output


class HoldsBCThenDiesOnSeqC(DiesOnSeqC):
    """Holds ``seq_b`` and ``seq_c``, and once released ends ``seq_c``'s worker."""

    hold_on = _HELD


class HoldsBDiesOnSeqC(HoldsUntilReleased):
    """Runs ``seq_a`` to ``seq_c`` in three workers, holds ``seq_b``, kills ``seq_c``'s.

    ``seq_a``'s worker waits until ``seq_c`` is applied, so the three entries run in
    three workers. Once ``seq_a`` is computed and ``seq_b`` is held, ``seq_c``'s
    worker touches ``killed`` in the hold directory and ends itself with SIGKILL, so
    at its death one worker is idle and one holds ``seq_b``.
    """

    hold_on = ("seq_b",)

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        sequence = str(df["sequence"].iloc[0]) if not df.empty else ""
        marks = Path(os.environ[HOLD_DIRECTORY_VARIABLE])
        if sequence == "seq_a":
            wait_for(
                (marks / "applied-seq_c").exists,
                CHILD_TIMEOUT_SECONDS,
                "seq_c was never applied",
            )
        output = super().apply(df)
        if sequence == "seq_c":
            wait_for(
                lambda: _computed(marks, "seq_a"),
                CHILD_TIMEOUT_SECONDS,
                "seq_a was never computed",
            )
            wait_for(
                (marks / "seq_b").exists,
                CHILD_TIMEOUT_SECONDS,
                "seq_b was never held",
            )
            (marks / "killed").touch()
            os.kill(os.getpid(), signal.SIGKILL)
        return output


def _wrap_pools(
    monkeypatch: pytest.MonkeyPatch,
    marks: Path,
    around_result: Callable[[OnResult], OnResult] | None = None,
) -> None:
    """Have each pool of a run mark the calls it computed, and wrap its result callback.

    The pool touches ``computed-<sequence>`` in *marks* once its ``on_computed``
    hears a call, which a future calls after its outcome is set. Its ``on_result``
    becomes ``around_result(on_result)`` when *around_result* is given. The run
    builds its pool through the name ``EntryPool`` of its module, which this
    replaces with a builder of the same keywords, because the ``__init__`` of the
    specialized pool cannot be named with its types. Every keyword is required, so
    a run that stops passing one fails here instead of taking a stale default.
    """

    def build(
        *,
        workers: int,
        mode: ParallelMode,
        on_result: OnResult,
        on_error: Callable[[FeatureMeta, Exception], None],
        min_free_memory_fraction: float,
        on_resources: ResourceReporter | None,
        watch: MemoryWatch | None,
        floor_at_submit: bool,
        on_dispatch: Callable[[FeatureMeta], None] | None,
        on_computed: Callable[[FeatureMeta], None] | None,
        cancel_check: Callable[[], None] | None,
        hand_back_on_break: bool,
        initializer: Callable[..., object] | None,
        initargs: tuple[object, ...],
    ) -> RunPool:
        def computed(tag: FeatureMeta) -> None:
            if on_computed is not None:
                on_computed(tag)
            (marks / f"computed-{tag.sequence}").touch()

        return RunPool(
            workers=workers,
            mode=mode,
            on_result=on_result if around_result is None else around_result(on_result),
            on_error=on_error,
            min_free_memory_fraction=min_free_memory_fraction,
            on_resources=on_resources,
            watch=watch,
            floor_at_submit=floor_at_submit,
            on_dispatch=on_dispatch,
            on_computed=computed,
            cancel_check=cancel_check,
            hand_back_on_break=hand_back_on_break,
            initializer=initializer,
            initargs=initargs,
        )

    monkeypatch.setattr(run_module, "EntryPool", build)


def _no_spawned_worker_alive() -> bool:
    """Whether every worker that ``multiprocessing`` spawned below this process is gone."""
    for child in psutil.Process().children(recursive=True):
        try:
            if child.status() == psutil.STATUS_ZOMBIE:
                continue
            command = child.cmdline()
        except psutil.Error:
            continue
        if any("spawn_main" in part for part in command):
            return False
    return True


def _once_the_workers_are_gone() -> None:
    wait_for(_no_spawned_worker_alive, CHILD_TIMEOUT_SECONDS, "the workers never ended")


def test_a_worker_death_hands_back_the_calls_that_finished_before_it(
    exits: Exits, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Four workers, so seq_d's dispatch finds a free worker and meets the broken
    # pool, with seq_a and seq_b computed and not yet handed back.
    options = RunOptions(workers=4)
    _wrap_pools(monkeypatch, exits.marks)

    with pytest.MonkeyPatch.context() as faults:
        _once_seq_c_is_dispatched(
            faults, exits, "process", _once_the_workers_are_gone, holding=False
        )
        with pytest.raises(WorkerProcessDied):
            _ = exits.run(DiesOnSeqC(), "process", "TESTEXITDIED", options)

    _check_exit(
        exits,
        "TESTEXITDIED",
        status="failed",
        failed=set(),
        rows=frozenset({"seq_a", "seq_b"}),
        mode="process",
        options=options,
    )
    assert exits.settles == [True]


def test_a_worker_death_during_the_settle_ends_it_with_the_results_handed_back(
    exits: Exits, monkeypatch: pytest.MonkeyPatch
) -> None:
    _wrap_pools(monkeypatch, exits.marks)

    with pytest.MonkeyPatch.context() as faults:
        _once_seq_c_is_dispatched(faults, exits, "process", _cancel(exits))
        with pytest.raises(Cancelled):
            _ = exits.run(HoldsBCThenDiesOnSeqC(), "process", "TESTEXITSETTLEDIED")

    assert time.time() - (exits.marks / "killed").stat().st_mtime < (
        CHILD_TIMEOUT_SECONDS
    )
    _check_exit(
        exits,
        "TESTEXITSETTLEDIED",
        status="cancelled",
        failed=set(),
        rows=frozenset({"seq_a", "seq_b"}),
        mode="process",
    )
    assert exits.settles == [True]


class KillsInTheWriteOfSeqC(HoldsUntilReleased):
    """Ends ``seq_c``'s worker in its write, once ``seq_a`` and ``seq_b`` are computed.

    The worker dies after the temporary file is written and before its rename.
    """

    fault = "kill-in-write"
    fault_on = ("seq_c",)
    kill_once_computed = ("seq_a", "seq_b")


class KillsAfterTheRenameOfSeqC(KillsInTheWriteOfSeqC):
    """Ends ``seq_c``'s worker once its write has returned."""

    fault = "kill-after-rename"


@pytest.mark.parametrize(
    ("feature", "unrecorded", "temporary"),
    [
        (KillsInTheWriteOfSeqC(), frozenset[str](), frozenset({"seq_c"})),
        (KillsAfterTheRenameOfSeqC(), frozenset({"seq_c"}), frozenset[str]()),
    ],
    ids=["in-the-write", "after-the-rename"],
)
def test_a_worker_killed_in_its_write_leaves_a_temporary_file_or_an_output_without_a_row(
    exits: Exits,
    monkeypatch: pytest.MonkeyPatch,
    feature: HoldsUntilReleased,
    unrecorded: frozenset[str],
    temporary: frozenset[str],
) -> None:
    # Four workers, so seq_d's dispatch finds a free worker and meets the broken
    # pool, with seq_a and seq_b computed and not yet handed back. A worker killed
    # before its rename leaves seq_c's temporary file and no output, and one killed
    # after it seq_c's output, which the rerun serves.
    options = RunOptions(workers=4)
    _wrap_pools(monkeypatch, exits.marks)

    with pytest.MonkeyPatch.context() as faults:
        _once_seq_c_is_dispatched(
            faults, exits, "process", _once_the_workers_are_gone, holding=False
        )
        with pytest.raises(WorkerProcessDied):
            _ = exits.run(feature, "process", "TESTEXITKILLEDWRITING", options)

    assert (exits.marks / "killed").exists()
    _check_exit(
        exits,
        "TESTEXITKILLEDWRITING",
        status="failed",
        failed=set(),
        rows=frozenset({"seq_a", "seq_b"}),
        mode="process",
        unrecorded=unrecorded,
        temporary=temporary,
        options=options,
    )
    assert exits.settles == [True]


_SECOND_INTERRUPT = "a second interrupt"


def _interrupt_after_seq_a_in_the_settle(
    exits: Exits,
) -> Callable[[OnResult], OnResult]:
    """Raise a second ``KeyboardInterrupt`` once the settle has handed back ``seq_a``.

    It stands for a second signal, and is raised whether or not the result callback
    raised.
    """

    def around(on_result: OnResult) -> OnResult:
        def interrupting(tag: FeatureMeta, outcome: RunOutcome) -> None:
            if not exits.settles or tag.sequence != "seq_a":
                on_result(tag, outcome)
                return
            try:
                on_result(tag, outcome)
            finally:
                raise KeyboardInterrupt(_SECOND_INTERRUPT)

        return interrupting

    return around


def _pool_threads(before: Iterable[threading.Thread]) -> list[threading.Thread]:
    """The threads of a thread pool started since *before* was listed."""
    known = set(before)
    return [
        thread
        for thread in threading.enumerate()
        if thread not in known and thread.name.startswith("ThreadPoolExecutor")
    ]


_WORKERS_GONE_AFTER_THE_RAISE_SECONDS = 10.0
"""How long a held process worker may outlive the raise of a second interrupt."""


def _interrupt_twice(
    exits: Exits, execution_id: str, mode: Literal["thread", "process"] = "thread"
) -> KeyboardInterrupt:
    """Run until a second interrupt abandons the settle, then check the held workers.

    The settle does not release ``seq_b`` and ``seq_c``, so it hands back ``seq_a``
    alone before the second interrupt. In thread mode, once the release exists,
    after the run has raised, the held threads write ``seq_b`` and ``seq_c``, which
    get no row, and end. In process mode the held workers are gone within
    :data:`_WORKERS_GONE_AFTER_THE_RAISE_SECONDS` of the raise, without a release.
    """
    exits.releases = False
    threads = threading.enumerate()
    with pytest.MonkeyPatch.context() as faults:
        _wrap_pools(faults, exits.marks, _interrupt_after_seq_a_in_the_settle(exits))
        _once_seq_c_is_dispatched(faults, exits, mode, _interrupt)
        with pytest.raises(KeyboardInterrupt) as raised:
            _ = exits.run(HoldsBC(), mode, execution_id)
        held = _pool_threads(threads)

    if mode == "process":
        workers = [int((exits.marks / sequence).read_text()) for sequence in _HELD]
        wait_for(
            lambda: not any(process_alive(worker) for worker in workers),
            _WORKERS_GONE_AFTER_THE_RAISE_SECONDS,
            "a held worker outlived the second interrupt",
        )
    else:
        assert len(held) >= len(_HELD)
        (exits.marks / "release").touch()
        for thread in held:
            thread.join(CHILD_TIMEOUT_SECONDS)
            assert not thread.is_alive()
    first = raised.value.__context__
    assert raised.value.args == (_SECOND_INTERRUPT,)
    assert isinstance(first, KeyboardInterrupt)
    assert first.args == ()
    return raised.value


def test_a_second_interrupt_abandons_the_settle_and_keeps_what_it_recorded(
    exits: Exits,
) -> None:
    _ = _interrupt_twice(exits, "TESTEXITTWICE")

    _check_exit(
        exits,
        "TESTEXITTWICE",
        status="cancelled",
        failed=set(),
        rows=frozenset({"seq_a"}),
        unrecorded=frozenset({"seq_b", "seq_c"}),
        mode="thread",
    )
    assert exits.settles == [True]


def test_a_second_interrupt_kills_the_process_workers_it_abandoned(
    exits: Exits,
) -> None:
    _ = _interrupt_twice(exits, "TESTEXITTWICEPROCESS", "process")

    _check_exit(
        exits,
        "TESTEXITTWICEPROCESS",
        status="cancelled",
        failed=set(),
        rows=frozenset({"seq_a"}),
        mode="process",
    )
    assert exits.settles == [True]


def test_a_failed_flush_after_an_abandoned_settle_is_a_note_on_the_second_interrupt(
    exits: Exits,
) -> None:
    # The first flush has no rows, since seq_a is handed back only in the settle,
    # and the second fails on seq_a's row.
    with pytest.MonkeyPatch.context() as faults:
        _fail_appends(faults, None)
        second = _interrupt_twice(exits, "TESTEXITTWICEFLUSH")

    (note,) = _notes(second, "index.csv is locked")
    assert "second flush" in note
    first = second.__context__
    assert first is not None
    assert _notes(first, "index.csv is locked") == []
    _check_exit(
        exits,
        "TESTEXITTWICEFLUSH",
        status="cancelled",
        failed=set(),
        rows=frozenset(),
        unrecorded=frozenset({"seq_a", "seq_b", "seq_c"}),
        mode="thread",
    )


# --- Signals to the process group -----------------------------------------------

_WAITING_AT_SEQ_D = "waiting-seq_d"
"""The mark a signalled child touches once it waits for the signal after the dispatch
of ``seq_c``."""

_CANCELLED_EXIT = 3
"""The exit status of a signalled child whose run raised :class:`Cancelled`."""

_INTERRUPTED_EXIT = -signal.SIGINT
"""The status of a child that an unhandled ``KeyboardInterrupt`` ended."""

_EXIT_AFTER_THE_SIGNAL_SECONDS = 10.0
"""How long a process-mode child may take to exit after an interrupt of its group."""

type Handler = Literal["install-handler", "default-handler"]
"""Whether a signalled child routes SIGTERM and SIGINT to its run's cancel token."""

_SIGNALLED_CHILD = """
import sys
from tests.core.pipeline.test_run_exits import run_signalled
run_signalled(sys.argv[1:])
"""


def _waiting_for_the_cancel(exits: Exits) -> Fault:
    def fault() -> None:
        (exits.marks / _WAITING_AT_SEQ_D).touch()
        wait_for(
            exits.token.is_cancelled,
            CHILD_TIMEOUT_SECONDS,
            "the run was never cancelled",
        )

    return fault


def _waiting_for_the_interrupt(exits: Exits) -> Fault:
    def fault() -> None:
        (exits.marks / _WAITING_AT_SEQ_D).touch()
        wait_for(lambda: False, CHILD_TIMEOUT_SECONDS, "the run was never interrupted")

    return fault


def run_signalled(arguments: Sequence[str]) -> None:
    """Run :class:`HoldsBC` in a child process until a signal to it or its group ends it.

    *arguments* are the dataset's manifest, the mode, the :data:`Handler` and the
    execution id. The pools mark their computed calls and their settles release the
    held entries, as in the tests of this module, and the run waits for the signal
    once the dispatch of ``seq_c`` returns. With ``install-handler`` the signals
    cancel the run's token, and the child exits with :data:`_CANCELLED_EXIT` once
    the run raises :class:`Cancelled`. Otherwise SIGINT raises
    ``KeyboardInterrupt``, which ends the child. The release is touched again on
    the way out, so a run that never settles leaves no thread holding.
    """
    manifest, mode_name, handler, execution_id = arguments
    modes: dict[str, Mode] = {candidate: candidate for candidate in _PARALLEL}
    mode = modes[mode_name]
    # A child started from a process that ignores SIGINT inherits the ignore, and
    # Python then raises no KeyboardInterrupt.
    _ = signal.signal(signal.SIGINT, signal.default_int_handler)
    exits = Exits(
        ds=Dataset(manifest_path=Path(manifest)).load(),
        marks=Path(os.environ[HOLD_DIRECTORY_VARIABLE]),
    )
    patches = pytest.MonkeyPatch()
    _record_settles(patches, exits)
    _wrap_pools(patches, exits.marks)
    if handler == "install-handler":
        install_signal_handler(exits.token)
        fault = _waiting_for_the_cancel(exits)
    else:
        fault = _waiting_for_the_interrupt(exits)
    _once_seq_c_is_dispatched(patches, exits, mode, fault)
    try:
        _ = exits.run(HoldsBC(), mode, execution_id)
    except Cancelled:
        sys.exit(_CANCELLED_EXIT)
    finally:
        (exits.marks / "release").touch()


def _signal_held_child(
    exits: Exits,
    mode: Mode,
    handler: Handler,
    signum: signal.Signals,
    execution_id: str,
    *,
    child_only: bool = False,
) -> SignalledGroup:
    """Run :func:`run_signalled` in its own process group, and signal the group.

    The signal goes out once ``seq_b`` and ``seq_c`` are held, ``seq_a`` is
    computed and the run waits after the dispatch of ``seq_c``. With *child_only* it
    goes to the child's process alone.
    """
    return signal_group_when(
        [
            sys.executable,
            "-c",
            _SIGNALLED_CHILD,
            str(exits.ds.manifest_path),
            mode,
            handler,
            execution_id,
        ],
        lambda: (
            exits.held()
            and _computed(exits.marks, "seq_a")
            and (exits.marks / _WAITING_AT_SEQ_D).exists()
        ),
        signum,
        extra_env={HOLD_DIRECTORY_VARIABLE: str(exits.marks)},
        errors=exits.marks.parent / f"{execution_id}.stderr",
        waiting="the child held seq_b and seq_c with seq_a computed",
        child_only=child_only,
    )


def test_a_cancel_of_the_group_records_the_outputs_of_the_threads_in_flight(
    exits: Exits,
) -> None:
    child = _signal_held_child(
        exits, "thread", "install-handler", signal.SIGTERM, "TESTEXITGROUPTERM"
    )

    assert child.status == _CANCELLED_EXIT, child.errors
    _check_exit(
        exits,
        "TESTEXITGROUPTERM",
        status="cancelled",
        failed=set(),
        rows=_COMPUTED_BEFORE_THE_EXIT,
        mode="thread",
    )


def test_an_interrupt_of_the_group_records_the_outputs_of_the_threads_in_flight(
    exits: Exits,
) -> None:
    child = _signal_held_child(
        exits, "thread", "default-handler", signal.SIGINT, "TESTEXITGROUPINT"
    )

    assert child.status == _INTERRUPTED_EXIT, child.errors
    _check_exit(
        exits,
        "TESTEXITGROUPINT",
        status="cancelled",
        failed=set(),
        rows=_COMPUTED_BEFORE_THE_EXIT,
        mode="thread",
    )


def test_an_interrupt_of_the_group_records_the_outputs_of_the_process_workers_in_flight(
    exits: Exits,
) -> None:
    # The interrupt reaches the workers too, and they ignore it: the run's process
    # raises it, its settle releases the held entries, and the workers finish them,
    # so seq_b and seq_c are recorded beside seq_a.
    child = _signal_held_child(
        exits, "process", "default-handler", signal.SIGINT, "TESTEXITGROUPINTPROCESS"
    )

    assert child.status == _INTERRUPTED_EXIT, child.errors
    assert child.seconds < _EXIT_AFTER_THE_SIGNAL_SECONDS
    workers = [int((exits.marks / sequence).read_text()) for sequence in _HELD]
    assert not any(process_alive(worker) for worker in workers)
    _check_exit(
        exits,
        "TESTEXITGROUPINTPROCESS",
        status="cancelled",
        failed=set(),
        rows=_COMPUTED_BEFORE_THE_EXIT,
        mode="process",
    )


_WORKERS_GONE_AFTER_THE_PARENT_SECONDS = 2.0
"""How long a process worker may outlive the process of its run once SIGKILL ends it."""


def test_a_sigkill_of_the_runs_process_ends_its_process_workers(
    exits: Exits,
) -> None:
    # seq_a's worker wrote its output, and its result waits to be drained before
    # the dispatch of seq_d, which waits until the kill. No row is flushed
    # before then, so seq_a's output is left without a row. seq_b and seq_c are
    # held in apply and write nothing.
    child = _signal_held_child(
        exits,
        "process",
        "default-handler",
        signal.SIGKILL,
        "TESTEXITPARENTKILLED",
        child_only=True,
    )
    try:
        assert child.status == -signal.SIGKILL, child.errors
        workers = [int((exits.marks / sequence).read_text()) for sequence in _HELD]
        wait_for(
            lambda: not any(process_alive(worker) for worker in workers),
            _WORKERS_GONE_AFTER_THE_PARENT_SECONDS,
            "a held worker outlived the process of its run",
        )
        (exits.marks / "release").touch()
        snapshot = read_run(run_log_dir(exits.ds.base_dir), "TESTEXITPARENTKILLED")
        assert snapshot is not None
        # A killed run writes no terminal status and no count of the entries
        # written, so only its outputs, its rows and the rerun are checked.
        _check_disk_and_rerun(
            exits,
            "TESTEXITPARENTKILLED",
            snapshot["run_id"],
            rows=frozenset(),
            unrecorded=frozenset({"seq_a"}),
            mode="process",
        )
    finally:
        with suppress(ProcessLookupError):
            os.killpg(child.pid, signal.SIGKILL)


# --- A worker death with a worker idle and one held -----------------------------

_DIED_EXIT = 4
"""The exit status of a child whose run raised :class:`WorkerProcessDied`."""

type DeathCase = Literal["dispatch", "settle"]
"""Where a child of :func:`run_until_a_worker_dies` meets the death: in the dispatch
of ``seq_d``, or in the settle of a cancel."""

_DYING_CHILD = """
import sys
from tests.core.pipeline.test_run_exits import run_until_a_worker_dies
run_until_a_worker_dies(sys.argv[1:])
"""


def _seq_c_failed(exits: Exits) -> Fault:
    def fault() -> None:
        wait_for(
            lambda: _computed(exits.marks, "seq_c"),
            CHILD_TIMEOUT_SECONDS,
            "seq_c never failed with the break",
        )

    return fault


def run_until_a_worker_dies(arguments: Sequence[str]) -> None:
    """Run :class:`HoldsBDiesOnSeqC` in a child process until ``seq_c``'s worker dies.

    *arguments* are the dataset's manifest, the :data:`DeathCase` and the execution
    id. Three process workers compute the run, the pools mark their computed calls,
    and the settles are recorded and release nothing, so ``seq_b`` stays held. With
    ``dispatch`` the run waits after the dispatch of ``seq_c`` until ``seq_c``'s
    call fails with the break, and the wait for a worker before ``seq_d`` meets the
    broken pool. The child exits with :data:`_DIED_EXIT` on
    :class:`WorkerProcessDied`. With ``settle`` the run is cancelled once ``seq_c``
    is dispatched, and the settle of the cancel meets the break. The child exits
    with :data:`_CANCELLED_EXIT` on :class:`Cancelled`.
    """
    manifest, case, execution_id = arguments
    exits = Exits(
        ds=Dataset(manifest_path=Path(manifest)).load(),
        marks=Path(os.environ[HOLD_DIRECTORY_VARIABLE]),
        releases=False,
    )
    patches = pytest.MonkeyPatch()
    _record_settles(patches, exits)
    _wrap_pools(patches, exits.marks)
    fault = _seq_c_failed(exits) if case == "dispatch" else _cancel(exits)
    _once_seq_c_is_dispatched(patches, exits, "process", fault, holding=False)
    try:
        _ = exits.run(HoldsBDiesOnSeqC(), "process", execution_id)
    except Cancelled:
        sys.exit(_CANCELLED_EXIT)
    except WorkerProcessDied:
        sys.exit(_DIED_EXIT)


def _written_pids(marks: Path) -> list[int]:
    """The process ids that the held entries wrote into *marks*."""
    texts = (path.read_text() for path in marks.iterdir() if path.is_file())
    return [int(text) for text in texts if text.isdigit()]


_EXIT_AFTER_THE_DEATH_SECONDS = 10.0
"""How long a child may take to exit after one of its process workers died."""


@pytest.mark.parametrize(
    ("case", "status", "exit_status"),
    [("dispatch", "failed", _DIED_EXIT), ("settle", "cancelled", _CANCELLED_EXIT)],
    ids=["dispatch", "settle"],
)
def test_a_worker_death_ends_the_runs_process_with_a_worker_idle_and_one_held(
    exits: Exits, case: DeathCase, status: str, exit_status: int
) -> None:
    # The held worker ignores the executor's SIGTERM, and the idle one waits on its
    # call queue: the pool kills both, so the child exits soon after the death.
    execution_id = f"TESTEXITBROKEN{case.upper()}"
    errors = exits.marks.parent / f"{execution_id}.stderr"
    try:
        child_status = run_child_to_files(
            [
                sys.executable,
                "-c",
                _DYING_CHILD,
                str(exits.ds.manifest_path),
                case,
                execution_id,
            ],
            errors=errors,
            timeout=CHILD_TIMEOUT_SECONDS,
        )
        exited = time.time()

        assert child_status == exit_status, errors.read_text()
        killed = (exits.marks / "killed").stat().st_mtime
        assert exited - killed < _EXIT_AFTER_THE_DEATH_SECONDS
        held = _written_pids(exits.marks)
        assert held
        assert not any(process_alive(worker) for worker in held)
        _check_exit(
            exits,
            execution_id,
            status=status,
            failed=set(),
            rows=frozenset({"seq_a"}),
            mode="process",
        )
    finally:
        kill_running(_written_pids(exits.marks))


# --- Who writes the index and the run-log ---------------------------------------


def _record_writer_threads(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[str]]:
    """Record the thread of each write of an index or the run-log, by the method called.

    The methods are the append of rows under an index's lock, the mark of a
    finished run and the run-log's append of one line.
    """
    threads: dict[str, list[str]] = {}
    for owner, name in (
        (IndexCSV, "_append_locked"),
        (IndexCSV, "mark_finished"),
        (JsonlRunLog, "_append"),
    ):
        calls = threads.setdefault(name, [])
        real: Callable[..., object] = getattr(owner, name)
        monkeypatch.setattr(owner, name, observed(real, _thread_noted(calls)))
    return threads


def _thread_noted(threads: list[str]) -> Callable[..., None]:
    """A watcher that appends the name of the thread that calls it to *threads*."""

    def note(*_args: object, **_kwargs: object) -> None:
        threads.append(threading.current_thread().name)

    return note


def test_no_thread_of_the_pool_writes_the_index_or_the_run_log(
    exits: Exits, monkeypatch: pytest.MonkeyPatch
) -> None:
    threads = _record_writer_threads(monkeypatch)

    result = exits.run(HoldsUntilReleased(), "thread", "TESTEXITWRITERS")

    assert result.run_id is not None
    assert sorted(_indexed(exits.ds, result.run_id)["sequence"]) == list(_SEQUENCES)
    assert all(threads.values()), threads
    from_the_pool = {
        name: [thread for thread in names if thread.startswith("ThreadPoolExecutor")]
        for name, names in threads.items()
    }
    assert from_the_pool == {name: [] for name in threads}


# --- What is not an exit, and exits with nothing in flight ----------------------


@pytest.mark.parametrize("mode", _PARALLEL)
def test_an_apply_error_is_an_entry_error_and_not_an_exit(
    exits: Exits, mode: Mode
) -> None:
    result = exits.run(LosesSeqB(), mode, "TESTEXITLOST")

    assert exits.settles == []
    assert [parse_entry_key(key)[1] for key in result.failed_entries] == ["seq_b"]
    errors = logged_events(exits.ds.base_dir, "TESTEXITLOST", "entry_error")
    assert [parse_entry_key(str(error["key"]))[1] for error in errors] == ["seq_b"]
    assert result.run_id is not None
    index = _indexed(exits.ds, result.run_id)
    assert sorted(index["sequence"]) == ["seq_a", "seq_c", "seq_d"]


class _EveryEntryIsTooLarge(RowCostEstimate):
    """Predicts every entry far over any memory limit, from the first."""

    @property
    def bytes_per_row(self) -> int | None:
        return 1

    def predicted_peak(
        self, rows_in_flight: int, rows_next: int, workers_after: int, margin: float
    ) -> int:
        return 2**62


def _no_free_memory(watch: MemoryWatch) -> float:
    return 0.01


def _under_the_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(MemoryWatch, "free_fraction", _no_free_memory)


def _over_the_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run_module, "RowCostEstimate", _EveryEntryIsTooLarge)


@pytest.mark.parametrize(
    ("setup", "options", "raised"),
    [
        (
            _under_the_floor,
            RunOptions(min_free_memory_fraction=0.5),
            MemoryFloorReached,
        ),
        (
            _over_the_limit,
            RunOptions(admission_margin=1.5),
            EntryOverMemoryLimit,
        ),
    ],
    ids=["memory-floor", "over-the-limit"],
)
def test_a_memory_exit_settles_a_pool_with_nothing_in_flight(
    exits: Exits,
    capsys: pytest.CaptureFixture[str],
    setup: Callable[[pytest.MonkeyPatch], None],
    options: RunOptions,
    raised: type[Exception],
) -> None:
    with pytest.MonkeyPatch.context() as faults:
        setup(faults)
        with pytest.raises(raised):
            _ = exits.run(HoldsUntilReleased(), "thread", "TESTEXITMEMORY", options)

    assert "waits for the entries in flight" not in capsys.readouterr().err
    # The rerun leaves the memory options out: they do not move the run id, and
    # the host it runs on decides whether it would meet them.
    _check_exit(
        exits,
        "TESTEXITMEMORY",
        status="failed",
        failed=set(),
        rows=frozenset(),
        mode="thread",
    )
    assert exits.settles == [True]
