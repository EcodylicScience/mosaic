"""Append-only JSONL run-log + attempt-id helpers -- a dependency-light leaf.

This is the one progress + attempt-status sink per job, plus the two attempt-id
helpers (:func:`now_iso`, :func:`new_execution_id`) it is keyed by. It replaces the
per-dataset SQLite ``.mosaic.db`` (execution-layer.md 3). Every unit of compute that
enters the Job Contract (:mod:`mosaic.core.pipeline.job`) -- a feature run, a tracking op
(``extract-frames`` / ``train-*`` / ``infer-*``), TREx, and future payloads (media
transcode, a SLEAP plugin) -- emits its lifecycle and coarse progress to **one** append-only
JSONL file named by its ``execution_id``, under ``<dataset_root>/.mosaic/runs/``.

**Why this module is a top-level leaf.** The readers (:func:`read_run` / :func:`read_runs`
/ :func:`read_run_progress`) are on a hot path for external tools -- mosaic-api's ledger
sweeper reduces every non-terminal run's log on every tick, and ``mosaic-queue`` workers read
them from inside the web/worker process. So this module imports **stdlib only** (no numpy /
pandas / pydantic) and lives at ``mosaic.runlog`` -- ``mosaic/__init__.py`` is empty, so
``import mosaic.runlog`` pulls nothing heavy. ``mosaic.core.pipeline.run_log`` re-exports
these names for back-compat (its package ``__init__`` eagerly imports the heavy pipeline).

Design properties:

* **Job-kind-agnostic** -- the job's ``kind`` is a *field* in the log, never part of the
  path; the filename is the ``execution_id`` alone. A new payload kind plugs in with zero
  change here.
* **NFS-safe** -- one writer, append-only, flush-per-line, so a remote reader sees a
  partial line only at the end of the file or where a write failed, and the reducer skips
  it. Exactly where SQLite/WAL failed. On a network filesystem (a CIFS share was seen) a
  zero-filled gap may also appear in mid-file, glued to the front of the next record.
  The reader strips it (:func:`run_log_lines`) and still reads that record.
* **Bounded + ephemeral** -- finite entries*2 (per-entry) or epochs*1 (per-epoch) lines
  (KB-MB), plus at most one ``alive`` line per :data:`LIVENESS_INTERVAL_SECONDS` of
  silence, and it stops growing at a terminal event. The durable truth lives elsewhere
  (``index.csv`` + parquet for results); the JSONL is scratch.

**Events.** ``started`` opens an attempt and ``run_id``, ``total``, ``tracks_variant``,
``phase``, ``heartbeat``, ``entry_start``, ``entry_end``, ``epoch`` and ``class_start``
report its progress. ``entry_end`` carries an entry's cost (:class:`EntryMeasure`) when
the writer measured it. ``resources`` carries a :class:`ResourceReport`: the workers,
the memory and the cost the attempt has measured so far. ``probe`` marks an attempt
that runs a probe or a named subset of a step. ``entry_error``, ``entries_written``,
``cache_hit``, ``frame_axis_mismatch``, ``frame_tail_short``, ``columns_dropped`` and
``rows_dropped`` report what it lost and left, ``alive`` its liveness, and
``finished``, ``failed`` and ``cancelled`` end it. The removal of a superseded run adds
``run_removing``, ``run_removed`` and ``run_kept``. A reader ignores an event it does
not know, and a field that a log lacks reads as empty, so an old log reduces as it
did.

Two event shapes ride the same sink: **per-entry** (features / tracking -- a count, since
several entries can be in flight) and **per-epoch** (training -- a monotonic cursor). Bracketing
lifecycle events (``started`` / ``run_id`` / ``total`` / ``heartbeat`` / ``alive`` /
``finished`` / ``failed`` / ``cancelled``) carry the attempt status the old ``runs`` table
held.

**Liveness.** A job's ``started`` event records the host's boot (``boot_id``), its pid
namespace (``pid_ns``) and the ``liveness_interval`` it promises. While the attempt runs,
a thread (:meth:`JsonlRunLog.start_liveness`) appends an ``alive`` event after each
interval in which nothing else was written, so a reader on any host can tell a silent
attempt from a dead one. ``alive`` is an event and not a status. A consumer judging an
attempt by its silence uses a time-to-live of at least twice
:data:`LIVENESS_INTERVAL_SECONDS`; mosaic's own rule
(``mosaic.core.pipeline.graph.liveness``) waits :data:`LIVENESS_STALE_INTERVALS`
intervals.

The reader helpers reduce a log back to the same dict shape the SQLite ``runs`` row exposed,
so the ``mosaic status`` / ``runs`` / ``cancel`` CLI contract is unchanged.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import os
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, TypedDict, cast

__all__ = [
    "ALIVE_EVENT",
    "ERROR_MESSAGE_LIMIT",
    "LIVENESS_INTERVAL_SECONDS",
    "LIVENESS_STALE_INTERVALS",
    "MTIME_TOLERANCE_SECONDS",
    "AttemptLiveness",
    "AttemptOutcome",
    "EntryMeasure",
    "EntryErrorPage",
    "EntryErrorRecord",
    "ErrorSummary",
    "JsonlRunLog",
    "ProbeRecord",
    "ProgressEventRecord",
    "ResourceReport",
    "REMOVAL_EVENTS",
    "RemovalRecord",
    "RemovalState",
    "RunLogPage",
    "RunLogSnapshot",
    "TimeIrregularPage",
    "TimeIrregularRecord",
    "TERMINAL_STATUSES",
    "attempt_outcome",
    "changed_at_or_after",
    "error_fields",
    "error_summary",
    "fold_run_records",
    "last_attempt_records",
    "list_runs",
    "new_execution_id",
    "now_iso",
    "parse_instant",
    "read_entry_error_keys",
    "read_entry_errors",
    "read_removal_records",
    "read_run",
    "read_run_records",
    "read_run_progress",
    "read_time_irregular",
    "record_count",
    "time_irregular_sentence",
    "record_exited_attempt",
    "record_flag",
    "record_seconds",
    "record_text",
    "read_runs",
    "read_runs_by_run_id",
    "reduce_run_log",
    "run_log_dir",
    "run_log_lines",
    "run_log_path",
    "written_since",
]


class RunLogSnapshot(TypedDict):
    """One attempt-status snapshot, folded from a run-log JSONL by ``reduce_run_log``.

    ``run_id`` / ``target`` / the timestamps are empty strings until the job emits
    them; ``progress_done`` / ``progress_total`` / ``pid`` are zero until reported.

    ``entries_failed`` counts the entities that raised while the attempt carried
    on. A non-zero count on an otherwise ``finished`` run is the partial outcome:
    the run did what it could and lost the rest. Consumers index this TypedDict
    with ``.get``, so the key is additive rather than breaking.

    ``entries_written`` counts the entries the attempt leaves holding a valid
    output row -- cache hits included, so a resumed run and a fresh one report the
    same number over the same scope. Note the asymmetry with the field above it:
    ``entries_failed`` *accumulates* one per event, while ``entries_written`` is
    last-write-wins, because a writer reports a total it already holds rather than
    an increment it would have to be trusted to add up.

    ``cache_hit`` is the attempt's own claim that it found the whole of its work
    already done. A job that never considered the question writes nothing and
    folds to ``False``: silence and a denial are not the same thing, but no
    consumer can act on the difference and a tri-state would make every reader
    special-case it.

    ``entries_frame_axis_mismatch`` counts the entries whose tool read a different
    number of frames than their media holds. It accumulates like
    ``entries_failed`` and, unlike it, says nothing about the attempt's success:
    the entry published, and everything computed inside its table is right. What
    may be wrong is only the correspondence between a ``frame`` in that table and
    a frame of the video, on which every consumer reading pixels depends.

    ``entries_frame_tail_short`` counts the entries whose tool read short of their
    media by no more than the tool is known to lose at the end of a file, as TRex
    does. It accumulates like ``entries_frame_axis_mismatch`` and is kept apart
    from it, because frames missing at the end of a file leave every frame of the
    table on the video.

    ``entries_columns_dropped`` counts the entries whose table was published
    without some of its columns: columns that the mapping from a media variant
    into source space cannot correct, or columns computed against the single
    frame rate of a joined TREx conversion. It accumulates like
    ``entries_frame_axis_mismatch`` and, like it, does not bear on the attempt's
    success. The entry published. The ``columns_dropped`` event names the columns
    that it lost.

    ``entries_time_irregular`` counts the ``time_irregular`` events: one for each
    entry whose method read a table with uneven time steps as evenly spaced, as the
    wavelets, ``temporal-stack`` and the Savitzky-Golay features do, and so
    misplaced what follows a gap. It accumulates like ``entries_columns_dropped``
    and, like it, does not bear on the attempt's success. The event names the
    feature, the largest step and the frames it lies between, and
    ``resample-tracks`` as the remedy. A reader that predates the event folds the
    log as before.

    ``tracks_variant`` names the tracks recipes the attempt *read*, comma-joined
    and sorted, empty for a run that reads none. Never what an op *produced* --
    those are different relations and one key cannot hold both, which is what
    would make a downstream column unqueryable.

    ``workers`` and ``parallel_mode`` are how many entries the attempt computed at
    once and how, as the last ``resources`` event recorded them. ``peak_rss_bytes``
    is the largest resident memory of the attempt's process tree that the attempt
    measured, ``min_available_bytes`` the least memory the host had available and
    ``total_memory_bytes`` the host's physical memory. They are last-write-wins like
    ``entries_written``, because the writer reports extremes it already holds. A
    process-mode worker counts toward the peak, and a page that several processes
    map counts once for each.

    Zero, ``False`` and ``""`` are indistinguishable from "not reported", the same
    convention as the tracks index's blank ``n_keypoints`` cell meaning *unknown*
    rather than zero.

    ``probe`` is whether the attempt wrote a ``probe`` event of kind ``probe`` since
    its last ``started``: it runs a few of a step's entries to measure them, and
    its status view can mark it. A ``probe`` of kind ``named`` (a chosen subset of
    a step) leaves it ``False``. The rest of a ``resources`` event and of an
    ``entry_end`` event (:class:`ResourceReport`, :class:`EntryMeasure`) stays out
    of the snapshot. :func:`mosaic.run_cost.read_run_cost` reads it.

    A process that resumes an attempt whose process died appends a second
    ``started`` event to its run-log. The counters, the progress, ``cache_hit``
    and ``tracks_variant`` then report the events after the last ``started``, and
    the entries that the dead process lost are not counted twice.

    ``boot_id``, ``pid_ns`` and ``liveness_interval`` are what the last
    ``started`` event recorded of the process's host: its boot, its pid namespace
    and the seconds between the ``alive`` events it promises (0 for a writer that
    promises none, as every writer before them). ``gpu_card`` is the card the
    process was leased, ``<index>:<uuid>``, and ``""`` for a process that was leased
    none or wrote its ``started`` before the field existed. It names a card by its
    index and uuid and is apart from the fit environment's ``gpu``, the card's model
    name. ``phase`` is the last phase the attempt declared since that ``started``,
    ``""`` before one.

    The run-log is the only report a queued attempt makes, because a queue starts
    ``mosaic run`` with its standard output and standard error discarded. A writer
    emits these events before the terminal event. mosaic-api's sweeper stops
    reading a log once its status is terminal, and :class:`JsonlRunLog` drops a
    write made after it is closed.
    """

    execution_id: str
    kind: str
    target: str
    run_id: str
    status: str
    owner: str
    host: str
    pid: int
    created_at: str
    started_at: str
    heartbeat_at: str
    finished_at: str
    error_json: str
    progress_done: int
    progress_total: int
    entries_failed: int
    entries_written: int
    entries_frame_axis_mismatch: int
    entries_frame_tail_short: int
    entries_columns_dropped: int
    entries_time_irregular: int
    cache_hit: bool
    tracks_variant: str
    boot_id: str
    pid_ns: str
    liveness_interval: float
    gpu_card: str
    phase: str
    workers: int
    parallel_mode: str
    peak_rss_bytes: int
    min_available_bytes: int
    total_memory_bytes: int
    probe: bool


@dataclass(frozen=True, slots=True)
class ResourceReport:
    """What an attempt has measured of its workers, memory and cost so far.

    One ``resources`` event carries every field. The five without a default are the
    ones a snapshot folds. The rest are read by :func:`mosaic.run_cost.read_run_cost`
    and are empty (0, ``0.0``, ``False`` or ``""``) until the writer measures them,
    so a log written without them reads as empty.

    Attributes:
        workers: How many entries the attempt computes at once.
        parallel_mode: How, ``""`` for one at a time.
        peak_rss_bytes: The largest resident memory of its process tree.
        min_available_bytes: The least memory the host had available.
        total_memory_bytes: The host's physical memory.
        feature: The feature's registered name.
        feature_version: The feature's version.
        process_capable: Whether the feature may run in worker processes: it is
            parallelizable and sets no scope.
        cpu_seconds: The CPU time, user and system, of the process tree since
            measuring began.
        wall_seconds: The wall time since measuring began.
        memory_limit_bytes: The memory limit of the container, or of the host
            where there is none.
        memory_limit_source: Where that limit came from (``cgroup_v2``,
            ``cgroup_v1`` or ``host``).
        limit_peak_bytes: The most memory the limit's accounting reported in use.
        base_rss_bytes: The tree's resident memory before the first entry loads.
        fit_seconds: The wall time of the fit.
        fit_peak_rss_bytes: The largest resident memory of the tree during the fit.
        peak_bytes_per_row: The run's estimate of an entry's peak memory per input
            row, 0 before it has one.
        admission_margin: The factor the admission gate multiplies a prediction by,
            0 when the gate is off.
        admission_waits: How many times the gate held an entry back.
        fit_entries: The entries of the stream that the fit was offered.
        fit_input_rows: Their input rows, counted as the footer rows of each
            entry's input files. A fitter that fits from pinned artifacts, such
            as a params-level fitter, ignores that stream, so for it both figures
            describe the stream it was offered and not what it read.
        worker_base_rss_bytes: One process worker's own memory, 0 when not
            measured.
        apply_wall_seconds: The wall time from the first entry's load start to the
            last output written.
    """

    workers: int
    parallel_mode: str
    peak_rss_bytes: int
    min_available_bytes: int
    total_memory_bytes: int
    feature: str = ""
    feature_version: str = ""
    process_capable: bool = False
    cpu_seconds: float = 0.0
    wall_seconds: float = 0.0
    memory_limit_bytes: int = 0
    memory_limit_source: str = ""
    limit_peak_bytes: int = 0
    base_rss_bytes: int = 0
    fit_seconds: float = 0.0
    fit_peak_rss_bytes: int = 0
    peak_bytes_per_row: int = 0
    admission_margin: float = 0.0
    admission_waits: int = 0
    fit_entries: int = 0
    fit_input_rows: int = 0
    worker_base_rss_bytes: int = 0
    apply_wall_seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class EntryMeasure:
    """What one computed entry cost, written with its ``entry_end`` event.

    Attributes:
        wall_seconds: From the start of its input load to its output written.
        apply_seconds: From its dispatch to the end of its apply.
        input_rows: The rows of its input files.
        in_flight: The most entries loaded or computing at once during its span,
            itself included. One means it ran alone, so its ``cpu_seconds`` are its
            own.
        cpu_seconds: The CPU time of the process tree between its load start and
            its output written.
    """

    wall_seconds: float
    apply_seconds: float
    input_rows: int
    in_flight: int
    cpu_seconds: float


@dataclass(frozen=True, slots=True)
class ProbeRecord:
    """An attempt that runs only some entries of a step, written as its ``probe`` event.

    Attributes:
        request_id: The request the step belongs to.
        step: The step.
        entries: The entries the attempt computes, as ``group:sequence`` tokens.
        kind: ``probe`` for the largest entries of the step's attempt, ``named``
            for entries a caller chose.
    """

    request_id: str
    step: str
    entries: tuple[str, ...]
    kind: Literal["probe", "named"]

    def write(self, log: JsonlRunLog) -> None:
        """Append this record to *log* as its ``probe`` event."""
        log.probe(
            request_id=self.request_id,
            step=self.step,
            entries=self.entries,
            kind=self.kind,
        )


ERROR_MESSAGE_LIMIT: Final = 2000
"""The most characters of a recorded message that a report quotes."""


class ErrorSummary(TypedDict):
    """The exception type and message of an ``error_json`` blob, read by ``error_summary``.

    Attributes:
        type: The exception class name, ``""`` for a refusal or a blob that names
            none.
        message: The sentence a person reads, at most
            :data:`ERROR_MESSAGE_LIMIT` characters.
    """

    type: str
    message: str


class EntryErrorRecord(TypedDict):
    """One entity that failed while its attempt continued, read by ``read_entry_errors``.

    Attributes:
        key: The entity, as the attempt named it.
        error_json: The blob the attempt recorded, shaped as an attempt-level
            ``error_json``: a refusal's reason, step and message, or an exception's
            type, message and traceback.
        error_type: The exception class name in the blob, ``""`` for a refusal.
        message: The message in the blob (:func:`error_summary`).
        recorded_at: When the attempt recorded the failure.
    """

    key: str
    error_json: str
    error_type: str
    message: str
    recorded_at: str


class EntryErrorPage(TypedDict):
    """The first entities that one attempt lost, read by ``read_entry_errors``.

    Attributes:
        items: The records, in the order the attempt wrote them.
        total: How many entities the attempt lost, ``entries_failed`` of its
            snapshot.
        has_more: Whether *total* exceeds the records listed.
    """

    items: list[EntryErrorRecord]
    total: int
    has_more: bool


class ProgressEventRecord(TypedDict):
    """One progress event of an attempt, read by ``read_run_progress``.

    Attributes:
        step_type: ``entry``, ``entry_start``, ``epoch``, ``class`` or ``phase``.
        step_index: The entry index, epoch or class index, 0 for a phase.
        step_total: The count that *step_index* runs to, 0 for a phase.
        metrics: An epoch's metrics, empty for every other event.
        message: The entry key, class name, or ``<phase>: <message>``.
        timestamp: When the attempt recorded the event.
    """

    step_type: str
    step_index: int
    step_total: int
    metrics: dict[str, float]
    message: str
    timestamp: str


class TimeIrregularRecord(TypedDict):
    """One uneven time axis that a method read as even, read by ``read_time_irregular``.

    Attributes:
        key: The entry, as the attempt named it.
        who: The feature that read the table as evenly spaced.
        largest_step: The longest step in seconds.
        first_frame: The frame before that step.
        last_frame: The frame after it.
        median_step: The median seconds per frame step.
        rate: The frames per second the method used.
        message: The sentence a person reads, naming the gap and ``resample-tracks``.
        recorded_at: When the attempt recorded the event.
    """

    key: str
    who: str
    largest_step: float
    first_frame: int
    last_frame: int
    median_step: float
    rate: float
    message: str
    recorded_at: str


class TimeIrregularPage(TypedDict):
    """The first uneven-time events of one attempt, read by ``read_time_irregular``.

    Attributes:
        items: The records, in the order the attempt wrote them.
        total: How many events the attempt recorded.
        has_more: Whether *total* exceeds the records listed.
    """

    items: list[TimeIrregularRecord]
    total: int
    has_more: bool


class RunLogPage(TypedDict):
    """One page of attempt snapshots, read by ``list_runs``.

    Attributes:
        items: The snapshots, newest first.
        next_after: The ``after`` that reads the next page, ``None`` on the last
            page. It is the execution id of the last run-log that the page read.
    """

    items: list[RunLogSnapshot]
    next_after: str | None


LIVENESS_INTERVAL_SECONDS: Final = 60.0
"""The most seconds a running job leaves its run-log without a line.

A job's liveness thread appends an ``alive`` event after this long without any
other write, so a job that writes events often writes no ``alive`` at all.
"""

LIVENESS_STALE_INTERVALS: Final = 10
"""How many declared intervals of silence prove that an attempt's process is gone."""

ALIVE_EVENT: Final = "alive"
"""The event that a job's liveness thread appends after an interval of silence."""


type AttemptLiveness = Literal["alive", "unknown", "dead", "ended"]
"""Whether an attempt's process may still be running, as a reader can tell.

``ended`` is an attempt with a terminal event. ``dead`` is one whose process is
provably gone: its pid exited, its host rebooted, or it declared a liveness
interval and fell silent for :data:`LIVENESS_STALE_INTERVALS` of them. ``alive``
is one whose process is provably running, or whose run-log is current. ``unknown``
is one that this reader cannot check, as a process on another host whose run-log
declares no interval. The rule is ``mosaic.core.pipeline.graph.liveness``.
"""


type AttemptOutcome = Literal[
    "pending", "running", "finished", "partial", "failed", "cancelled", "bound"
]
"""How an attempt stands, as a report names it.

``pending`` is an attempt with no run-log yet. ``partial`` is a ``finished``
attempt that lost entries. ``bound`` is a step of a request that binds it to a
run, which is never executed. All three are reporting words and none is a run-log
status, so none joins :data:`TERMINAL_STATUSES`, which mosaic-api's sweeper reaps.
"""


def attempt_outcome(status: str, entries_failed: int) -> AttemptOutcome:
    """Return the outcome that a run-log *status* and its failed-entry count report.

    Args:
        status: The run-log status, or ``""`` for an attempt that has written none.
        entries_failed: How many entries the attempt lost while it continued.

    Returns:
        ``partial`` for a finished attempt that lost entries, ``pending`` for one
        with no status, and the status otherwise. A status outside the run-log's
        vocabulary reads ``running``, as ``reduce_run_log`` folds one.
    """
    if status == "finished":
        return "partial" if entries_failed else "finished"
    if status == "failed" or status == "cancelled" or status == "running":
        return status
    return "pending" if not status else "running"


# ---------------------------------------------------------------------------
# Attempt-id helpers (stdlib-only; the run-log is keyed by execution_id)
# ---------------------------------------------------------------------------


def now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def parse_instant(stamp: str) -> datetime.datetime | None:
    """Return the ISO-8601 instant *stamp*, read as UTC when it names no zone.

    A stamp that does not parse returns ``None``.
    """
    try:
        moment = datetime.datetime.fromisoformat(stamp)
    except ValueError:
        return None
    if moment.tzinfo is None:
        return moment.replace(tzinfo=datetime.timezone.utc)
    return moment


# Crockford base32 (excludes I, L, O, U to avoid ambiguity) -- the ULID alphabet.
_CROCKFORD_B32 = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def new_execution_id() -> str:
    """Return a fresh ULID identifying one execution *attempt*.

    A ULID is a 128-bit id: a 48-bit millisecond timestamp in the high bits
    followed by 80 bits of randomness, rendered as 26 Crockford-base32
    characters. The timestamp prefix makes ids lexicographically sortable by
    creation time (unlike ``uuid4``), which the run-attempt ledger and the
    downstream Dolt ``Run`` primary key both rely on.

    This is the identity of an *attempt* and is intentionally
    non-deterministic. It is never hashed into a ``run_id`` and never written
    into a feature/model output -- only into the attempt's JSONL run-log filename
    and its ``job_id`` progress events. Determinism of ``run_id`` is therefore
    unaffected.
    """
    timestamp_ms = int(time.time() * 1000) & ((1 << 48) - 1)
    randomness = int.from_bytes(os.urandom(10), "big")  # 80 bits
    value = (timestamp_ms << 80) | randomness
    chars = [""] * 26
    for i in range(25, -1, -1):
        chars[i] = _CROCKFORD_B32[value & 0x1F]
        value >>= 5
    return "".join(chars)


# ---------------------------------------------------------------------------
# Path helpers (plain Path in, no Dataset dependency)
# ---------------------------------------------------------------------------


def run_log_dir(base_dir: Path | str) -> Path:
    """Return the run-log directory for a dataset (``<base_dir>/.mosaic/runs``)."""
    return Path(base_dir) / ".mosaic" / "runs"


def run_log_path(base_dir: Path | str, execution_id: str) -> Path:
    """Return the JSONL path for one attempt (``.../<execution_id>.jsonl``)."""
    return run_log_dir(base_dir) / f"{execution_id}.jsonl"


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


class JsonlRunLog:
    """Append-only JSONL sink for one job attempt.

    Implements the :class:`~mosaic.core.pipeline.progress.ProgressCallback` protocol
    (``on_entry_*`` / ``on_epoch_end`` / ``on_class_start`` / ``on_phase``) **and** carries
    the attempt-lifecycle methods (``started`` / ``set_run_id`` / ``set_total`` /
    ``heartbeat`` / ``finished`` / ``failed`` / ``cancelled``). One file, one writer, flush
    after every line, so a concurrent reader sees a partial line only at the end of the
    file or where a write failed. The first line after a failed write, or after a file
    that ends mid-line, begins with a newline, so the lines after the partial one decode.

    :meth:`start_liveness` adds a daemon thread that appends ``alive`` after each
    interval of silence. Writes from it and from the job go through one lock, and
    nothing is written after a terminal event or after :meth:`close`.

    Parameters
    ----------
    path : Path
        The ``<execution_id>.jsonl`` file to append to (created if absent).
    execution_id : str
        The attempt this log belongs to (also the file stem).
    """

    def __init__(self, path: Path, execution_id: str) -> None:
        self.path = Path(path)
        self.execution_id = execution_id
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Text append mode; each ``_emit`` writes one full line and flushes.
        self._file: Any = open(self.path, "a", encoding="utf-8")
        self._lock = threading.Lock()
        self._last_write: float = time.monotonic()
        self._ended: bool = False
        # Whether the file may end in a partial line: an earlier writer's write
        # failed, or this one's last write raised.
        self._torn: bool = _ends_mid_line(self.path)
        self._stop_liveness = threading.Event()
        self._liveness: threading.Thread | None = None
        self._progress_writes: int = 0
        self._last_progress: tuple[int, int] | None = None

    @property
    def progress_writes(self) -> int:
        """Count the ``heartbeat``, ``entry_end`` and ``epoch`` lines written so far.

        Each of these events sets the progress count that :func:`reduce_run_log`
        folds. A caller compares two readings to learn whether a count was written
        between them.
        """
        return self._progress_writes

    @property
    def last_progress(self) -> tuple[int, int] | None:
        """The ``(done, total)`` that the last progress line folds to, or None."""
        return self._last_progress

    def _emit(self, event: str, **fields: Any) -> None:
        _ = self._write(event, fields)

    def _write(
        self,
        event: str,
        fields: dict[str, Any],
        *,
        end: bool = False,
        progress: tuple[int, int] | None = None,
    ) -> bool:
        """Append one event line, returning ``False`` when the log is closed.

        Args:
            event: The event kind.
            fields: The event's fields.
            end: Whether the event is terminal. No ``alive`` line follows one.
            progress: The ``(done, total)`` that the event folds to, for an
                event that sets the progress count. A written one increments
                :attr:`progress_writes` and becomes :attr:`last_progress`.
        """
        if self._file is None:
            return False
        record = {"t": now_iso(), "ev": event, **fields}
        line = json.dumps(record, default=str) + "\n"
        with self._lock:
            if self._file is None:
                return False
            self._append(line)
            if end:
                self._ended = True
            if progress is not None:
                self._progress_writes += 1
                self._last_progress = progress
        return True

    def _write_alive(self) -> bool:
        """Append an ``alive`` event, returning ``False`` once the attempt has ended."""
        line = json.dumps({"t": now_iso(), "ev": ALIVE_EVENT}) + "\n"
        with self._lock:
            if self._file is None or self._ended:
                return False
            self._append(line)
        return True

    def _append(self, line: str) -> None:
        """Write and flush *line*, with the lock held.

        After a write that raised, or on a file that ends mid-line, *line* begins
        with a newline, which ends the partial line. The reader skips the partial
        line and the blank one and decodes *line*.
        """
        if self._torn:
            line = "\n" + line
        self._torn = True
        self._file.write(line)
        self._file.flush()
        self._torn = False
        self._last_write = time.monotonic()

    def start_liveness(self, interval: float) -> None:
        """Append ``alive`` after each *interval* seconds without another write.

        A daemon thread waits out the time left of the interval since the last
        line, so two lines are never further apart than *interval* while the job
        runs. It stops at the terminal event and at :meth:`close`. A failed
        ``alive`` write is swallowed and retried after the next interval:
        liveness is a report, a failed report must not fail the job, and one
        transient write error must not make a running job read as dead. A
        second call, or an *interval* of 0 or below, starts nothing.

        Args:
            interval: The seconds of silence after which ``alive`` is written,
                normally :data:`LIVENESS_INTERVAL_SECONDS`, the value the
                ``started`` event declares.
        """
        if interval <= 0 or self._liveness is not None:
            return
        self._liveness = threading.Thread(
            target=self._keep_alive,
            args=(interval,),
            name=f"runlog-alive-{self.execution_id}",
            daemon=True,
        )
        self._liveness.start()

    def _keep_alive(self, interval: float) -> None:
        """Write ``alive`` whenever *interval* seconds pass without a line."""
        wait = interval
        while not self._stop_liveness.wait(wait):
            silent = time.monotonic() - self._last_write
            if silent < interval:
                wait = interval - silent
                continue
            wait = interval
            try:
                if not self._write_alive():
                    return
            except Exception:
                continue

    # -- lifecycle ----------------------------------------------------------

    def started(
        self,
        *,
        kind: str,
        target: str,
        owner: str = "",
        host: str = "",
        pid: int = 0,
        created_at: str | None = None,
        boot_id: str = "",
        pid_ns: str = "",
        liveness_interval: float = 0.0,
        gpu_card: str = "",
    ) -> None:
        """Bracket the start of the attempt (kind/target/owner/host/pid).

        *boot_id*, *pid_ns* and *liveness_interval* describe the process for a
        reader judging whether it still runs. *gpu_card* names the card the
        process was leased, ``<index>:<uuid>``. Each is written only when set, so
        a record of a process that runs nothing (a refusal, a cache hit) reads as
        before.
        """
        process: dict[str, Any] = {}
        if boot_id:
            process["boot_id"] = boot_id
        if pid_ns:
            process["pid_ns"] = pid_ns
        if liveness_interval > 0:
            process["liveness_interval"] = liveness_interval
        if gpu_card:
            process["gpu_card"] = gpu_card
        self._emit(
            "started",
            kind=kind,
            target=target,
            owner=owner,
            host=host,
            pid=pid,
            created_at=created_at or now_iso(),
            **process,
        )

    def set_run_id(self, run_id: str) -> None:
        """Record the content-addressed ``run_id`` once the job computes it."""
        self._emit("run_id", run_id=run_id)

    def set_total(self, total: int) -> None:
        """Declare the total number of entries/epochs (progress denominator)."""
        self._emit("total", total=total)

    def heartbeat(self, done: int, total: int) -> None:
        """Refresh liveness and the coarse completed count."""
        _ = self._write(
            "heartbeat", {"done": done, "total": total}, progress=(done, total)
        )

    def finished(self) -> None:
        """Terminal: the attempt completed successfully."""
        _ = self._write("finished", {}, end=True)

    def failed(self, error_json: str = "") -> None:
        """Terminal: the attempt raised (``error_json`` is a captured-error blob)."""
        _ = self._write("failed", {"error": error_json}, end=True)

    def cancelled(self) -> None:
        """Terminal: the attempt was cooperatively cancelled."""
        _ = self._write("cancelled", {}, end=True)

    def entry_failed(self, key: str, error_json: str = "") -> None:
        """One entity failed; the attempt continues.

        Deliberately NOT a terminal event and deliberately not a new status. A
        status would have to join :data:`TERMINAL_STATUSES` to be folded, and
        mosaic-api's sweeper treats everything in that set as terminal -- so a
        live run would be reaped mid-flight. An ordinary event kind costs nothing
        instead: ``reduce_run_log`` is an if/elif fold in which an unrecognised
        ``ev`` advances liveness and changes nothing else, so a reader that
        predates this event still folds a log containing it correctly.

        Not part of the progress protocol either. ``ProgressCallback`` is a
        runtime-checkable Protocol with four implementations and a documented
        extension point, so a required sixth method would break any backend
        outside this repository. The failure goes to the run-log, which is the
        channel that survives the queue sending the child's stderr to DEVNULL.
        """
        self._emit("entry_error", key=key, error=error_json)

    def frame_axis_mismatch(self, key: str, *, read: int, media: int) -> None:
        """One entry's tool read *read* frames of media that holds *media*.

        Recorded, never raised. The table is real and its own contents are
        sound -- dropout counts, neighbour distances, body size and coverage are
        all computed *within* the frame axis and do not care which video frame a
        row corresponds to. What breaks is anything that takes a ``frame`` from
        the table and reads a pixel at it, and that is a reason to tell someone,
        not a reason to throw the table away.

        The numbers travel with the event because the gap is the whole content of
        the report: "read 1788 of 1800" says how far a late crop may be off, where
        a bare flag says only that something is wrong somewhere.

        An ordinary event kind, for the reason :meth:`entry_failed` gives at
        length -- an unrecognised ``ev`` falls off ``reduce_run_log``'s if/elif
        chain, advancing liveness and changing nothing else -- and emphatically
        not a status: this attempt is `finished`, and a run whose every entry
        mismatched is still a run that did what it was asked.
        """
        self._emit("frame_axis_mismatch", key=key, read=read, media=media)

    def frame_tail_short(self, key: str, *, read: int, media: int) -> None:
        """One entry's tool read *read* frames of *media*, short by a known tail loss.

        The shortfall is within what the producer's tool is known to lose at the
        end of the file it read (``TrackingRoot.tail_loss``), so it is recorded
        apart from :meth:`frame_axis_mismatch`. The event records both numbers,
        because a count cannot show that the missing frames were at the end and
        a reader may want to check. An ordinary event kind, like that one, and
        not a status.
        """
        self._emit("frame_tail_short", key=key, read=read, media=media)

    def columns_dropped(self, key: str, columns: Sequence[str]) -> None:
        """Record that one entry's table was published without *columns*.

        Mapping a table from a media variant into source space drops each column
        that the variant changed and the table cannot recompute: a timestamp
        minted at the variant's rate, a speed computed against a relabeled rate,
        a distance to the border of a cropped image. Retiming a joined TREx
        conversion on its clips' measured rates drops the same kinds of column.
        The table is still published, and the event reports the loss without
        failing the entry. It contains the names, because a reader needs them to
        know the missing columns.

        It is an ordinary event kind, for the reason that :meth:`entry_failed`
        gives. A reader that predates it folds a log with this event correctly.
        """
        self._emit("columns_dropped", key=key, columns=list(columns))

    def rows_dropped(self, key: str, counts: Mapping[str, int]) -> None:
        """Record that one entry's table was published without some of its rows.

        ``assign-arenas`` drops a row outside every arena, inside an excluded
        rect or above its arena's capacity, and counts the rows of each reason.
        The table is still published, and the event reports the loss without
        failing the entry. It is an ordinary event kind, for the reason that
        :meth:`entry_failed` gives.
        """
        self._emit("rows_dropped", key=key, counts=dict(counts))

    def time_irregular(
        self,
        key: str,
        *,
        who: str,
        largest_step: float,
        first_frame: int,
        last_frame: int,
        median_step: float,
        rate: float,
    ) -> None:
        """Record that a method read one entry's uneven time steps as even ones.

        The entry published, and its table is a real one. What is off is the
        position in time of every sample after a gap, in a method that reads its
        rows as evenly spaced. The numbers travel with the event because the gap is
        the whole content of the report. It is an ordinary event kind, for the
        reason :meth:`entry_failed` gives, and never a status.
        """
        self._emit(
            "time_irregular",
            key=key,
            who=who,
            largest_step=float(largest_step),
            first_frame=int(first_frame),
            last_frame=int(last_frame),
            median_step=float(median_step),
            rate=float(rate),
        )

    def entries_written(self, count: int) -> None:
        """How many entries this attempt leaves holding a valid output row.

        Cache hits are counted. The question a reader asks is "what does this
        run's scope hold now", and a resumed run that recomputed two of five
        entries holds five -- so a fresh run and a resumed one over one scope
        report one number. That is why the count comes from the single index-row
        choke point rather than from work done.

        Deliberately not derivable from ``progress_done``: a trainer overwrites
        that with an epoch cursor, a feature writing one global artifact never
        advances it at all, and neither sees the entries a manifest dropped for
        having no table.

        An ordinary event kind, for the reason :meth:`entry_failed` gives at
        length: an ``ev`` a reader does not recognise falls off the end of
        ``reduce_run_log``'s if/elif chain, advancing liveness and changing
        nothing else.
        """
        self._emit("entries_written", entries_written=count)

    def cache_hit(self) -> None:
        """This attempt found the whole of its work already done.

        A presence event carrying no payload, like ``finished`` and
        ``cancelled``: a writer either makes the claim or stays silent, and the
        fold defaults it to ``False`` -- so a job that never asked the question is
        not recorded as having answered no.

        Separate from :meth:`entries_written` rather than one combined event
        because the two have separate knowers. An op reusing a trained model knows
        it was a cache hit and has no entry count at all; a partial feature run has
        a count and no claim to make.
        """
        self._emit("cache_hit")

    def tracks_variant(self, variants: Sequence[str]) -> None:
        """Which tracks recipes this attempt read.

        Sorted, deduplicated and joined into one cell, so one set of variants has
        one spelling however the caller ordered them. That is the
        ``consumed_source_roots`` rule from the tracks index, re-spelled rather
        than imported: ``encode_source_roots`` sits beside pandas and this module
        is stdlib-only on purpose (see the module docstring).

        Emitted early, beside :meth:`set_run_id`, rather than at the end. Which
        tables a run was reading is exactly what is wanted from an attempt that
        was killed halfway, and a fact written at the end is one such an attempt
        never records.
        """
        self._emit(
            "tracks_variant",
            tracks_variant=",".join(sorted({v for v in variants if v})),
        )

    def resources(self, report: ResourceReport) -> None:
        """Record how many entries the attempt computes at once, its memory and its cost.

        Each event carries the extremes measured so far, so the last one before the
        terminal event is the attempt's peak. A writer emits it repeatedly while the
        attempt runs, because a reader polling a long step needs the memory now, and
        a figure written only at the end is one that an attempt the host killed
        never records.
        """
        self._emit("resources", **dataclasses.asdict(report))

    def probe(
        self,
        *,
        request_id: str,
        step: str,
        entries: Sequence[str],
        kind: Literal["probe", "named"],
    ) -> None:
        """Record that this attempt runs only some entries of a step.

        Written right after ``started``. A ``probe`` measures the largest entries
        of a step to size the rest, and a ``named`` attempt computes the entries a
        caller chose. Neither is the step's own attempt, and status reads only that
        one. An ordinary event kind, for the reason :meth:`entry_failed` gives.

        Args:
            request_id: The request the step belongs to.
            step: The step.
            entries: The entries the attempt computes, as ``group:sequence`` tokens.
            kind: ``probe`` or ``named``.
        """
        self._emit(
            "probe",
            request_id=request_id,
            step=step,
            entries=list(entries),
            kind=kind,
        )

    # -- removal of a superseded run -----------------------------------------

    def run_removing(
        self,
        *,
        feature: str,
        run_id: str,
        reason: str,
        superseded_by: str,
        params: Mapping[str, Any],
        retired: str = "",
    ) -> None:
        """Record the intent to remove one feature run, before anything is removed.

        *params* is the run's whole ``params.json`` document, which rebuilds the
        feature and its run id, so the run can be computed again from this log
        after its directory is gone. An intent without a :meth:`run_removed`
        after it reads as interrupted (:func:`read_removal_records`): a removal
        never happens without its record, and a record can outlive a removal that
        did not finish.

        An ordinary event kind, for the reason :meth:`entry_failed` gives.

        Args:
            feature: The run's storage name under ``features/``.
            run_id: The run.
            reason: Why it is removed.
            superseded_by: The run that replaces it.
            params: Its recorded ``params.json``.
            retired: The digest of the retired recipe whose request wrote it,
                ``""`` for an ordinary superseded run. The event carries the
                key only when it is set, so a log without a retirement is
                what an older mosaic wrote.
        """
        extra = {"retired": retired} if retired else {}
        self._emit(
            "run_removing",
            feature=feature,
            run_id=run_id,
            reason=reason,
            superseded_by=superseded_by,
            params=dict(params),
            **extra,
        )

    def run_removed(
        self, *, feature: str, run_id: str, bytes: int, files: int, rows: int
    ) -> None:
        """Record that one feature run's index rows and files are gone.

        Args:
            feature: The run's storage name.
            run_id: The run.
            bytes: The bytes its directory held.
            files: The files its directory held.
            rows: The index rows dropped.
        """
        self._emit(
            "run_removed",
            feature=feature,
            run_id=run_id,
            bytes=bytes,
            files=files,
            rows=rows,
        )

    def run_kept(self, *, feature: str, run_id: str, reason: str, detail: str) -> None:
        """Record that a run approved for removal was kept when the removal ran.

        Args:
            feature: The run's storage name.
            run_id: The run.
            reason: Why, such as a run read since the plan or one already gone.
            detail: What made it so, in a sentence.
        """
        self._emit(
            "run_kept", feature=feature, run_id=run_id, reason=reason, detail=detail
        )

    # -- progress protocol --------------------------------------------------

    def on_entry_start(self, index: int, total: int, key: str) -> None:
        self._emit("entry_start", index=index, total=total, key=key)

    def on_entry_end(
        self,
        index: int,
        total: int,
        key: str,
        *,
        measure: EntryMeasure | None = None,
    ) -> None:
        """Record an entry that ended, with its cost when the writer measured it."""
        fields: dict[str, Any] = {"index": index, "total": total, "key": key}
        if measure is not None:
            fields.update(dataclasses.asdict(measure))
        _ = self._write("entry_end", fields, progress=(index, total))

    def on_epoch_end(
        self, epoch: int, total_epochs: int, metrics: dict[str, float]
    ) -> None:
        _ = self._write(
            "epoch",
            {"epoch": epoch, "total_epochs": total_epochs, "metrics": dict(metrics)},
            progress=(epoch + 1, total_epochs),
        )

    def on_class_start(
        self, class_idx: int, total_classes: int, class_name: str
    ) -> None:
        self._emit(
            "class_start",
            class_idx=class_idx,
            total_classes=total_classes,
            class_name=class_name,
        )

    def on_phase(self, phase: str, message: str) -> None:
        self._emit("phase", phase=phase, message=message)

    # -- lifecycle (resource) -----------------------------------------------

    def close(self) -> None:
        """Stop the liveness thread, and close the file.

        Writes after this are dropped.
        """
        self._stop_liveness.set()
        thread = self._liveness
        if (
            thread is not None
            and thread is not threading.current_thread()
            and thread.is_alive()
        ):
            thread.join()
        with self._lock:
            if self._file is not None:
                try:
                    self._file.close()
                finally:
                    self._file = None

    def __enter__(self) -> JsonlRunLog:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _ends_mid_line(path: Path) -> bool:
    """Whether *path* is non-empty and its last byte is not a newline."""
    try:
        with path.open("rb") as file:
            if file.seek(0, os.SEEK_END) == 0:
                return False
            _ = file.seek(-1, os.SEEK_END)
            return file.read(1) != b"\n"
    except OSError:
        return False


def record_exited_attempt(
    run_dir: Path | str,
    execution_id: str,
    status: Literal["failed", "cancelled"],
    error_json: str = "",
) -> None:
    """Append the terminal event of an attempt whose process exited without one.

    Call it only after that process has exited. A run-log has one writer, and an
    exited writer leaves the caller as the only one.

    Args:
        run_dir: The ``.mosaic/runs`` directory.
        execution_id: The attempt.
        status: The terminal status to record.
        error_json: The failure to record with ``failed``.
    """
    with JsonlRunLog(Path(run_dir) / f"{execution_id}.jsonl", execution_id) as log:
        if status == "failed":
            log.failed(error_json)
        else:
            log.cancelled()


# ---------------------------------------------------------------------------
# Readers (stdlib-only; reduce a log back to the old ``runs`` row shape)
# ---------------------------------------------------------------------------

# The statuses that prove an attempt is over. Public because it is the only
# way another module can ask "is this execution still alive?" without
# re-deriving the answer from a second record that could disagree.
TERMINAL_STATUSES: Final[frozenset[str]] = frozenset(
    {"finished", "failed", "cancelled"}
)
_PROGRESS_EVENTS = {"entry_start", "entry_end", "epoch", "class_start", "phase"}


def run_log_lines(text: str) -> list[str]:
    """Split run-log text into its lines, without the NUL bytes of a zero-filled gap.

    A network filesystem (a CIFS share was seen) can leave a zero-filled gap in
    mid-file, glued to the front of the next record. A raw NUL never belongs to a
    record, because ``json.dumps`` writes U+0000 as ``\\u0000``, so every NUL is
    removed. Blank lines, including one the gap leaves empty, are dropped.

    Args:
        text: The content of a run-log.

    Returns:
        The non-empty lines in file order, stripped of surrounding whitespace.
    """
    lines = (line.replace("\x00", "").strip() for line in text.splitlines())
    return [line for line in lines if line]


def read_run_records(
    run_dir: Path | str, execution_id: str
) -> list[Mapping[str, object]]:
    """Parse one attempt's run-log JSON lines, tolerating a torn or partial line.

    Returns records in file order, empty when the log is absent. An unparseable
    line (a partial append caught mid-write on NFS, or the remains of a write that
    failed) is skipped rather than raising. The writer begins the line after a
    failed write on a fresh line, so only the torn line itself is lost. On a network
    filesystem a zero-filled gap may also sit in mid-file, glued to the front of the
    next record. The gap is stripped, and that record is read.

    Args:
        run_dir: The run-log directory.
        execution_id: The attempt.

    Returns:
        The records of the log, one per line that parses as a JSON object.
    """
    path = Path(run_dir) / f"{execution_id}.jsonl"
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, FileNotFoundError):
        return []
    out: list[Mapping[str, object]] = []
    for line in run_log_lines(text):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict):
            out.append(cast("dict[str, Any]", rec))
    return out


def reduce_run_log(path: Path) -> RunLogSnapshot | None:
    """Fold a run-log JSONL into a :class:`RunLogSnapshot` (``None`` if empty/unreadable)."""
    path = Path(path)
    return fold_run_records(path.stem, read_run_records(path.parent, path.stem))


def fold_run_records(
    execution_id: str, records: Sequence[Mapping[str, object]]
) -> RunLogSnapshot | None:
    """Fold an attempt's records into a :class:`RunLogSnapshot`.

    Args:
        execution_id: The attempt the records belong to.
        records: Its records in file order, as :func:`read_run_records` returns
            them.

    Returns:
        The snapshot, ``None`` for no records.
    """
    snap: RunLogSnapshot = {
        "execution_id": execution_id,
        "kind": "",
        "target": "",
        "run_id": "",
        "status": "running",
        "owner": "",
        "host": "",
        "pid": 0,
        "created_at": "",
        "started_at": "",
        "heartbeat_at": "",
        "finished_at": "",
        "error_json": "",
        "progress_done": 0,
        "progress_total": 0,
        "entries_failed": 0,
        "entries_written": 0,
        "entries_frame_axis_mismatch": 0,
        "entries_frame_tail_short": 0,
        "entries_columns_dropped": 0,
        "entries_time_irregular": 0,
        "cache_hit": False,
        "tracks_variant": "",
        "boot_id": "",
        "pid_ns": "",
        "liveness_interval": 0.0,
        "gpu_card": "",
        "phase": "",
        "workers": 0,
        "parallel_mode": "",
        "peak_rss_bytes": 0,
        "min_available_bytes": 0,
        "total_memory_bytes": 0,
        "probe": False,
    }
    if not records:
        return None
    for rec in records:
        ev = record_text(rec, "ev")
        ts = record_text(rec, "t")
        if ts:
            snap["heartbeat_at"] = ts  # last-seen event time == liveness
        if ev == "started":
            _reset_for_start(snap)
            snap["kind"] = record_text(rec, "kind")
            snap["target"] = record_text(rec, "target")
            snap["owner"] = record_text(rec, "owner")
            snap["host"] = record_text(rec, "host")
            snap["pid"] = record_count(rec, "pid")
            snap["created_at"] = _text_or(rec, "created_at", ts)
            snap["started_at"] = _text_or(rec, "created_at", ts)
            # Set from every ``started``, so a later one written without them
            # (a resume by an older mosaic) clears what an earlier one declared.
            snap["boot_id"] = record_text(rec, "boot_id")
            snap["pid_ns"] = record_text(rec, "pid_ns")
            snap["liveness_interval"] = record_seconds(rec, "liveness_interval")
            snap["gpu_card"] = record_text(rec, "gpu_card")
        elif ev == "phase":
            snap["phase"] = record_text(rec, "phase")
        elif ev == "run_id":
            snap["run_id"] = record_text(rec, "run_id")
        elif ev == "total":
            snap["progress_total"] = _count_or(rec, "total", snap["progress_total"])
        elif ev == "heartbeat":
            snap["progress_done"] = _count_or(rec, "done", snap["progress_done"])
            snap["progress_total"] = _count_or(rec, "total", snap["progress_total"])
        elif ev == "entry_end":
            snap["progress_done"] = _count_or(rec, "index", snap["progress_done"])
            snap["progress_total"] = _count_or(rec, "total", snap["progress_total"])
        elif ev == "epoch":
            snap["progress_done"] = _count_or(rec, "epoch", -1) + 1
            snap["progress_total"] = _count_or(
                rec, "total_epochs", snap["progress_total"]
            )
        elif ev == "entry_error":
            snap["entries_failed"] += 1
        elif ev == "frame_axis_mismatch":
            snap["entries_frame_axis_mismatch"] += 1
        elif ev == "frame_tail_short":
            snap["entries_frame_tail_short"] += 1
        elif ev == "columns_dropped":
            snap["entries_columns_dropped"] += 1
        elif ev == "time_irregular":
            snap["entries_time_irregular"] += 1
        elif ev == "entries_written":
            snap["entries_written"] = _count_or(
                rec, "entries_written", snap["entries_written"]
            )
        elif ev == "cache_hit":
            snap["cache_hit"] = True
        elif ev == "tracks_variant":
            snap["tracks_variant"] = _text_or(
                rec, "tracks_variant", snap["tracks_variant"]
            )
        elif ev == "resources":
            snap["workers"] = record_count(rec, "workers")
            snap["parallel_mode"] = record_text(rec, "parallel_mode")
            snap["peak_rss_bytes"] = record_count(rec, "peak_rss_bytes")
            snap["min_available_bytes"] = record_count(rec, "min_available_bytes")
            snap["total_memory_bytes"] = record_count(rec, "total_memory_bytes")
        elif ev == "probe":
            snap["probe"] = rec.get("kind") == "probe"
        elif ev in TERMINAL_STATUSES:
            snap["status"] = ev
            snap["finished_at"] = ts
            if ev == "failed":
                snap["error_json"] = record_text(rec, "error")
    return snap


def _count_or(rec: Mapping[str, object], key: str, default: int) -> int:
    """An integer field of a run-log record, *default* when absent or not an integer."""
    value = rec.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else default


def _text_or(rec: Mapping[str, object], key: str, default: str) -> str:
    """A string field of a run-log record, *default* when absent or not a string."""
    value = rec.get(key)
    return value if isinstance(value, str) else default


def _reset_for_start(snap: RunLogSnapshot) -> None:
    """Reset the fields of *snap* that report one process of a resumed attempt."""
    snap["progress_done"] = 0
    snap["progress_total"] = 0
    snap["entries_failed"] = 0
    snap["entries_written"] = 0
    snap["entries_frame_axis_mismatch"] = 0
    snap["entries_frame_tail_short"] = 0
    snap["entries_columns_dropped"] = 0
    snap["entries_time_irregular"] = 0
    snap["cache_hit"] = False
    snap["tracks_variant"] = ""
    snap["phase"] = ""
    snap["workers"] = 0
    snap["parallel_mode"] = ""
    snap["peak_rss_bytes"] = 0
    snap["min_available_bytes"] = 0
    snap["total_memory_bytes"] = 0
    snap["probe"] = False


def read_run(run_dir: Path | str, execution_id: str) -> RunLogSnapshot | None:
    """Read one attempt snapshot by ``execution_id`` (or ``None`` if absent)."""
    return reduce_run_log(Path(run_dir) / f"{execution_id}.jsonl")


def read_runs(
    run_dir: Path | str,
    *,
    kind: str | None = None,
    status: str | None = None,
) -> list[RunLogSnapshot]:
    """Read every attempt snapshot (newest first), optionally filtered by kind/status."""
    return list_runs(run_dir, kind=kind, status=status)["items"]


def list_runs(
    run_dir: Path | str,
    *,
    limit: int | None = None,
    after: str | None = None,
    kind: str | None = None,
    status: str | None = None,
    target: str | None = None,
    changed_since: datetime.datetime | None = None,
) -> RunLogPage:
    """Read one page of attempt snapshots, newest first.

    The run-log file names are listed and sorted, and logs are reduced in that order
    until the page is full. Without a filter, a page reduces *limit* logs whatever
    the number of attempts. A filter also reduces the logs it passes over. An empty
    or unreadable log is skipped. ``execution_id`` is a ULID, and a descending sort
    of the names is creation-time descending. An attempt created after a page was
    read sorts before that page and leaves the next page unchanged.

    Args:
        run_dir: The run-log directory.
        limit: The most snapshots to return, at least 1. ``None`` returns all.
        after: The ``next_after`` of the previous page. The page starts at the
            first execution id that sorts before it.
        kind: Keep only attempts of this ``kind``.
        status: Keep only attempts with this ``status``.
        target: Keep only attempts with this ``target``.
        changed_since: Keep only attempts that changed at or after this
            instant (:func:`changed_at_or_after`). A log last written before it is
            not read.

    Returns:
        The page, and the ``after`` of the next page while unread logs remain.

    Raises:
        ValueError: *limit* is below 1.
    """
    if limit is not None and limit < 1:
        raise ValueError(f"limit must be at least 1, got {limit}")
    names = _run_log_names(Path(run_dir))
    if after is not None:
        names = [name for name in names if name < after]
    if changed_since is not None:
        names = [
            name
            for name in names
            if written_since(Path(run_dir) / f"{name}.jsonl", changed_since)
        ]
    items: list[RunLogSnapshot] = []
    for position, name in enumerate(names):
        if limit is not None and len(items) >= limit:
            return {"items": items, "next_after": names[position - 1]}
        snap = reduce_run_log(Path(run_dir) / f"{name}.jsonl")
        if snap is None:
            continue
        if changed_since is not None and not changed_at_or_after(snap, changed_since):
            continue
        if kind is not None and snap["kind"] != kind:
            continue
        if status is not None and snap["status"] != status:
            continue
        if target is not None and snap["target"] != target:
            continue
        items.append(snap)
    return {"items": items, "next_after": None}


MTIME_TOLERANCE_SECONDS: Final = 5.0
"""How far before an instant a file's modification time may read and still count.

A coarse file clock, or a file server whose clock runs behind, can stamp a write
made after the instant a little before it. The event stamps then decide.
"""


def written_since(path: Path, instant: datetime.datetime) -> bool:
    """Whether the file at *path* may have been written at or after *instant*.

    The modification time may read up to :data:`MTIME_TOLERANCE_SECONDS` before
    *instant*. An absent file was not written. A log written from a host whose
    clock runs further behind may still be passed over.
    """
    try:
        modified = path.stat().st_mtime
    except OSError:
        return False
    return modified >= instant.timestamp() - MTIME_TOLERANCE_SECONDS


def changed_at_or_after(snap: RunLogSnapshot, instant: datetime.datetime) -> bool:
    """Whether the attempt of *snap* was created, started or ended at or after *instant*.

    Progress, heartbeats and ``alive`` events are not changes. A blank or
    unparseable timestamp is not a change.
    """
    for stamp in (snap["created_at"], snap["started_at"], snap["finished_at"]):
        when = parse_instant(stamp) if stamp else None
        if when is not None and when >= instant:
            return True
    return False


def _run_log_names(run_dir: Path) -> list[str]:
    """The execution ids of the run-logs in *run_dir*, newest first."""
    if not run_dir.is_dir():
        return []
    return sorted((path.stem for path in run_dir.glob("*.jsonl")), reverse=True)


def read_runs_by_run_id(run_dir: Path | str, run_id: str) -> list[RunLogSnapshot]:
    """Read all attempts that produced (or targeted) a given content ``run_id``."""
    return [r for r in read_runs(run_dir) if r["run_id"] == run_id]


REMOVAL_EVENTS: Final[frozenset[str]] = frozenset(
    {"run_removing", "run_removed", "run_kept"}
)
"""The events with which a cleanup records each run it removed or kept."""

type RemovalState = Literal["removed", "interrupted", "failed", "kept"]
"""How one run a cleanup was approved to remove ended.

``interrupted`` is an intent with no outcome after it: the process stopped
between recording the intent and recording the removal. ``failed`` is an intent
followed by an ``entry_error`` naming the run.
"""


class RemovalRecord(TypedDict):
    """One run that a cleanup was approved to remove, as its run-log records it.

    Attributes:
        feature: The run's storage name under ``features/``.
        run_id: The run.
        state: How it ended (:data:`RemovalState`).
        reason: Why it was removed or kept.
        detail: For a kept run, what kept it; ``""`` otherwise.
        superseded_by: The run that replaced it, ``""`` when unrecorded.
        bytes: The bytes its directory held, 0 unless removed.
        params: Its recorded ``params.json``, empty for a run never removed.
        retired: The digest of the retired recipe whose request wrote the run,
            ``""`` when it was removed as superseded or the log predates
            retirement.
    """

    feature: str
    run_id: str
    state: RemovalState
    reason: str
    detail: str
    superseded_by: str
    bytes: int
    params: dict[str, Any]
    retired: str


def read_removal_records(run_dir: Path | str, execution_id: str) -> list[RemovalRecord]:
    """Fold a cleanup's run-log into one record per run, in the order first named.

    A cleanup that ran again after it was stopped writes a run's events twice,
    and the last outcome stands, except that a removal is never undone. An
    ``entry_error`` keyed ``<feature>/<run_id>`` after an intent is a failed
    removal.

    Args:
        run_dir: The run-log directory.
        execution_id: The cleanup's attempt.

    Returns:
        The records, empty when the log is absent or names no run.
    """
    records: dict[tuple[str, str], RemovalRecord] = {}
    for rec in read_run_records(run_dir, execution_id):
        ev = record_text(rec, "ev")
        if ev == "entry_error":
            feature, _, run_id = record_text(rec, "key").partition("/")
            found = records.get((feature, run_id))
            if found is not None and found["state"] == "interrupted":
                found["state"] = "failed"
            continue
        if ev not in REMOVAL_EVENTS:
            continue
        key = (record_text(rec, "feature"), record_text(rec, "run_id"))
        found = records.get(key)
        if found is None:
            found = records[key] = {
                "feature": key[0],
                "run_id": key[1],
                "state": "kept",
                "reason": "",
                "detail": "",
                "superseded_by": "",
                "bytes": 0,
                "params": {},
                "retired": "",
            }
        if found["state"] == "removed":
            continue
        if ev == "run_removing":
            found["state"] = "interrupted"
            found["reason"] = record_text(rec, "reason")
            found["superseded_by"] = record_text(rec, "superseded_by")
            found["retired"] = record_text(rec, "retired")
            params = rec.get("params")
            if isinstance(params, dict):
                found["params"] = cast("dict[str, Any]", params)
        elif ev == "run_removed":
            found["state"] = "removed"
            found["bytes"] = record_count(rec, "bytes")
        else:
            found["state"] = "kept"
            found["reason"] = record_text(rec, "reason")
            found["detail"] = record_text(rec, "detail")
    return list(records.values())


def read_entry_errors(
    run_dir: Path | str, execution_id: str, *, limit: int | None = None
) -> EntryErrorPage:
    """Return the entities that one attempt lost while it continued, in file order.

    The whole run-log is read to count the failures. Only the listed records parse
    their blobs.

    Args:
        run_dir: The run-log directory.
        execution_id: The attempt.
        limit: The most records to list, at least 1. ``None`` lists every one.

    Returns:
        One record per ``entry_error`` event up to *limit*, with the count of all
        of them. An absent run-log lists none.

    Raises:
        ValueError: *limit* is below 1.
    """
    if limit is not None and limit < 1:
        raise ValueError(f"limit must be at least 1, got {limit}")
    items: list[EntryErrorRecord] = []
    total = 0
    for rec in _entry_error_events(run_dir, execution_id):
        total += 1
        if limit is not None and len(items) >= limit:
            continue
        error_json = record_text(rec, "error")
        summary = error_summary(error_json)
        items.append(
            {
                "key": record_text(rec, "key"),
                "error_json": error_json,
                "error_type": summary["type"],
                "message": summary["message"],
                "recorded_at": record_text(rec, "t"),
            }
        )
    return {"items": items, "total": total, "has_more": total > len(items)}


def read_time_irregular(
    run_dir: Path | str, execution_id: str, *, limit: int | None = None
) -> TimeIrregularPage:
    """Return the uneven-time events that one attempt recorded, in file order.

    Args:
        run_dir: The run-log directory.
        execution_id: The attempt.
        limit: The most records to list, at least 1. ``None`` lists every one.

    Returns:
        One record per ``time_irregular`` event after the last ``started`` event,
        up to *limit*, with the count of all of them. An absent run-log lists none.

    Raises:
        ValueError: *limit* is below 1.
    """
    if limit is not None and limit < 1:
        raise ValueError(f"limit must be at least 1, got {limit}")
    events = [
        rec
        for rec in last_attempt_records(read_run_records(run_dir, execution_id))
        if rec.get("ev") == "time_irregular"
    ]
    listed = events if limit is None else events[:limit]
    items: list[TimeIrregularRecord] = [
        {
            "key": record_text(rec, "key"),
            "who": record_text(rec, "who"),
            "largest_step": record_seconds(rec, "largest_step"),
            "first_frame": record_count(rec, "first_frame"),
            "last_frame": record_count(rec, "last_frame"),
            "median_step": record_seconds(rec, "median_step"),
            "rate": record_seconds(rec, "rate"),
            "message": _irregular_message(rec),
            "recorded_at": record_text(rec, "t"),
        }
        for rec in listed
    ]
    return {"items": items, "total": len(events), "has_more": len(events) > len(items)}


def time_irregular_sentence(
    largest_step: float,
    first_frame: int,
    last_frame: int,
    median_step: float,
    rate: float,
) -> str:
    """The sentence that reports an uneven time axis read as even.

    The one wording of the warning that ``core.timebase`` logs and of the record that
    :func:`read_time_irregular` lists. It names the largest gap and ``resample-tracks``.
    """
    return (
        f"the time column steps unevenly: the largest step is {largest_step:.4g} s "
        f"(frame {first_frame} to {last_frame}) against a median of "
        f"{median_step:.4g} s per frame. The method read the table as evenly sampled "
        f"at {rate:.4g} frames per second, so it misplaces what follows a gap. Run "
        "resample-tracks first to place the tracks on one uniform rate."
    )


def _irregular_message(rec: Mapping[str, object]) -> str:
    """The sentence of one ``time_irregular`` record, built from its numbers."""
    sentence = time_irregular_sentence(
        record_seconds(rec, "largest_step"),
        record_count(rec, "first_frame"),
        record_count(rec, "last_frame"),
        record_seconds(rec, "median_step"),
        record_seconds(rec, "rate"),
    )
    return f"[{record_text(rec, 'who')}] {sentence}"


def read_entry_error_keys(run_dir: Path | str, execution_id: str) -> list[str]:
    """Return the entry keys of the entities that one attempt lost, in file order.

    The blobs are left unparsed.

    Args:
        run_dir: The run-log directory.
        execution_id: The attempt.

    Returns:
        One key per ``entry_error`` event, empty for an absent run-log.
    """
    return [
        record_text(rec, "key") for rec in _entry_error_events(run_dir, execution_id)
    ]


def last_attempt_records(
    records: Sequence[Mapping[str, object]],
) -> Sequence[Mapping[str, object]]:
    """The records from the last ``started`` event on, all of them when there is none.

    A process that resumes an attempt appends a second ``started``, and what
    follows it is the latest process's own.
    """
    for index in range(len(records) - 1, -1, -1):
        if records[index].get("ev") == "started":
            return records[index:]
    return records


def _entry_error_events(
    run_dir: Path | str, execution_id: str
) -> list[Mapping[str, object]]:
    """The ``entry_error`` records after the last ``started`` event, in file order.

    A resumed attempt lists the entities that its latest process lost, as
    :func:`reduce_run_log` counts them.
    """
    return [
        rec
        for rec in last_attempt_records(read_run_records(run_dir, execution_id))
        if rec.get("ev") == "entry_error"
    ]


def error_fields(error_json: str) -> Mapping[str, object] | None:
    """Return the fields of an ``error_json`` blob, ``None`` when it is not a JSON object.

    A crash's blob holds ``type``, ``message`` and ``traceback``. A refusal's holds
    ``reason``, ``step`` and ``message`` with its detail beside them.
    """
    try:
        blob: object = json.loads(error_json)
    except json.JSONDecodeError:
        return None
    if not isinstance(blob, dict):
        return None
    return cast("dict[str, object]", blob)


def error_summary(error_json: str) -> ErrorSummary:
    """Return the exception type and message that an ``error_json`` blob records.

    The message of a blob that is not a JSON object is the blob itself. An absent
    field, or one that is not a string, reads as ``""``.

    Args:
        error_json: The blob that an attempt or one of its entries recorded.

    Returns:
        The type and the message, each ``""`` for an empty blob.
    """
    fields = error_fields(error_json)
    if fields is None:
        return {"type": "", "message": error_json[:ERROR_MESSAGE_LIMIT]}
    return {
        "type": record_text(fields, "type")[:ERROR_MESSAGE_LIMIT],
        "message": record_text(fields, "message")[:ERROR_MESSAGE_LIMIT],
    }


def read_run_progress(
    run_dir: Path | str, execution_id: str
) -> list[ProgressEventRecord]:
    """Return one attempt's progress events in the legacy ``read_progress`` shape.

    Each row matches what the old SQLite ``training_progress`` reader returned, so
    ``mosaic status --progress`` is unchanged. A field of the wrong type in an
    event reads as 0, ``""`` or ``{}``.
    """
    out: list[ProgressEventRecord] = []
    for rec in read_run_records(run_dir, execution_id):
        ev = record_text(rec, "ev")
        if ev not in _PROGRESS_EVENTS:
            continue
        ts = record_text(rec, "t")
        if ev in ("entry_end", "entry_start"):
            out.append(
                {
                    "step_type": "entry" if ev == "entry_end" else "entry_start",
                    "step_index": record_count(rec, "index"),
                    "step_total": record_count(rec, "total"),
                    "metrics": {},
                    "message": record_text(rec, "key"),
                    "timestamp": ts,
                }
            )
        elif ev == "epoch":
            out.append(
                {
                    "step_type": "epoch",
                    "step_index": record_count(rec, "epoch"),
                    "step_total": record_count(rec, "total_epochs"),
                    "metrics": _record_metrics(rec),
                    "message": "",
                    "timestamp": ts,
                }
            )
        elif ev == "class_start":
            out.append(
                {
                    "step_type": "class",
                    "step_index": record_count(rec, "class_idx"),
                    "step_total": record_count(rec, "total_classes"),
                    "metrics": {},
                    "message": record_text(rec, "class_name"),
                    "timestamp": ts,
                }
            )
        elif ev == "phase":
            phase = record_text(rec, "phase")
            message = record_text(rec, "message")
            out.append(
                {
                    "step_type": "phase",
                    "step_index": 0,
                    "step_total": 0,
                    "metrics": {},
                    "message": f"{phase}: {message}",
                    "timestamp": ts,
                }
            )
    return out


def record_count(rec: Mapping[str, object], key: str) -> int:
    """An integer field of a run-log record, 0 when absent or not an integer."""
    value = rec.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def record_seconds(rec: Mapping[str, object], key: str) -> float:
    """A non-negative number field of a run-log record, 0.0 when absent or not one."""
    value = rec.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 0.0
    return max(float(value), 0.0)


def record_text(rec: Mapping[str, object], key: str) -> str:
    """A string field of a run-log record, ``""`` when absent or not a string."""
    value = rec.get(key)
    return value if isinstance(value, str) else ""


def record_flag(rec: Mapping[str, object], key: str) -> bool:
    """A boolean field of a run-log record, ``False`` when absent or not a boolean."""
    return rec.get(key) is True


def _record_metrics(rec: Mapping[str, object]) -> dict[str, float]:
    """An epoch's numeric metrics, leaving out every entry that is not a number."""
    metrics = rec.get("metrics")
    if not isinstance(metrics, dict):
        return {}
    return {
        str(name): float(value)
        for name, value in cast("dict[object, object]", metrics).items()
        if isinstance(value, int | float) and not isinstance(value, bool)
    }
