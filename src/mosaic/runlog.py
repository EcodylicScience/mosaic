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
  it. Exactly where SQLite/WAL failed.
* **Bounded + ephemeral** -- finite entries*2 (per-entry) or epochs*1 (per-epoch) lines
  (KB-MB), plus at most one ``alive`` line per :data:`LIVENESS_INTERVAL_SECONDS` of
  silence, and it stops growing at a terminal event. The durable truth lives elsewhere
  (``index.csv`` + parquet for results); the JSONL is scratch.

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

import datetime
import json
import os
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, Literal, TypedDict, cast

__all__ = [
    "ALIVE_EVENT",
    "ERROR_MESSAGE_LIMIT",
    "LIVENESS_INTERVAL_SECONDS",
    "LIVENESS_STALE_INTERVALS",
    "AttemptLiveness",
    "AttemptOutcome",
    "EntryErrorPage",
    "EntryErrorRecord",
    "ErrorSummary",
    "JsonlRunLog",
    "ProgressEventRecord",
    "REMOVAL_EVENTS",
    "RemovalRecord",
    "RemovalState",
    "RunLogPage",
    "RunLogSnapshot",
    "TERMINAL_STATUSES",
    "attempt_outcome",
    "error_summary",
    "list_runs",
    "new_execution_id",
    "now_iso",
    "read_entry_error_keys",
    "read_entry_errors",
    "read_removal_records",
    "read_run",
    "read_run_progress",
    "record_exited_attempt",
    "read_runs",
    "read_runs_by_run_id",
    "reduce_run_log",
    "run_log_dir",
    "run_log_path",
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

    ``tracks_variant`` names the tracks recipes the attempt *read*, comma-joined
    and sorted, empty for a run that reads none. Never what an op *produced* --
    those are different relations and one key cannot hold both, which is what
    would make a downstream column unqueryable.

    Zero, ``False`` and ``""`` are indistinguishable from "not reported", the same
    convention as the tracks index's blank ``n_keypoints`` cell meaning *unknown*
    rather than zero.

    A process that resumes an attempt whose process died appends a second
    ``started`` event to its run-log. The counters, the progress, ``cache_hit``
    and ``tracks_variant`` then report the events after the last ``started``, and
    the entries that the dead process lost are not counted twice.

    ``boot_id``, ``pid_ns`` and ``liveness_interval`` are what the last
    ``started`` event recorded of the process's host: its boot, its pid namespace
    and the seconds between the ``alive`` events it promises (0 for a writer that
    promises none, as every writer before them). ``phase`` is the last phase the
    attempt declared since that ``started``, ``""`` before one.

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
    cache_hit: bool
    tracks_variant: str
    boot_id: str
    pid_ns: str
    liveness_interval: float
    phase: str


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
    "pending", "running", "finished", "partial", "failed", "cancelled"
]
"""How an attempt stands, as a report names it.

``pending`` is an attempt with no run-log yet. ``partial`` is a ``finished``
attempt that lost entries. Both are reporting words and neither is a run-log status,
so neither joins :data:`TERMINAL_STATUSES`, which mosaic-api's sweeper reaps.
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

    def _emit(self, event: str, **fields: Any) -> None:
        _ = self._write(event, fields)

    def _write(self, event: str, fields: dict[str, Any], *, end: bool = False) -> bool:
        """Append one event line, returning ``False`` when the log is closed.

        Args:
            event: The event kind.
            fields: The event's fields.
            end: Whether the event is terminal. No ``alive`` line follows one.
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
    ) -> None:
        """Bracket the start of the attempt (kind/target/owner/host/pid).

        *boot_id*, *pid_ns* and *liveness_interval* describe the process for a
        reader judging whether it still runs. Each is written only when set, so
        a record of a process that runs nothing (a refusal, a cache hit) reads as
        before.
        """
        liveness: dict[str, Any] = {}
        if boot_id:
            liveness["boot_id"] = boot_id
        if pid_ns:
            liveness["pid_ns"] = pid_ns
        if liveness_interval > 0:
            liveness["liveness_interval"] = liveness_interval
        self._emit(
            "started",
            kind=kind,
            target=target,
            owner=owner,
            host=host,
            pid=pid,
            created_at=created_at or now_iso(),
            **liveness,
        )

    def set_run_id(self, run_id: str) -> None:
        """Record the content-addressed ``run_id`` once the job computes it."""
        self._emit("run_id", run_id=run_id)

    def set_total(self, total: int) -> None:
        """Declare the total number of entries/epochs (progress denominator)."""
        self._emit("total", total=total)

    def heartbeat(self, done: int, total: int) -> None:
        """Refresh liveness and the coarse completed count."""
        self._emit("heartbeat", done=done, total=total)

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

    # -- removal of a superseded run -----------------------------------------

    def run_removing(
        self,
        *,
        feature: str,
        run_id: str,
        reason: str,
        superseded_by: str,
        params: Mapping[str, Any],
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
        """
        self._emit(
            "run_removing",
            feature=feature,
            run_id=run_id,
            reason=reason,
            superseded_by=superseded_by,
            params=dict(params),
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

    def on_entry_end(self, index: int, total: int, key: str) -> None:
        self._emit("entry_end", index=index, total=total, key=key)

    def on_epoch_end(
        self, epoch: int, total_epochs: int, metrics: dict[str, float]
    ) -> None:
        self._emit(
            "epoch", epoch=epoch, total_epochs=total_epochs, metrics=dict(metrics)
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


def _iter_records(path: Path) -> list[dict[str, Any]]:
    """Parse a run-log's JSON lines, tolerating a torn/partial line.

    Returns records in file order. An un-parseable line (a partial append caught
    mid-write on NFS, or the remains of a write that failed) is skipped rather
    than raising. The writer begins the line after a failed write on a fresh
    line, so only the torn line itself is lost.
    """
    try:
        text = Path(path).read_text(encoding="utf-8")
    except (OSError, FileNotFoundError):
        return []
    out: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
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
    snap: RunLogSnapshot = {
        "execution_id": path.stem,
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
        "cache_hit": False,
        "tracks_variant": "",
        "boot_id": "",
        "pid_ns": "",
        "liveness_interval": 0.0,
        "phase": "",
    }
    records = _iter_records(path)
    if not records:
        return None
    for rec in records:
        ev = rec.get("ev", "")
        ts = rec.get("t", "")
        if ts:
            snap["heartbeat_at"] = ts  # last-seen event time == liveness
        if ev == "started":
            _reset_for_start(snap)
            snap["kind"] = rec.get("kind", "")
            snap["target"] = rec.get("target", "")
            snap["owner"] = rec.get("owner", "")
            snap["host"] = rec.get("host", "")
            snap["pid"] = rec.get("pid", 0)
            snap["created_at"] = rec.get("created_at", ts)
            snap["started_at"] = rec.get("created_at", ts)
            # Set from every ``started``, so a later one written without them
            # (a resume by an older mosaic) clears what an earlier one declared.
            snap["boot_id"] = _record_text(rec, "boot_id")
            snap["pid_ns"] = _record_text(rec, "pid_ns")
            snap["liveness_interval"] = _record_seconds(rec, "liveness_interval")
        elif ev == "phase":
            snap["phase"] = _record_text(rec, "phase")
        elif ev == "run_id":
            snap["run_id"] = rec.get("run_id", "")
        elif ev == "total":
            snap["progress_total"] = rec.get("total", snap["progress_total"])
        elif ev == "heartbeat":
            snap["progress_done"] = rec.get("done", snap["progress_done"])
            snap["progress_total"] = rec.get("total", snap["progress_total"])
        elif ev == "entry_end":
            snap["progress_done"] = rec.get("index", snap["progress_done"])
            snap["progress_total"] = rec.get("total", snap["progress_total"])
        elif ev == "epoch":
            snap["progress_done"] = rec.get("epoch", -1) + 1
            snap["progress_total"] = rec.get("total_epochs", snap["progress_total"])
        elif ev == "entry_error":
            snap["entries_failed"] += 1
        elif ev == "frame_axis_mismatch":
            snap["entries_frame_axis_mismatch"] += 1
        elif ev == "frame_tail_short":
            snap["entries_frame_tail_short"] += 1
        elif ev == "columns_dropped":
            snap["entries_columns_dropped"] += 1
        elif ev == "entries_written":
            snap["entries_written"] = rec.get(
                "entries_written", snap["entries_written"]
            )
        elif ev == "cache_hit":
            snap["cache_hit"] = True
        elif ev == "tracks_variant":
            snap["tracks_variant"] = rec.get("tracks_variant", snap["tracks_variant"])
        elif ev in TERMINAL_STATUSES:
            snap["status"] = ev
            snap["finished_at"] = ts
            if ev == "failed":
                snap["error_json"] = rec.get("error", "")
    return snap


def _reset_for_start(snap: RunLogSnapshot) -> None:
    """Reset the fields of *snap* that report one process of a resumed attempt."""
    snap["progress_done"] = 0
    snap["progress_total"] = 0
    snap["entries_failed"] = 0
    snap["entries_written"] = 0
    snap["entries_frame_axis_mismatch"] = 0
    snap["entries_frame_tail_short"] = 0
    snap["entries_columns_dropped"] = 0
    snap["cache_hit"] = False
    snap["tracks_variant"] = ""
    snap["phase"] = ""


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
    items: list[RunLogSnapshot] = []
    for position, name in enumerate(names):
        if limit is not None and len(items) >= limit:
            return {"items": items, "next_after": names[position - 1]}
        snap = reduce_run_log(Path(run_dir) / f"{name}.jsonl")
        if snap is None:
            continue
        if kind is not None and snap["kind"] != kind:
            continue
        if status is not None and snap["status"] != status:
            continue
        if target is not None and snap["target"] != target:
            continue
        items.append(snap)
    return {"items": items, "next_after": None}


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
    """

    feature: str
    run_id: str
    state: RemovalState
    reason: str
    detail: str
    superseded_by: str
    bytes: int
    params: dict[str, Any]


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
    for rec in _iter_records(Path(run_dir) / f"{execution_id}.jsonl"):
        ev = rec.get("ev", "")
        if ev == "entry_error":
            feature, _, run_id = _record_text(rec, "key").partition("/")
            found = records.get((feature, run_id))
            if found is not None and found["state"] == "interrupted":
                found["state"] = "failed"
            continue
        if ev not in REMOVAL_EVENTS:
            continue
        key = (_record_text(rec, "feature"), _record_text(rec, "run_id"))
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
            }
        if found["state"] == "removed":
            continue
        if ev == "run_removing":
            found["state"] = "interrupted"
            found["reason"] = _record_text(rec, "reason")
            found["superseded_by"] = _record_text(rec, "superseded_by")
            params = rec.get("params")
            if isinstance(params, dict):
                found["params"] = cast("dict[str, Any]", params)
        elif ev == "run_removed":
            found["state"] = "removed"
            found["bytes"] = _record_count(rec, "bytes")
        else:
            found["state"] = "kept"
            found["reason"] = _record_text(rec, "reason")
            found["detail"] = _record_text(rec, "detail")
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
        error_json = _record_text(rec, "error")
        summary = error_summary(error_json)
        items.append(
            {
                "key": _record_text(rec, "key"),
                "error_json": error_json,
                "error_type": summary["type"],
                "message": summary["message"],
                "recorded_at": _record_text(rec, "t"),
            }
        )
    return {"items": items, "total": total, "has_more": total > len(items)}


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
        _record_text(rec, "key") for rec in _entry_error_events(run_dir, execution_id)
    ]


def _entry_error_events(run_dir: Path | str, execution_id: str) -> list[dict[str, Any]]:
    """The ``entry_error`` records after the last ``started`` event, in file order.

    A resumed attempt lists the entities that its latest process lost, as
    :func:`reduce_run_log` counts them.
    """
    events: list[dict[str, Any]] = []
    for rec in _iter_records(Path(run_dir) / f"{execution_id}.jsonl"):
        if rec.get("ev") == "started":
            events.clear()
        elif rec.get("ev") == "entry_error":
            events.append(rec)
    return events


def error_summary(error_json: str) -> ErrorSummary:
    """Return the exception type and message that an ``error_json`` blob records.

    The message of a blob that is not a JSON object is the blob itself. An absent
    field, or one that is not a string, reads as ``""``.

    Args:
        error_json: The blob that an attempt or one of its entries recorded.

    Returns:
        The type and the message, each ``""`` for an empty blob.
    """
    try:
        blob: object = json.loads(error_json)
    except json.JSONDecodeError:
        return {"type": "", "message": error_json[:ERROR_MESSAGE_LIMIT]}
    if not isinstance(blob, dict):
        return {"type": "", "message": error_json[:ERROR_MESSAGE_LIMIT]}
    fields = cast("dict[str, Any]", blob)
    return {
        "type": _record_text(fields, "type")[:ERROR_MESSAGE_LIMIT],
        "message": _record_text(fields, "message")[:ERROR_MESSAGE_LIMIT],
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
    for rec in _iter_records(Path(run_dir) / f"{execution_id}.jsonl"):
        ev = rec.get("ev", "")
        if ev not in _PROGRESS_EVENTS:
            continue
        ts = _record_text(rec, "t")
        if ev in ("entry_end", "entry_start"):
            out.append(
                {
                    "step_type": "entry" if ev == "entry_end" else "entry_start",
                    "step_index": _record_count(rec, "index"),
                    "step_total": _record_count(rec, "total"),
                    "metrics": {},
                    "message": _record_text(rec, "key"),
                    "timestamp": ts,
                }
            )
        elif ev == "epoch":
            out.append(
                {
                    "step_type": "epoch",
                    "step_index": _record_count(rec, "epoch"),
                    "step_total": _record_count(rec, "total_epochs"),
                    "metrics": _record_metrics(rec),
                    "message": "",
                    "timestamp": ts,
                }
            )
        elif ev == "class_start":
            out.append(
                {
                    "step_type": "class",
                    "step_index": _record_count(rec, "class_idx"),
                    "step_total": _record_count(rec, "total_classes"),
                    "metrics": {},
                    "message": _record_text(rec, "class_name"),
                    "timestamp": ts,
                }
            )
        elif ev == "phase":
            phase = _record_text(rec, "phase")
            message = _record_text(rec, "message")
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


def _record_count(rec: dict[str, Any], key: str) -> int:
    """An integer field of a run-log record, 0 when absent or not an integer."""
    value = rec.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _record_seconds(rec: dict[str, Any], key: str) -> float:
    """A non-negative number field of a run-log record, 0.0 when absent or not one."""
    value = rec.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 0.0
    return max(float(value), 0.0)


def _record_text(rec: dict[str, Any], key: str) -> str:
    """A string field of a run-log record, ``""`` when absent or not a string."""
    value = rec.get(key)
    return value if isinstance(value, str) else ""


def _record_metrics(rec: dict[str, Any]) -> dict[str, float]:
    """An epoch's numeric metrics, leaving out every entry that is not a number."""
    metrics = rec.get("metrics")
    if not isinstance(metrics, dict):
        return {}
    return {
        str(name): float(value)
        for name, value in cast("dict[object, object]", metrics).items()
        if isinstance(value, int | float) and not isinstance(value, bool)
    }
