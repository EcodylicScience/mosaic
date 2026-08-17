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
* **NFS-safe** -- one writer, append-only, flush-per-line, so a remote reader only ever sees
  a partial *last* line (skipped by the reducer). Exactly where SQLite/WAL failed.
* **Bounded + ephemeral** -- finite entries*2 (per-entry) or epochs*1 (per-epoch) lines
  (KB-MB), and it stops growing at a terminal event. The durable truth lives elsewhere
  (``index.csv`` + parquet for results); the JSONL is scratch.

Two event shapes ride the same sink: **per-entry** (features / tracking -- a count, since
several entries can be in flight) and **per-epoch** (training -- a monotonic cursor). Bracketing
lifecycle events (``started`` / ``run_id`` / ``total`` / ``heartbeat`` / ``finished`` /
``failed`` / ``cancelled``) carry the attempt status the old ``runs`` table held.

The reader helpers reduce a log back to the same dict shape the SQLite ``runs`` row exposed,
so the ``mosaic status`` / ``runs`` / ``cancel`` CLI contract is unchanged.
"""

from __future__ import annotations

import datetime
import json
import os
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final, TypedDict, cast

__all__ = [
    "JsonlRunLog",
    "RunLogSnapshot",
    "TERMINAL_STATUSES",
    "new_execution_id",
    "now_iso",
    "read_run",
    "read_run_progress",
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

    ``tracks_variant`` names the tracks recipes the attempt *read*, comma-joined
    and sorted, empty for a run that reads none. Never what an op *produced* --
    those are different relations and one key cannot hold both, which is what
    would make a downstream column unqueryable.

    Zero, ``False`` and ``""`` are indistinguishable from "not reported", the same
    convention as the tracks index's blank ``n_keypoints`` cell meaning *unknown*
    rather than zero.
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
    cache_hit: bool
    tracks_variant: str


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
    after every line so a concurrent reader only ever sees a partial last line.

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

    def _emit(self, event: str, **fields: Any) -> None:
        if self._file is None:
            return
        record = {"t": now_iso(), "ev": event, **fields}
        line = json.dumps(record, default=str) + "\n"
        with self._lock:
            if self._file is None:
                return
            self._file.write(line)
            self._file.flush()

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
    ) -> None:
        """Bracket the start of the attempt (kind/target/owner/host/pid)."""
        self._emit(
            "started",
            kind=kind,
            target=target,
            owner=owner,
            host=host,
            pid=pid,
            created_at=created_at or now_iso(),
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
        self._emit("finished")

    def failed(self, error_json: str = "") -> None:
        """Terminal: the attempt raised (``error_json`` is a captured-error blob)."""
        self._emit("failed", error=error_json)

    def cancelled(self) -> None:
        """Terminal: the attempt was cooperatively cancelled."""
        self._emit("cancelled")

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
    """Parse a run-log's JSON lines, tolerating a torn/partial last line.

    Returns records in file order. An un-parseable line (a partial append caught
    mid-write on NFS) is skipped rather than raising -- the append-only invariant
    guarantees only the *last* line can ever be partial.
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
        "cache_hit": False,
        "tracks_variant": "",
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
            snap["kind"] = rec.get("kind", "")
            snap["target"] = rec.get("target", "")
            snap["owner"] = rec.get("owner", "")
            snap["host"] = rec.get("host", "")
            snap["pid"] = rec.get("pid", 0)
            snap["created_at"] = rec.get("created_at", ts)
            snap["started_at"] = rec.get("created_at", ts)
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


def read_run(run_dir: Path | str, execution_id: str) -> RunLogSnapshot | None:
    """Read one attempt snapshot by ``execution_id`` (or ``None`` if absent)."""
    return reduce_run_log(Path(run_dir) / f"{execution_id}.jsonl")


def read_runs(
    run_dir: Path | str,
    *,
    kind: str | None = None,
    status: str | None = None,
) -> list[RunLogSnapshot]:
    """Read attempt snapshots (newest first), optionally filtered by kind/status.

    ``execution_id`` is a ULID, so a plain lexicographic descending sort is
    creation-time descending.
    """
    run_dir = Path(run_dir)
    if not run_dir.exists():
        return []
    out: list[RunLogSnapshot] = []
    for p in run_dir.glob("*.jsonl"):
        snap = reduce_run_log(p)
        if snap is None:
            continue
        if kind is not None and snap["kind"] != kind:
            continue
        if status is not None and snap["status"] != status:
            continue
        out.append(snap)
    out.sort(key=lambda r: r["execution_id"], reverse=True)
    return out


def read_runs_by_run_id(run_dir: Path | str, run_id: str) -> list[RunLogSnapshot]:
    """Read all attempts that produced (or targeted) a given content ``run_id``."""
    return [r for r in read_runs(run_dir) if r["run_id"] == run_id]


def read_run_progress(run_dir: Path | str, execution_id: str) -> list[dict[str, Any]]:
    """Return one attempt's progress events in the legacy ``read_progress`` shape.

    Each row has ``step_type`` / ``step_index`` / ``step_total`` / ``metrics`` /
    ``message`` / ``timestamp`` -- matching what the old SQLite ``training_progress``
    reader returned, so ``mosaic status --progress`` is unchanged.
    """
    out: list[dict[str, Any]] = []
    for rec in _iter_records(Path(run_dir) / f"{execution_id}.jsonl"):
        ev = rec.get("ev", "")
        if ev not in _PROGRESS_EVENTS:
            continue
        ts = rec.get("t", "")
        if ev == "entry_end":
            out.append(
                {
                    "step_type": "entry",
                    "step_index": rec.get("index", 0),
                    "step_total": rec.get("total", 0),
                    "metrics": {},
                    "message": rec.get("key", ""),
                    "timestamp": ts,
                }
            )
        elif ev == "entry_start":
            out.append(
                {
                    "step_type": "entry_start",
                    "step_index": rec.get("index", 0),
                    "step_total": rec.get("total", 0),
                    "metrics": {},
                    "message": rec.get("key", ""),
                    "timestamp": ts,
                }
            )
        elif ev == "epoch":
            out.append(
                {
                    "step_type": "epoch",
                    "step_index": rec.get("epoch", 0),
                    "step_total": rec.get("total_epochs", 0),
                    "metrics": rec.get("metrics", {}),
                    "message": "",
                    "timestamp": ts,
                }
            )
        elif ev == "class_start":
            out.append(
                {
                    "step_type": "class",
                    "step_index": rec.get("class_idx", 0),
                    "step_total": rec.get("total_classes", 0),
                    "metrics": {},
                    "message": rec.get("class_name", ""),
                    "timestamp": ts,
                }
            )
        elif ev == "phase":
            out.append(
                {
                    "step_type": "phase",
                    "step_index": 0,
                    "step_total": 0,
                    "metrics": {},
                    "message": f"{rec.get('phase', '')}: {rec.get('message', '')}",
                    "timestamp": ts,
                }
            )
    return out
