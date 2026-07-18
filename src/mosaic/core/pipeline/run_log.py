"""Append-only JSONL run-log -- the one progress + attempt-status sink per job.

Replaces the per-dataset SQLite ``.mosaic.db`` (execution-layer.md 3). Every unit of
compute that enters the Job Contract (:mod:`mosaic.core.pipeline.job`) -- a feature run, a
tracking op (``extract-frames`` / ``train-*`` / ``infer-*``), TREx, and future payloads
(media transcode, a SLEAP plugin) -- emits its lifecycle and coarse progress to **one**
append-only JSONL file named by its ``execution_id``, under
``<dataset_root>/.mosaic/runs/``.

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

The reader helpers (:func:`read_run` / :func:`read_runs` / :func:`read_run_progress`) reduce a
log back to the same dict shape the SQLite ``runs`` row exposed, so the ``mosaic status`` /
``runs`` / ``cancel`` CLI contract is unchanged. They are stdlib-only, so an external tool
(mosaic-api's ledger sweeper) can read a log without importing the rest of mosaic.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, cast

from ._utils import now_iso

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

    # -- progress protocol --------------------------------------------------

    def on_entry_start(self, index: int, total: int, key: str) -> None:
        self._emit("entry_start", index=index, total=total, key=key)

    def on_entry_end(self, index: int, total: int, key: str) -> None:
        self._emit("entry_end", index=index, total=total, key=key)

    def on_epoch_end(
        self, epoch: int, total_epochs: int, metrics: dict[str, float]
    ) -> None:
        self._emit("epoch", epoch=epoch, total_epochs=total_epochs, metrics=dict(metrics))

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

_TERMINAL = {"finished", "failed", "cancelled"}
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


def reduce_run_log(path: Path) -> dict[str, Any] | None:
    """Fold a run-log JSONL into an attempt-status snapshot (``None`` if empty/unreadable).

    The returned keys mirror the old SQLite ``runs`` row so the CLI contract is unchanged::

        execution_id, kind, target, run_id, status, owner, host, pid, created_at,
        started_at, heartbeat_at, finished_at, error_json, progress_done, progress_total
    """
    path = Path(path)
    snap: dict[str, Any] = {
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
        elif ev in _TERMINAL:
            snap["status"] = ev
            snap["finished_at"] = ts
            if ev == "failed":
                snap["error_json"] = rec.get("error", "")
    return snap


def read_run(run_dir: Path | str, execution_id: str) -> dict[str, Any] | None:
    """Read one attempt snapshot by ``execution_id`` (or ``None`` if absent)."""
    return reduce_run_log(Path(run_dir) / f"{execution_id}.jsonl")


def read_runs(
    run_dir: Path | str,
    *,
    kind: str | None = None,
    status: str | None = None,
) -> list[dict[str, Any]]:
    """Read attempt snapshots (newest first), optionally filtered by kind/status.

    ``execution_id`` is a ULID, so a plain lexicographic descending sort is
    creation-time descending.
    """
    run_dir = Path(run_dir)
    if not run_dir.exists():
        return []
    out: list[dict[str, Any]] = []
    for p in run_dir.glob("*.jsonl"):
        snap = reduce_run_log(p)
        if snap is None:
            continue
        if kind is not None and snap.get("kind") != kind:
            continue
        if status is not None and snap.get("status") != status:
            continue
        out.append(snap)
    out.sort(key=lambda r: str(r.get("execution_id", "")), reverse=True)
    return out


def read_runs_by_run_id(run_dir: Path | str, run_id: str) -> list[dict[str, Any]]:
    """Read all attempts that produced (or targeted) a given content ``run_id``."""
    return [r for r in read_runs(run_dir) if r.get("run_id") == run_id]


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
