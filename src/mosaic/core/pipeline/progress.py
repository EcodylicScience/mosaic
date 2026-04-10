"""Training progress callback protocol and implementations.

Provides a lightweight callback that training loops call at regular
intervals (per-epoch, per-class, per-phase) to persist progress into the
``training_progress`` table of ``.mosaic.db``.  The same table is read by
the API layer (via SSE) for live monitoring.

Also provides ``CSVProgressCallback`` for append-mode CSV writing that
is readable mid-training (no database required).
"""

from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ._utils import now_iso


# ---------------------------------------------------------------------------
# Protocol (stable interface for future backends)
# ---------------------------------------------------------------------------


@runtime_checkable
class TrainingProgressCallback(Protocol):
    """Minimal callback contract for training progress reporting.

    Implement this protocol to create custom progress backends (e.g.
    MLflow, W&B).  The three methods correspond to different granularities
    of training progress.
    """

    def on_epoch_end(
        self,
        epoch: int,
        total_epochs: int,
        metrics: dict[str, float],
    ) -> None:
        """Called after each training epoch.

        Args:
            epoch: Zero-based epoch index.
            total_epochs: Total number of epochs planned.
            metrics: Metric name-value pairs (e.g. ``{"train_loss": 0.45}``).
        """
        ...

    def on_class_start(
        self,
        class_idx: int,
        total_classes: int,
        class_name: str,
    ) -> None:
        """Called when a one-vs-rest class training begins.

        Args:
            class_idx: Zero-based class index.
            total_classes: Total number of classes.
            class_name: Human-readable class identifier.
        """
        ...

    def on_phase(self, phase: str, message: str) -> None:
        """Called for coarse-grained phase transitions.

        Args:
            phase: Phase identifier (e.g. ``"data_prep"``, ``"training"``).
            message: Free-text description.
        """
        ...


# ---------------------------------------------------------------------------
# Null implementation (no-op default)
# ---------------------------------------------------------------------------


class NullProgressCallback:
    """No-op callback used when no monitoring is requested."""

    def on_epoch_end(
        self, epoch: int, total_epochs: int, metrics: dict[str, float]
    ) -> None:
        pass

    def on_class_start(
        self, class_idx: int, total_classes: int, class_name: str
    ) -> None:
        pass

    def on_phase(self, phase: str, message: str) -> None:
        pass


# ---------------------------------------------------------------------------
# CSV implementation (append-mode, readable mid-training)
# ---------------------------------------------------------------------------


class CSVProgressCallback:
    """Append-mode CSV writer for epoch metrics.

    Writes one row per ``on_epoch_end`` call.  The file is flushed after
    each write so it can be read from another process (e.g. a notebook
    cell or ``cat`` from the terminal) while training is still running.

    Example::

        cb = CSVProgressCallback(run_root / "summary.csv")
        for epoch in range(n_epochs):
            ...
            cb.on_epoch_end(epoch, n_epochs, {"train_loss": 0.3, "val_f1": 0.9})
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._fieldnames: list[str] | None = None
        self._file = None
        self._writer = None

    def on_epoch_end(
        self,
        epoch: int,
        total_epochs: int,
        metrics: dict[str, float],
    ) -> None:
        row = {"epoch": epoch, "total_epochs": total_epochs, **metrics}

        if self._file is None:
            # First call -- open file in append mode
            write_header = not (self.path.exists() and self.path.stat().st_size > 0)
            self._fieldnames = list(row.keys())
            self._file = open(self.path, "a", newline="")
            self._writer = csv.DictWriter(
                self._file, fieldnames=self._fieldnames, extrasaction="ignore"
            )
            if write_header:
                self._writer.writeheader()

        self._writer.writerow(row)  # type: ignore[union-attr]
        self._file.flush()  # type: ignore[union-attr]

    def on_class_start(
        self, class_idx: int, total_classes: int, class_name: str
    ) -> None:
        pass

    def on_phase(self, phase: str, message: str) -> None:
        pass

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None


# ---------------------------------------------------------------------------
# SQLite implementation
# ---------------------------------------------------------------------------


class SQLiteProgressCallback:
    """Writes progress rows into the ``training_progress`` table.

    Opens its own connection (WAL mode) so that the training thread and
    any reader (API, notebook) do not block each other.

    Parameters
    ----------
    db_path : Path
        Path to the ``.mosaic.db`` file.
    job_id : str
        The training job this progress belongs to.
    """

    def __init__(self, db_path: Path, job_id: str) -> None:
        self.db_path = db_path
        self.job_id = job_id
        self._conn = sqlite3.connect(str(db_path), timeout=10)
        self._conn.execute("PRAGMA journal_mode=WAL")

    # -- callback methods ---------------------------------------------------

    def on_epoch_end(
        self,
        epoch: int,
        total_epochs: int,
        metrics: dict[str, float],
    ) -> None:
        """Record an epoch completion with associated metrics.

        Args:
            epoch: Zero-based epoch index.
            total_epochs: Total epochs planned.
            metrics: Metric name-value pairs.
        """
        self._write("epoch", epoch, total_epochs, metrics)

    def on_class_start(
        self,
        class_idx: int,
        total_classes: int,
        class_name: str,
    ) -> None:
        """Record the start of a one-vs-rest class training iteration.

        Args:
            class_idx: Zero-based class index.
            total_classes: Total number of classes.
            class_name: Human-readable class identifier.
        """
        self._write("class", class_idx, total_classes, message=class_name)

    def on_phase(self, phase: str, message: str) -> None:
        """Record a coarse-grained phase transition.

        Args:
            phase: Phase identifier.
            message: Free-text description.
        """
        self._write("phase", 0, 0, message=f"{phase}: {message}")

    # -- query helper -------------------------------------------------------

    def get_progress(self) -> list[dict[str, Any]]:
        """Return all progress rows for this job, ordered chronologically."""
        cur = self._conn.execute(
            """\
            SELECT step_type, step_index, step_total, metric_json, message, timestamp
            FROM training_progress
            WHERE job_id = ?
            ORDER BY timestamp, step_index
            """,
            (self.job_id,),
        )
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
        out = []
        for row in rows:
            d = dict(zip(cols, row))
            d["metrics"] = json.loads(d.pop("metric_json", "{}"))
            out.append(d)
        return out

    # -- internals ----------------------------------------------------------

    def _write(
        self,
        step_type: str,
        step_index: int,
        step_total: int = 0,
        metrics: dict[str, float] | None = None,
        message: str = "",
    ) -> None:
        self._conn.execute(
            """\
            INSERT OR REPLACE INTO training_progress
                (job_id, step_type, step_index, step_total, metric_json,
                 message, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                self.job_id,
                step_type,
                step_index,
                step_total,
                json.dumps(metrics or {}),
                message,
                now_iso(),
            ),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> SQLiteProgressCallback:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Composite (for running multiple backends simultaneously)
# ---------------------------------------------------------------------------


class CompositeProgressCallback:
    """Fans out calls to multiple callback backends.

    Example::

        cb = CompositeProgressCallback(
            SQLiteProgressCallback(db, job_id),
            some_mlflow_callback,
        )
    """

    def __init__(self, *backends: TrainingProgressCallback) -> None:
        self._backends = backends

    def on_epoch_end(
        self, epoch: int, total_epochs: int, metrics: dict[str, float]
    ) -> None:
        for b in self._backends:
            b.on_epoch_end(epoch, total_epochs, metrics)

    def on_class_start(
        self, class_idx: int, total_classes: int, class_name: str
    ) -> None:
        for b in self._backends:
            b.on_class_start(class_idx, total_classes, class_name)

    def on_phase(self, phase: str, message: str) -> None:
        for b in self._backends:
            b.on_phase(phase, message)


# ---------------------------------------------------------------------------
# Standalone reader (for API / notebooks)
# ---------------------------------------------------------------------------


def read_progress(db_path: Path, job_id: str) -> list[dict[str, Any]]:
    """Read progress for a job without creating a full callback instance.

    Opens a read-only connection, queries, and closes immediately.
    Suitable for one-shot reads from an API endpoint or notebook.

    Args:
        db_path: Path to the ``.mosaic.db`` file.
        job_id: The training job to query.

    Returns:
        List of progress dicts, each with keys ``step_type``,
        ``step_index``, ``step_total``, ``metrics``, ``message``,
        and ``timestamp``.
    """
    conn = sqlite3.connect(str(db_path), timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        cur = conn.execute(
            """\
            SELECT step_type, step_index, step_total, metric_json, message, timestamp
            FROM training_progress
            WHERE job_id = ?
            ORDER BY timestamp, step_index
            """,
            (job_id,),
        )
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
        out = []
        for row in rows:
            d = dict(zip(cols, row))
            d["metrics"] = json.loads(d.pop("metric_json", "{}"))
            out.append(d)
        return out
    finally:
        conn.close()
