"""SQLite-backed training job queue with a single worker thread.

Manages a queue of model training jobs.  Job metadata (status, config,
timestamps, errors) is persisted in the ``training_jobs`` table of
``.mosaic.db`` so that any reader (API, notebook, frontend) can inspect
progress.  Model instances are held in an in-memory queue for the worker
thread to consume.

Typical usage::

    queue = TrainingQueue(ds)
    queue.submit(BehaviorXGBoostModel(), config_a)
    queue.submit(BehaviorXGBoostModel(), config_b)
    queue.start()          # spawns worker thread
    queue.list_jobs()      # returns DataFrame of all jobs
    queue.stop()           # waits for current job, stops worker
"""

from __future__ import annotations

import json
import os
import queue
import sqlite3
import sys
import threading
import traceback
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from ._utils import hash_params, json_ready, now_iso
from .progress import NullProgressCallback, SQLiteProgressCallback

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset

# sentinel to signal the worker thread to stop
_STOP = object()


class TrainingQueue:
    """Persistent job queue backed by SQLite with an in-memory worker.

    Parameters
    ----------
    ds : Dataset
        The dataset whose ``.mosaic.db`` stores job metadata.
    """

    def __init__(self, ds: Dataset) -> None:
        self._ds = ds
        self._db_path = ds.get_root("features") / ".mosaic.db"
        self._conn = sqlite3.connect(str(self._db_path), timeout=10)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        # Ensure tables exist (idempotent — uses CREATE IF NOT EXISTS)
        from .registry import _SCHEMA
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

        self._memory_queue: queue.Queue[Any] = queue.Queue()
        self._worker_thread: threading.Thread | None = None
        self._running = False

    # -- submit / cancel ----------------------------------------------------

    def submit(
        self,
        model: Any,
        config: str | Path | dict[str, object] | None = None,
        priority: int = 0,
    ) -> str:
        """Add a training job to the queue.

        Parameters
        ----------
        model
            Model instance implementing name, version, and train().
        config
            Path to JSON config, a dict, or None.
        priority
            Higher numbers run first (default 0).

        Returns
        -------
        str
            The job_id (UUID) for tracking.
        """
        from .models import load_model_config

        model_name = getattr(
            model, "storage_model_name", getattr(model, "name", None)
        )
        if not model_name:
            raise ValueError("Model must define 'name' or 'storage_model_name'.")
        model_version = getattr(model, "version", "unknown")
        config_dict = load_model_config(config)

        job_id = uuid.uuid4().hex[:12]
        self._conn.execute(
            """\
            INSERT INTO training_jobs
                (job_id, model_name, model_version, config_json, status,
                 priority, created_at)
            VALUES (?, ?, ?, ?, 'pending', ?, ?)
            """,
            (
                job_id,
                model_name,
                model_version,
                json.dumps(json_ready(config_dict)),
                priority,
                now_iso(),
            ),
        )
        self._conn.commit()

        # Enqueue the in-memory work item
        self._memory_queue.put((job_id, model, config_dict))

        print(
            f"[queue] submitted job {job_id} "
            f"(model={model_name}, priority={priority})"
        )
        return job_id

    def cancel(self, job_id: str) -> bool:
        """Cancel a pending job.

        Only jobs with ``status='pending'`` can be cancelled.  Running
        jobs are not interrupted.

        Args:
            job_id: The job to cancel.

        Returns:
            True if the job was cancelled, False if it was already
            running, completed, or not found.
        """
        cur = self._conn.execute(
            """\
            UPDATE training_jobs SET status = 'cancelled', finished_at = ?
            WHERE job_id = ? AND status = 'pending'
            """,
            (now_iso(), job_id),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def clear_stale(self) -> int:
        """Mark orphaned ``running`` and ``pending`` jobs as ``cancelled``.

        After a kernel restart, jobs that were running or waiting in the
        in-memory queue will never complete.  This method cleans them up
        so :meth:`list_jobs` shows an accurate picture.

        Returns:
            Number of jobs marked as cancelled.
        """
        cur = self._conn.execute(
            """\
            UPDATE training_jobs SET status = 'cancelled', finished_at = ?
            WHERE status IN ('running', 'pending')
            """,
            (now_iso(),),
        )
        self._conn.commit()
        n = cur.rowcount
        if n:
            print(f"[queue] cleared {n} stale job(s)")
        return n

    def resubmit(self, job_id: str, model: Any) -> str:
        """Re-submit a failed or cancelled job with the same config.

        Parameters
        ----------
        job_id
            The job to resubmit.
        model
            A fresh model instance (same class as the original).

        Returns
        -------
        str
            The new job_id.
        """
        row = self._conn.execute(
            "SELECT config_json, priority FROM training_jobs WHERE job_id = ?",
            (job_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Job not found: {job_id}")
        config_dict = json.loads(row[0])
        priority = row[1]
        return self.submit(model, config=config_dict, priority=priority)

    # -- worker lifecycle ---------------------------------------------------

    def start(self) -> None:
        """Start the background worker thread (idempotent)."""
        if self._running:
            return
        self._running = True
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="mosaic-training-worker",
            daemon=True,
        )
        self._worker_thread.start()
        print("[queue] worker started")

    def stop(self, wait: bool = True) -> None:
        """Signal the worker to stop after the current job finishes.

        Args:
            wait: Block until the worker thread exits (default True).
        """
        if not self._running:
            return
        self._running = False
        self._memory_queue.put(_STOP)
        if wait and self._worker_thread is not None:
            self._worker_thread.join(timeout=5)
        print("[queue] worker stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    # -- queries ------------------------------------------------------------

    def list_jobs(self) -> pd.DataFrame:
        """Return all jobs as a DataFrame.

        Rows are ordered: running first, then pending (by priority),
        then completed, failed, and cancelled.

        Returns:
            DataFrame with columns ``job_id``, ``model_name``,
            ``model_version``, ``status``, ``priority``, ``created_at``,
            ``started_at``, ``finished_at``, ``run_id``, ``error``.
        """
        return pd.read_sql_query(
            """\
            SELECT job_id, model_name, model_version, status, priority,
                   created_at, started_at, finished_at, run_id, error
            FROM training_jobs
            ORDER BY
                CASE status
                    WHEN 'running' THEN 0
                    WHEN 'pending' THEN 1
                    WHEN 'success' THEN 2
                    WHEN 'failed'  THEN 3
                    WHEN 'cancelled' THEN 4
                END,
                priority DESC, created_at ASC
            """,
            self._conn,
        )

    def get_job(self, job_id: str) -> dict[str, Any]:
        """Return full detail for a single job.

        Args:
            job_id: The job to look up.

        Returns:
            Dict with all job columns.  The ``config_json`` column is
            parsed into a ``config`` dict.

        Raises:
            ValueError: If the job_id does not exist.
        """
        row = self._conn.execute(
            "SELECT * FROM training_jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"Job not found: {job_id}")
        cols = [d[0] for d in self._conn.execute(
            "SELECT * FROM training_jobs LIMIT 0"
        ).description]
        d = dict(zip(cols, row))
        d["config"] = json.loads(d.pop("config_json", "{}"))
        return d

    def get_progress(self, job_id: str) -> pd.DataFrame:
        """Return epoch-level progress for a running or completed job.

        Reads from the ``training_progress`` table, which is written to
        in real time by the training callback.  Safe to call while
        training is in progress.

        Args:
            job_id: The job to query.

        Returns:
            DataFrame with columns ``step_type``, ``step_index``,
            ``step_total``, ``metrics``, ``message``, ``timestamp``.
            Metrics is a dict of metric name-value pairs.  Returns an
            empty DataFrame with these columns if no progress has been
            recorded yet.
        """
        from .progress import read_progress

        _empty_cols = [
            "step_type", "step_index", "step_total",
            "metrics", "message", "timestamp",
        ]
        rows = read_progress(self._db_path, job_id)
        if not rows:
            return pd.DataFrame(columns=_empty_cols)
        return pd.DataFrame(rows)

    def pending_count(self) -> int:
        """Return the number of pending jobs."""
        row = self._conn.execute(
            "SELECT COUNT(*) FROM training_jobs WHERE status = 'pending'"
        ).fetchone()
        return int(row[0]) if row else 0

    # -- worker loop --------------------------------------------------------

    def _worker_loop(self) -> None:
        """Main loop for the worker thread.

        Opens its own SQLite connection (required — sqlite3 objects are
        thread-local) and processes jobs from the in-memory queue.
        """
        from .models import train_model

        # Worker thread needs its own connection
        conn = sqlite3.connect(str(self._db_path), timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")

        try:
            while self._running:
                try:
                    item = self._memory_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                if item is _STOP:
                    break

                job_id, model, config_dict = item

                # Check if this job was cancelled while waiting
                row = conn.execute(
                    "SELECT status FROM training_jobs WHERE job_id = ?",
                    (job_id,),
                ).fetchone()
                if row is None or row[0] == "cancelled":
                    continue

                # Mark as running
                conn.execute(
                    """\
                    UPDATE training_jobs
                    SET status = 'running', started_at = ?, worker_pid = ?
                    WHERE job_id = ?
                    """,
                    (now_iso(), os.getpid(), job_id),
                )
                conn.commit()

                model_name = getattr(
                    model, "storage_model_name", getattr(model, "name", "?")
                )
                print(f"[queue] running job {job_id} (model={model_name})")

                # Create progress callback (opens its own connection)
                progress = SQLiteProgressCallback(self._db_path, job_id)

                try:
                    run_id = train_model(
                        self._ds,
                        model,
                        config=config_dict,
                        job_id=job_id,
                        progress_callback=progress,
                    )
                    conn.execute(
                        """\
                        UPDATE training_jobs
                        SET status = 'success', finished_at = ?, run_id = ?
                        WHERE job_id = ?
                        """,
                        (now_iso(), run_id, job_id),
                    )
                    conn.commit()
                    print(f"[queue] job {job_id} completed (run_id={run_id})")
                except Exception as exc:
                    tb = traceback.format_exc()
                    error_text = tb[-5000:]  # truncate to last 5000 chars
                    conn.execute(
                        """\
                        UPDATE training_jobs
                        SET status = 'failed', finished_at = ?, error = ?
                        WHERE job_id = ?
                        """,
                        (now_iso(), error_text, job_id),
                    )
                    conn.commit()
                    print(
                        f"[queue] job {job_id} failed: {exc}",
                        file=sys.stderr,
                    )
                finally:
                    progress.close()
        finally:
            conn.close()

    # -- cleanup ------------------------------------------------------------

    def close(self) -> None:
        self.stop(wait=True)
        self._conn.close()

    def __enter__(self) -> TrainingQueue:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
