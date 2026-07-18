"""The Job Contract -- the one attempt/progress/cancel spine for compute.

Every unit of backend compute (a feature run, a tracking op, and -- via
``tracking.run_trex`` -- an external tracking run) enters :func:`job_context`. It
gives each job a single, uniform lifecycle:

* an **``execution_id``** (ULID) identifying this *attempt*, distinct from the
  content-addressed ``run_id`` identifying a *result*. Two attempts of the same
  work share a ``run_id`` but get different ``execution_id``\\s;
* one **append-only JSONL run-log** per attempt (:class:`JsonlRunLog`, under
  ``<dataset_root>/.mosaic/runs/<execution_id>.jsonl``) recording the lifecycle
  ``running`` -> ``finished`` | ``failed`` | ``cancelled`` with an error capture
  and a liveness heartbeat -- replacing the retired ``runs`` SQLite table;
* a :class:`ProgressCallback` bound to ``job_id = execution_id`` (so feature
  entries, training epochs, and trex phases all land in that same one log);
* a cooperative :class:`CancelToken` (hard-kill of blocked subprocesses stays
  the executor's job at a higher layer).

Glossary (names collide easily, so pin them down):

* ``run_id``       -- content hash of a *result*; the key of the ``index.csv`` ledger.
* ``execution_id`` -- ULID of one *attempt*; the run-log filename and the
  ``job_id`` of every progress event. The join key across all layers.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import sys
import threading
import time
import traceback
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ._utils import new_execution_id
from .progress import NullProgressCallback, ProgressCallback
from .run_log import JsonlRunLog, run_log_path

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset


# ---------------------------------------------------------------------------
# Cooperative cancellation
# ---------------------------------------------------------------------------


class Cancelled(Exception):
    """Raised inside a job when a cooperative cancel has been requested."""


class CancelToken:
    """A cooperative, thread-safe cancellation signal.

    The default token is inert -- it never fires unless some holder calls
    :meth:`cancel` -- so ordinary (notebook) runs are unaffected. Job code polls
    it at safe checkpoints (e.g. between manifest entries) via
    :meth:`raise_if_cancelled`.
    """

    def __init__(self, event: threading.Event | None = None) -> None:
        self._event = event or threading.Event()

    def cancel(self) -> None:
        self._event.set()

    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        if self._event.is_set():
            raise Cancelled()


def install_signal_handler(token: CancelToken) -> None:
    """Route SIGTERM/SIGINT to ``token.cancel()`` (for ``mosaic run``).

    No-ops silently when not on the main thread (where signal handlers can't be
    installed), e.g. inside a worker pool.
    """

    def _handler(signum: int, frame: object) -> None:
        token.cancel()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _handler)
        except (ValueError, OSError):
            pass


# ---------------------------------------------------------------------------
# Job context
# ---------------------------------------------------------------------------

_HEARTBEAT_EVERY = 10  # write a heartbeat at least every N entries
_HEARTBEAT_SECONDS = 15.0  # ...and at least this often for slow single-entry jobs


def _capture_error(exc: BaseException) -> str:
    return json.dumps(
        {
            "type": type(exc).__name__,
            "message": str(exc)[:2000],
            "traceback": traceback.format_exc()[:8000],
        }
    )


@dataclass
class JobContext:
    """Handle yielded by :func:`job_context` for the duration of one attempt."""

    execution_id: str
    kind: str  # "feature" | "trex" | a tracking-op kind | ...
    target: str  # storage feature name / op label
    run_log: JsonlRunLog | None
    progress: ProgressCallback
    cancel_token: CancelToken
    owner: str = ""
    run_id: str | None = None
    _total: int = 0
    _done: int = 0
    _hb_count: int = 0
    _hb_last: float = field(default=0.0, repr=False)

    @property
    def total(self) -> int:
        """Total number of entries declared for this attempt (0 if unknown)."""
        return self._total

    @property
    def done(self) -> int:
        """Entries completed so far in this attempt."""
        return self._done

    def set_run_id(self, run_id: str) -> None:
        """Record the content-addressed ``run_id`` once the job computes it."""
        self.run_id = run_id
        if self.run_log is not None:
            self.run_log.set_run_id(run_id)

    def set_total(self, total: int) -> None:
        """Declare the total number of entries (for progress denominators)."""
        self._total = total
        if self.run_log is not None:
            self.run_log.set_total(total)

    def heartbeat(self, done: int | None = None) -> None:
        """Refresh liveness (throttled) and, optionally, the completed count."""
        if done is not None:
            self._done = done
        if self.run_log is None:
            return
        self._hb_count += 1
        now = time.monotonic()
        if (
            self._hb_count % _HEARTBEAT_EVERY == 0
            or (now - self._hb_last) >= _HEARTBEAT_SECONDS
        ):
            self.run_log.heartbeat(self._done, self._total)
            self._hb_last = now

    def check_cancel(self) -> None:
        """Raise :class:`Cancelled` if a cancel has been requested."""
        self.cancel_token.raise_if_cancelled()


@contextmanager
def job_context(
    ds: "Dataset",
    *,
    kind: str,
    target: str,
    execution_id: str | None = None,
    owner: str = "",
    track: bool = True,
    progress_callback: ProgressCallback | None = None,
    cancel_token: CancelToken | None = None,
    total: int = 0,
) -> Generator[JobContext]:
    """Run a block as a tracked job attempt.

    Parameters
    ----------
    ds:
        Dataset whose ``.mosaic/runs/`` directory records the attempt's run-log.
    kind, target:
        Job classification (``"feature"`` / ``"trex"`` / a tracking-op kind /
        future payload kinds) and its subject. ``kind`` is a field in the log,
        never part of the path -- a new payload kind needs no change here.
    execution_id:
        Reuse an externally minted ULID (how a Layer-2 subprocess inherits its
        identity); otherwise a fresh one is generated.
    track:
        When ``False`` (or the dataset base dir is unresolvable) nothing is
        recorded -- the "bare run leaves no trace" behaviour.
    progress_callback, cancel_token:
        Optional injected backends; sensible defaults are provided. When a
        progress callback is injected, per-entry/per-epoch events go to it and
        only coarse progress (via :meth:`JobContext.heartbeat`) lands in the log.
    """
    execution_id = execution_id or new_execution_id()

    run_log: JsonlRunLog | None = None
    if track:
        try:
            run_log = JsonlRunLog(
                run_log_path(ds.base_dir, execution_id), execution_id
            )
        except Exception as exc:  # dataset without a base dir, unwritable FS, etc.
            print(
                f"[job] tracking disabled (could not open run-log): {exc}",
                file=sys.stderr,
            )
            run_log = None

    progress: ProgressCallback
    if progress_callback is not None:
        progress = progress_callback
    elif run_log is not None:
        progress = run_log
    else:
        progress = NullProgressCallback()

    token = cancel_token or CancelToken()
    host = socket.gethostname()
    pid = os.getpid()
    ctx = JobContext(
        execution_id=execution_id,
        kind=kind,
        target=target,
        run_log=run_log,
        progress=progress,
        cancel_token=token,
        owner=owner,
        _total=total,
    )

    if run_log is not None:
        run_log.started(kind=kind, target=target, owner=owner, host=host, pid=pid)
        if total:
            run_log.set_total(total)

    try:
        yield ctx
    except Cancelled:
        if run_log is not None:
            run_log.cancelled()
        raise
    except Exception as exc:
        if run_log is not None:
            run_log.failed(_capture_error(exc))
        raise
    else:
        if run_log is not None:
            run_log.finished()
    finally:
        if run_log is not None:
            try:
                run_log.close()
            except Exception:
                pass
