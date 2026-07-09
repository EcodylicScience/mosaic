"""The Job Contract -- the one attempt/progress/cancel spine for compute.

Every unit of backend compute (a feature run, and -- via ``tracking.run_trex``
-- an external tracking run) enters :func:`job_context`. It gives each job a
single, uniform lifecycle:

* an **``execution_id``** (ULID) identifying this *attempt*, distinct from the
  content-addressed ``run_id`` identifying a *result*. Two attempts of the same
  work share a ``run_id`` but get different ``execution_id``\\s;
* a durable attempt row in the ``runs`` table of ``.mosaic.db``, transitioning
  ``running`` -> ``finished`` | ``failed`` | ``cancelled`` with an error capture
  and a liveness heartbeat;
* a :class:`ProgressCallback` bound to ``job_id = execution_id`` (so feature
  entries, training epochs, and trex phases all land in one progress store);
* a cooperative :class:`CancelToken` (hard-kill of blocked subprocesses stays
  the executor's job at a higher layer).

Glossary (names collide easily, so pin them down):

* ``run_id``       -- content hash of a *result*; the key of ``feature_runs``.
* ``execution_id`` -- ULID of one *attempt*; the key of ``runs`` and the
  ``job_id`` of ``training_progress``. The join key across all layers.
* ``runs``         -- the attempt ledger (this module). Not ``feature_runs``.
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
from .progress import (
    NullProgressCallback,
    ProgressCallback,
    SQLiteProgressCallback,
)
from .registry import FeatureRegistry, open_registry

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
    kind: str  # "feature" | "trex"
    target: str  # storage feature name / trex operation
    registry: FeatureRegistry | None
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
        if self.registry is not None:
            self.registry.set_attempt_run_id(self.execution_id, run_id)

    def set_total(self, total: int) -> None:
        """Declare the total number of entries (for progress denominators)."""
        self._total = total
        if self.registry is not None:
            self.registry.heartbeat_attempt(self.execution_id, progress_total=total)

    def heartbeat(self, done: int | None = None) -> None:
        """Refresh liveness (throttled) and, optionally, the completed count."""
        if done is not None:
            self._done = done
        if self.registry is None:
            return
        self._hb_count += 1
        now = time.monotonic()
        if (
            self._hb_count % _HEARTBEAT_EVERY == 0
            or (now - self._hb_last) >= _HEARTBEAT_SECONDS
        ):
            self.registry.heartbeat_attempt(
                self.execution_id, progress_done=self._done, progress_total=self._total
            )
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
    registry: FeatureRegistry | None = None,
    track: bool = True,
    progress_callback: ProgressCallback | None = None,
    cancel_token: CancelToken | None = None,
    total: int = 0,
) -> Generator[JobContext]:
    """Run a block as a tracked job attempt.

    Parameters
    ----------
    ds:
        Dataset whose ``.mosaic.db`` records the attempt.
    kind, target:
        Job classification (``"feature"``/``"trex"``) and its subject.
    execution_id:
        Reuse an externally minted ULID (how a Layer-2 subprocess inherits its
        identity); otherwise a fresh one is generated.
    registry:
        A caller-owned registry (e.g. from ``Pipeline.run``). It is used but
        **not** closed here. When ``None`` and ``track`` is set, a registry is
        opened and owned (closed on exit).
    track:
        When ``False`` (and no registry is passed) nothing is recorded -- the
        legacy "bare run leaves no trace" behaviour.
    progress_callback, cancel_token:
        Optional injected backends; sensible defaults are provided.
    """
    execution_id = execution_id or new_execution_id()

    owns_registry = False
    reg = registry
    if reg is None and track:
        try:
            reg = open_registry(ds.get_root("features"))
            owns_registry = True
        except Exception as exc:  # dataset without a features root, etc.
            print(
                f"[job] tracking disabled (could not open registry): {exc}",
                file=sys.stderr,
            )
            reg = None

    owns_progress = False
    progress: ProgressCallback
    if progress_callback is not None:
        progress = progress_callback
    elif reg is not None:
        progress = SQLiteProgressCallback(reg.db_path, execution_id)
        owns_progress = True
    else:
        progress = NullProgressCallback()

    token = cancel_token or CancelToken()
    ctx = JobContext(
        execution_id=execution_id,
        kind=kind,
        target=target,
        registry=reg,
        progress=progress,
        cancel_token=token,
        owner=owner,
        _total=total,
    )

    if reg is not None:
        reg.record_attempt(
            execution_id,
            kind,
            target,
            owner=owner,
            host=socket.gethostname(),
            pid=os.getpid(),
            status="running",
            progress_total=total,
        )

    try:
        yield ctx
    except Cancelled:
        if reg is not None:
            reg.finish_attempt(execution_id, "cancelled")
        raise
    except Exception as exc:
        if reg is not None:
            reg.finish_attempt(execution_id, "failed", error_json=_capture_error(exc))
        raise
    else:
        if reg is not None:
            reg.finish_attempt(execution_id, "finished")
    finally:
        if owns_progress:
            try:
                progress.close()  # type: ignore[attr-defined]
            except Exception:
                pass
        if owns_registry and reg is not None:
            try:
                reg.close()
            except Exception:
                pass
