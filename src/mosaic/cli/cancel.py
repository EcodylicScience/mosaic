"""``mosaic cancel``: best-effort, single-box cooperative cancel of a running attempt.

The library cannot cooperatively signal *another* process, so this sends
SIGTERM to the pid recorded in the attempt's run-log (which ``install_signal_handler``
in the target process converts to a cooperative cancel). It lands at the op's
next checkpoint; real process-group hard-kill is the Layer-2 executor's job.
Cross-host cancel is refused.
"""

from __future__ import annotations

import os
import signal
import socket
from pathlib import Path
from typing import Annotated

import typer

from mosaic.cli._context import load_dataset, run_log_dir_for
from mosaic.cli._io import emit_json, fail, log


def cancel_command(
    manifest: Annotated[
        Path,
        typer.Option(
            "--manifest", "-m", help="Path to the dataset manifest (dataset.yaml)."
        ),
    ],
    execution_id: Annotated[
        str, typer.Option("--execution-id", help="Attempt ULID to cancel.")
    ],
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit the result as JSON on stdout.")
    ] = False,
) -> None:
    """Request cancellation of a running attempt (SIGTERM to its recorded pid)."""
    from mosaic.core.pipeline.run_log import read_run

    ds = load_dataset(manifest)
    run_dir = run_log_dir_for(ds)
    if not run_dir.exists():
        fail("No run-logs found (nothing has run yet).")
    row = read_run(run_dir, execution_id)
    if row is None:
        fail(f"No run found with execution_id={execution_id}.")

    status = row["status"]
    if status not in ("running", "queued"):
        _emit(
            {"execution_id": execution_id, "status": status, "signalled": False},
            as_json,
            f"[mosaic] run {execution_id} already {status}; nothing to cancel.",
        )
        return

    host = row["host"]
    if host and host != socket.gethostname():
        fail(f"Run is on host {host!r}; cross-host cancel is a Layer-2 concern.")

    pid = row["pid"]
    if pid <= 0:
        fail("No pid recorded for this run; cannot signal it.")

    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        _reap(run_dir, execution_id)
        _emit(
            {
                "execution_id": execution_id,
                "pid": pid,
                "status": "cancelled",
                "signalled": False,
            },
            as_json,
            f"[mosaic] pid {pid} for run {execution_id} had already exited; "
            f"recorded the run as cancelled.",
        )
        return
    except PermissionError:
        fail(f"Not permitted to signal process {pid}.")

    _emit(
        {
            "execution_id": execution_id,
            "pid": pid,
            "status": "cancelling",
            "signalled": True,
        },
        as_json,
        f"[mosaic] sent SIGTERM to pid {pid} for run {execution_id}.",
    )


def _reap(run_dir: Path, execution_id: str) -> None:
    """Record the terminal event for an attempt whose process is already gone.

    Reported as success, not failure: what the user asked for has happened,
    just by something other than the signal. Leaving the run-log alone is what
    made this the one state that is neither live nor reclaimable -- the attempt
    stays ``running`` forever, and ``inflight_state`` reads a holder as
    reclaimable only once its run-log has gone terminal, so the run root stays
    claimed until the marker expires.

    **Scoped to the branch where the process is gone, and safe only there.**
    ``JsonlRunLog`` is one file, one writer; writing another process's log is
    safe here because there is no other process. In the branch where SIGTERM
    was delivered the writer is alive and still holds the file, and the cancel
    it received is what makes it write its own terminal event.
    """
    from mosaic.runlog import JsonlRunLog

    with JsonlRunLog(run_dir / f"{execution_id}.jsonl", execution_id) as run_log:
        run_log.cancelled()


def _emit(payload: dict[str, object], as_json: bool, human: str) -> None:
    if as_json:
        emit_json(payload)
    else:
        log(human)
