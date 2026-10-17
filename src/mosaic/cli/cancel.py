"""``mosaic cancel``: send SIGTERM to a running attempt, or to a request's driver.

The process that receives SIGTERM cancels at its next checkpoint and records its
own ending. :mod:`mosaic.core.pipeline.cancel` decides what to signal and what to
record, and refuses a process on another host unless its run-log proves it gone.
``--assume-exited`` records one whose run-log cannot show it, on the user's word.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from mosaic.cli._context import load_dataset, run_log_dir_for
from mosaic.cli._io import emit_json, fail, fail_classified, log
from mosaic.core.pipeline.error_kind import ClassifiedError

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset
    from mosaic.core.pipeline.cancel import AttemptCancel, RequestCancel


def cancel_command(
    manifest: Annotated[
        Path,
        typer.Option(
            "--manifest", "-m", help="Path to the dataset manifest (dataset.yaml)."
        ),
    ],
    execution_id: Annotated[
        str | None, typer.Option("--execution-id", help="Attempt ULID to cancel.")
    ] = None,
    request_id: Annotated[
        str | None,
        typer.Option(
            "--request",
            help=(
                "Pipeline request to cancel: SIGTERM to its driver, or to its "
                "running steps without one, and its unstarted steps recorded "
                "cancelled. Not for a request whose steps a queue dispatches."
            ),
        ),
    ] = None,
    assume_exited: Annotated[
        bool,
        typer.Option(
            "--assume-exited",
            help=(
                "Record as cancelled an attempt on another host whose run-log "
                "declares no heartbeat, asserting that its process has exited. "
                "Refused for one whose heartbeat is current."
            ),
        ),
    ] = False,
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit the result as JSON on stdout.")
    ] = False,
) -> None:
    """Request cancellation of a running attempt (SIGTERM to its recorded pid).

    With `--request`, cancel a pipeline request instead. A process on another
    host is not signalled. It is recorded cancelled when its run-log proves it
    gone, or with `--assume-exited`, and the command otherwise exits 1.
    """
    if request_id is not None:
        if execution_id is not None:
            fail("Provide exactly one of --execution-id or --request.")
        _cancel_request(
            load_dataset(manifest),
            request_id,
            assume_exited=assume_exited,
            as_json=as_json,
        )
        return
    if execution_id is None:
        fail("Provide exactly one of --execution-id or --request.")
    from mosaic.core.pipeline.cancel import cancel_attempt

    run_dir = run_log_dir_for(load_dataset(manifest))
    if not run_dir.exists():
        fail("No run-logs found (nothing has run yet).")
    try:
        result = cancel_attempt(run_dir, execution_id, assume_exited=assume_exited)
    except ClassifiedError as error:
        fail_classified(error, as_json=as_json)
    _report_attempt(result, as_json=as_json)


def _cancel_request(
    ds: Dataset, request_id: str, *, assume_exited: bool, as_json: bool
) -> None:
    """Cancel one pipeline request, and report what was done."""
    from mosaic.core.pipeline.cancel import cancel_request

    try:
        result = cancel_request(ds, request_id, assume_exited=assume_exited)
    except ClassifiedError as error:
        fail_classified(error, as_json=as_json)
    _report_request(result, as_json=as_json)


def _report_attempt(result: AttemptCancel, *, as_json: bool) -> None:
    """Print *result*, and exit 1 when its process is on another host."""
    if result.outcome == "other_host":
        fail(result.message)
    if as_json:
        emit_json(result.model_dump(mode="json"))
    else:
        log(f"[mosaic] {result.message}")


def _report_request(result: RequestCancel, *, as_json: bool) -> None:
    """Print *result*, and exit 1 when one of its processes is on another host."""
    if result.outcome == "other_host":
        fail(result.message)
    if as_json:
        emit_json(result.model_dump(mode="json"))
        return
    log(f"[mosaic] {result.message}")
    for step in result.steps:
        log(f"[mosaic] {step.message}")
