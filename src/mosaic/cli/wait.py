"""``mosaic wait``: block until pipeline requests end, then print their verdicts.

Read-only and import-light, as ``mosaic pipeline status`` is. Run in the
background, it wakes its caller once, when the requests end.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Final, Literal

import typer

from mosaic.cli._context import load_dataset
from mosaic.cli._io import emit_json, fail

if TYPE_CHECKING:
    from mosaic.core.pipeline.graph import Request, RequestWaitWire, WaitOutcome

WAIT_TIMED_OUT_EXIT_CODE: Final = 124
"""The exit code of a wait whose ``--timeout`` ran out, as ``timeout(1)`` exits."""

UNKNOWN_REQUEST_EXIT_CODE: Final = 2
"""The exit code when a request id names no request, or a file that does not read as one."""

_EXIT_CODES: Final[dict[WaitOutcome, int]] = {
    "finished": 0,
    "failed": 1,
    "timed_out": WAIT_TIMED_OUT_EXIT_CODE,
}


def wait_command(
    request_ids: Annotated[
        list[str], typer.Argument(help="The pipeline requests to wait for.")
    ],
    manifest: Annotated[
        Path,
        typer.Option(
            "--manifest", "-m", help="Path to the dataset manifest (dataset.yaml)."
        ),
    ],
    until: Annotated[
        Literal["any", "all"],
        typer.Option(
            "--until",
            help="'all' waits for every request to end, 'any' for the first.",
        ),
    ] = "all",
    timeout: Annotated[
        float | None,
        typer.Option(
            "--timeout", min=0, help="The most seconds to wait. Unset waits on."
        ),
    ] = None,
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit the result as JSON on stdout.")
    ] = False,
) -> None:
    """Wait until pipeline requests end, and print each one's verdict.

    A request ends when its verdict is `finished`, `partial`, `failed`,
    `cancelled` or `orphaned`. The command exits 0 when every request that
    ended is `finished`, 1 when one is not, 124 when `--timeout` ran out first
    and 2 when an id names no request or an unreadable one. Run it as a
    background job to be woken once, when the requests end, instead of
    polling.
    """
    from mosaic.core.pipeline.graph import (
        RequestUnreadable,
        load_request,
        wait_for_requests,
    )

    ds = load_dataset(manifest, read_only=True)
    requests: list[Request] = []
    for request_id in dict.fromkeys(request_ids):
        try:
            requests.append(load_request(ds.base_dir, request_id))
        except FileNotFoundError:
            fail(
                f"No request {request_id!r} in this dataset.", UNKNOWN_REQUEST_EXIT_CODE
            )
        except RequestUnreadable as error:
            fail(str(error), UNKNOWN_REQUEST_EXIT_CODE)
    waited = wait_for_requests(ds.base_dir, requests, until=until, timeout=timeout)
    if as_json:
        emit_json(waited.model_dump(mode="json"))
    else:
        _render(waited)
    raise typer.Exit(code=_EXIT_CODES[waited.outcome])


def _render(waited: RequestWaitWire) -> None:
    """Print one line per request, and a last line when the wait timed out."""
    for report in waited.requests:
        typer.echo(f"request {report.request_id}: {report.verdict}")
    if waited.outcome == "timed_out":
        typer.echo("timed out")
