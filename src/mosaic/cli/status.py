"""``mosaic status``: read one run attempt (and optionally its progress) from its run-log.

Read-only and import-light -- it never loads the feature/tracking stacks. The store
is the append-only JSONL run-log under ``<dataset_root>/.mosaic/runs/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Final

import typer

from mosaic.cli._context import load_dataset, run_log_dir_for
from mosaic.cli._io import emit_json, fail
from mosaic.cli._render import render_json_block, render_kv

if TYPE_CHECKING:
    from mosaic.core.pipeline.graph import RecordedErrorWire, RunStatusWire

_STRUCTURED_FIELDS: Final = frozenset(
    {"error", "failed_entries", "failed_entries_has_more", "progress"}
)
"""The report fields that the plain output prints on lines of their own."""


def status_command(
    manifest: Annotated[
        Path,
        typer.Option(
            "--manifest", "-m", help="Path to the dataset manifest (dataset.yaml)."
        ),
    ],
    execution_id: Annotated[
        str, typer.Option("--execution-id", help="Attempt ULID to look up.")
    ],
    progress: Annotated[
        bool, typer.Option("--progress", help="Attach the per-step progress stream.")
    ] = False,
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit the row as JSON on stdout.")
    ] = False,
) -> None:
    """Show the status of one run attempt by execution_id.

    Besides the run-log snapshot, the report gives the outcome, which is
    ``partial`` for a finished attempt that lost entries, the recorded error
    parsed, and the first 20 failed entries.
    """
    from mosaic.core.pipeline.graph import run_status_wire

    ds = load_dataset(manifest, read_only=True)
    if not run_log_dir_for(ds).exists():
        fail("No run-logs found (nothing has run yet).")
    report = run_status_wire(ds.base_dir, execution_id, progress=progress)
    if report is None:
        fail(f"No run found with execution_id={execution_id}.")
    if as_json:
        emit_json(report.model_dump(mode="json"))
    else:
        _render_run_status(report)


def _render_run_status(report: RunStatusWire) -> None:
    """Print *report* as ``key: value`` lines, its error and failed entries as text."""
    payload = report.model_dump(mode="json")
    render_kv({k: v for k, v in payload.items() if k not in _STRUCTURED_FIELDS})
    if report.error is not None:
        typer.echo(f"error: {_error_text(report.error)}")
    for entry in report.failed_entries:
        message = "" if entry.error is None else _error_text(entry.error)
        typer.echo(f"failed entry {entry.key}: {message}")
    if report.failed_entries_has_more:
        hidden = report.entries_failed - len(report.failed_entries)
        typer.echo(f"failed entries not listed: {hidden}")
    if report.progress is not None:
        typer.echo("progress:")
        render_json_block(payload["progress"])


def _error_text(error: RecordedErrorWire) -> str:
    """One line for a recorded failure: the refusal's reason, or the exception type."""
    if error.refusal is not None:
        return f"refused ({error.refusal.reason}): {error.message}"
    return f"{error.type}: {error.message}" if error.type else error.message
