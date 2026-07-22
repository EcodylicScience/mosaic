"""``mosaic status``: read one run attempt (and optionally its progress) from its run-log.

Read-only and import-light -- it never loads the feature/tracking stacks. The store
is the append-only JSONL run-log under ``<dataset_root>/.mosaic/runs/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from mosaic.cli._context import load_dataset, run_log_dir_for
from mosaic.cli._io import emit_json, fail
from mosaic.cli._render import render_kv


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
    """Show the status of one run attempt by execution_id."""
    from mosaic.core.pipeline.run_log import read_run

    ds = load_dataset(manifest)
    run_dir = run_log_dir_for(ds)
    if not run_dir.exists():
        fail("No run-logs found (nothing has run yet).")
    row = read_run(run_dir, execution_id)
    if row is None:
        fail(f"No run found with execution_id={execution_id}.")
    payload: dict[str, object] = dict(row)
    if progress:
        from mosaic.core.pipeline.run_log import read_run_progress

        payload["progress"] = read_run_progress(run_dir, execution_id)
    if as_json:
        emit_json(payload)
    else:
        render_kv(payload)
