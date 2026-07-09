"""``mosaic status``: read one run attempt (and optionally its progress) from ``.mosaic.db``.

Read-only and import-light -- it never loads the feature/tracking stacks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from mosaic.cli._context import features_db_path, load_dataset
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
    from mosaic.core.pipeline.registry import read_run

    ds = load_dataset(manifest)
    db = features_db_path(ds)
    if not db.exists():
        fail("No run database found (nothing has run yet).")
    row = read_run(db, execution_id)
    if row is None:
        fail(f"No run found with execution_id={execution_id}.")
    if progress:
        from mosaic.core.pipeline.progress import read_progress

        row = {**row, "progress": read_progress(db, execution_id)}
    if as_json:
        emit_json(row)
    else:
        render_kv(row)
