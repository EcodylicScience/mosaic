"""``mosaic runs``: list run attempts from ``.mosaic.db`` (newest first).

``runs.kind`` is ``"feature"`` for *all* features (the feature slug lives in
``target``, e.g. ``speed-angvel__from__tracks``), the op kind for tracking ops,
and ``"trex"`` for TREx. Use ``--target`` to filter to one feature's runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from mosaic.cli._context import features_db_path, load_dataset
from mosaic.cli._io import emit_json
from mosaic.cli._render import render_table


def runs_command(
    manifest: Annotated[
        Path,
        typer.Option(
            "--manifest", "-m", help="Path to the dataset manifest (dataset.yaml)."
        ),
    ],
    kind: Annotated[
        str | None,
        typer.Option(
            "--kind", help="Filter by runs.kind ('feature', 'trex', or an op kind)."
        ),
    ] = None,
    status: Annotated[
        str | None,
        typer.Option(
            "--status", help="Filter by status (running|finished|failed|cancelled)."
        ),
    ] = None,
    target: Annotated[
        str | None,
        typer.Option(
            "--target", help="Filter by exact runs.target (feature-slug filter)."
        ),
    ] = None,
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit the rows as a JSON array.")
    ] = False,
) -> None:
    """List run attempts, optionally filtered by kind / status / target."""
    from mosaic.core.pipeline.registry import read_runs

    ds = load_dataset(manifest)
    db = features_db_path(ds)
    rows: list[dict[str, object]] = (
        [] if not db.exists() else read_runs(db, kind=kind, status=status)
    )
    if target is not None:
        rows = [r for r in rows if str(r.get("target", "")) == target]
    if as_json:
        emit_json(rows)
    else:
        render_table(
            rows, ["execution_id", "kind", "target", "run_id", "status", "owner"]
        )
