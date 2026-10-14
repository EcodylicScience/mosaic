"""``mosaic runs``: list run attempts from the JSONL run-logs (newest first).

Each attempt's ``kind`` is ``"feature"`` for *all* features (the feature slug lives in
``target``, e.g. ``speed-angvel__from__tracks``), the op kind for tracking ops, and
``"trex"`` for TREx (plus any future payload kind). Use ``--target`` to filter to one
feature's runs.

``--limit`` and ``--after`` read one page and reduce only the run-logs that it
reads. Under ``--json`` a paged listing prints an object, ``items`` and
``next_after``, and an unpaged one the array of rows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from mosaic.cli._context import load_dataset, run_log_dir_for
from mosaic.cli._io import emit_json, log
from mosaic.cli._render import render_table

_COLUMNS = ["execution_id", "kind", "target", "run_id", "status", "owner"]


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
    limit: Annotated[
        int | None,
        typer.Option("--limit", help="List at most this many attempts.", min=1),
    ] = None,
    after: Annotated[
        str | None,
        typer.Option(
            "--after", help="Start after the page that printed this next_after."
        ),
    ] = None,
    as_json: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Emit the rows as a JSON array, or a page object with --limit/--after.",
        ),
    ] = False,
) -> None:
    """List run attempts newest first, optionally filtered by kind / status / target."""
    from mosaic.runlog import list_runs

    ds = load_dataset(manifest, read_only=True)
    page = list_runs(
        run_log_dir_for(ds),
        limit=limit,
        after=after,
        kind=kind,
        status=status,
        target=target,
    )
    paged = limit is not None or after is not None
    if as_json:
        emit_json(page if paged else page["items"])
        return
    render_table(page["items"], _COLUMNS)
    if page["next_after"] is not None:
        log(f"note: --after {page['next_after']} lists the next page")
