"""``mosaic preview``: one entry's table of a feature run or a tracks variant.

The verb prints the table's schema, row count and frame extent, its first rows
and statistics of each numeric column read. ``--json`` prints the
:class:`~mosaic.core.pipeline.preview.ArtifactPreview` that
:func:`~mosaic.core.pipeline.preview.preview_artifact` returns. The verb writes
nothing to the dataset.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from mosaic.cli._context import load_dataset
from mosaic.cli._io import emit_json, fail_classified
from mosaic.cli._render import render_kv, render_table

if TYPE_CHECKING:
    from mosaic.core.pipeline.preview import ArtifactPreview, PreviewColumn


def preview_command(
    ref: Annotated[
        str,
        typer.Argument(
            help=(
                "The artifact's ref, as 'mosaic inventory' prints it: "
                "feature/<storage>/<run_id> or tracks-variant/<run_id>."
            )
        ),
    ],
    manifest: Annotated[
        Path,
        typer.Option(
            "--manifest", "-m", help="Path to the dataset manifest (dataset.yaml)."
        ),
    ],
    entry: Annotated[
        str,
        typer.Option(
            "--entry",
            help=(
                "The group:sequence entry whose table to read. A bare name is a "
                "sequence in the empty group."
            ),
        ),
    ],
    column: Annotated[
        list[str] | None,
        typer.Option(
            "--column",
            help=(
                "Read this column for the rows and statistics (repeatable). "
                "Default: every column."
            ),
        ),
    ] = None,
    limit: Annotated[
        int | None,
        typer.Option(
            "--limit", help="How many rows to print, at most 50 (default 20)."
        ),
    ] = None,
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit as a JSON object.")
    ] = False,
) -> None:
    """Preview one entry's table: schema, row count, frame extent, rows and statistics."""
    # Imported in the body, so --help and the other verbs stay import-light.
    from mosaic.core.entry import parse_entry_tokens
    from mosaic.core.pipeline.error_kind import ClassifiedError
    from mosaic.core.pipeline.preview import (
        DEFAULT_PREVIEW_ROWS,
        preview_artifact,
        preview_ref,
    )

    ds = load_dataset(manifest, read_only=True)
    try:
        preview = preview_artifact(
            ds,
            preview_ref(ref),
            parse_entry_tokens([entry])[0],
            columns=column,
            limit=DEFAULT_PREVIEW_ROWS if limit is None else limit,
        )
    except ClassifiedError as error:
        fail_classified(error, as_json=as_json)
    if as_json:
        emit_json(preview.model_dump(mode="json"))
        return
    _render(preview)


def _render(preview: ArtifactPreview) -> None:
    """Print the facts, the statistics of each numeric column and the rows."""
    render_kv(
        {
            "ref": preview.ref,
            "entry": f"{preview.group}:{preview.sequence}",
            "path": preview.path,
            "rows": preview.n_rows,
            "frames": "none"
            if preview.frame_min is None
            else f"{preview.frame_min}-{preview.frame_max}",
        }
    )
    typer.echo()
    render_table(
        [_column_row(column) for column in preview.columns],
        ["column", "type", "min", "max", "mean", "nan_fraction"],
    )
    if not preview.rows:
        return
    typer.echo()
    render_table(
        [dict(zip(preview.read_columns, row, strict=True)) for row in preview.rows],
        list(preview.read_columns),
    )
    if preview.has_more:
        typer.echo(
            f"note: {len(preview.rows)} of {preview.n_rows} rows printed", err=True
        )


def _column_row(column: PreviewColumn) -> dict[str, object]:
    """Return one column's name, Arrow type and statistics, blank when not computed."""
    row: dict[str, object] = {"column": column.name, "type": column.arrow_type}
    if column.statistics is not None:
        row.update(column.statistics.model_dump())
    return row
