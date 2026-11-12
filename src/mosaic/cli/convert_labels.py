"""``mosaic convert-labels``: convert raw labels into mosaic's label format (labels/<kind>/)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, cast

import typer

from mosaic.cli._context import load_dataset
from mosaic.cli._io import emit_json, fail, load_json_arg, stdout_to_stderr


def convert_labels_command(
    manifest: Annotated[
        Path,
        typer.Option(
            "--manifest", "-m", help="Path to the dataset manifest (dataset.yaml)."
        ),
    ],
    kind: Annotated[
        str, typer.Option("--kind", help="Label kind, e.g. 'behavior' or 'id_tags'.")
    ] = "behavior",
    source_format: Annotated[
        str | None,
        typer.Option(
            "--source-format", help="Source format (e.g. 'calms21_npy', 'boris_csv')."
        ),
    ] = None,
    params: Annotated[
        str | None,
        typer.Option("--params", help="Converter params as JSON, @file.json, or @-."),
    ] = None,
    overwrite: Annotated[
        bool, typer.Option("--overwrite", help="Overwrite existing label files.")
    ] = False,
    tracks_run_id: Annotated[
        str | None,
        typer.Option(
            "--tracks-run-id",
            help=(
                "The tracks variant that second-based events are placed on. The "
                "labels variant names it. Unset uses the one variant every entry "
                "has, and refuses entries that have different ones."
            ),
        ),
    ] = None,
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit the result as JSON.")
    ] = False,
) -> None:
    """Convert all raw labels for a kind into mosaic's label format."""
    ds = load_dataset(manifest)
    params_value = load_json_arg(params)
    params_dict: dict[str, object] | None = None
    if params_value is not None:
        if not isinstance(params_value, dict):
            fail("--params must be a JSON object.")
        params_dict = cast("dict[str, object]", params_value)
    try:
        with stdout_to_stderr():
            outcome = ds.convert_all_labels(
                kind=kind,
                overwrite=overwrite,
                params=params_dict,
                source_format=source_format,
                tracks_run_id=tracks_run_id,
            )
    except Exception as exc:  # noqa: BLE001 - surface conversion errors cleanly
        fail(f"convert-labels failed: {exc}")
    status = "failed" if outcome.lost_everything else "ok" if outcome.ok else "partial"
    if as_json:
        emit_json(
            {
                "status": status,
                "variant": outcome.variant,
                "written": outcome.written,
                "skipped": outcome.skipped,
                "failed": len(outcome.failed_entries),
            }
        )
    else:
        typer.echo(
            f"Converted labels (kind={kind}) into {outcome.variant}: "
            f"{outcome.written} written, {outcome.skipped} already present, "
            f"{len(outcome.failed_entries)} without tracks.",
            err=status == "failed",
        )
    if status == "failed":
        # Losing every entry is the AllEntriesFailed rule: the exit status says so.
        raise typer.Exit(code=1)
