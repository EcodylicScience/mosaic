"""``mosaic convert-tracks``: convert raw tracks (tracks_raw/) to standard parquet (tracks/)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, cast

import typer

from mosaic.cli._context import load_dataset
from mosaic.cli._io import emit_json, fail, load_json_arg, stdout_to_stderr


def convert_tracks_command(
    manifest: Annotated[
        Path,
        typer.Option(
            "--manifest", "-m", help="Path to the dataset manifest (dataset.yaml)."
        ),
    ],
    params: Annotated[
        str | None,
        typer.Option("--params", help="Converter params as JSON, @file.json, or @-."),
    ] = None,
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite", help="Overwrite existing output parquet files."),
    ] = False,
    merge_per_sequence: Annotated[
        bool | None,
        typer.Option(
            "--merge-per-sequence/--no-merge-per-sequence",
            help="Merge rows per (group, sequence).",
        ),
    ] = None,
    group_from: Annotated[
        str | None, typer.Option("--group-from", help="'infile' | 'filename' | 'both'.")
    ] = None,
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit the result as JSON.")
    ] = False,
) -> None:
    """Convert all raw tracks into schema-valid parquet under tracks/."""
    ds = load_dataset(manifest)
    params_value = load_json_arg(params)
    params_dict: dict[str, object] | None = None
    if params_value is not None:
        if not isinstance(params_value, dict):
            fail("--params must be a JSON object.")
        params_dict = cast("dict[str, object]", params_value)
    try:
        with stdout_to_stderr():
            ds.convert_all_tracks(  # pyright: ignore[reportUnknownMemberType]
                params=params_dict,
                overwrite=overwrite,
                merge_per_sequence=merge_per_sequence,
                group_from=group_from,
            )
    except Exception as exc:  # noqa: BLE001 - surface conversion errors cleanly
        fail(f"convert-tracks failed: {exc}")
    if as_json:
        emit_json({"status": "ok"})
    else:
        typer.echo("Converted tracks.")
