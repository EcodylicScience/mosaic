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
            help=(
                "Merge rows per (group, sequence). Default: each format's "
                "converter decides. --merge-per-sequence forces it for every "
                "format, --no-merge-per-sequence for none."
            ),
        ),
    ] = None,
    group_from: Annotated[
        str | None, typer.Option("--group-from", help="'infile' | 'filename' | 'both'.")
    ] = None,
    strict_schema: Annotated[
        bool,
        typer.Option(
            "--strict-schema/--no-strict-schema",
            help=(
                "Refuse a table that fails schema validation instead of warning "
                "and skipping its sequence. Off by default, which is the "
                "converter default."
            ),
        ),
    ] = False,
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
    if strict_schema:
        # An explicit flag wins over the same key in --params, and is the surface a
        # caller should reach for: the key was only ever reachable through
        # `--params '{"strict_schema": true}'`, which reads like an obscure
        # converter setting rather than the refusal it is.
        params_dict = {**(params_dict or {}), "strict_schema": True}
    try:
        with stdout_to_stderr():
            outcome = ds.convert_all_tracks(
                params=params_dict,
                overwrite=overwrite,
                merge_per_sequence=merge_per_sequence,
                group_from=group_from,
            )
    except Exception as exc:  # noqa: BLE001 - surface conversion errors cleanly
        fail(f"convert-tracks failed: {exc}")
    if as_json:
        # "ok" was emitted unconditionally, so a run in which every sequence
        # failed to convert exited 0 reporting success.
        emit_json(
            {
                "status": "ok" if outcome.ok else "partial",
                "converted": outcome.converted,
                "failed": outcome.failed,
            }
        )
    elif outcome.ok:
        typer.echo(f"Converted {outcome.converted} sequence(s).")
    else:
        typer.echo(
            f"Converted {outcome.converted} sequence(s); {outcome.failed} failed."
        )
