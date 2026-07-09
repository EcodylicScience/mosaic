"""``mosaic sequences``: list the sequences in a dataset (from tracks/index.csv)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from mosaic.cli._context import load_dataset
from mosaic.cli._io import emit_json, fail


def sequences_command(
    manifest: Annotated[
        Path,
        typer.Option(
            "--manifest", "-m", help="Path to the dataset manifest (dataset.yaml)."
        ),
    ],
    group: Annotated[
        str | None, typer.Option("--group", help="Restrict to one group namespace.")
    ] = None,
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit as a JSON object.")
    ] = False,
) -> None:
    """List sequences (optionally filtered by group) from tracks/index.csv."""
    ds = load_dataset(manifest)
    try:
        seqs = ds.list_sequences(group=group)
    except FileNotFoundError:
        fail(
            "tracks/index.csv not found; convert tracks first (mosaic convert-tracks)."
        )
    if as_json:
        emit_json({"sequences": seqs})
    else:
        for seq in seqs:
            typer.echo(seq)
