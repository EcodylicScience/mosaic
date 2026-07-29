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
    # Keyed on the *unfiltered* index having no rows, which treats an absent
    # index and a header-only one alike -- they are one dataset state. A
    # --group that narrows to nothing is a different thing entirely and still
    # exits 0 with no output.
    if not ds.list_sequences():
        fail(
            "tracks/index.csv is empty or absent; convert tracks first "
            "(mosaic convert-tracks)."
        )
    seqs = ds.list_sequences(group=group)
    # The only place mosaic lists sequence strings to a human, so the only place
    # a recorded label should displace the token. One read for the whole listing
    # rather than one per row; sequences with no label keep their token, which is
    # every sequence in a dataset that has never named one.
    labels = ds.display_names()
    labelled = [(seq, labels.get((group or "", seq), "")) for seq in seqs]
    if as_json:
        # Both, and named: a script consuming this needs the token to pass back
        # to --sequence, and a human reading it wants the label.
        emit_json(
            {
                "sequences": seqs,
                "labelled": [
                    {"sequence": seq, "display_name": label or seq}
                    for seq, label in labelled
                ],
            }
        )
    else:
        for seq, label in labelled:
            typer.echo(f"{seq}  ({label})" if label else seq)
