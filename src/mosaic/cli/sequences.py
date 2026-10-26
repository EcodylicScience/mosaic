"""``mosaic sequences``: list the entries in a dataset (from tracks/index.csv)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from mosaic.cli._context import load_dataset
from mosaic.cli._io import emit_json, fail
from mosaic.core.entry import entry_token
from mosaic.core.recording import recording_name


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
    """List the entries (group:sequence) in tracks/index.csv."""
    ds = load_dataset(manifest, read_only=True)
    # Keyed on the *unfiltered* index having no rows, which treats an absent
    # index and a header-only one alike -- they are one dataset state. A
    # --group that narrows to nothing is a different thing entirely and still
    # exits 0 with no output.
    if not ds.list_sequences():
        fail(
            "tracks/index.csv is empty or absent; convert tracks first "
            "(mosaic convert-tracks)."
        )
    entries = ds.list_entries(group=group)
    # The only place mosaic lists sequence strings to a human, so the only place
    # a recorded label should displace the token. One read for the whole listing
    # rather than one per row; sequences with no label keep their token, which is
    # every sequence in a dataset that has never named one.
    labels = ds.display_names()
    listed = [
        (
            entry_group,
            sequence,
            recording_name(ds.recording_of(entry_group, sequence)),
            labels.get((entry_group, sequence), ""),
        )
        for entry_group, sequence in entries
    ]
    if as_json:
        # Both, and named: a script consuming this needs the token to pass back
        # to --sequence, and a human reading it wants the label.
        emit_json(
            {
                "sequences": sorted({sequence for _, sequence in entries}),
                "labelled": [
                    {"sequence": sequence, "display_name": label or sequence}
                    for _, sequence, _, label in listed
                ],
                "entries": [
                    {
                        "group": entry_group,
                        "sequence": sequence,
                        "recording": recording,
                        "display_name": label or sequence,
                    }
                    for entry_group, sequence, recording, label in listed
                ],
            }
        )
    else:
        for entry_group, sequence, _, label in listed:
            token = entry_token(entry_group, sequence)
            typer.echo(f"{token}  ({label})" if label else token)
