"""``mosaic index-media``: scan directories for media and write media/index.csv."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from mosaic_media import VIDEO_EXTENSIONS

from mosaic.cli._context import load_dataset
from mosaic.cli._io import emit_json, fail, stdout_to_stderr


def index_media_command(
    manifest: Annotated[
        Path,
        typer.Option(
            "--manifest", "-m", help="Path to the dataset manifest (dataset.yaml)."
        ),
    ],
    search_dir: Annotated[
        list[str],
        typer.Option("--search-dir", help="Directory to scan for media (repeatable)."),
    ],
    extensions: Annotated[
        str,
        typer.Option(
            "--extensions",
            help="Comma-separated file extensions. Defaults to mosaic-media's supported set.",
        ),
    ] = "",
    recursive: Annotated[
        bool,
        typer.Option("--recursive/--no-recursive", help="Recurse into subdirectories."),
    ] = True,
    sequence_match_mode: Annotated[
        str, typer.Option("--sequence-match-mode", help="'exact' or 'prefix'.")
    ] = "exact",
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit the index path as JSON.")
    ] = False,
) -> None:
    """Index media files under one or more directories into media/index.csv."""
    ds = load_dataset(manifest)
    extensions = extensions or ",".join(sorted(VIDEO_EXTENSIONS))
    exts = tuple(
        e if e.startswith(".") else f".{e}"
        for e in (part.strip() for part in extensions.split(","))
        if e
    )
    try:
        with stdout_to_stderr():
            path = ds.index_media(
                search_dir,
                extensions=exts,
                recursive=recursive,
                sequence_match_mode=sequence_match_mode,
            )
    except Exception as exc:  # noqa: BLE001 - surface indexing errors cleanly
        fail(f"index-media failed: {exc}")
    if as_json:
        emit_json({"index": str(path)})
    else:
        typer.echo(str(path))
