"""``mosaic index-tracks``: scan directories for raw tracks and write tracks_raw/index.csv."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from mosaic.cli._context import load_dataset
from mosaic.cli._io import emit_json, fail, stdout_to_stderr


def index_tracks_command(
    manifest: Annotated[
        Path,
        typer.Option(
            "--manifest", "-m", help="Path to the dataset manifest (dataset.yaml)."
        ),
    ],
    search_dir: Annotated[
        list[str],
        typer.Option(
            "--search-dir", help="Directory to scan for track files (repeatable)."
        ),
    ],
    patterns: Annotated[
        str, typer.Option("--patterns", help="Comma-separated glob patterns.")
    ] = "*.npy,*.h5,*.csv",
    src_format: Annotated[
        str,
        typer.Option(
            "--src-format",
            help="Source format identifier (e.g. 'calms21_npy', 'trex_npz').",
        ),
    ] = "calms21_npy",
    group_from: Annotated[
        str | None,
        typer.Option("--group-from", help="How to derive group (converter-specific)."),
    ] = None,
    group_pattern: Annotated[
        str | None,
        typer.Option(
            "--group-pattern", help="Regex/pattern to extract group from a path."
        ),
    ] = None,
    recursive: Annotated[
        bool,
        typer.Option("--recursive/--no-recursive", help="Recurse into subdirectories."),
    ] = True,
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit the index path as JSON.")
    ] = False,
) -> None:
    """Index raw track files under one or more directories into tracks_raw/index.csv."""
    ds = load_dataset(manifest)
    pattern_list = [p.strip() for p in patterns.split(",") if p.strip()]
    try:
        with stdout_to_stderr():
            path = ds.index_tracks_raw(
                search_dir,
                patterns=pattern_list,
                src_format=src_format,
                recursive=recursive,
                group_from=group_from,
                group_pattern=group_pattern,
            )
    except Exception as exc:  # noqa: BLE001 - surface indexing errors cleanly
        fail(f"index-tracks failed: {exc}")
    if as_json:
        emit_json({"index": str(path)})
    else:
        typer.echo(str(path))
