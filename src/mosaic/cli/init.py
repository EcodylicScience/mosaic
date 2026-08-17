"""``mosaic init``: create a dataset.

Until this existed a dataset could only be created from Python, which meant the
one gesture that starts everything else was the one gesture the CLI could not
do.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from mosaic.cli._io import emit_json, fail, terse
from mosaic.user_paths import user_path


def _parse_assignments(pairs: list[str] | None, what: str) -> dict[str, str]:
    """``KEY=VALUE`` strings as a mapping, or fail naming the bad one."""
    parsed: dict[str, str] = {}
    for pair in pairs or []:
        key, separator, value = pair.partition("=")
        if not separator or not key.strip():
            fail(f"--{what} expects KEY=VALUE, got {pair!r}")
        parsed[key.strip()] = value
    return parsed


def init_command(
    directory: Annotated[
        Path,
        typer.Argument(help="Dataset directory. Created if it does not exist."),
    ] = Path(),
    name: Annotated[
        str | None,
        typer.Option("--name", help="Dataset name. Defaults to the directory name."),
    ] = None,
    version: Annotated[
        str, typer.Option("--version", help="The dataset's own version string.")
    ] = "0.1.0",
    root: Annotated[
        list[str] | None,
        typer.Option(
            "--root",
            help="Override one root as KEY=PATH, relative to the dataset (repeatable).",
        ),
    ] = None,
    tag: Annotated[
        list[str] | None,
        typer.Option(
            "--tag",
            help=(
                "Add a text tag as NAME=VALUE (repeatable). For a typed tag, "
                "use 'mosaic tags define' afterwards."
            ),
        ),
    ] = None,
    note: Annotated[
        str | None, typer.Option("--note", help="Notes text for the dataset.")
    ] = None,
    notes_file: Annotated[
        Path | None,
        typer.Option("--notes-file", help="Read the notes from this file instead."),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", help="Overwrite an existing manifest."),
    ] = False,
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit the manifest path as JSON.")
    ] = False,
) -> None:
    """Create a dataset: write dataset.yaml and the roots it declares."""
    from mosaic.core.dataset import new_dataset_manifest
    from mosaic.core.manifest import DatasetTag, default_roots

    target = user_path(directory).resolve()
    manifest_path = target / "dataset.yaml"
    if manifest_path.exists() and not force:
        fail(f"{manifest_path} already exists; pass --force to overwrite it.")

    if note is not None and notes_file is not None:
        fail("Pass --note or --notes-file, not both.")
    notes = note or ""
    if notes_file is not None:
        notes_path = user_path(notes_file)
        if not notes_path.exists():
            fail(f"Notes file not found: {notes_path}")
        notes = notes_path.read_text(encoding="utf-8")

    roots = dict(default_roots)
    roots.update(_parse_assignments(root, "root"))
    tags = tuple(
        DatasetTag(name=key, type="text", value=value)
        for key, value in _parse_assignments(tag, "tag").items()
    )

    try:
        written = new_dataset_manifest(
            name=name or target.name,
            base_dir=target,
            roots=roots,
            version=version,
            notes=notes,
            tags=tags,
        )
    except ValueError as exc:
        fail(f"init failed: {terse(exc)}")

    if as_json:
        emit_json({"manifest": str(written), "name": name or target.name})
    else:
        typer.echo(str(written))
