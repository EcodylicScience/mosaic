"""``mosaic scan``: refresh a dataset's indexes from the sources it declares.

Replaces ``mosaic index-media`` and ``mosaic index-tracks``, which each took the
directories to walk and the recipe to walk them with as arguments remembered
nowhere. Here the manifest holds both, so a bare ``mosaic scan`` rescans exactly
what the dataset says it draws from -- and it finally reaches labels, whose
indexer never had a command at all.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import typer

from mosaic.cli._context import load_dataset
from mosaic.cli._io import emit_json, fail, stdout_to_stderr

ScanKind = Literal["media", "tracks", "labels"]
ALL_KINDS: tuple[ScanKind, ...] = ("media", "tracks", "labels")


def _check_kinds(kinds: list[str] | None) -> tuple[ScanKind, ...]:
    if not kinds:
        return ALL_KINDS
    chosen: list[ScanKind] = []
    for kind in kinds:
        if kind == "media":
            chosen.append("media")
        elif kind == "tracks":
            chosen.append("tracks")
        elif kind == "labels":
            chosen.append("labels")
        else:
            fail(f"--kind must be media, tracks or labels, got {kind!r}")
    return tuple(chosen)


def scan_command(
    manifest: Annotated[
        Path,
        typer.Option(
            "--manifest", "-m", help="Path to the dataset manifest (dataset.yaml)."
        ),
    ],
    kind: Annotated[
        list[str] | None,
        typer.Option("--kind", help="Restrict to one kind (repeatable). Default: all."),
    ] = None,
    only: Annotated[
        list[str] | None,
        typer.Option(
            "--only",
            help="Restrict to these source ids (repeatable). The declaration is unchanged.",
        ),
    ] = None,
    reassign: Annotated[
        bool,
        typer.Option(
            "--reassign",
            help=(
                "Let the scan re-derive identity for rows a caller assigned. Off "
                "by default: a scan's identity is a guess and an assignment is not."
            ),
        ),
    ] = False,
    prune_unsourced: Annotated[
        bool,
        typer.Option(
            "--prune-unsourced",
            help=(
                "Also drop rows no scanned source claims. Off by default: those "
                "are usually an assignment or a reference to a file elsewhere."
            ),
        ),
    ] = False,
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit the written index paths as JSON.")
    ] = False,
) -> None:
    """Rescan every declared source and rewrite the indexes it feeds."""
    dataset = load_dataset(manifest)
    kinds = _check_kinds(kind)
    restrict = list(only or [])

    if restrict and len(kinds) > 1:
        wanted = set(restrict)
        matching: list[ScanKind] = [
            one
            for one in kinds
            if {source.id for source in dataset.scan_sources(one)} & wanted
        ]
        kinds = tuple(matching)
        if not kinds:
            fail(f"no declared source matches --only {restrict}")

    written: dict[str, str] = {}
    for one in kinds:
        if not dataset.scan_sources(one):
            continue
        try:
            with stdout_to_stderr():
                if one == "media":
                    path = dataset.scan_media(
                        only=restrict,
                        reassign=reassign,
                        prune_unsourced=prune_unsourced,
                    )
                elif one == "tracks":
                    path = dataset.scan_tracks_raw(
                        only=restrict, prune_unsourced=prune_unsourced
                    )
                else:
                    path = dataset.scan_labels_raw(
                        only=restrict, prune_unsourced=prune_unsourced
                    )
        except KeyError as exc:
            fail(f"scan failed: {exc}")
        except Exception as exc:  # noqa: BLE001 - surface scan failures cleanly
            fail(f"scan failed: {exc}")
        written[one] = str(path)

    if not written:
        fail(
            "no scan sources are declared for the selected kinds. "
            "Declare one with 'mosaic sources add'."
        )

    if as_json:
        emit_json({"indexes": written})
    else:
        for one, path in written.items():
            typer.echo(f"{one:<7} {path}")
