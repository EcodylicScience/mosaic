"""``mosaic libraries``: the datasets a trained model may be resolved from.

A model is named by its run identifier, which carries no location. A library
link is what lets that name resolve when the model was trained in another
dataset -- a group's shared library, say -- so an inference run here can name it
exactly as the dataset that trained it would.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from mosaic.cli._context import load_dataset
from mosaic.cli._io import emit_json, fail
from mosaic.core.dataset import LibraryMismatchError
from mosaic.core.manifest import LibraryLink
from mosaic.core.strict_model import terse

libraries_app = typer.Typer(
    name="libraries",
    help="Link the datasets a trained model may be resolved from, by run id.",
    no_args_is_help=True,
    add_completion=False,
)

ManifestOption = Annotated[
    Path,
    typer.Option(
        "--manifest", "-m", help="Path to the dataset manifest (dataset.yaml)."
    ),
]


@libraries_app.command("list")
def list_libraries(
    manifest: ManifestOption,
    as_json: Annotated[bool, typer.Option("--json", help="Emit JSON.")] = False,
) -> None:
    """Show every linked library, in the order a model lookup tries them."""
    dataset = load_dataset(manifest)
    listed = [
        {
            "id": link.id,
            "path": link.path,
            "uuid": link.uuid,
            "added_at": link.added_at,
        }
        for link in dataset.libraries
    ]
    if as_json:
        emit_json({"libraries": listed})
        return
    if not listed:
        typer.echo("No libraries linked.")
        return
    for entry in listed:
        typer.echo(f"{entry['id']}\t{entry['path']}\t{entry['uuid'] or '(no uuid)'}")


@libraries_app.command("add")
def add_library(
    manifest: ManifestOption,
    link_id: Annotated[str, typer.Option("--id", help="What to call the link.")],
    path: Annotated[
        str,
        typer.Option(
            "--path",
            help=(
                "The library dataset: its directory or its manifest. Relative "
                "to this dataset, which is the spelling that survives the tree "
                "being mounted somewhere else."
            ),
        ),
    ],
    uuid: Annotated[
        str,
        typer.Option(
            "--uuid",
            help=(
                "The library's own uuid, to insist on. Left out, the uuid found "
                "at the path is recorded."
            ),
        ),
    ] = "",
) -> None:
    """Link a library, recording its uuid so a moved one is told from a wrong one."""
    dataset = load_dataset(manifest)
    try:
        stored = dataset.add_library(LibraryLink(id=link_id, path=path, uuid=uuid))
    except FileNotFoundError as exc:
        fail(f"libraries add failed: no dataset manifest at {exc}")
    except (ValueError, LibraryMismatchError) as exc:
        fail(f"libraries add failed: {terse(exc)}")
    typer.echo(f"Linked {stored.id!r} -> {stored.path} ({stored.uuid or 'no uuid'})")


@libraries_app.command("remove")
def remove_library(
    manifest: ManifestOption,
    link_id: Annotated[str, typer.Option("--id", help="The link to drop.")],
) -> None:
    """Drop a link. Nothing on disk changes; its models stop resolving from here."""
    dataset = load_dataset(manifest)
    if not dataset.remove_library(link_id):
        declared = sorted(link.id for link in dataset.libraries)
        fail(f"no library named {link_id!r}; linked: {declared or 'none'}")
    typer.echo(f"Unlinked {link_id!r}.")
