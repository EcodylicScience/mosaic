"""``mosaic links``: the datasets whose derived data this one may read.

A model is named by its run identifier, which carries no location. A link is
what lets that name resolve when the model was trained in another dataset -- a
group's shared library, say -- so an inference run here can name it exactly as
the dataset that trained it would.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from mosaic.cli._context import load_dataset
from mosaic.cli._io import emit_json, fail
from mosaic.core.dataset import LinkMismatchError
from mosaic.core.manifest import DatasetLink
from mosaic.core.pipeline.artifact_kinds import LINKED_KINDS, LinkedKind
from mosaic.core.strict_model import terse

links_app = typer.Typer(
    name="links",
    help="Link the datasets whose derived data this one may read, such as models.",
    no_args_is_help=True,
    add_completion=False,
)

ManifestOption = Annotated[
    Path,
    typer.Option(
        "--manifest", "-m", help="Path to the dataset manifest (dataset.yaml)."
    ),
]


@links_app.command("list")
def list_links(
    manifest: ManifestOption,
    as_json: Annotated[bool, typer.Option("--json", help="Emit JSON.")] = False,
) -> None:
    """Show every link, in the order a lookup tries them."""
    dataset = load_dataset(manifest, read_only=True)
    listed = [
        {
            "id": link.id,
            "path": link.path,
            "uuid": link.uuid,
            "added_at": link.added_at,
            "provides": list(link.provides),
        }
        for link in dataset.links
    ]
    if as_json:
        emit_json({"links": listed})
        return
    if not listed:
        typer.echo("No datasets linked.")
        return
    for link in dataset.links:
        typer.echo(
            f"{link.id}\t{link.path}\t{link.uuid or '(no uuid)'}\t"
            f"{','.join(link.provides)}"
        )


def _check_provides(kinds: list[str] | None) -> tuple[LinkedKind, ...]:
    """The kinds named by ``--provides``, in :data:`LINKED_KINDS` order."""
    if not kinds:
        return LINKED_KINDS
    unknown = [kind for kind in kinds if kind not in LINKED_KINDS]
    if unknown:
        fail(f"--provides must be one of {', '.join(LINKED_KINDS)}, got {unknown[0]!r}")
    return tuple(kind for kind in LINKED_KINDS if kind in kinds)


@links_app.command("add")
def add_link(
    manifest: ManifestOption,
    link_id: Annotated[str, typer.Option("--id", help="What to call the link.")],
    path: Annotated[
        str,
        typer.Option(
            "--path",
            help=(
                "The linked dataset: its directory or its manifest. Relative "
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
                "The linked dataset's own uuid, to insist on. Left out, the uuid "
                "found at the path is recorded."
            ),
        ),
    ] = "",
    provides: Annotated[
        list[str] | None,
        typer.Option(
            "--provides",
            help=(
                "A kind of derived data to read through the link: "
                f"{', '.join(LINKED_KINDS)}. Repeatable. Default: every kind."
            ),
        ),
    ] = None,
) -> None:
    """Link a dataset, recording its uuid so a moved one is told from a wrong one."""
    kinds = _check_provides(provides)
    dataset = load_dataset(manifest)
    try:
        stored = dataset.add_link(
            DatasetLink(id=link_id, path=path, uuid=uuid, provides=kinds)
        )
    except FileNotFoundError as exc:
        fail(f"links add failed: no dataset manifest at {exc}")
    except (ValueError, LinkMismatchError) as exc:
        fail(f"links add failed: {terse(exc)}")
    typer.echo(f"Linked {stored.id!r} -> {stored.path} ({stored.uuid or 'no uuid'})")


@links_app.command("remove")
def remove_link(
    manifest: ManifestOption,
    link_id: Annotated[str, typer.Option("--id", help="The link to drop.")],
) -> None:
    """Drop a link. Nothing on disk changes; what it served stops resolving here."""
    dataset = load_dataset(manifest)
    if not dataset.remove_link(link_id):
        declared = sorted(link.id for link in dataset.links)
        fail(f"no link named {link_id!r}; linked: {declared or 'none'}")
    typer.echo(f"Unlinked {link_id!r}.")
