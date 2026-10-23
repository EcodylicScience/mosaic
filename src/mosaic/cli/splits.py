"""``mosaic splits``: the role each group has in a held-out protocol.

A role is ``train``, ``test`` or ``sealed`` and lives in the manifest, so it
travels with the dataset. It is an operator's declaration. The agent tools read
it and never write it, because an agent that could set a role could also unseal
what the role protects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

import typer

from mosaic.cli._context import load_dataset
from mosaic.cli._io import emit_json, fail
from mosaic.cli.tags import ManifestOption
from mosaic.core.manifest import SPLIT_ROLES
from mosaic.core.strict_model import terse

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset

splits_app = typer.Typer(
    name="splits",
    help="Give groups a role in a held-out protocol: train, test or sealed.",
    no_args_is_help=True,
    add_completion=False,
)


@splits_app.command("list")
def list_splits(
    manifest: ManifestOption,
    as_json: Annotated[bool, typer.Option("--json", help="Emit JSON.")] = False,
) -> None:
    """Show every group that has a role."""
    dataset = load_dataset(manifest, read_only=True)
    held = dict(sorted(dataset.splits.items()))
    if as_json:
        emit_json({"splits": held})
        return
    if not held:
        typer.echo("No group has a role. Give one with 'mosaic splits set'.")
        return
    for group, role in held.items():
        typer.echo(f"{group:<24} {role}")


@splits_app.command("set")
def set_splits(
    manifest: ManifestOption,
    role: Annotated[str, typer.Argument(help="train, test or sealed.")],
    groups: Annotated[list[str], typer.Argument(help="The groups to give the role.")],
) -> None:
    """Give each group a role, replacing the one it had."""
    if role not in SPLIT_ROLES:
        fail(f"{role!r} is not a role; use one of {', '.join(SPLIT_ROLES)}.")
    dataset = load_dataset(manifest)
    named = list(dict.fromkeys(groups))
    try:
        dataset.set_splits(role, named)
    except ValueError as exc:
        fail(f"splits set failed: {terse(exc)}")
    unknown = [group for group in named if group not in _indexed_groups(dataset)]
    if unknown:
        # A warning and not a refusal: the held-out data may be added later.
        typer.echo(
            f"Warning: no entry of {', '.join(map(repr, unknown))} is indexed in "
            "this dataset. Check the spelling if the group should hold data now.",
            err=True,
        )
    typer.echo(f"Set {role}: {', '.join(named)}.")


def _indexed_groups(dataset: Dataset) -> set[str]:
    """The groups that hold a row of the tracks index or of the raw tracks index."""
    from mosaic.core.pipeline.tracks_index import read_tracks_index

    groups = {str(group) for group in read_tracks_index(dataset)["group"]}
    groups.update(row.get("group", "") for row in dataset.read_tracks_raw_index())
    return groups


@splits_app.command("remove")
def remove_splits(
    manifest: ManifestOption,
    groups: Annotated[
        list[str], typer.Argument(help="The groups to take the role from.")
    ],
) -> None:
    """Take the role from each group. A group with no role is skipped."""
    dataset = load_dataset(manifest)
    removed = dataset.remove_splits(groups)
    if not removed:
        fail(f"none of {', '.join(groups)} has a role.")
    typer.echo(f"Removed: {', '.join(removed)}.")
