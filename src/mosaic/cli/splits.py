"""``mosaic splits``: the role each group has in a held-out protocol.

A role is ``train``, ``test`` or ``sealed`` and lives in the manifest, so it
travels with the dataset. It is an operator's declaration. The agent tools read
it and never write it, because an agent that could set a role could also unseal
what the role protects.
"""

from __future__ import annotations

from typing import Annotated, cast, get_args

import typer

from mosaic.cli._context import load_dataset
from mosaic.cli._io import emit_json, fail
from mosaic.cli.tags import ManifestOption
from mosaic.core.manifest import SplitRole
from mosaic.core.strict_model import terse

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
    allowed = get_args(SplitRole)
    if role not in allowed:
        fail(f"{role!r} is not a role; use one of {', '.join(allowed)}.")
    dataset = load_dataset(manifest)
    try:
        dataset.set_splits(cast("SplitRole", role), groups)
    except ValueError as exc:
        fail(f"splits set failed: {terse(exc)}")
    typer.echo(f"Set {role}: {', '.join(groups)}.")


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
