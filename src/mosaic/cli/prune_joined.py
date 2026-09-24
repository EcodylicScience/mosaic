"""``mosaic prune-joined``: delete the joined exports no tracker reads any more."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

import typer

from mosaic.cli._context import load_dataset
from mosaic.cli._io import emit_json, fail, stdout_to_stderr


def _echo_group(header: str, details: Sequence[str]) -> None:
    """Emit one group of paths under a counted header, or nothing."""
    if not details:
        return
    typer.echo(f"{len(details)} {header}:")
    for detail in details:
        typer.echo(f"  {detail}")


def prune_joined_command(
    manifest: Annotated[
        Path,
        typer.Option(
            "--manifest", "-m", help="Path to the dataset manifest (dataset.yaml)."
        ),
    ],
    apply: Annotated[
        bool,
        typer.Option(
            "--apply/--dry-run",
            help="Delete superseded joins. Default is a dry-run report.",
        ),
    ] = False,
    min_age_hours: Annotated[
        float,
        typer.Option(
            "--min-age-hours",
            help=(
                "Never delete a file modified inside this window. A join being "
                "written looks exactly like one left behind, so this is what "
                "keeps a prune from racing a running export-joined."
            ),
        ),
    ] = 24.0,
    include_stray: Annotated[
        bool,
        typer.Option(
            "--include-stray",
            help=(
                "Also delete files under the joined directory that are not "
                "joins, such as the partial a failed join keeps for inspection. "
                "Subdirectories and symlinks are never deleted."
            ),
        ),
    ] = False,
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit the result as JSON.")
    ] = False,
) -> None:
    """Delete joined exports an earlier version of export-joined made.

    A tracker reads a join only under a recipe the current export-joined writes,
    so after an upgrade that changes it, every existing join is read by nothing
    and re-joining writes the current one beside it. This deletes the old ones.
    A join whose clips no entry resolves to is never deleted, since it may be the
    last copy of the session. Neither is either of two current joins of one
    entry. Dry-run by default.
    """
    from mosaic.core.media.prune_joined import JoinedPruneClass, joined_decline_text

    ds = load_dataset(manifest)
    try:
        with stdout_to_stderr():
            report = ds.prune_joined_exports(
                apply=apply,
                min_age_hours=min_age_hours,
                include_stray=include_stray,
            )
    except OSError as error:
        fail(f"prune-joined failed: {error}")

    if as_json:
        emit_json(report.payload())
        return

    if report.declined is not None:
        typer.echo(f"prune-joined: declined -- {joined_decline_text(report.declined)}")
        return

    verb = "deleted" if report.applied else "would delete"
    typer.echo(
        f"prune-joined: {verb} {len(report.files_deleted)} file(s), "
        f"{report.bytes_reclaimed} byte(s)."
    )
    # The recipes decide what is superseded, and they come from the installed
    # mosaic. A shell on a different version from the worker's would call the
    # worker's fresh joins superseded; this line is what makes that visible.
    for recipe in report.current_recipes:
        typer.echo(f"  current recipe: {recipe}")
    _echo_group("file(s) to delete", [str(p) for p in report.files_deleted])
    if report.held_for_age:
        typer.echo(
            f"{report.held_for_age} file(s) held back as newer than "
            f"{min_age_hours}h; re-run later or lower --min-age-hours."
        )

    # Each kept class wants a different action from a person, so they are
    # listed apart rather than summed.
    reported: tuple[tuple[JoinedPruneClass, str], ...] = (
        ("competing", "current join(s) sharing an entry with another; keep one"),
        ("unsourced", "join(s) of clips no entry resolves to, kept"),
        ("stray", "non-join entr(ies) under joined/"),
    )
    for verdict, header in reported:
        _echo_group(
            header,
            [
                f"{entry.path}  ({entry.entry})" if entry.entry else str(entry.path)
                for entry in report.of(verdict)
            ],
        )
    _echo_group("entr(ies) whose media could not be resolved", report.unresolved)
