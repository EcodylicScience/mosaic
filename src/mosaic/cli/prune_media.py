"""``mosaic prune-media``: delete the transcode derivatives nothing reaches."""

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


def prune_media_command(
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
            help="Delete unreferenced derivatives. Default is a dry-run report.",
        ),
    ] = False,
    min_age_hours: Annotated[
        float,
        typer.Option(
            "--min-age-hours",
            help=(
                "Never delete a file modified inside this window. An in-flight "
                "encode's working file looks exactly like a stranded one, so "
                "this is what keeps a prune from racing a running job."
            ),
        ),
    ] = 24.0,
    relink: Annotated[
        bool,
        typer.Option(
            "--relink",
            help=(
                "Also repair: point a link at an unreferenced derivative a "
                "current recipe would reproduce, and clear a link whose file is "
                "gone. Turns the next run's re-encode into a skip."
            ),
        ),
    ] = False,
    include_stray: Annotated[
        bool,
        typer.Option(
            "--include-stray",
            help=(
                "Also delete files under the transcode directory that are not "
                "derivatives, such as an interrupted encode's working file. "
                "Subdirectories and symlinks are never deleted."
            ),
        ),
    ] = False,
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit the result as JSON.")
    ] = False,
) -> None:
    """Delete transcode derivatives that no forward link reaches.

    A retuned recipe writes a new derivative and overwrites the link cell,
    leaving the old file and its index row behind with nothing addressing them.
    Deleting one costs a re-encode that was going to happen anyway; passing
    ``--relink`` avoids that re-encode where a current recipe would reproduce
    the file. A derivative whose source is no longer indexed is never deleted --
    it may be the last copy of an archived video. Dry-run by default.
    """
    from mosaic.core.media.prune import PruneClass, decline_text

    ds = load_dataset(manifest)
    try:
        with stdout_to_stderr():
            report = ds.prune_media(
                apply=apply,
                min_age_hours=min_age_hours,
                relink=relink,
                include_stray=include_stray,
            )
    except OSError as error:
        fail(f"prune-media failed: {error}")

    if as_json:
        emit_json(report.payload())
        return

    if report.declined is not None:
        # Distinct from "would prune 0": this dataset can never hold a
        # derivative, so telling someone to re-run with --apply would be a lie.
        typer.echo(f"prune-media: declined -- {decline_text(report.declined)}")
        return

    verb = "deleted" if report.applied else "would delete"
    typer.echo(
        f"prune-media: {verb} {len(report.files_deleted)} file(s), "
        f"{report.bytes_reclaimed} byte(s), and "
        f"{report.rows_dropped} index row(s)."
    )
    # The recipes are the one input an operator cannot see, and they read the
    # environment -- media_thresholds() is env-driven and the recipe folds it
    # whole. A shell that differs from the worker's classifies fresh output as
    # superseded, and this line is what makes that diagnosable.
    for target, recipe in sorted(report.live_recipes.items()):
        typer.echo(f"  current recipe, {target}: {recipe}")
    _echo_group("file(s) to delete", [str(p) for p in report.files_deleted])
    if report.held_for_age:
        typer.echo(
            f"{report.held_for_age} file(s) held back as newer than "
            f"{min_age_hours}h; re-run later or lower --min-age-hours."
        )
    if report.links_relinked:
        _echo_group("link(s) restored", report.links_relinked)
    if report.links_cleared:
        _echo_group("link(s) cleared", report.links_cleared)

    # Each refused class names a repair a person has to make, so they are listed
    # apart rather than summed: an unsourced derivative wants its original found,
    # a legacy-named one wants the one-off sweep, an unrowed one wants a
    # re-transcode. One count would hide which.
    reported: tuple[tuple[PruneClass, str], ...] = (
        ("unsourced", "derivative(s) whose source is no longer indexed, kept"),
        ("unrowed", "linked derivative(s) with no index row, kept"),
        ("outside_kind_directory", "row(s) naming media outside transcode/, kept"),
        ("foreign", "row(s) carrying no recipe, kept"),
        ("live_legacy_recipe", "live derivative(s) under a superseded recipe"),
        ("relinkable", "unreferenced derivative(s) a current recipe would rebuild"),
        ("stray", "non-derivative entr(ies) under transcode/"),
    )
    for verdict, header in reported:
        _echo_group(header, [str(entry.path) for entry in report.of(verdict)])
    if not relink and (report.of("relinkable") or report.of("dangling")):
        typer.echo("Pass --relink to repair the two classes above.")
    for backup in report.backups:
        typer.echo(f"backup: {backup}")
