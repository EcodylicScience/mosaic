"""``mosaic inventory``: what has been computed in this dataset.

``mosaic sequences`` lists the sequences the tracks index names, ``mosaic runs``
and ``mosaic status`` report attempts from the run-log, and ``mosaic features
list`` is the registry of what the installation can compute. This verb reports
the artifacts that the dataset contains. ``--json`` prints the
:class:`~mosaic.core.pipeline.inventory.InventoryWire` that
:func:`~mosaic.core.pipeline.inventory.inventory_wire` returns.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from mosaic.cli._context import load_dataset
from mosaic.cli._io import (
    EntriesOption,
    emit_json,
    entries_scope,
    fail,
    fail_classified,
)
from mosaic.cli._render import render_json_block, render_table
from mosaic.core.pipeline.artifact_kinds import ARTIFACT_KINDS, is_artifact_kind

if TYPE_CHECKING:
    from mosaic.core.pipeline.inventory import (
        ArtifactKind,
        ArtifactStatus,
        InventoryWire,
    )

inventory_app = typer.Typer(
    name="inventory",
    help="Report every computed artifact, its identity and its coverage.",
    invoke_without_command=True,
    add_completion=False,
)

TracksRunIdOption = Annotated[
    str | None,
    typer.Option(
        "--tracks-run-id",
        help=(
            "Keep the feature runs that read this tracks variant, measured over "
            "its entries. '' keeps the runs over the unlabeled tables alone."
        ),
    ),
]
JsonOption = Annotated[bool, typer.Option("--json", help="Emit as a JSON object.")]


@inventory_app.callback()
def list_inventory(
    ctx: typer.Context,
    manifest: Annotated[
        Path | None,
        typer.Option(
            "--manifest", "-m", help="Path to the dataset manifest (dataset.yaml)."
        ),
    ] = None,
    kind: Annotated[
        list[str] | None,
        typer.Option(
            "--kind",
            help=(
                "Restrict to an artifact kind (repeatable): "
                f"{', '.join(ARTIFACT_KINDS)}."
            ),
        ),
    ] = None,
    entry: EntriesOption = None,
    tracks_run_id: TracksRunIdOption = None,
    status_not: Annotated[
        list[str] | None,
        typer.Option(
            "--status-not",
            help=(
                "Leave out a status (repeatable): absent, partial, complete, "
                "complete-but-drifted, inconsistent."
            ),
        ),
    ] = None,
    limit: Annotated[
        int | None,
        typer.Option("--limit", help="List at most this many artifacts.", min=1),
    ] = None,
    cursor: Annotated[
        str | None,
        typer.Option("--cursor", help="Start after the page that printed this cursor."),
    ] = None,
    as_json: JsonOption = False,
) -> None:
    """Report every computed artifact, its identity and its coverage."""
    if ctx.invoked_subcommand is not None:
        listing = {
            "--kind": kind,
            "--entry": entry,
            "--tracks-run-id": tracks_run_id,
            "--status-not": status_not,
            "--limit": limit,
            "--cursor": cursor,
            "--json": as_json or None,
        }
        given = [name for name, value in listing.items() if value is not None]
        if given:
            fail(
                f"{', '.join(given)} given before '{ctx.invoked_subcommand}'; "
                "give the subcommand's options after it.",
                code=2,
            )
        ctx.obj = manifest
        return
    if manifest is None:
        fail("Missing option '--manifest' / '-m'.", code=2)
    wire = _read(
        manifest,
        kinds=_kinds(kind),
        entry=entry,
        tracks_run_id=tracks_run_id,
        status_not=_statuses(status_not),
        limit=limit,
        cursor=cursor,
        ref=None,
        as_json=as_json,
    )
    if as_json:
        emit_json(wire.model_dump(mode="json"))
        return

    # The ref names the kind, the name and the run id, and ``show`` takes it.
    rows: list[dict[str, object]] = [
        {
            "ref": artifact.ref,
            "status": artifact.status,
            "coverage": "all"
            if artifact.coverage.covers_all
            else f"{artifact.coverage.covered}/{artifact.coverage.target}",
            "drift": artifact.drift or "",
        }
        for artifact in wire.artifacts
    ]
    if not rows:
        typer.echo("No artifacts recorded in this dataset.")
    else:
        render_table(rows, ["ref", "status", "coverage", "drift"])
    _print_notes(wire)
    if wire.next_cursor is not None:
        typer.echo(
            f"note: {len(wire.artifacts)} of {wire.total} artifacts listed; "
            f"--cursor {wire.next_cursor} lists the next page",
            err=True,
        )


@inventory_app.command("show")
def show_artifact(
    ref: Annotated[
        str,
        typer.Argument(
            help=(
                "The artifact's ref, as a listing prints it: its kind and the "
                "parts of its key joined by '/', e.g. feature/<storage>/<run_id>."
            )
        ),
    ],
    ctx: typer.Context,
    manifest: Annotated[
        Path | None,
        typer.Option(
            "--manifest",
            "-m",
            help=(
                "Path to the dataset manifest (dataset.yaml). Defaults to the one "
                "given before 'show'."
            ),
        ),
    ] = None,
    entry: EntriesOption = None,
    tracks_run_id: TracksRunIdOption = None,
    as_json: JsonOption = False,
) -> None:
    """Report one artifact in depth: its recorded params, orphans and drift."""
    if manifest is None and isinstance(ctx.obj, Path):
        manifest = ctx.obj
    if manifest is None:
        fail("Missing option '--manifest' / '-m'.", code=2)
    kind = ref.split("/", 1)[0]
    wire = _read(
        manifest,
        kinds=[kind] if is_artifact_kind(kind) else None,
        entry=entry,
        tracks_run_id=tracks_run_id,
        status_not=None,
        limit=None,
        cursor=None,
        ref=ref,
        as_json=as_json,
    )
    if as_json:
        emit_json(wire.model_dump(mode="json"))
        return
    render_json_block(wire.artifacts[0].model_dump(mode="json"))
    _print_notes(wire)


def _read(
    manifest: Path,
    *,
    kinds: list[ArtifactKind] | None,
    entry: list[str] | None,
    tracks_run_id: str | None,
    status_not: list[ArtifactStatus] | None,
    limit: int | None,
    cursor: str | None,
    ref: str | None,
    as_json: bool,
) -> InventoryWire:
    """Read the dataset's inventory and project it as the options ask."""
    # Imported in the body, so --help and the other read verbs stay import-light.
    from mosaic.core.pipeline.error_kind import ClassifiedError
    from mosaic.core.pipeline.inventory import inventory_wire
    from mosaic.tracking import dataset_inventory

    ds = load_dataset(manifest, read_only=True)
    found = dataset_inventory(ds, kinds=kinds)
    try:
        return inventory_wire(
            ds,
            found,
            kinds=kinds,
            scope=entries_scope(entry),
            tracks_run_id=tracks_run_id,
            status_not=status_not,
            limit=limit,
            cursor=cursor,
            ref=ref,
        )
    except ClassifiedError as error:
        fail_classified(error, as_json=as_json)


def _kinds(names: list[str] | None) -> list[ArtifactKind] | None:
    """Return the ``--kind`` values, refusing an unknown one.

    A misspelled kind would report an empty list, the output of a dataset that
    contains none of that kind.
    """
    if not names:
        return None
    unknown = sorted(name for name in names if not is_artifact_kind(name))
    if unknown:
        fail(
            f"unknown artifact kind(s): {', '.join(unknown)}. "
            f"Known kinds: {', '.join(sorted(ARTIFACT_KINDS))}."
        )
    return [name for name in names if is_artifact_kind(name)]


def _statuses(names: list[str] | None) -> list[ArtifactStatus] | None:
    """Return the ``--status-not`` values, refusing an unknown one."""
    from mosaic.core.pipeline.inventory import ARTIFACT_STATUSES, is_artifact_status

    if not names:
        return None
    unknown = sorted(name for name in names if not is_artifact_status(name))
    if unknown:
        fail(
            f"unknown status(es): {', '.join(unknown)}. "
            f"Known statuses: {', '.join(ARTIFACT_STATUSES)}."
        )
    return [name for name in names if is_artifact_status(name)]


def _print_notes(wire: InventoryWire) -> None:
    """Print the frame-axis notes, the unavailable kinds and the scan's errors."""
    for artifact in wire.artifacts:
        # A frame-axis finding describes a table that is otherwise complete, so
        # it is a note beside the table and leaves the status as it is.
        mismatched = artifact.notes.get("frame_axis_mismatch", 0)
        if mismatched:
            typer.echo(
                f"note: {mismatched} entry(ies) of {artifact.run_id} were read "
                "short of or past their media, so anything reading pixels at a "
                "track frame may be off; see mosaic measure-tracks",
                err=True,
            )
        tail_short = artifact.notes.get("frame_tail_short", 0)
        if tail_short:
            typer.echo(
                f"note: {tail_short} entry(ies) of {artifact.run_id} were read "
                "short by no more than their tool is known to leave unread at the "
                "end of a file, which a count cannot confirm; see mosaic "
                "measure-tracks",
                err=True,
            )
    for missing in wire.unavailable_kinds:
        typer.echo(f"note: {missing} was not reported (no producer imported)", err=True)
    for message in wire.errors:
        typer.echo(f"warning: {message}", err=True)
