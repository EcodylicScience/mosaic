"""``mosaic runs``: list run attempts from the JSONL run-logs (newest first).

Each attempt's ``kind`` is ``"feature"`` for *all* features (the feature slug lives in
``target``, e.g. ``speed-angvel__from__tracks``), the op kind for tracking ops, and
``"trex"`` for TREx (plus any future payload kind). Use ``--target`` to filter to one
feature's runs.

``--limit`` and ``--after`` read one page and reduce only the run-logs that it
reads. Under ``--json`` a paged listing prints an object, ``items`` and
``next_after``, and an unpaged one the array of rows.

``mosaic runs errors <execution_id>`` lists the entities that one attempt lost,
each with its failure record. ``--json`` prints the
:class:`~mosaic.core.pipeline.graph.EntryErrorsWire` that
:func:`~mosaic.core.pipeline.graph.entry_errors_wire` returns.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Final

import typer

from mosaic.cli._context import load_dataset, run_log_dir_for
from mosaic.cli._io import emit_json, fail, log
from mosaic.cli._render import recorded_error_text, render_kv, render_table

if TYPE_CHECKING:
    from mosaic.core.pipeline.graph import EntryErrorsWire

_COLUMNS = ["execution_id", "kind", "target", "run_id", "status", "owner"]
_ERROR_COLUMNS: Final = ["key", "retry", "attempts", "not_before", "error"]

runs_app = typer.Typer(
    name="runs",
    help="List run attempts newest first, or the entities that one attempt lost.",
    invoke_without_command=True,
    add_completion=False,
)

JsonOption = Annotated[bool, typer.Option("--json", help="Emit as JSON on stdout.")]


@runs_app.callback()
def list_attempts(
    ctx: typer.Context,
    manifest: Annotated[
        Path | None,
        typer.Option(
            "--manifest", "-m", help="Path to the dataset manifest (dataset.yaml)."
        ),
    ] = None,
    kind: Annotated[
        str | None,
        typer.Option(
            "--kind", help="Filter by runs.kind ('feature', 'trex', or an op kind)."
        ),
    ] = None,
    status: Annotated[
        str | None,
        typer.Option(
            "--status", help="Filter by status (running|finished|failed|cancelled)."
        ),
    ] = None,
    target: Annotated[
        str | None,
        typer.Option(
            "--target", help="Filter by exact runs.target (feature-slug filter)."
        ),
    ] = None,
    limit: Annotated[
        int | None,
        typer.Option("--limit", help="List at most this many attempts.", min=1),
    ] = None,
    after: Annotated[
        str | None,
        typer.Option(
            "--after", help="Start after the page that printed this next_after."
        ),
    ] = None,
    as_json: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Emit the rows as a JSON array, or a page object with --limit/--after.",
        ),
    ] = False,
) -> None:
    """List run attempts newest first, optionally filtered by kind / status / target."""
    if ctx.invoked_subcommand is not None:
        listing = {
            "--kind": kind,
            "--status": status,
            "--target": target,
            "--limit": limit,
            "--after": after,
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
    from mosaic.runlog import list_runs

    ds = load_dataset(manifest, read_only=True)
    page = list_runs(
        run_log_dir_for(ds),
        limit=limit,
        after=after,
        kind=kind,
        status=status,
        target=target,
    )
    paged = limit is not None or after is not None
    if as_json:
        emit_json(page if paged else page["items"])
        return
    render_table(page["items"], _COLUMNS)
    if page["next_after"] is not None:
        log(f"note: --after {page['next_after']} lists the next page")


@runs_app.command("errors")
def list_entry_errors(
    execution_id: Annotated[
        str, typer.Argument(help="The attempt ULID whose lost entities to list.")
    ],
    ctx: typer.Context,
    manifest: Annotated[
        Path | None,
        typer.Option(
            "--manifest",
            "-m",
            help=(
                "Path to the dataset manifest (dataset.yaml). Defaults to the one "
                "given before 'errors'."
            ),
        ),
    ] = None,
    limit: Annotated[
        int | None,
        typer.Option(
            "--limit", help="List at most this many entities (default 20).", min=1
        ),
    ] = None,
    as_json: JsonOption = False,
) -> None:
    """List the entities that one attempt lost, each with its failure record.

    Each entity's retry state says whether a pipeline request attempts it again:
    ``ready``, ``waiting`` until ``not_before``, ``quarantined`` until a request
    with ``allow_partial`` proceeds without it, or ``unrecorded`` when its record
    counts zero failures.
    """
    from mosaic.core.pipeline.graph import FAILED_ENTRY_SAMPLE, entry_errors_wire

    if manifest is None and isinstance(ctx.obj, Path):
        manifest = ctx.obj
    if manifest is None:
        fail("Missing option '--manifest' / '-m'.", code=2)
    ds = load_dataset(manifest, read_only=True)
    report = entry_errors_wire(
        ds.base_dir, execution_id, limit=limit or FAILED_ENTRY_SAMPLE
    )
    if report is None:
        fail(f"No run found with execution_id={execution_id}.")
    if as_json:
        emit_json(report.model_dump(mode="json"))
        return
    _render_entry_errors(report)


def _render_entry_errors(report: EntryErrorsWire) -> None:
    """Print the attempt as ``key: value`` lines and a row per lost entity."""
    render_kv(
        {
            "execution_id": report.execution_id,
            "target": report.target,
            "run_id": report.run_id,
            "outcome": report.outcome,
            "total": report.total,
            "held_back_by_quarantine": report.held_back_by_quarantine,
        }
    )
    if not report.entries:
        return
    rows: list[dict[str, object]] = [
        {
            "key": entry.key,
            "retry": entry.retry,
            "attempts": entry.attempts,
            "not_before": entry.not_before,
            "error": "" if entry.error is None else recorded_error_text(entry.error),
        }
        for entry in report.entries
    ]
    render_table(rows, _ERROR_COLUMNS)
    if report.has_more:
        log(f"note: {len(report.entries)} of {report.total} listed; --limit lists more")
