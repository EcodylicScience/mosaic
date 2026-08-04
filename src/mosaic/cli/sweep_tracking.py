"""``mosaic sweep-tracking``: reclaim finished tracker intermediates."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from mosaic.cli._context import load_dataset
from mosaic.cli._io import emit_json, fail, stdout_to_stderr


def sweep_tracking_command(
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
            help="Delete what is reclaimable. Default is a dry-run report.",
        ),
    ] = False,
    root: Annotated[
        list[str] | None,
        typer.Option(
            "--root",
            help="Restrict to one tracker root (repeatable): trex, sleap, ...",
        ),
    ] = None,
    tracker_days: Annotated[
        float,
        typer.Option(
            "--tracker-days",
            help="Keep finished tracker output this long (default 14).",
        ),
    ] = 14.0,
    inference_days: Annotated[
        float,
        typer.Option(
            "--inference-days",
            help="Keep finished inference output this long (default 3).",
        ),
    ] = 3.0,
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit the result as JSON.")
    ] = False,
) -> None:
    """Delete tracker working directories that are finished and past their window.

    Never touches work in progress: a directory a live execution holds, or one
    this dataset's index does not yet name, is reported and left alone. Rows go
    before files, so an interrupted sweep leaves rows naming absent directories
    -- which ``mosaic reindex`` repairs -- rather than files nothing names.

    Dry-run by default; pass ``--apply`` to delete.
    """
    # A tracker root's index is opened through the reconcilable-index registry,
    # which each tracker fills as a side effect of being imported -- ``core``
    # does not import ``tracking``, so nothing else fills it. Without this the
    # registry is empty here, every working directory reads as ``unrowed``, and
    # unrowed is refused: the sweep reclaims nothing at all, while reporting a
    # well-formed result that blames the index.
    from mosaic.tracking import register_ops

    register_ops()
    ds = load_dataset(manifest)
    try:
        with stdout_to_stderr():
            report = ds.sweep_tracking(
                apply=apply,
                roots=root or None,
                retention_overrides={
                    "tracker": tracker_days,
                    "inference": inference_days,
                },
            )
    except Exception as exc:  # noqa: BLE001 - surface sweep errors cleanly
        fail(f"sweep-tracking failed: {exc}")

    if as_json:
        emit_json(report.payload())
        return

    if not report.considered:
        from mosaic.core.pipeline.sweep import decline_text

        if report.declined is None:
            fail("sweep-tracking declined without a reason; this is a bug.")
        typer.echo(f"sweep-tracking declined: {decline_text(report.declined)}")
        return

    from mosaic.core.pipeline.sweep import REFUSED_NOTES, deletable

    verb = "deleted" if report.applied else "would delete"
    reclaimable = [e for e in report.entries if deletable(e.verdict)]
    typer.echo(f"{verb} {len(reclaimable)} entry directory/ies.")
    if report.applied:
        mib = report.bytes_reclaimed / (1024 * 1024)
        typer.echo(f"reclaimed {mib:.1f} MiB, dropped {report.rows_dropped} row(s).")

    # Each refused class is grouped and counted rather than summed into one
    # number, because they mean different things: held work will be reclaimable
    # later, a foreign directory never will, and an unrowed one wants a reindex.
    for verdict, note in REFUSED_NOTES.items():
        found = report.of(verdict)
        if found:
            typer.echo(f"\n{len(found)} {verdict} ({note}):")
            for entry in found:
                typer.echo(f"  {entry.path}")
    if report.held_for_age and not report.applied:
        typer.echo(
            "\nSome finished output is newer than its window; re-run later or "
            "lower --tracker-days / --inference-days."
        )
