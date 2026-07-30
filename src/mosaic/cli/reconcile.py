"""``mosaic reconcile``: recompute artifact identifiers and re-address what moved.

The forward pass over the identity machinery. Where ``mosaic reindex`` reconciles
an index against the files on disk, this reconciles the on-disk identifiers against
the current code: it recomputes each run's ``run_id`` and, where the recorded
provenance confirms the inputs did not change, re-addresses the artifact under its
new identifier instead of recomputing it. Dry-run by default; ``--apply`` performs
the confirmed moves and marker refreshes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from mosaic.cli._context import load_dataset
from mosaic.cli._io import emit_json, fail, stdout_to_stderr


def reconcile_command(
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
            help="Perform the re-addresses and marker refreshes. Default is a "
            "dry-run report.",
        ),
    ] = False,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Reserved for the destructive path (deleting derivatives whose "
            "identity moved but could not be re-addressed); not yet wired.",
        ),
    ] = False,
    only: Annotated[
        list[str] | None,
        typer.Option(
            "--only",
            help="Restrict to one artifact kind (repeatable), e.g. --only features.",
        ),
    ] = None,
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit the result as JSON.")
    ] = False,
) -> None:
    """Recompute every artifact's identifier and re-address what the code moved.

    For each feature (and, as their reconcilers land, tracks and labels) run, the
    ``run_id`` is recomputed from the current code and compared with the recorded
    one. A run whose identifier moved *and* whose recorded provenance confirms its
    inputs unchanged is re-addressed -- its directory renamed, its index rows
    restamped, its marker refreshed, the index backed up first. A run that cannot
    be confirmed is reported and left, to be recomputed by an ordinary run.

    Idempotent and resumable: it reads the ``.identity_scheme`` marker each run was
    minted under, so a re-run over an already-migrated dataset reports every run
    ``ok``. Dry-run by default; ``--apply`` writes.
    """
    from mosaic.core.pipeline.reconcile import known_keys

    only_keys = tuple(only or ())
    valid = set(known_keys())
    unknown = [key for key in only_keys if key not in valid]
    if unknown:
        fail(
            f"unknown --only key(s) {', '.join(unknown)}; "
            f"known kinds: {', '.join(sorted(valid))}"
        )

    ds = load_dataset(manifest)
    try:
        with stdout_to_stderr():
            report = ds.reconcile(apply=apply, force=force, only=only_keys)
    except Exception as exc:  # noqa: BLE001 - surface reconcile errors cleanly
        fail(f"reconcile failed: {exc}")

    if as_json:
        emit_json(report.payload())
        return

    if not report.changed:
        typer.echo("reconcile: every artifact is current (nothing to do).")
        _echo_errors(report.errors)
        return

    verb = "applied" if report.applied else "would apply"
    counts = report.counts()
    summary = ", ".join(f"{verdict}={n}" for verdict, n in sorted(counts.items()))
    typer.echo(f"reconcile ({verb}): {summary}")
    for finding in report.findings:
        if finding.verdict == "ok":
            continue
        moved = (
            f"  {finding.old_address} -> {finding.new_address}"
            if finding.new_address != finding.old_address
            else ""
        )
        action = f" [{finding.action}]" if finding.action != "none" else ""
        typer.echo(
            f"  {finding.verdict}: {finding.key}{action}{moved}\n"
            f"      {finding.run_root}\n"
            f"      {finding.reason}"
        )
    verb_hygiene = "dropped" if report.applied else "would drop"
    for index_path, count in report.pruned.items():
        typer.echo(f"  dangling: {verb_hygiene} {count} row(s)\t{index_path}")
    verb_path = "rewrote" if report.applied else "would rewrite"
    for target, count in report.repathed.items():
        typer.echo(f"  non_portable: {verb_path} {count} path(s)\t{target}")
    for backup in report.backups:
        typer.echo(f"backup: {backup}")
    _echo_errors(report.errors)


def _echo_errors(errors: tuple[str, ...]) -> None:
    """List every artifact the sweep could not read, apart from the findings."""
    for message in errors:
        typer.echo(f"error: {message}")
