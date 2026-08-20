"""``mosaic upgrade-tracks``: rescale centimetre-era TRex tables into pixels."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from mosaic.cli._context import load_dataset
from mosaic.cli._io import emit_json, fail, stdout_to_stderr


def upgrade_tracks_command(
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
            help="Write the rescaled tables. Default is a dry-run report.",
        ),
    ] = False,
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit the result as JSON.")
    ] = False,
) -> None:
    """Convert this dataset's TRex tables from centimeters to pixels, in place.

    **Reconverting is the better route** and this is not it. ``convert-tracks``
    re-reads the ``.npz``, so it depends on nothing a previous conversion
    happened to keep. Use this only when the raw export is gone -- typically
    reclaimed by ``sweep-tracking`` once its run passed the retention window.

    It works because TRex records the factor it scaled by inside every export,
    and the old conversion copied that field through to the table. A table that
    does not carry it is refused rather than assumed to be unscaled: centimeters
    and pixels are indistinguishable once the number is lost, so guessing would
    be wrong silently.

    Results land under the variant a reconversion would have produced, so
    converting the same entry properly later finds its table already in place
    rather than writing a second one beside it.

    Dry-run by default, because this rewrites data rather than an index.
    """
    from mosaic.core.pipeline.upgrade_tracks import upgrade_trex_tables

    ds = load_dataset(manifest)
    try:
        with stdout_to_stderr():
            report = upgrade_trex_tables(ds, apply=apply)
    except Exception as exc:  # noqa: BLE001 - surface migration errors cleanly
        fail(f"upgrade-tracks failed: {exc}")

    if as_json:
        emit_json(
            {
                "status": "ok",
                "applied": apply,
                "target_variant": report.target_variant,
                "upgraded": len(report.upgraded),
                "refused": [
                    {
                        "group": o.group,
                        "sequence": o.sequence,
                        "detail": o.detail,
                    }
                    for o in report.refused
                ],
                "skipped": len(report.skipped),
            }
        )
        return

    verb = "upgraded" if apply else "would upgrade"
    typer.echo(f"{verb} {len(report.upgraded)} table(s) -> {report.target_variant}")
    if report.skipped:
        typer.echo(f"skipped {len(report.skipped)} table(s) that need no rescaling")
    for outcome in report.refused:
        typer.echo(
            f"refused {outcome.group}/{outcome.sequence}: {outcome.detail}", err=True
        )
    if report.refused and apply:
        fail(
            f"{len(report.refused)} table(s) could not be upgraded. They are "
            "unchanged; reconvert them from their raw exports."
        )
    if not apply and report.upgraded:
        typer.echo("dry run; pass --apply to write")
