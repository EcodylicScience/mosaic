"""``mosaic reprobe-media``: re-probe the media an index already lists."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Annotated

import typer

from mosaic.cli._context import load_dataset
from mosaic.cli._io import emit_json, fail, log, stdout_to_stderr


def _echo_group(
    emit: Callable[[str], None], header: str, details: Sequence[str]
) -> None:
    """Emit one group of rows under a counted header, or nothing.

    The two unreadable conditions get their own group each, wherever they are
    printed: an archived file and a corrupt one call for different responses, and
    one merged list would hide which is which. *emit* is the sink, so the abort
    can report to stderr while the human report goes to stdout.
    """
    if not details:
        return
    emit(f"{len(details)} row(s) {header}:")
    for detail in details:
        emit(f"  {detail}")


def reprobe_media_command(
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
            help="Rewrite the media index in place. Default is a dry-run report.",
        ),
    ] = False,
    skip_unreadable: Annotated[
        bool,
        typer.Option(
            "--skip-unreadable",
            help=(
                "Leave rows untouched instead of aborting when their media is "
                "missing from disk or is present but cannot be probed. The two "
                "are reported separately either way."
            ),
        ),
    ] = False,
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit the result as JSON.")
    ] = False,
) -> None:
    """Re-probe the media files the media index already lists, in place.

    Where ``index-media`` scans directories to BUILD a media index, this re-probes
    the media index that ALREADY EXISTS: every file it lists is measured again and
    the measurement written back into that row. The index is authoritative --
    group, sequence, order and paths are read from it and never re-derived.

    It is also the migration path for an index written before a column existed.
    An index whose header lacks the identity columns outright is widened to the
    current schema and written back complete, so one run takes an old media index
    to the full current column set with ``video_uuid`` and ``content_digest``
    populated for every row that can carry them.

    A row whose media is missing from disk, or present but impossible to probe,
    aborts the run before anything is written, so the media index is never left
    half-identified; ``--skip-unreadable`` leaves those rows verbatim and
    migrates the rest. The two conditions are always counted and listed
    separately: a missing file is expected on an old corpus, while one that will
    not probe is a corruption signal. A file whose measured identity has drifted
    from a recorded one is counted and reported, and the run continues with the
    fresh measurement; a row that recorded no identity at all is not drift, it is
    the ordinary migration case.

    Dry-run by default; pass ``--apply`` to write. An applied run leaves a
    timestamped backup of every index it modifies.
    """
    from mosaic.core.media.reprobe import ReprobeAbort

    ds = load_dataset(manifest)
    try:
        with stdout_to_stderr():
            report = ds.reprobe_media(apply=apply, skip_unreadable=skip_unreadable)
    except ReprobeAbort as error:
        _echo_group(log, "point at media missing from disk", error.missing)
        _echo_group(log, "point at media present but unprobeable", error.unprobeable)
        fail(f"reprobe-media failed: {error}")
    except OSError as error:
        fail(f"reprobe-media failed: {error}")

    if as_json:
        emit_json(report.payload())
        return

    typer.echo(f"index: {report.index_path}")
    if not report.changed:
        typer.echo("reprobe-media: already fully probed; no changes.")
        return
    verb = "minted" if report.applied else "would mint"
    summary = (
        f"{verb} identity for {report.minted} of {report.rows_total} row(s) "
        f"({report.unchanged} already current, {report.unmintable} unmintable); "
        f"{report.rows_patched} row(s) rewritten."
    )
    typer.echo(summary)
    if report.schema_columns_added:
        typer.echo(f"schema columns added: {', '.join(report.schema_columns_added)}")
    for stored in (
        *report.content_digest_changed,
        *report.derivative.content_digest_changed,
    ):
        typer.echo(f"content_digest changed: {stored}")
    for stored in (*report.video_uuid_changed, *report.derivative.video_uuid_changed):
        typer.echo(f"video_uuid changed at an unchanged content_digest: {stored}")
    _echo_group(
        typer.echo,
        "left untouched -- media missing from disk",
        [*report.missing, *report.derivative.missing],
    )
    _echo_group(
        typer.echo,
        "left untouched -- media present but unprobeable",
        [*report.unprobeable, *report.derivative.unprobeable],
    )
    # Dropping a curated column is the one thing this command destroys, so it is
    # named per file and per column, and the backup is the only way back.
    for label, columns in (
        ("media_raw index", report.unknown_columns),
        ("media index", report.derivative.unknown_columns),
    ):
        for column in columns:
            dropped = (
                f"column outside the media-index schema, dropped from the "
                f"{label}: {column}"
            )
            typer.echo(dropped)
    if report.derivative.media_index is not None:
        derivative_summary = (
            f"derivatives: {report.derivative.minted} minted, "
            f"{report.derivative.relinked} back-link(s) re-keyed."
        )
        typer.echo(derivative_summary)
    for unresolved in report.derivative.unresolved:
        message = "derivative back-link resolves to no indexed original"
        typer.echo(f"{message}: {unresolved}")
    for backup in report.backups:
        typer.echo(f"backup: {backup}")
