"""``mosaic describe``: what a dataset is and how much it holds.

The verb prints the manifest's fields, the counts of sequences, recordings,
videos and artifacts, the names of the divided recordings, the rule and span of
each recording divided from its stores, the track schemas, the frame rates and the
saved selections.
``--raw`` adds what the raw indexes, the tracks index and the run-logs record
about the raw data.
``--json`` prints the :class:`~mosaic.core.dataset_summary.DatasetSummary` that
:func:`~mosaic.core.dataset_summary.dataset_summary` returns, whose
``raw_report`` is ``null`` without ``--raw``. The verb writes nothing to the
dataset.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from mosaic.cli._context import load_dataset
from mosaic.cli._io import emit_json
from mosaic.cli._render import render_kv, render_table
from mosaic.cli.selection import render_selections

if TYPE_CHECKING:
    from mosaic.core.dataset_summary import DatasetSummary
    from mosaic.core.raw_report import RawReport
    from mosaic.core.strict_model import StrictModel


def describe_command(
    manifest: Annotated[
        Path,
        typer.Option(
            "--manifest", "-m", help="Path to the dataset manifest (dataset.yaml)."
        ),
    ],
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit as a JSON object.")
    ] = False,
    raw: Annotated[
        bool,
        typer.Option(
            "--raw",
            help=(
                "Add the raw report: duplicate files, unhashed files, files a "
                "group pattern misses, frame-axis and span outliers, uneven time "
                "steps and missing sources."
            ),
        ),
    ] = False,
) -> None:
    """Describe a dataset: its manifest, counts, track schemas and frame rates."""
    # Imported in the body, so --help and the other verbs stay import-light.
    from mosaic.core.dataset_summary import SUMMARY_KINDS, dataset_summary
    from mosaic.tracking import dataset_inventory

    ds = load_dataset(manifest, read_only=True)
    summary = dataset_summary(
        ds, found=dataset_inventory(ds, kinds=SUMMARY_KINDS), raw=raw
    )
    if as_json:
        emit_json(summary.model_dump(mode="json"))
        return
    _render(summary)


def _render(summary: DatasetSummary) -> None:
    """Print the manifest's fields and counts, then each table the summary holds.

    The tables are the schemas, rates, arena layouts, selections, tags and libraries.
    """
    counts = summary.counts
    render_kv(
        {
            "name": summary.name,
            "uuid": summary.uuid or "none",
            "manifest": summary.manifest,
            "manifest_version": summary.manifest_version,
            "managed_by": summary.managed_by if summary.api_managed else "none",
            "splits": ", ".join(
                f"{group}={role}" for group, role in summary.splits.items()
            )
            or "none",
            "sequences": f"{counts.sequences} with a tracks table",
            "recordings": _recordings(summary),
            "videos": counts.videos,
            "tracks_variants": _shown(counts.tracks_variants),
            "feature_runs": _shown(counts.feature_runs),
            "models": _shown(counts.models),
        }
    )
    for line in _division_lines(summary):
        typer.echo(line)
    if summary.track_schemas:
        typer.echo()
        render_table(
            [schema.model_dump() for schema in summary.track_schemas],
            ["name", "tables"],
        )
    if summary.frame_rates:
        typer.echo()
        render_table(
            [rate.model_dump() for rate in summary.frame_rates], ["fps", "videos"]
        )
    if summary.arena_layouts:
        typer.echo()
        render_table(
            [
                {
                    "recording": layout.recording,
                    "camera": layout.camera,
                    "arenas": len(layout.arenas),
                }
                for layout in summary.arena_layouts
            ],
            ["recording", "camera", "arenas"],
        )
    if summary.selections:
        typer.echo()
        render_selections(summary.selections)
    if summary.tags:
        typer.echo()
        render_table(
            [tag.model_dump() for tag in summary.tags], ["name", "type", "value"]
        )
    if summary.libraries:
        typer.echo()
        render_table(
            [link.model_dump() for link in summary.libraries], ["id", "path", "uuid"]
        )
    if summary.raw_report is not None:
        typer.echo()
        _render_raw(summary.raw_report)
    if summary.notes:
        typer.echo()
        typer.echo(summary.notes)
    if summary.next_steps:
        typer.echo()
        for step in summary.next_steps:
            typer.echo(f"next: {step}")
    for error in summary.inventory_errors:
        typer.echo(f"note: inventory read failed: {error}", err=True)


def _render_raw(report: RawReport) -> None:
    """Print the report's counts, then a table for each list that holds items."""
    render_kv(
        {
            "duplicate_sets": report.duplicate_sets,
            "unhashed": ", ".join(f"{u.kind}={u.count}" for u in report.unhashed)
            or "none",
            "pattern_miss_sources": report.pattern_miss_sources,
            "frame_axis_tables": report.frame_axis_tables,
            "span_outlier_tables": report.span_outlier_tables,
            "time_irregular_entries": report.time_irregular_entries,
            "missing_source_paths": report.missing_source_paths,
        }
    )
    duplicates = [
        {**found.model_dump(), "sources": ", ".join(found.sources)}
        for found in report.duplicates
    ]
    if duplicates:
        typer.echo()
        render_table(duplicates, ["md5", "count", "sources"])
    tables: tuple[tuple[Sequence[StrictModel], list[str]], ...] = (
        (report.pattern_misses, ["kind", "source", "pattern", "count"]),
        (
            report.frame_axis,
            ["verdict", "run_id", "group", "sequence", "frames_read", "media_frames"],
        ),
        (
            report.span_outliers,
            ["run_id", "group", "sequence", "span", "median_span"],
        ),
        (
            report.time_irregular,
            ["entry", "count", "largest_step", "execution_id", "recorded_at"],
        ),
        (report.missing_sources, ["kind", "source", "path"]),
    )
    for rows, columns in tables:
        if rows:
            typer.echo()
            render_table([row.model_dump() for row in rows], columns)


def _recordings(summary: DatasetSummary) -> str:
    """Return the recordings count, what it counts over, and the divided ones named."""
    divided = f"{summary.counts.divided_recordings} divided"
    names = ", ".join(
        recording.name
        for recording in summary.recordings
        if recording.kind == "divided"
    )
    if names:
        divided = f"{divided}: {names}"
    return f"{summary.counts.recordings} in the media or tracks index ({divided})"


def _division_lines(summary: DatasetSummary) -> list[str]:
    """Return one line per recording divided from its stores, with its rule and span."""
    lines: list[str] = []
    for recording in summary.recordings:
        if recording.kind != "divided" or recording.division is None:
            continue
        division = recording.division
        lines.append(
            f"division {recording.name}: {recording.n_sequences} sequences of "
            f"{division.frames} frames (chunks of {division.chunksize}) by "
            f"{division.identity}, camera {division.camera_serial or 'none'}, from "
            f"{division.first_start_utc} to {division.last_start_utc}, "
            f"{division.n_breaks} breaks"
        )
    return lines


def _shown(count: int | None) -> int | str:
    """Return *count*, or ``unknown`` for a kind the inventory did not report."""
    return "unknown" if count is None else count
