"""``mosaic describe``: what a dataset is and how much it holds.

The verb prints the manifest's fields, the counts of sequences, recordings,
videos and artifacts, the names of the divided recordings, the track schemas
and the frame rates. ``--json`` prints the
:class:`~mosaic.core.dataset_summary.DatasetSummary` that
:func:`~mosaic.core.dataset_summary.dataset_summary` returns. The verb writes
nothing to the dataset.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from mosaic.cli._context import load_dataset
from mosaic.cli._io import emit_json
from mosaic.cli._render import render_kv, render_table

if TYPE_CHECKING:
    from mosaic.core.dataset_summary import DatasetSummary


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
) -> None:
    """Describe a dataset: its manifest, counts, track schemas and frame rates."""
    # Imported in the body, so --help and the other verbs stay import-light.
    from mosaic.core.dataset_summary import SUMMARY_KINDS, dataset_summary
    from mosaic.tracking import dataset_inventory

    ds = load_dataset(manifest, read_only=True)
    summary = dataset_summary(ds, found=dataset_inventory(ds, kinds=SUMMARY_KINDS))
    if as_json:
        emit_json(summary.model_dump(mode="json"))
        return
    _render(summary)


def _render(summary: DatasetSummary) -> None:
    """Print the manifest's fields and counts, then the schemas, rates, tags and libraries."""
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
            "sequences": counts.sequences,
            "recordings": _recordings(summary),
            "videos": counts.videos,
            "tracks_variants": _shown(counts.tracks_variants),
            "feature_runs": _shown(counts.feature_runs),
            "models": _shown(counts.models),
        }
    )
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
    if summary.notes:
        typer.echo()
        typer.echo(summary.notes)
    if summary.next_steps:
        typer.echo()
        for step in summary.next_steps:
            typer.echo(f"next: {step}")
    for error in summary.inventory_errors:
        typer.echo(f"note: inventory read failed: {error}", err=True)


def _recordings(summary: DatasetSummary) -> str:
    """Return the recordings count with the divided ones counted and named."""
    divided = f"{summary.counts.divided_recordings} divided"
    names = ", ".join(recording.name for recording in summary.divided_recordings)
    if names:
        divided = f"{divided}: {names}"
    return f"{summary.counts.recordings} ({divided})"


def _shown(count: int | None) -> int | str:
    """Return *count*, or ``unknown`` for a kind the inventory did not report."""
    return "unknown" if count is None else count
