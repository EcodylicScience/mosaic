"""``mosaic inventory``: what has been computed in this dataset.

The verb nothing answered. ``mosaic sequences`` lists the sequences the tracks
index names, ``mosaic runs`` and ``mosaic status`` report *attempts* from the
run-log, and ``mosaic features list`` is the registry -- what the installation
knows how to compute. None of them say what this dataset holds.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from mosaic.cli._context import load_dataset
from mosaic.cli._io import emit_json, fail
from mosaic.cli._render import render_table

if TYPE_CHECKING:
    from mosaic.core.pipeline.inventory.model import AnyRecord, ArtifactKind


MISSING_SAMPLE = 20
"""How many missing keys cross the wire before the count stands in for them."""


def inventory_command(
    manifest: Annotated[
        Path,
        typer.Option(
            "--manifest", "-m", help="Path to the dataset manifest (dataset.yaml)."
        ),
    ],
    kind: Annotated[
        list[str] | None,
        typer.Option(
            "--kind",
            help=(
                "Restrict to an artifact kind (repeatable): feature, "
                "tracks-variant, labels-variant, tracker-run, frame-run, "
                "trained-model, media-derivative."
            ),
        ),
    ] = None,
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit as a JSON object.")
    ] = False,
) -> None:
    """Report every computed artifact, its identity and its coverage."""
    # Imported in the body, the discipline this CLI states: --help and the
    # read-only verbs stay import-light. The tracking import is not incidental --
    # it is what registers the contributors for tracker runs, frame runs and
    # trained models, without which those kinds would honestly report as
    # unavailable to a user who never asked about layering.
    from mosaic.core.pipeline.inventory import inventory as read_inventory
    from mosaic.tracking import register_ops

    # Called for its import side effect, which is what registers the
    # contributors for tracker runs, frame runs and trained models. Without it
    # those kinds report as unavailable -- a true answer about the process, and
    # a useless one to a user who never asked about layering.
    register_ops()
    from mosaic.core.pipeline.inventory.model import (
        ARTIFACT_KINDS,
        is_artifact_kind,
    )

    known = sorted(ARTIFACT_KINDS)
    unknown = sorted(name for name in (kind or ()) if name not in known)
    if unknown:
        # Refused rather than ignored: a misspelled kind that silently reports
        # nothing reads as "this dataset holds none of those", which is the same
        # output as the true answer and indistinguishable from it.
        fail(
            f"unknown artifact kind(s): {', '.join(unknown)}. "
            f"Known kinds: {', '.join(known)}."
        )
    wanted: list[ArtifactKind] = []
    for name in kind or []:
        if is_artifact_kind(name):
            wanted.append(name)

    ds = load_dataset(manifest)
    found = read_inventory(ds, kinds=wanted or None)

    ordered = sorted(found.records, key=_sort_key)
    rows: list[dict[str, object]] = [
        {
            "kind": record.ref.kind,
            "name": record.name,
            "run_id": record.run_id,
            "status": record.status,
            "coverage": _coverage_cell(record),
            "drift": len(record.drift) or "",
        }
        for record in ordered
    ]

    if as_json:
        emit_json(
            {
                "dataset": str(found.dataset_root),
                "artifacts": [
                    {**row, **_detail(record)}
                    for row, record in zip(rows, ordered, strict=True)
                ],
                # Never silently empty: a kind nobody can report on is a
                # different answer from a kind with nothing in it.
                "unavailable_kinds": sorted(found.unavailable_kinds),
                "errors": list(found.errors),
            }
        )
        return

    if not rows:
        typer.echo("No artifacts recorded in this dataset.")
    else:
        render_table(rows, ["kind", "name", "run_id", "status", "coverage", "drift"])
    for missing in sorted(found.unavailable_kinds):
        typer.echo(f"note: {missing} was not reported (no producer imported)", err=True)
    for message in found.errors:
        typer.echo(f"warning: {message}", err=True)


def _coverage_cell(record: AnyRecord) -> str:
    """``covered/target``, or ``all`` for an artifact answering for every key."""
    if record.coverage.covers_all:
        return "all"
    return f"{len(record.coverage.covered)}/{len(record.coverage.target)}"


def _detail(record: AnyRecord) -> dict[str, object]:
    """The per-artifact JSON fields the table has no column for.

    Missing entries are sampled rather than listed in full: a wide dataset under
    several kinds would otherwise emit megabytes, and the count beside the sample
    is what a reader actually acts on.
    """
    missing = sorted(str(key) for key in record.coverage.missing)
    detail: dict[str, object] = {
        "params": record.params_state,
        "started_at": record.started_at,
        "finished_at": record.finished_at,
        "covered": len(record.coverage.covered),
        "target": len(record.coverage.target),
        "missing_sample": missing[:MISSING_SAMPLE],
        "has_more_missing": len(missing) > MISSING_SAMPLE,
    }
    for name, keys in sorted(record.extra.items()):
        listed = sorted(keys)
        detail[name] = listed[:MISSING_SAMPLE]
        detail[f"n_{name}"] = len(listed)
    return detail


def _sort_key(record: AnyRecord) -> tuple[str, str, str]:
    """One ordering for the table and the JSON, so the two cannot disagree."""
    return (record.ref.kind, record.name, record.run_id)
