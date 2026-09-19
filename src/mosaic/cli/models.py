"""``mosaic models``: questions about the models a dataset can reach."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from mosaic.cli._context import load_dataset
from mosaic.cli._io import emit_json, fail

models_app = typer.Typer(
    name="models",
    help="Ask about trained models: this dataset's, and its linked libraries'.",
    no_args_is_help=True,
    add_completion=False,
)

ManifestOption = Annotated[
    Path,
    typer.Option(
        "--manifest", "-m", help="Path to the dataset manifest (dataset.yaml)."
    ),
]


@models_app.command("provenance")
def provenance(
    manifest: ManifestOption,
    run_id: Annotated[str, typer.Argument(help="The model's run id.")],
    kind: Annotated[
        str | None,
        typer.Option(
            "--kind",
            help="The training op. Read from the run id when it is left out.",
        ),
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Emit JSON.")] = False,
) -> None:
    """Show what a model was trained on, back to the exact annotation revisions."""
    from mosaic.core.pipeline.op_identity import parse_op_run_id
    from mosaic.tracking.training_provenance import training_provenance

    dataset = load_dataset(manifest)
    parsed = parse_op_run_id(run_id)
    resolved_kind = kind or (parsed.kind if parsed is not None else None)
    if resolved_kind is None:
        fail(f"{run_id!r} does not name its op; pass --kind, e.g. --kind train-pose")
    try:
        found = training_provenance(dataset, resolved_kind, run_id)
    except KeyError as exc:
        fail(str(exc.args[0]))

    if as_json:
        emit_json(found.as_json())
        return
    where = f" (served by library {found.served_by!r})" if found.served_by else ""
    typer.echo(f"{found.run_id}{where}")
    typer.echo(f"  trained on   {found.data_path or '(not recorded)'}")
    if found.data_fingerprint:
        typer.echo(f"  fingerprint  {found.data_fingerprint}")
    if found.base_run_id:
        typer.echo(f"  fine-tuned   {found.base_run_id}")
    if found.prepared_run_id:
        typer.echo(f"  prepared by  {found.prepared_run_id}")
    for item in found.sets:
        state = "" if item.reachable else "  [not on disk]"
        typer.echo(
            f"    {item.set_key} rev{item.revision}  {item.digest}  "
            f"from {item.origin_uuid or '(unknown dataset)'}{state}"
        )
        for name in sorted(item.origin):
            typer.echo(f"      {name}: {item.origin[name]}")
    if found.stopped_at:
        typer.echo(f"  stopped: {found.stopped_at}")
