"""``mosaic reindex``: reconcile feature index CSVs with the parquet files on disk."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from mosaic.cli._context import load_dataset
from mosaic.cli._io import emit_json, fail, stdout_to_stderr


def reindex_command(
    manifest: Annotated[
        Path,
        typer.Option(
            "--manifest", "-m", help="Path to the dataset manifest (dataset.yaml)."
        ),
    ],
    feature: Annotated[
        str | None,
        typer.Option("--feature", help="Restrict to a single feature storage name."),
    ] = None,
    apply: Annotated[
        bool,
        typer.Option(
            "--apply/--dry-run",
            help="Rewrite indexes (drop stale rows). Default is a dry-run report.",
        ),
    ] = False,
    reconcile_registry: Annotated[
        bool,
        typer.Option(
            "--reconcile-registry/--no-reconcile-registry",
            help="Also prune the SQLite mirror (features/.mosaic.db).",
        ),
    ] = True,
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit the result as JSON.")
    ] = False,
) -> None:
    """Drop feature-index rows whose parquet files no longer exist.

    Relocated-but-present rows (a moved or synced dataset) are kept -- use
    ``make_portable`` / ``rewrite_index_paths`` for those. Never deletes parquet
    files. Dry-run by default; pass ``--apply`` to write.
    """
    ds = load_dataset(manifest)
    try:
        with stdout_to_stderr():
            dropped = ds.reindex_features(
                feature,
                dry_run=not apply,
                reconcile_registry=reconcile_registry,
            )
    except Exception as exc:  # noqa: BLE001 - surface reconcile errors cleanly
        fail(f"reindex failed: {exc}")
    total = sum(dropped.values())
    if as_json:
        emit_json({"applied": apply, "total_dropped": total, "by_index": dropped})
        return
    verb = "dropped" if apply else "would drop"
    if not dropped:
        typer.echo("reindex: all feature indexes are clean (no missing files).")
        return
    for idx_path, n in dropped.items():
        typer.echo(f"{verb} {n}\t{idx_path}")
    typer.echo(f"total: {verb} {total} row(s).")
