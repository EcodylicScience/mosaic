"""``mosaic reindex``: reconcile index CSVs with the files on disk."""

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
    root: Annotated[
        str | None,
        typer.Option(
            "--root",
            help=(
                "Restrict to one root key (tracks, features, trex, sleap, ...). "
                "Default is every root with a reconcilable index."
            ),
        ),
    ] = None,
    apply: Annotated[
        bool,
        typer.Option(
            "--apply/--dry-run",
            help="Rewrite indexes (drop stale rows). Default is a dry-run report.",
        ),
    ] = False,
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit the result as JSON.")
    ] = False,
) -> None:
    """Drop index rows whose files no longer exist, in every root or one.

    Covers ``tracks``, ``features``, and each tracker and inference root under
    ``_tracking`` -- which until item 6.1 were reached by no reindex, prune or
    portability pass at all, so a working directory removed by hand left a row
    naming it forever.

    Relocated-but-present rows (a moved or synced dataset) are kept -- use
    ``make_portable`` / ``rewrite_index_paths`` for those. Never deletes output
    files. Dry-run by default; pass ``--apply`` to write.
    """
    if feature is not None and root not in (None, "features"):
        fail(
            "--feature restricts the features root; it cannot be combined with --root."
        )
    # Each tracker root's index is opened through the reconcilable-index
    # registry, which each tracker fills as a side effect of being imported --
    # ``core`` does not import ``tracking``, so nothing else fills it. Without
    # this the registry is empty here and every ``_tracking`` root is skipped in
    # silence, which is exactly the coverage this command's docstring promises.
    from mosaic.tracking import register_ops

    register_ops()
    ds = load_dataset(manifest)
    try:
        with stdout_to_stderr():
            dropped = (
                ds.reindex_features(feature, dry_run=not apply)
                if feature is not None
                else ds.reindex(root, dry_run=not apply)
            )
    except Exception as exc:  # noqa: BLE001 - surface reconcile errors cleanly
        fail(f"reindex failed: {exc}")
    total = sum(dropped.values())
    if as_json:
        emit_json({"applied": apply, "total_dropped": total, "by_index": dropped})
        return
    verb = "dropped" if apply else "would drop"
    if not dropped:
        typer.echo("reindex: every index is clean (no missing files).")
        return
    for idx_path, n in dropped.items():
        typer.echo(f"{verb} {n}\t{idx_path}")
    typer.echo(f"total: {verb} {total} row(s).")
