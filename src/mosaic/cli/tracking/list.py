"""``mosaic tracking list``: enumerate registered tracking ops (kind/category/version)."""

from __future__ import annotations

from typing import Annotated

import typer

from mosaic.cli._io import emit_json
from mosaic.cli._render import render_table


def list_command(
    category: Annotated[
        str | None,
        typer.Option(
            "--category", help="Filter by category (extract|train|infer|convert)."
        ),
    ] = None,
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit the list as a JSON array.")
    ] = False,
) -> None:
    """List every registered tracking op, sorted by kind."""
    from mosaic.tracking import list_tracking_ops

    ops = list_tracking_ops(category)
    if as_json:
        emit_json(ops)
    else:
        render_table(ops, ["kind", "category", "version"])
