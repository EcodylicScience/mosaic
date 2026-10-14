"""``mosaic features list``: every registered feature, as its catalog index row."""

from __future__ import annotations

from typing import Annotated

import typer

from mosaic.cli._io import emit_json
from mosaic.cli._render import render_table


def list_command(
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit the list as a JSON array.")
    ] = False,
) -> None:
    """List every registered feature, sorted by slug, with its lane and columns.

    Under --json each row also names what the installation lacks to run the
    feature and any use its license forbids.
    """
    from mosaic.core.pipeline.graph import catalog_index

    rows = [
        entry.model_dump(mode="json")
        for entry in catalog_index().entries
        if entry.type == "feature"
    ]
    if as_json:
        emit_json(rows)
        return
    render_table(
        [
            {
                **row,
                "available": "no" if row["unavailable"] else "yes",
                "restricted": "yes" if row["license_note"] else "no",
            }
            for row in rows
        ],
        ["name", "version", "category", "lane", "available", "restricted"],
    )
