"""``mosaic features list``: every registered feature, as its catalog index row."""

from __future__ import annotations

from typing import Annotated

import typer

from mosaic.cli._io import emit_json
from mosaic.cli._render import render_table


def list_command(
    query: Annotated[
        str,
        typer.Option(
            "--query",
            help=(
                "List only the features whose name, summary or category contains "
                "every word, best match first."
            ),
        ),
    ] = "",
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit the list as a JSON array.")
    ] = False,
) -> None:
    """List the registered features with their lanes and columns.

    Without --query the list is sorted by slug. With --query it holds only the
    features that match, best match first: the name, then words of the name,
    then the summary or category. Under --json each row also names what the
    installation lacks to run the feature and any use its license forbids.
    """
    from mosaic.core.pipeline.graph import catalog_index, search_catalog

    rows = [
        match.entry.model_dump(mode="json")
        for match in search_catalog(catalog_index().entries, query)
        if match.entry.type == "feature"
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
