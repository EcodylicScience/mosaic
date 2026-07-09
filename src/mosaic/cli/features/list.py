"""``mosaic features list``: enumerate registered features (name/version/category)."""

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
    """List every registered feature, sorted by slug."""
    from mosaic.behavior.feature_library import FEATURES

    rows: list[dict[str, object]] = []
    for cls in FEATURES.values():
        name_raw = getattr(cls, "name", None)
        version_raw = getattr(cls, "version", None)
        category_raw = getattr(cls, "category", None)
        rows.append(
            {
                "name": name_raw if isinstance(name_raw, str) else cls.__name__,
                "version": version_raw if isinstance(version_raw, str) else "0.0",
                "category": category_raw if isinstance(category_raw, str) else None,
            }
        )
    rows.sort(key=lambda r: str(r["name"]))
    if as_json:
        emit_json(rows)
    else:
        render_table(rows, ["name", "version", "category"])
