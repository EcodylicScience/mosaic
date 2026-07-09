"""Plain-text renderers for human (non-``--json``) output, printed to stdout."""

from __future__ import annotations

import json

import typer


def render_kv(payload: dict[str, object]) -> None:
    """Print ``key: value`` lines to stdout."""
    for key, value in payload.items():
        typer.echo(f"{key}: {value}")


def render_table(rows: list[dict[str, object]], columns: list[str]) -> None:
    """Print a tab-separated table (header + one row per dict) to stdout."""
    typer.echo("\t".join(columns))
    for row in rows:
        typer.echo("\t".join(str(row.get(col, "")) for col in columns))


def render_json_block(payload: object) -> None:
    """Pretty-print a JSON block to stdout (used for schema/describe in human mode)."""
    typer.echo(json.dumps(payload, indent=2, default=str))
