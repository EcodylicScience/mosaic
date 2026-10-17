"""Plain-text renderers for human (non-``--json``) output, printed to stdout."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, cast

import typer

if TYPE_CHECKING:
    from mosaic.core.pipeline.graph import RecordedErrorWire, StepTypeDetail


def render_kv(payload: Mapping[str, object]) -> None:
    """Print ``key: value`` lines to stdout."""
    for key, value in payload.items():
        typer.echo(f"{key}: {value}")


def render_table(rows: Sequence[Mapping[str, object]], columns: list[str]) -> None:
    """Print a tab-separated table (header + one row per dict) to stdout."""
    typer.echo("\t".join(columns))
    for row in rows:
        typer.echo("\t".join(str(row.get(col, "")) for col in columns))


def render_json_block(payload: object) -> None:
    """Pretty-print a JSON block to stdout (used for schema/describe in human mode)."""
    typer.echo(json.dumps(payload, indent=2, default=str))


def recorded_error_text(error: RecordedErrorWire) -> str:
    """One line for a recorded failure: the refusal's reason, or the exception type."""
    if error.refusal is not None:
        return f"refused ({error.refusal.reason}): {error.message}"
    return f"{error.type}: {error.message}" if error.type else error.message


_STEP_TYPE_HEADER = (
    "name",
    "type",
    "version",
    "category",
    "summary",
    "lane",
    "emits",
    "consumes",
    "resource_class",
    "scope_dependent",
    "requires",
    "provides",
    "passthrough",
    "payload_families",
    "unavailable",
    "license_note",
)


def render_step_type_detail(detail: StepTypeDetail) -> None:
    """Print a step type's facts, a table of its params, and any schema.

    A default reads ``required`` for a field a recipe must set and
    ``constructed`` for one whose default the model builds.
    """
    facts: dict[str, object] = {}
    for key in _STEP_TYPE_HEADER:
        value: object = getattr(detail, key)
        if isinstance(value, tuple):
            items = cast("tuple[object, ...]", value)
            value = ", ".join(str(item) for item in items)
        facts[key] = value
    render_kv(facts)
    typer.echo("")
    if not detail.params:
        typer.echo("No params.")
    else:
        render_table(
            [
                {
                    "path": row.path,
                    "type": row.type,
                    "default": (
                        "required"
                        if row.required
                        else "constructed"
                        if row.constructed
                        else json.dumps(row.default)
                    ),
                    "constraints": row.constraints,
                    "description": row.description,
                }
                for row in detail.params
            ],
            ["path", "type", "default", "constraints", "description"],
        )
    if detail.params_schema is not None:
        typer.echo("")
        render_json_block(detail.params_schema)
