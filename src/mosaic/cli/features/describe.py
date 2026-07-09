"""``mosaic features describe <slug>``: the feature's params JSON-Schema + metadata."""

from __future__ import annotations

from typing import Annotated

import typer

from mosaic.cli._features import feature_class_for_slug
from mosaic.cli._io import emit_json
from mosaic.cli._render import render_json_block


def describe_command(
    slug: Annotated[str, typer.Argument(help="Feature slug, e.g. 'speed-angvel'.")],
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit the description as JSON.")
    ] = False,
) -> None:
    """Describe one feature: name, version, category, and its params JSON-Schema."""
    from pydantic import BaseModel

    cls = feature_class_for_slug(slug)
    name_raw = getattr(cls, "name", None)
    version_raw = getattr(cls, "version", None)
    category_raw = getattr(cls, "category", None)
    params_cls = getattr(cls, "Params", None)
    schema: dict[str, object] | None = None
    if isinstance(params_cls, type) and issubclass(params_cls, BaseModel):
        schema = params_cls.model_json_schema()

    payload: dict[str, object] = {
        "name": name_raw if isinstance(name_raw, str) else cls.__name__,
        "version": version_raw if isinstance(version_raw, str) else "0.0",
        "category": category_raw if isinstance(category_raw, str) else None,
        "params_schema": schema,
    }
    if as_json:
        emit_json(payload)
    else:
        render_json_block(payload)
