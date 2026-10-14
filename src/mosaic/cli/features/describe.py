"""``mosaic features describe <slug>``: one feature's catalog entry, params schema included."""

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
    """Describe one feature: its params JSON-Schema and the columns it reads and writes.

    Also names its version, category and lane, what the installation lacks to
    run it, and any use its license forbids.
    """
    from mosaic.core.pipeline.graph import describe_step_type, step_type_lane

    _ = feature_class_for_slug(slug)
    entry = describe_step_type(slug)
    payload = entry.model_dump(
        mode="json",
        include={
            "name",
            "version",
            "category",
            "summary",
            "params_schema",
            "requires",
            "provides",
            "unavailable",
            "license_note",
        },
    )
    payload["lane"] = step_type_lane(entry)
    if as_json:
        emit_json(payload)
    else:
        render_json_block(payload)
