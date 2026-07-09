"""``mosaic tracking describe <kind>``: the op's params JSON-Schema + metadata."""

from __future__ import annotations

from typing import Annotated

import typer

from mosaic.cli._io import emit_json, fail
from mosaic.cli._render import render_json_block


def describe_command(
    kind: Annotated[str, typer.Argument(help="Tracking-op kind, e.g. 'infer-pose'.")],
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit the description as JSON.")
    ] = False,
) -> None:
    """Describe one tracking op: kind, category, version, and its params JSON-Schema."""
    from mosaic.tracking import describe_tracking_op

    try:
        info = describe_tracking_op(kind)
    except KeyError as exc:
        fail(str(exc))
    if as_json:
        emit_json(info)
    else:
        render_json_block(info)
