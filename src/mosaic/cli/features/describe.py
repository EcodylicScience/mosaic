"""``mosaic features describe <slug>``: one feature's params as a table."""

from __future__ import annotations

from typing import Annotated, Literal

import typer

from mosaic.cli._context import definitions_of
from mosaic.cli._features import feature_class_for_slug
from mosaic.cli._io import emit_json
from mosaic.cli._render import render_step_type_detail
from mosaic.cli.tags import DefinitionsManifestOption

DescribeDetail = Literal["fields", "schema"]
"""How much of a step type's params ``--detail`` prints."""

DetailOption = Annotated[
    DescribeDetail,
    typer.Option(
        "--detail",
        help=(
            "'fields' prints a row per params field, nested ones as dotted "
            "paths. 'schema' also prints the params JSON Schema."
        ),
    ),
]


def describe_command(
    slug: Annotated[str, typer.Argument(help="Feature slug, e.g. 'speed-angvel'.")],
    detail: DetailOption = "fields",
    manifest: DefinitionsManifestOption = None,
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit the description as JSON.")
    ] = False,
) -> None:
    """Describe one feature: its params and the columns it reads and writes.

    Also names its version, category and lane, what it consumes, whether its
    output depends on its scope, what the installation lacks to run it, and any
    use its license forbids. Each params field is a row, nested ones as dotted
    paths; ``--detail schema`` adds the params JSON Schema.

    With --manifest the slug may be def:<name>, the current version of a feature
    definition of the dataset, described from its stored sidecar without
    importing its source.
    """
    from mosaic.core.pipeline.graph import step_type_detail

    with definitions_of(manifest):
        _ = feature_class_for_slug(slug, as_json=as_json)
        described = step_type_detail(slug, detail=detail)
    if as_json:
        emit_json(described.model_dump(mode="json"))
    else:
        render_step_type_detail(described)
