"""``mosaic features define <name> <file>``: store a feature's source in a dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from mosaic.cli._context import load_dataset
from mosaic.cli._io import emit_json, fail_classified
from mosaic.cli._render import DIGEST_PREFIX_LENGTH
from mosaic.cli.tags import ManifestOption


def define_command(
    name: Annotated[
        str, typer.Argument(help="The definition's name, lowercase kebab-case.")
    ],
    file: Annotated[
        Path,
        typer.Argument(
            help="The Python file that declares the feature class.",
            exists=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    manifest: ManifestOption,
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit the record as JSON.")
    ] = False,
) -> None:
    """Define a feature in the dataset from a source file.

    The file is checked in a child interpreter and stored under the dataset's
    feature_definitions directory as a version of the name, which the slug
    def:<name> then names. Defining the same source again reuses its version.
    Defining other source adds a version and makes it current. A refusal prints
    every finding and exits 1.
    """
    from mosaic.core.pipeline.definitions.define import define_feature
    from mosaic.core.pipeline.definitions.store import DefinitionInvalid
    from mosaic.core.pipeline.error_kind import ClassifiedError

    dataset = load_dataset(manifest)
    try:
        record = define_feature(dataset, name, file.read_bytes())
    except DefinitionInvalid as error:
        lines = [f"the definition {error.name!r} is invalid:"]
        lines.extend(
            f"  {finding.check}"
            f"{'' if finding.field is None else f' ({finding.field})'}: "
            f"{finding.message}"
            for finding in error.findings
        )
        fail_classified(error, as_json=as_json, message="\n".join(lines))
    except ClassifiedError as error:
        fail_classified(error, as_json=as_json)
    if as_json:
        emit_json(record.model_dump(mode="json"))
        return
    version = f"version {record.digest[:DIGEST_PREFIX_LENGTH]} of {record.defined_on}"
    if record.reused:
        typer.echo(f"Reused {record.name!r}: {version} was already defined.")
    else:
        typer.echo(f"Defined {record.name!r}: {version}.")
