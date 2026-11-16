"""``mosaic features list``: every registered feature, as its catalog index row."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

import typer

from mosaic.cli._context import definitions_of
from mosaic.cli._io import emit_json
from mosaic.cli._render import DIGEST_PREFIX_LENGTH, render_table
from mosaic.cli.tags import DefinitionsManifestOption
from mosaic.core.spaces import TRACK_SPACES, is_track_space

if TYPE_CHECKING:
    from mosaic.core.pipeline.graph.wire import DefinitionWire


def list_command(
    query: Annotated[
        str,
        typer.Option(
            "--query",
            help=(
                "List only the features whose name, summary, keywords or category "
                "contain every word, best match first."
            ),
        ),
    ] = "",
    track_space: Annotated[
        str | None,
        typer.Option(
            "--track-space",
            help=(
                "List only the features that compute in this track space (image_px, "
                "plane_cm or world_m)."
            ),
        ),
    ] = None,
    manifest: DefinitionsManifestOption = None,
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit the list as a JSON array.")
    ] = False,
) -> None:
    """List the registered features with their lanes and columns.

    Without --query the list is sorted by slug. With --query it holds only the
    features that match, best match first: the name, then words of the name,
    then the summary, keywords or category. Under --json each row also lists its
    keywords and names what the installation lacks to run the feature and any
    use its license forbids, and its track_spaces.

    With --track-space the list holds only the features that compute in that
    space. The spaces column lists the spaces each feature computes in, which
    are image_px and plane_cm for a feature that declares none.

    With --manifest the dataset's feature definitions are listed beside the
    built-in features as def:<name>, each with its version, the first 12
    characters of its digest and the date it was defined. Listing imports no
    definition. Under --json a definition's row carries its name, full digest
    and date in `definition`, which is null for a built-in feature.
    """
    from mosaic.core.pipeline.graph import (
        catalog_index,
        computes_in_track_space,
        search_catalog,
    )

    if track_space is not None and not is_track_space(track_space):
        msg = f"{track_space!r} is not one of {', '.join(TRACK_SPACES)}"
        raise typer.BadParameter(msg, param_hint="--track-space")
    with definitions_of(manifest):
        entries = [
            match.entry
            for match in search_catalog(catalog_index().entries, query)
            if match.entry.type == "feature"
            and (
                track_space is None or computes_in_track_space(match.entry, track_space)
            )
        ]
    if as_json:
        emit_json([entry.model_dump(mode="json") for entry in entries])
        return
    columns = [
        "name",
        "version",
        "category",
        "lane",
        "spaces",
        "available",
        "restricted",
    ]
    if any(entry.definition is not None for entry in entries):
        columns.extend(["digest", "defined"])
    render_table(
        [
            {
                **entry.model_dump(mode="json"),
                "spaces": ",".join(entry.track_spaces),
                "available": "no" if entry.unavailable else "yes",
                "restricted": "yes" if entry.license_note else "no",
                **_definition_cells(entry.definition),
            }
            for entry in entries
        ],
        columns,
    )


def _definition_cells(definition: DefinitionWire | None) -> dict[str, str]:
    """The ``digest`` and ``defined`` cells of a row, blank for a built-in feature."""
    if definition is None:
        return {"digest": "", "defined": ""}
    return {
        "digest": (definition.digest or "")[:DIGEST_PREFIX_LENGTH],
        "defined": "" if definition.defined_on is None else str(definition.defined_on),
    }
