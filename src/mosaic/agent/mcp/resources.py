"""The resources that the local server offers, and how each one is read.

A resource is a document a client reads by URI: the agent guide, the step-type
catalog, and per dataset its manifest, inventory, stored recipes, requests,
attempts, run-logs and recorded params. :data:`RESOURCES` declares each by its
URI template. The server matches a URI against the templates and calls the
spec's reader with the variables that the match bound. A reader returns the
text and its media type, or raises
:class:`~mosaic.agent.mcp.local.ToolFailure` with the envelope that names the
fault.

The module imports no MCP SDK, and imports the library inside its readers.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal

from pydantic import BaseModel

from mosaic.core.pipeline.artifact_kinds import (
    ARTIFACT_STATUSES,
    ArtifactKind,
    ArtifactStatus,
    is_artifact_kind,
    is_artifact_status,
)
from mosaic.core.pipeline.error_envelope import ErrorEnvelope

from .local import LocalBackend, ToolFailure
from .spec import (
    MAX_PAGE_SIZE,
    GetInventoryInput,
    GetRequestStatusInput,
)

if TYPE_CHECKING:
    from mosaic.core.json_value import JsonValue

__all__ = [
    "DEFAULT_LOG_TAIL",
    "MAX_LOG_TAIL",
    "RESOURCES",
    "MimeType",
    "ResourceSpec",
    "ResourceText",
    "UriVariables",
]

type UriVariables = Mapping[str, str | Sequence[str]]
"""The variables that matching a URI against a template binds."""

type MimeType = Literal["application/json", "application/x-ndjson", "text/markdown"]
"""The media type of a resource's text."""

DEFAULT_LOG_TAIL: Final = 50
"""The run-log lines that a log resource returns when its URI names no ``tail``."""

MAX_LOG_TAIL: Final = 1000
"""The most run-log lines that a log resource returns."""

_SEGMENT: Final = re.compile(r"^(?!\.{1,2}$)[^/\\\x00]+$")
"""One path component: no separator, no NUL, and neither ``.`` nor ``..``."""


@dataclass(frozen=True, slots=True)
class ResourceText:
    """A resource as read: its text and the media type of the text.

    Attributes:
        text: The document.
        mime_type: Its media type.
    """

    text: str
    mime_type: MimeType


@dataclass(frozen=True, slots=True)
class ResourceSpec:
    """One resource or resource template that the server lists.

    Attributes:
        uri: The URI, or an RFC 6570 template of URIs.
        name: A short name for it.
        description: What it holds, written for a model.
        mime_type: The media type of its text.
        read: Reads it from the backend and the variables that the URI bound.
    """

    uri: str
    name: str
    description: str
    mime_type: MimeType
    read: Callable[[LocalBackend, UriVariables], ResourceText]

    @property
    def is_template(self) -> bool:
        """Whether :attr:`uri` names a family of resources by its variables."""
        return "{" in self.uri


def _json(document: JsonValue | BaseModel) -> ResourceText:
    """Return *document* as indented JSON text."""
    if isinstance(document, BaseModel):
        return ResourceText(document.model_dump_json(indent=2), "application/json")
    return ResourceText(json.dumps(document, indent=2), "application/json")


def _answered[T: BaseModel](result: T | ErrorEnvelope) -> T:
    """Return a tool's output, raising its envelope when it failed."""
    if isinstance(result, ErrorEnvelope):
        raise ToolFailure(result)
    return result


def _variable(variables: UriVariables, name: str) -> str:
    """Return the variable *name* as one path component.

    Raises:
        ToolFailure: It is absent, a list, or not one path component.
    """
    value = variables.get(name)
    if isinstance(value, str) and _SEGMENT.match(value):
        return value
    raise ToolFailure(
        ErrorEnvelope(
            kind="invalid",
            reason="invalid_uri",
            message=f"the URI's {name} is not one path component",
            detail={"variable": name},
        )
    )


def _listed(variables: UriVariables, name: str) -> list[str]:
    """Return the comma-separated values of the query variable *name*."""
    value = variables.get(name)
    if value is None:
        return []
    joined = value if isinstance(value, str) else ",".join(value)
    return [part for part in joined.split(",") if part]


def _not_found(what: str, detail: dict[str, JsonValue]) -> ToolFailure:
    return ToolFailure(
        ErrorEnvelope(
            kind="not_found",
            reason="resource_not_found",
            message=f"{what} does not exist",
            detail=detail,
        )
    )


def _read_guide(_backend: LocalBackend, _variables: UriVariables) -> ResourceText:
    from mosaic.agent import guide_text

    try:
        return ResourceText(guide_text(), "text/markdown")
    except FileNotFoundError as error:
        raise _not_found("the agent guide of this installation", {}) from error


def _read_catalog(backend: LocalBackend, _variables: UriVariables) -> ResourceText:
    return _json(backend.catalog())


def _read_catalog_entry(backend: LocalBackend, variables: UriVariables) -> ResourceText:
    return _json(backend.step_type_entry(_variable(variables, "name")))


def _read_manifest(backend: LocalBackend, variables: UriVariables) -> ResourceText:
    from mosaic.core.manifest import manifest_payload, read_manifest

    dataset = backend.dataset(_variable(variables, "id"))
    return _json(manifest_payload(read_manifest(dataset.manifest_path)))


def _read_inventory(backend: LocalBackend, variables: UriVariables) -> ResourceText:
    kinds = _listed(variables, "kind")
    statuses = _listed(variables, "status")
    known_kinds: list[ArtifactKind] = [kind for kind in kinds if is_artifact_kind(kind)]
    known_statuses: list[ArtifactStatus] = [
        status for status in statuses if is_artifact_status(status)
    ]
    unknown = sorted({*kinds} - {*known_kinds} | {*statuses} - {*known_statuses})
    if unknown:
        raise ToolFailure(
            ErrorEnvelope(
                kind="invalid",
                reason="invalid_uri",
                message=f"no artifact kind or status is named {', '.join(unknown)}",
                detail={"unknown": list(unknown)},
            )
        )
    status_not: list[ArtifactStatus] = (
        [status for status in ARTIFACT_STATUSES if status not in known_statuses]
        if known_statuses
        else []
    )
    request = GetInventoryInput(
        dataset=_variable(variables, "id"),
        kinds=known_kinds or None,
        status_not=status_not,
        limit=MAX_PAGE_SIZE,
    )
    return _json(backend.inventory_document(request))


def _read_recipe(backend: LocalBackend, variables: UriVariables) -> ResourceText:
    from mosaic.core.pipeline.graph import load_recipe, recipe_path

    dataset = backend.dataset(_variable(variables, "id"))
    digest = _variable(variables, "digest")
    try:
        recipe = load_recipe(recipe_path(dataset.base_dir, digest))
    except FileNotFoundError as error:
        raise _not_found(f"recipe {digest}", {"digest": digest}) from error
    except ValueError as error:
        raise ToolFailure(
            ErrorEnvelope(
                kind="invalid",
                reason="recipe_unreadable",
                message=f"recipe {digest} does not read: {error}",
                detail={"digest": digest},
            )
        ) from error
    return _json(recipe.model_dump(mode="json"))


def _read_request(backend: LocalBackend, variables: UriVariables) -> ResourceText:
    from mosaic.core.pipeline.cancel import read_request

    uuid = _variable(variables, "id")
    request_id = _variable(variables, "rid")
    dataset = backend.dataset(uuid)
    status = _answered(
        backend.get_request_status(
            GetRequestStatusInput(dataset=uuid, request_id=request_id)
        )
    )
    request = read_request(dataset.base_dir, request_id)
    return _json(
        {
            "request": request.model_dump(mode="json"),
            "status": status.model_dump(mode="json"),
        }
    )


def _read_run(backend: LocalBackend, variables: UriVariables) -> ResourceText:
    from mosaic.core.pipeline.graph import run_status_wire

    dataset = backend.dataset(_variable(variables, "id"))
    execution_id = _variable(variables, "eid")
    found = run_status_wire(dataset.base_dir, execution_id)
    if found is None:
        raise _not_found(f"attempt {execution_id}", {"execution_id": execution_id})
    return _json(found)


def _read_run_log(backend: LocalBackend, variables: UriVariables) -> ResourceText:
    from mosaic.runlog import run_log_path

    dataset = backend.dataset(_variable(variables, "id"))
    execution_id = _variable(variables, "eid")
    tail = _tail(variables)
    path = run_log_path(dataset.base_dir, execution_id)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as error:
        raise _not_found(
            f"the run-log of attempt {execution_id}", {"execution_id": execution_id}
        ) from error
    kept = lines[-tail:] if tail else []
    return ResourceText("".join(f"{line}\n" for line in kept), "application/x-ndjson")


def _tail(variables: UriVariables) -> int:
    """Return the ``tail`` query variable, :data:`DEFAULT_LOG_TAIL` when absent.

    Raises:
        ToolFailure: It is not a whole number from 0 to :data:`MAX_LOG_TAIL`.
    """
    given = _listed(variables, "tail")
    if not given:
        return DEFAULT_LOG_TAIL
    if len(given) == 1 and given[0].isdecimal() and int(given[0]) <= MAX_LOG_TAIL:
        return int(given[0])
    raise ToolFailure(
        ErrorEnvelope(
            kind="invalid",
            reason="invalid_uri",
            message=f"tail is a whole number from 0 to {MAX_LOG_TAIL}",
            detail={"variable": "tail"},
        )
    )


def _read_params(backend: LocalBackend, variables: UriVariables) -> ResourceText:
    from mosaic.core.pipeline.index import feature_run_root
    from mosaic.core.pipeline.inventory import read_run_params

    dataset = backend.dataset(_variable(variables, "id"))
    kind = _variable(variables, "kind")
    if kind != "feature":
        raise ToolFailure(
            ErrorEnvelope(
                kind="invalid",
                reason="invalid_uri",
                message=(
                    f"artifacts of kind {kind} record no params.json that this "
                    "resource reads; it reads feature runs. get_inventory reports "
                    "the params of every artifact by its ref"
                ),
                detail={"variable": "kind"},
            )
        )
    name = _variable(variables, "name")
    run_id = _variable(variables, "run_id")
    read = read_run_params(feature_run_root(dataset, name, run_id))
    params = (
        None
        if read.params is None
        else read.params.model_dump(mode="json", by_alias=True)
    )
    return _json(
        {
            "kind": kind,
            "name": name,
            "run_id": run_id,
            "state": read.state,
            "error": read.error,
            "params": params,
        }
    )


_DATASET_URI: Final = "mosaic://dataset/{id}"

RESOURCES: Final[tuple[ResourceSpec, ...]] = (
    ResourceSpec(
        uri="mosaic://guide",
        name="guide",
        description=(
            "The guide to mosaic for an agent: the workflow, the dataset rules and "
            "the mistakes to avoid. Read it before building a pipeline."
        ),
        mime_type="text/markdown",
        read=_read_guide,
    ),
    ResourceSpec(
        uri="mosaic://catalog",
        name="catalog",
        description=(
            "Every feature and op a recipe may run, without params schemas. "
            "mosaic://catalog/{name} has one in full."
        ),
        mime_type="application/json",
        read=_read_catalog,
    ),
    ResourceSpec(
        uri="mosaic://catalog/{name}",
        name="step-type",
        description="One feature or op in full, with its params schema.",
        mime_type="application/json",
        read=_read_catalog_entry,
    ),
    ResourceSpec(
        uri=f"{_DATASET_URI}/manifest",
        name="manifest",
        description="A dataset's manifest, dataset.yaml, as JSON.",
        mime_type="application/json",
        read=_read_manifest,
    ),
    ResourceSpec(
        uri=f"{_DATASET_URI}/inventory{{?kind,status}}",
        name="inventory",
        description=(
            "The first 100 artifacts a dataset holds, narrowed to the "
            "comma-separated kinds and statuses the query names."
        ),
        mime_type="application/json",
        read=_read_inventory,
    ),
    ResourceSpec(
        uri=f"{_DATASET_URI}/recipes/{{digest}}",
        name="recipe",
        description="A recipe stored in a dataset, by its digest.",
        mime_type="application/json",
        read=_read_recipe,
    ),
    ResourceSpec(
        uri=f"{_DATASET_URI}/requests/{{rid}}",
        name="request",
        description="A submitted request and its status.",
        mime_type="application/json",
        read=_read_request,
    ),
    ResourceSpec(
        uri=f"{_DATASET_URI}/runs/{{eid}}",
        name="run",
        description="One attempt's status, as its run-log reports it.",
        mime_type="application/json",
        read=_read_run,
    ),
    ResourceSpec(
        uri=f"{_DATASET_URI}/runs/{{eid}}/log{{?tail}}",
        name="run-log",
        description=(
            f"The last lines of one attempt's JSON-lines run-log, {DEFAULT_LOG_TAIL} "
            f"unless tail names up to {MAX_LOG_TAIL}."
        ),
        mime_type="application/x-ndjson",
        read=_read_run_log,
    ),
    ResourceSpec(
        uri=f"{_DATASET_URI}/artifacts/{{kind}}/{{name}}/{{run_id}}/params",
        name="params",
        description=(
            "The params.json that a feature run recorded, by kind feature, its "
            "storage name and its run id."
        ),
        mime_type="application/json",
        read=_read_params,
    ),
)
"""The resources and resource templates, in the order a server lists them."""
