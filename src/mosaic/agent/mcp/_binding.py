"""The MCP server bound to the SDK, behind :mod:`~mosaic.agent.mcp.server`.

:func:`bind_server` binds the shared tools of :mod:`~mosaic.agent.mcp.spec`,
the resources of :mod:`~mosaic.agent.mcp.resources` and the prompts of
:mod:`~mosaic.agent.mcp.prompts` to a :class:`~mosaic.agent.mcp.local.LocalBackend`
through the low-level server of the MCP SDK. :func:`serve_stdio` runs it on the
process's standard streams.

A tool call returns the output model as ``structuredContent``, which validates
against the tool's ``outputSchema``, and a paragraph from
:func:`~mosaic.agent.mcp.summaries.summarize` as its text. A failure returns
``isError`` with the paragraph and the error envelope as JSON text. Backend
calls block, so each runs on a worker thread.

A tool that asks for confirmation is confirmed by the user through elicitation
when the client declares form elicitation, and the ``confirm`` argument that the
model passes is then ignored. A client on the 2026-07-28 protocol is returned
``input_required`` with the elicitation and a digest of the call as its request
state, and retries the call with the answer. The answer confirms only a retry
whose tool and arguments match the digest. A client on an earlier protocol is
sent an ``elicitation/create`` request during the call. A client without form
elicitation is refused until the call carries ``confirm: true``, and the tool's
description tells the model to ask the user first.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import AsyncGenerator, Mapping
from contextlib import asynccontextmanager
from functools import partial
from typing import Final

import anyio
import anyio.to_thread
import mcp_types as types
from mcp.server import Server, ServerRequestContext
from mcp.shared.exceptions import MCPError
from mcp.shared.uri_template import UriTemplate
from mcp_types.version import MODERN_PROTOCOL_VERSIONS
from pydantic import BaseModel

from mosaic.core.json_value import JsonValue
from mosaic.core.narrowing import is_object_list, is_str_mapping
from mosaic.core.pipeline.error_envelope import ErrorEnvelope, error_envelope
from mosaic.core.pipeline.error_kind import ClassifiedError
from mosaic.version import installed_version

from .local import LocalBackend, ToolFailure
from .prompts import PROMPTS, PromptArgumentsMissing, prompt_spec, render_prompt
from .resources import RESOURCES, ResourceSpec, ResourceText, UriVariables
from .spec import (
    CONFIRMATION_REQUIRED,
    TOOLS,
    ToolSpec,
    call_tool,
    confirmation_declined,
    tool_spec,
)
from .summaries import summarize

__all__ = [
    "CONFIRM_FIELD",
    "CONFIRM_INPUT",
    "INSTRUCTIONS",
    "SERVER_NAME",
    "bind_server",
]

SERVER_NAME: Final = "mosaic"
"""The name the server reports to a client."""

INSTRUCTIONS: Final = (
    "mosaic computes behavioral features from animal tracking data, in datasets "
    "named by uuid (list_datasets). Read mosaic://guide first. To compute "
    "something: describe_step_types (a query, then names), draft a recipe, "
    "validate_recipe, plan_recipe, show the user the plan and wait for their "
    "approval, then submit_pipeline and poll get_request_status. Previews and "
    "diagnoses change nothing. Never set confirm without the user's agreement, "
    "and propose the mosaic CLI command for anything these tools do not offer."
)
"""What the server tells a client's model at initialization."""

CONFIRM_INPUT: Final = "confirm"
"""The key of the elicitation that an ``input_required`` result asks for."""

CONFIRM_FIELD: Final = "confirm"
"""The boolean field of the confirmation form."""

_CONFIRM_ARGUMENT: Final = "confirm"
"""The argument of a tool that asks for confirmation, set when the user confirms."""

_CONFIRM_SCHEMA: Final[dict[str, JsonValue]] = {
    "type": "object",
    "properties": {
        CONFIRM_FIELD: {
            "type": "boolean",
            "title": "Proceed",
            "description": "Whether to go ahead despite the conditions listed.",
        }
    },
    "required": [CONFIRM_FIELD],
}
"""The form that a confirmation elicitation shows: one boolean."""

RESOURCE_NOT_FOUND: Final = -32002
"""The JSON-RPC error code of a resource that does not exist."""


class _Tools:
    """The tool listing, built once on a worker thread.

    Building it resolves every output model, which imports the editor wires,
    and generates every input and output schema. The first ``tools/list``
    builds it, so initialization does neither.
    """

    def __init__(self) -> None:
        self._lock: threading.Lock = threading.Lock()
        self._listed: list[types.Tool] | None = None

    def listed(self) -> list[types.Tool]:
        with self._lock:
            if self._listed is None:
                self._listed = [_tool(spec) for spec in TOOLS]
            return self._listed


def _tool(spec: ToolSpec) -> types.Tool:
    """Return the listing of the shared tool *spec*."""
    hints = spec.annotations
    return types.Tool(
        name=spec.name,
        title=spec.title,
        description=spec.description,
        input_schema=spec.input_schema(),
        output_schema=spec.output_schema(),
        annotations=types.ToolAnnotations(
            title=spec.title,
            read_only_hint=hints.read_only,
            destructive_hint=hints.destructive,
            idempotent_hint=hints.idempotent,
            open_world_hint=hints.open_world,
        ),
    )


@asynccontextmanager
async def _no_state(_server: Server[None]) -> AsyncGenerator[None]:
    """The server's lifespan, which holds no state: the backend is bound at build."""
    yield


def bind_server(backend: LocalBackend) -> Server[None]:
    """Bind the shared tools, resources and prompts to *backend*.

    Args:
        backend: What answers the tools and reads the resources.

    Returns:
        The low-level server, which ``Server.run`` serves over a pair of
        streams and the SDK's ``Client`` connects to in process.
    """
    tools = _Tools()

    async def list_tools(
        _ctx: ServerRequestContext[None], _params: types.PaginatedRequestParams | None
    ) -> types.ListToolsResult:
        return types.ListToolsResult(tools=await anyio.to_thread.run_sync(tools.listed))

    async def call(
        ctx: ServerRequestContext[None], params: types.CallToolRequestParams
    ) -> types.CallToolResult | types.InputRequiredResult:
        return await _call_tool(backend, ctx, params)

    async def list_resources(
        _ctx: ServerRequestContext[None], _params: types.PaginatedRequestParams | None
    ) -> types.ListResourcesResult:
        return types.ListResourcesResult(
            resources=[
                types.Resource(
                    uri=spec.uri,
                    name=spec.name,
                    description=spec.description,
                    mime_type=spec.mime_type,
                )
                for spec in RESOURCES
                if not spec.is_template
            ]
        )

    async def list_resource_templates(
        _ctx: ServerRequestContext[None], _params: types.PaginatedRequestParams | None
    ) -> types.ListResourceTemplatesResult:
        return types.ListResourceTemplatesResult(
            resource_templates=[
                types.ResourceTemplate(
                    uri_template=spec.uri,
                    name=spec.name,
                    description=spec.description,
                    mime_type=spec.mime_type,
                )
                for spec in RESOURCES
                if spec.is_template
            ]
        )

    async def read_resource(
        _ctx: ServerRequestContext[None], params: types.ReadResourceRequestParams
    ) -> types.ReadResourceResult:
        spec, variables = _resource_for(params.uri)
        read = await anyio.to_thread.run_sync(
            partial(_read_resource, spec, backend, variables)
        )
        return types.ReadResourceResult(
            contents=[
                types.TextResourceContents(
                    uri=params.uri, mime_type=read.mime_type, text=read.text
                )
            ]
        )

    async def list_prompts(
        _ctx: ServerRequestContext[None], _params: types.PaginatedRequestParams | None
    ) -> types.ListPromptsResult:
        return types.ListPromptsResult(
            prompts=[
                types.Prompt(
                    name=spec.name,
                    title=spec.title,
                    description=spec.description,
                    arguments=[
                        types.PromptArgument(
                            name=argument.name,
                            description=argument.description,
                            required=True,
                        )
                        for argument in spec.arguments
                    ],
                )
                for spec in PROMPTS
            ]
        )

    async def get_prompt(
        _ctx: ServerRequestContext[None], params: types.GetPromptRequestParams
    ) -> types.GetPromptResult:
        spec = prompt_spec(params.name)
        if spec is None:
            raise MCPError(types.INVALID_PARAMS, f"no prompt named {params.name!r}")
        try:
            text = render_prompt(spec, params.arguments or {})
        except PromptArgumentsMissing as error:
            raise MCPError(types.INVALID_PARAMS, str(error)) from error
        return types.GetPromptResult(
            description=spec.description,
            messages=[
                types.PromptMessage(role="user", content=types.TextContent(text=text))
            ],
        )

    return Server(
        SERVER_NAME,
        version=installed_version(),
        title="mosaic",
        instructions=INSTRUCTIONS,
        lifespan=_no_state,
        on_list_tools=list_tools,
        on_call_tool=call,
        on_list_resources=list_resources,
        on_list_resource_templates=list_resource_templates,
        on_read_resource=read_resource,
        on_list_prompts=list_prompts,
        on_get_prompt=get_prompt,
    )


async def _call_tool(
    backend: LocalBackend,
    ctx: ServerRequestContext[None],
    params: types.CallToolRequestParams,
) -> types.CallToolResult | types.InputRequiredResult:
    """Answer one tool call, asking the user to confirm where the tool refuses.

    Where the client can elicit, only the user's answer confirms the call: a
    ``confirm`` that the model passed is dropped before the tool runs.
    """
    arguments: dict[str, object] = dict(params.arguments or {})
    eliciting = _can_elicit(ctx)
    spec = tool_spec(params.name)
    if eliciting and spec is not None and spec.needs_confirmation:
        _ = arguments.pop(_CONFIRM_ARGUMENT, None)
    call_digest = _call_digest(params.name, arguments)
    answered = (
        _answered_confirmation(params.input_responses)
        if params.request_state == call_digest
        else None
    )
    if answered is not None:
        if not answered:
            return _declined(params.name)
        arguments[_CONFIRM_ARGUMENT] = True
    output = await _run_tool(backend, params.name, arguments)
    if not (_asks_to_confirm(output) and eliciting):
        return _tool_result(output)
    assert isinstance(output, ErrorEnvelope)
    question = _confirmation_question(params.name, output)
    if ctx.protocol_version in MODERN_PROTOCOL_VERSIONS:
        return types.InputRequiredResult(
            input_requests={
                CONFIRM_INPUT: types.ElicitRequest(
                    params=types.ElicitRequestFormParams(
                        message=question, requested_schema=_CONFIRM_SCHEMA
                    )
                )
            },
            request_state=call_digest,
        )
    elicited = await ctx.session.elicit_form(
        question, _CONFIRM_SCHEMA, related_request_id=ctx.request_id
    )
    if not _accepted(elicited):
        return _declined(params.name)
    arguments[_CONFIRM_ARGUMENT] = True
    return _tool_result(await _run_tool(backend, params.name, arguments))


async def _run_tool(
    backend: LocalBackend, name: str, arguments: Mapping[str, object]
) -> BaseModel:
    """Answer the tool *name* on a worker thread."""
    return await anyio.to_thread.run_sync(partial(call_tool, backend, name, arguments))


def _asks_to_confirm(output: BaseModel) -> bool:
    return isinstance(output, ErrorEnvelope) and output.reason == CONFIRMATION_REQUIRED


def _can_elicit(ctx: ServerRequestContext[None]) -> bool:
    """Whether the client declared form elicitation and can be asked during this call."""
    if not _elicits_forms(ctx.session.client_capabilities):
        return False
    return (
        ctx.protocol_version in MODERN_PROTOCOL_VERSIONS or ctx.session.can_send_request
    )


def _elicits_forms(declared: types.ClientCapabilities | None) -> bool:
    """Whether *declared* includes form elicitation.

    An elicitation capability naming neither mode means form; one naming only
    ``url`` does not.
    """
    elicitation = None if declared is None else declared.elicitation
    if elicitation is None:
        return False
    return elicitation.form is not None or elicitation.url is None


def _call_digest(name: str, arguments: Mapping[str, object]) -> str:
    """Return the digest of a call to *name* with *arguments*, ``confirm`` aside.

    The request state of a confirmation, so that the answer confirms the call
    that the user was shown and no other.
    """
    asked = {key: value for key, value in arguments.items() if key != _CONFIRM_ARGUMENT}
    canonical = json.dumps(
        {"tool": name, "arguments": asked}, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def _confirmation_question(name: str, refusal: ErrorEnvelope) -> str:
    """Return the question shown to the user: the tool and each condition met."""
    findings: object = (
        None if refusal.detail is None else refusal.detail.get("findings")
    )
    reasons: list[str] = []
    listed_findings: list[object] = findings if is_object_list(findings) else []
    for finding in listed_findings:
        reason = finding.get("message") if is_str_mapping(finding) else None
        if isinstance(reason, str):
            reasons.append(reason)
    listed = "; ".join(reasons) if reasons else refusal.message
    return f"mosaic asks you to confirm {name}: {listed}. Proceed?"


def _answered_confirmation(responses: Mapping[str, object] | None) -> bool | None:
    """Return the user's answer to a confirmation asked by ``input_required``.

    ``None`` when the call carries no answer to it.
    """
    answer = None if responses is None else responses.get(CONFIRM_INPUT)
    if not isinstance(answer, types.ElicitResult):
        return None
    return _accepted(answer)


def _accepted(answer: types.ElicitResult) -> bool:
    """Whether the user accepted the form and ticked the box."""
    return (
        answer.action == "accept"
        and answer.content is not None
        and answer.content.get(CONFIRM_FIELD) is True
    )


def _declined(name: str) -> types.CallToolResult:
    spec = tool_spec(name)
    if spec is None:
        msg = f"no tool named {name!r} asks to confirm"
        raise MCPError(types.INVALID_PARAMS, msg)
    return _tool_result(confirmation_declined(spec.name))


def _tool_result(output: BaseModel) -> types.CallToolResult:
    """Return *output* as a tool result: structured content and its summary."""
    summary = types.TextContent(text=summarize(output))
    if isinstance(output, ErrorEnvelope):
        envelope = types.TextContent(text=output.model_dump_json())
        return types.CallToolResult(content=[summary, envelope], is_error=True)
    return types.CallToolResult(
        content=[summary], structured_content=output.model_dump(mode="json")
    )


def _resource_for(uri: str) -> tuple[ResourceSpec, UriVariables]:
    """Return the resource that *uri* names and the variables it binds.

    Raises:
        MCPError: No resource or template matches *uri*.
    """
    for spec in RESOURCES:
        if not spec.is_template:
            if spec.uri == uri:
                return spec, {}
            continue
        matched = UriTemplate.parse(spec.uri).match(uri)
        if matched is not None:
            return spec, matched
    raise MCPError(RESOURCE_NOT_FOUND, f"no resource is named {uri}")


def _read_resource(
    spec: ResourceSpec, backend: LocalBackend, variables: UriVariables
) -> ResourceText:
    """Read *spec*, raising the envelope of a failure as an ``MCPError``."""
    try:
        return spec.read(backend, variables)
    except ToolFailure as failure:
        raise _resource_error(failure.envelope) from failure
    except ClassifiedError as error:
        raise _resource_error(error_envelope(error)) from error


def _resource_error(envelope: ErrorEnvelope) -> MCPError:
    code = RESOURCE_NOT_FOUND if envelope.kind == "not_found" else types.INVALID_PARAMS
    return MCPError(code, envelope.message, envelope.model_dump(mode="json"))
