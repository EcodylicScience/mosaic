"""The MCP server in process: its tools, confirmation, resources and prompts.

A client of the SDK connects to the server built over a local backend. Each
submission starts a recording stand-in for the detached driver, so nothing runs.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Iterator
from pathlib import Path

import pytest

pytest.importorskip("mcp")

import anyio
import mcp_types as types
from mcp.client import Client, ClientRequestContext
from mcp.client.session import ClientSession, ElicitationFnT
from mcp.shared.exceptions import MCPError

from mosaic.agent.mcp.local import LocalBackend
from mosaic.agent.mcp.prompts import PROMPTS
from mosaic.agent.mcp.server import build_server
from mosaic.agent.mcp.spec import MAX_SCHEMA_BYTES, SHARED_TOOL_LIMIT, TOOLS
from mosaic.core.dataset import Dataset
from mosaic.core.json_value import JsonValue
from mosaic.core.pipeline.graph import submit_request
from mosaic.runlog import JsonlRunLog, run_log_path
from tests.helpers import (
    SPEED_RECIPE,
    RecordingStarter,
    add_tracks_variant,
    make_dataset,
    run_fresh,
    speed_recipe,
)

VARIANT = "convert-trex.0.2-1111111111"

STORAGE = "speed-angvel__from__tracks"

PARAMS = "artifacts/feature/speed-angvel__from__tracks/0.1-0000000000/params"
"""The params resource of a feature run that never ran."""

INITIALIZE_SECONDS = 1.5
"""The longest a client waits for initialization while the catalog warms."""


@pytest.fixture
def dataset(tmp_path: Path) -> Dataset:
    """Two schema-valid track tables, and nothing computed."""
    ds = make_dataset(tmp_path / "tracked", name="tracked")
    add_tracks_variant(ds, VARIANT, "seq_a", "seq_b", std_format="mosaic_v1")
    return ds


@pytest.fixture
def starter() -> RecordingStarter:
    return RecordingStarter()


@pytest.fixture
def backend(dataset: Dataset, starter: RecordingStarter) -> LocalBackend:
    return LocalBackend([dataset.manifest_path], warm=False, start_driver=starter)


def _uuid(dataset: Dataset) -> str:
    uuid = dataset.manifest.uuid
    assert uuid is not None
    return uuid


def _answering(accept: bool) -> ElicitationFnT:
    """An elicitation callback that accepts or declines every confirmation."""

    async def answer(
        context: ClientRequestContext, params: types.ElicitRequestParams
    ) -> types.ElicitResult:
        if accept:
            return types.ElicitResult(action="accept", content={"confirm": True})
        return types.ElicitResult(action="decline")

    return answer


def _session[T](
    backend: LocalBackend,
    body: Callable[[Client], Awaitable[T]],
    *,
    mode: str = "auto",
    elicitation: ElicitationFnT | None = None,
) -> T:
    """Run *body* with a client connected in process to a server over *backend*."""

    async def run() -> T:
        async with Client(
            build_server(backend), mode=mode, elicitation_callback=elicitation
        ) as client:
            return await body(client)

    return anyio.run(run)


def _structured(result: types.CallToolResult) -> dict[str, JsonValue]:
    """The structured content of a result that did not fail."""
    assert not result.is_error, result.content
    content: dict[str, JsonValue] = result.structured_content
    assert isinstance(content, dict)
    return content


def _text(result: types.CallToolResult) -> str:
    block = result.content[0]
    assert isinstance(block, types.TextContent)
    return block.text


def _error(result: types.CallToolResult) -> dict[str, JsonValue]:
    """The error envelope of a failed result, from its second text block."""
    assert result.is_error
    block = result.content[1]
    assert isinstance(block, types.TextContent)
    envelope: dict[str, JsonValue] = json.loads(block.text)
    return envelope


def _refs(node: JsonValue) -> Iterator[str]:
    """Every ``$ref`` that a JSON schema holds, at any depth."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "$ref" and isinstance(value, str):
                yield value
            else:
                yield from _refs(value)
    elif isinstance(node, list):
        for item in node:
            yield from _refs(item)


def _resolves(schema: dict[str, JsonValue], ref: str) -> bool:
    """Whether the local JSON pointer *ref* names a node of *schema*."""
    if not ref.startswith("#/"):
        return False
    node: JsonValue = schema
    for part in ref[2:].split("/"):
        if not isinstance(node, dict) or part not in node:
            return False
        node = node[part]
    return True


def test_the_shared_tools_are_listed_in_order_with_compact_resolving_schemas(
    backend: LocalBackend,
) -> None:
    async def body(client: Client) -> list[types.Tool]:
        return (await client.list_tools()).tools

    tools = _session(backend, body)

    assert [tool.name for tool in tools] == [spec.name for spec in TOOLS]
    assert len(tools) == len(TOOLS) <= SHARED_TOOL_LIMIT
    for tool in tools:
        assert tool.output_schema is not None
        assert tool.annotations is not None
        assert tool.annotations.open_world_hint is False
        for schema in (tool.input_schema, tool.output_schema):
            size = len(json.dumps(schema, separators=(",", ":")))
            assert size < MAX_SCHEMA_BYTES, (tool.name, size)
            assert schema["type"] == "object"
            unresolved = [ref for ref in _refs(schema) if not _resolves(schema, ref)]
            assert unresolved == [], (tool.name, unresolved)


def test_initialization_answers_while_the_catalog_warms(dataset: Dataset) -> None:
    program = f"""
import json, threading, time
import anyio
from mcp.client import Client
from mosaic.agent.mcp.local import LocalBackend
from mosaic.agent.mcp.server import build_server

backend = LocalBackend([{str(dataset.manifest_path)!r}])
server = build_server(backend)

async def connect():
    started = time.monotonic()
    async with Client(server, mode="legacy") as client:
        elapsed = time.monotonic() - started
        warming = any(
            thread.name == "mosaic-mcp-warm" and thread.is_alive()
            for thread in threading.enumerate()
        )
        return {{"elapsed": elapsed, "warming": warming, "version": client.protocol_version}}

print(json.dumps(anyio.run(connect)))
"""
    measured = json.loads(run_fresh(program))

    assert measured["warming"], (
        "the catalog finished warming before the client connected"
    )
    assert measured["elapsed"] < INITIALIZE_SECONDS, measured


def test_a_tool_returns_structured_content_and_a_summary(
    backend: LocalBackend, dataset: Dataset
) -> None:
    async def body(client: Client) -> types.CallToolResult:
        return await client.call_tool("list_sequences", {"dataset": _uuid(dataset)})

    result = _session(backend, body)

    content = _structured(result)
    assert content["total"] == 2
    assert _text(result).startswith("2 entry(s) match, 2 on this page.")


def test_a_failure_is_an_error_result_carrying_its_envelope(
    backend: LocalBackend,
) -> None:
    async def body(client: Client) -> types.CallToolResult:
        return await client.call_tool("describe_dataset", {"dataset": "elsewhere"})

    result = _session(backend, body)

    envelope = _error(result)
    assert (envelope["kind"], envelope["reason"]) == ("not_found", "unknown_dataset")
    assert result.structured_content is None
    assert _text(result).startswith("Failed (not_found, unknown_dataset)")


def _plan_then_submit(
    dataset: Dataset, **extra: JsonValue
) -> Callable[[Client], Awaitable[types.CallToolResult]]:
    """A session body that plans the speed recipe and submits it over every entry."""

    async def body(client: Client) -> types.CallToolResult:
        uuid = _uuid(dataset)
        recipe: dict[str, JsonValue] = json.loads(json.dumps(SPEED_RECIPE))
        plan = await client.call_tool(
            "plan_recipe", {"dataset": uuid, "recipe": recipe}
        )
        return await client.call_tool(
            "submit_pipeline",
            {
                "dataset": uuid,
                "recipe": recipe,
                "plan_fingerprint": _structured(plan)["fingerprint"],
                **extra,
            },
        )

    return body


def test_a_client_without_elicitation_is_refused_until_it_confirms(
    backend: LocalBackend, dataset: Dataset, starter: RecordingStarter
) -> None:
    refused = _session(backend, _plan_then_submit(dataset))
    confirmed = _session(backend, _plan_then_submit(dataset, confirm=True))

    envelope = _error(refused)
    assert envelope["reason"] == "confirmation_required"
    assert _structured(confirmed)["request_id"] in starter.started
    assert len(starter.started) == 1


@pytest.mark.parametrize("mode", ["auto", "legacy"])
def test_an_eliciting_client_is_asked_and_its_answer_decides(
    backend: LocalBackend, dataset: Dataset, starter: RecordingStarter, mode: str
) -> None:
    declined = _session(
        backend,
        _plan_then_submit(dataset),
        mode=mode,
        elicitation=_answering(accept=False),
    )
    assert _error(declined)["reason"] == "confirmation_declined"
    assert starter.started == []

    accepted = _session(
        backend,
        _plan_then_submit(dataset),
        mode=mode,
        elicitation=_answering(accept=True),
    )

    assert starter.started == [_structured(accepted)["request_id"]]


@pytest.mark.parametrize("mode", ["auto", "legacy"])
def test_an_eliciting_client_is_asked_even_when_the_model_confirms(
    backend: LocalBackend, dataset: Dataset, starter: RecordingStarter, mode: str
) -> None:
    asked: list[types.ElicitRequestParams] = []

    async def decline(
        context: ClientRequestContext, params: types.ElicitRequestParams
    ) -> types.ElicitResult:
        asked.append(params)
        return types.ElicitResult(action="decline")

    declined = _session(
        backend,
        _plan_then_submit(dataset, confirm=True),
        mode=mode,
        elicitation=decline,
    )

    assert _error(declined)["reason"] == "confirmation_declined"
    assert len(asked) == 1
    assert starter.started == []


@pytest.mark.parametrize("mode", ["auto", "legacy"])
@pytest.mark.parametrize(
    ("declared", "asked_reason"),
    [
        (types.ElicitationCapability(), "confirmation_declined"),
        (
            types.ElicitationCapability(url=types.UrlElicitationCapability()),
            "confirmation_required",
        ),
    ],
    ids=["neither-mode-means-form", "url-only"],
)
def test_only_a_client_declaring_form_elicitation_is_asked(
    backend: LocalBackend,
    dataset: Dataset,
    starter: RecordingStarter,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    declared: types.ElicitationCapability,
    asked_reason: str,
) -> None:
    def capabilities(_self: ClientSession, _version: str) -> types.ClientCapabilities:
        return types.ClientCapabilities(elicitation=declared)

    monkeypatch.setattr(ClientSession, "_build_capabilities", capabilities)
    asked: list[types.ElicitRequestParams] = []

    async def decline(
        context: ClientRequestContext, params: types.ElicitRequestParams
    ) -> types.ElicitResult:
        asked.append(params)
        return types.ElicitResult(action="decline")

    result = _session(
        backend, _plan_then_submit(dataset), mode=mode, elicitation=decline
    )

    assert _error(result)["reason"] == asked_reason
    assert len(asked) == (asked_reason == "confirmation_declined")
    assert starter.started == []


def test_a_confirmation_answers_only_the_call_it_was_asked_for(
    backend: LocalBackend, dataset: Dataset, starter: RecordingStarter
) -> None:
    accepted = types.ElicitResult(action="accept", content={"confirm": True})

    async def body(
        client: Client,
    ) -> tuple[types.CallToolResult | types.InputRequiredResult, ...]:
        uuid = _uuid(dataset)
        recipe: dict[str, JsonValue] = json.loads(json.dumps(SPEED_RECIPE))
        plan = await client.call_tool(
            "plan_recipe", {"dataset": uuid, "recipe": recipe}
        )
        shown: dict[str, object] = {
            "dataset": uuid,
            "recipe": recipe,
            "plan_fingerprint": _structured(plan)["fingerprint"],
        }
        asked = await client.session.call_tool(
            "submit_pipeline", shown, allow_input_required=True
        )
        assert isinstance(asked, types.InputRequiredResult)
        changed = await client.session.call_tool(
            "submit_pipeline",
            {**shown, "owner": "someone else"},
            input_responses={"confirm": accepted},
            request_state=asked.request_state,
            allow_input_required=True,
        )
        forged = await client.session.call_tool(
            "submit_pipeline",
            shown,
            input_responses={"confirm": accepted},
            request_state="0" * 64,
            allow_input_required=True,
        )
        confirmed = await client.session.call_tool(
            "submit_pipeline",
            shown,
            input_responses={"confirm": accepted},
            request_state=asked.request_state,
            allow_input_required=True,
        )
        return changed, forged, confirmed

    changed, forged, confirmed = _session(
        backend, body, elicitation=_answering(accept=True)
    )

    assert isinstance(changed, types.InputRequiredResult)
    assert isinstance(forged, types.InputRequiredResult)
    assert isinstance(confirmed, types.CallToolResult)
    assert starter.started == [_structured(confirmed)["request_id"]]


def test_the_resources_and_templates_are_listed_and_read(
    backend: LocalBackend, dataset: Dataset
) -> None:
    uuid = _uuid(dataset)
    request = submit_request(dataset, speed_recipe()).request
    execution_id = request.execution_of("speed")
    with JsonlRunLog(run_log_path(dataset.base_dir, execution_id), execution_id) as log:
        log.started(kind="feature", target=STORAGE)
    base = f"mosaic://dataset/{uuid}"

    async def body(
        client: Client,
    ) -> tuple[list[str], list[str], dict[str, str]]:
        listed = [str(item.uri) for item in (await client.list_resources()).resources]
        templates = [
            item.uri_template
            for item in (await client.list_resource_templates()).resource_templates
        ]
        read: dict[str, str] = {}
        for uri in (
            "mosaic://guide",
            "mosaic://catalog",
            "mosaic://catalog/speed-angvel",
            f"{base}/manifest",
            f"{base}/inventory?status=absent,partial",
            f"{base}/recipes/{request.recipe_digest}",
            f"{base}/requests/{request.request_id}",
            f"{base}/runs/{execution_id}",
            f"{base}/runs/{execution_id}/log?tail=1",
            f"{base}/{PARAMS}",
        ):
            (contents,) = (await client.read_resource(uri)).contents
            assert isinstance(contents, types.TextResourceContents)
            read[uri] = contents.text
        return listed, templates, read

    listed, templates, read = _session(backend, body)

    assert listed == ["mosaic://guide", "mosaic://catalog"]
    assert len(templates) == 8
    assert read["mosaic://guide"].startswith("---")
    assert json.loads(read[f"{base}/manifest"])["uuid"] == uuid
    request_read = json.loads(read[f"{base}/requests/{request.request_id}"])
    assert request_read["status"]["request_id"] == request.request_id
    assert len(read[f"{base}/runs/{execution_id}/log?tail=1"].splitlines()) == 1
    params = json.loads(read[f"{base}/{PARAMS}"])
    assert params["state"] == "absent"


@pytest.mark.parametrize(
    ("uri", "code"),
    [
        ("mosaic://nothing", -32002),
        ("mosaic://dataset/elsewhere/manifest", -32002),
        ("mosaic://dataset/{uuid}/runs/..", -32602),
        ("mosaic://dataset/{uuid}/runs/missing/log?tail=5000", -32602),
        ("mosaic://dataset/{uuid}/inventory?kind=nonsense", -32602),
    ],
)
def test_a_resource_that_cannot_be_read_is_an_error(
    backend: LocalBackend, dataset: Dataset, uri: str, code: int
) -> None:
    async def body(client: Client) -> MCPError:
        with pytest.raises(MCPError) as raised:
            _ = await client.read_resource(uri.format(uuid=_uuid(dataset)))
        return raised.value

    assert _session(backend, body).code == code


def test_the_prompts_are_listed_and_filled_in(
    backend: LocalBackend, dataset: Dataset
) -> None:
    uuid = _uuid(dataset)

    async def text_of(client: Client, name: str, arguments: dict[str, str]) -> str:
        content = (await client.get_prompt(name, arguments)).messages[0].content
        assert isinstance(content, types.TextContent)
        return content.text

    async def body(client: Client) -> tuple[dict[str, list[str]], str, str]:
        listed = {
            prompt.name: [argument.name for argument in prompt.arguments or []]
            for prompt in (await client.list_prompts()).prompts
        }
        built = await text_of(
            client, "build_pipeline", {"dataset": uuid, "goal": "speed per animal"}
        )
        explained = await text_of(
            client, "explain_request", {"dataset": uuid, "rid": "r-0001"}
        )
        return listed, built, explained

    listed, built, explained = _session(backend, body)

    assert list(listed) == [spec.name for spec in PROMPTS]
    assert listed == {
        "build_pipeline": ["dataset", "goal"],
        "diagnose_dataset": ["dataset"],
        "explain_request": ["dataset", "rid"],
        "compare_runs": ["dataset", "feature", "a", "b"],
        "summarize_dataset": ["dataset"],
    }
    assert uuid in built and "speed per animal" in built
    assert "Do not call submit_pipeline until" in built
    assert "r-0001" in explained and uuid in explained
