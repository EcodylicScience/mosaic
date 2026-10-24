"""``mosaic agent serve --http``: the server as a process, reached over streamable HTTP.

A real server listens on a loopback address and an ephemeral port behind a bearer
token. A request without the token, or with another, is refused before the MCP
layer. With it, the tools are the stdio server's, a costly submission is
confirmed by asking the user, a model's ``confirm`` is refused unless the
operator allows it, and a submitted request keeps running after its client goes.
"""

from __future__ import annotations

import socket
import stat
import subprocess
import sys
from collections.abc import Awaitable, Callable, Generator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Final

import pytest

pytest.importorskip("mcp")

import anyio
import httpx2
import mcp_types as types
from mcp.client import Client, ClientRequestContext
from mcp.client.session import ElicitationFnT
from mcp.client.streamable_http import streamable_http_client
from mcp.shared._httpx_utils import create_mcp_http_client
from typer.testing import CliRunner

from mosaic.agent.mcp.http_auth import (
    MIN_TOKEN_CHARS,
    BearerGuard,
    HttpAddress,
    TokenFileInvalid,
    is_loopback,
    parse_http_address,
    read_token_file,
)
from mosaic.agent.mcp.spec import TOOLS
from mosaic.cli import app
from mosaic.core.dataset import Dataset
from mosaic.core.json_value import JsonValue
from tests.helpers import SPEED_RECIPE, calms21_dataset, wait_for

TOKEN: Final = "t" * MIN_TOKEN_CHARS

START_SECONDS: Final = 60.0
"""The longest the server takes to listen."""

REQUEST_SECONDS: Final = 360.0
"""The longest the detached driver takes to run the speed step."""


def _token_file(directory: Path, token: str = TOKEN, mode: int = 0o600) -> Path:
    path = directory / "token"
    _ = path.write_text(token + "\n")
    path.chmod(mode)
    return path


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port: int = probe.getsockname()[1]
        return port


@pytest.fixture
def dataset(tmp_path: Path) -> Dataset:
    ds = calms21_dataset(tmp_path / "ds", {"train": {"a": {"s1": 20}}}, seed=1)
    _ = ds.convert_all_tracks()
    return ds


@contextmanager
def _server(dataset: Dataset, tmp_path: Path, *extra: str) -> Generator[str]:
    """Run ``mosaic agent serve --http`` and yield its endpoint URL."""
    port = _free_port()
    log = (tmp_path / "server.log").open("w")
    argv = [
        "agent",
        "serve",
        "--dataset",
        str(dataset.manifest_path),
        "--http",
        f"127.0.0.1:{port}",
        "--token-file",
        str(_token_file(tmp_path)),
        *extra,
    ]
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            f"from mosaic.cli import app; app({argv!r}, prog_name='mosaic')",
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    url = f"http://127.0.0.1:{port}/mcp"

    def listening() -> bool:
        assert process.poll() is None, (tmp_path / "server.log").read_text()
        try:
            _ = httpx2.get(url, timeout=1.0)
        except httpx2.TransportError:
            return False
        return True

    try:
        wait_for(listening, START_SECONDS, "the server did not listen")
        yield url
    finally:
        process.terminate()
        _ = process.wait(timeout=30)
        log.close()


async def _accept(
    context: ClientRequestContext, params: types.ElicitRequestParams
) -> types.ElicitResult:
    _ = (context, params)
    return types.ElicitResult(action="accept", content={"confirm": True})


type _Body[T] = Callable[[Client], Awaitable[T]]
"""What a test does with a connected client."""


def _call[T](url: str, body: _Body[T], *, mode: str = "auto", elicit: bool = True) -> T:
    """Connect to *url* with the token, run *body*, and return what it returns."""

    async def run() -> T:
        http = create_mcp_http_client(headers={"Authorization": f"Bearer {TOKEN}"})
        async with http:
            elicitation: ElicitationFnT | None = _accept if elicit else None
            async with Client(
                streamable_http_client(url, http_client=http),
                mode=mode,
                elicitation_callback=elicitation,
            ) as client:
                return await body(client)

    return anyio.run(run)


def _structured(result: types.CallToolResult) -> dict[str, JsonValue]:
    """The structured content of a result that did not fail."""
    assert not result.is_error, result.content
    content: dict[str, JsonValue] = result.structured_content
    assert isinstance(content, dict)
    return content


def _plan_arguments(dataset: Dataset) -> dict[str, object]:
    return {"dataset": dataset.manifest.uuid, "recipe": SPEED_RECIPE}


def _submit(dataset: Dataset, **extra: object) -> _Body[types.CallToolResult]:
    async def body(client: Client) -> types.CallToolResult:
        plan = await client.call_tool("plan_recipe", _plan_arguments(dataset))
        assert not plan.is_error, plan.content
        return await client.call_tool(
            "submit_pipeline",
            {
                **_plan_arguments(dataset),
                "plan_fingerprint": _structured(plan)["fingerprint"],
                **extra,
            },
        )

    return body


@pytest.mark.parametrize(
    ("header", "status"),
    [(None, 401), ("Bearer wrong", 401), (f"Bearer {TOKEN}x", 401), (TOKEN, 401)],
)
def test_a_request_without_the_token_is_refused(
    dataset: Dataset, tmp_path: Path, header: str | None, status: int
) -> None:
    with _server(dataset, tmp_path) as url:
        headers = {} if header is None else {"Authorization": header}
        response = httpx2.post(url, json={}, headers=headers)

    assert response.status_code == status
    assert response.headers["www-authenticate"].startswith("Bearer")


def test_the_tools_are_listed_and_a_plan_is_made_over_http(
    dataset: Dataset, tmp_path: Path
) -> None:
    async def body(client: Client) -> tuple[list[str], types.CallToolResult]:
        listed = await client.list_tools()
        plan = await client.call_tool("plan_recipe", _plan_arguments(dataset))
        return [tool.name for tool in listed.tools], plan

    with _server(dataset, tmp_path) as url:
        names, plan = _call(url, body)

    assert names == [spec.name for spec in TOOLS]
    assert not plan.is_error
    assert _structured(plan)["fingerprint"]


@pytest.mark.parametrize("mode", ["legacy", "auto"])
def test_a_client_that_asks_the_user_confirms_a_costly_submission(
    dataset: Dataset, tmp_path: Path, mode: str
) -> None:
    with _server(dataset, tmp_path) as url:
        submitted = _call(url, _submit(dataset, allow_partial=True), mode=mode)

    assert not submitted.is_error, submitted.content
    assert _structured(submitted)["request_id"]


def test_a_models_confirm_is_refused_unless_the_operator_allows_it(
    dataset: Dataset, tmp_path: Path
) -> None:
    with _server(dataset, tmp_path) as url:
        refused = _call(
            url,
            _submit(dataset, allow_partial=True, confirm=True),
            mode="legacy",
            elicit=False,
        )
    with _server(dataset, tmp_path, "--allow-model-confirm") as url:
        allowed = _call(
            url,
            _submit(dataset, allow_partial=True, confirm=True),
            mode="legacy",
            elicit=False,
        )

    assert refused.is_error
    assert "confirmation_unavailable" in str(refused.content)
    assert "allow_partial" in str(refused.content)
    assert not allowed.is_error, allowed.content


def test_a_request_runs_on_after_its_client_disconnects(
    dataset: Dataset, tmp_path: Path
) -> None:
    def status_of(request_id: str) -> _Body[Mapping[str, JsonValue]]:
        async def body(client: Client) -> Mapping[str, JsonValue]:
            result = await client.call_tool(
                "get_request_status",
                {
                    "dataset": dataset.manifest.uuid,
                    "request_id": request_id,
                    "wait_seconds": 5,
                },
            )
            return _structured(result)

        return body

    with _server(dataset, tmp_path) as url:
        submitted = _call(url, _submit(dataset))
        request_id = str(_structured(submitted)["request_id"])

        def finished() -> bool:
            status = _call(url, status_of(request_id))
            return status["verdict"] == "finished"

        wait_for(finished, REQUEST_SECONDS, "the request did not finish")


def test_an_http_address_is_read_with_ipv6_in_brackets() -> None:
    assert parse_http_address("127.0.0.1:8765") == HttpAddress("127.0.0.1", 8765)
    assert parse_http_address("[::1]:80") == HttpAddress("::1", 80)
    assert parse_http_address("box.tail1234.ts.net:9000").host == "box.tail1234.ts.net"
    for bad in ("8765", "::1:80", "host:0", "host:http", ":80", "[]:80"):
        with pytest.raises(ValueError, match="HOST:PORT|port|brackets"):
            _ = parse_http_address(bad)


def test_loopback_is_this_machine_only() -> None:
    assert is_loopback("127.0.0.1")
    assert is_loopback("::1")
    assert is_loopback("localhost")
    assert not is_loopback("0.0.0.0")
    assert not is_loopback("100.64.0.1")
    assert not is_loopback("box.tail1234.ts.net")


def test_a_token_file_must_be_private_and_long(tmp_path: Path) -> None:
    assert read_token_file(_token_file(tmp_path)) == TOKEN
    with pytest.raises(TokenFileInvalid, match="chmod 600"):
        _ = read_token_file(_token_file(tmp_path, mode=0o644))
    with pytest.raises(TokenFileInvalid, match="at least"):
        _ = read_token_file(_token_file(tmp_path, token="short"))
    with pytest.raises(TokenFileInvalid, match="does not read"):
        _ = read_token_file(tmp_path / "absent")
    assert stat.S_IMODE(_token_file(tmp_path).stat().st_mode) == 0o600


def test_the_guard_compares_the_whole_header() -> None:
    reached: list[str] = []

    async def inner(scope: object, receive: object, send: object) -> None:
        _ = (scope, receive, send)
        reached.append("app")

    sent: list[object] = []

    async def send(message: object) -> None:
        sent.append(message)

    async def receive() -> dict[str, object]:
        return {}

    guard = BearerGuard(inner, TOKEN)  # pyright: ignore[reportArgumentType]
    good = {"type": "http", "headers": [(b"authorization", f"Bearer {TOKEN}".encode())]}
    bad = {"type": "http", "headers": [(b"authorization", f"Basic {TOKEN}".encode())]}

    anyio.run(guard, good, receive, send)  # pyright: ignore[reportArgumentType]
    anyio.run(guard, bad, receive, send)  # pyright: ignore[reportArgumentType]
    anyio.run(guard, {"type": "lifespan"}, receive, send)  # pyright: ignore[reportArgumentType]

    assert reached == ["app", "app"]
    assert sent and isinstance(sent[0], dict) and sent[0]["status"] == 401


@pytest.mark.parametrize(
    ("flags", "fragment"),
    [
        (["--http", "127.0.0.1:8765"], "--token-file"),
        (["--token-file", "t"], "applies with --http only"),
        (["--allow-model-confirm"], "applies with --http only"),
        (["--http", "nonsense"], "HOST:PORT"),
    ],
)
def test_the_http_options_are_checked_before_serving(
    dataset: Dataset, flags: list[str], fragment: str
) -> None:
    result = CliRunner().invoke(
        app,
        ["agent", "serve", "--dataset", str(dataset.manifest_path), *flags],
    )

    assert result.exit_code != 0
    assert fragment in result.output


def test_a_tls_file_that_is_absent_is_refused(dataset: Dataset, tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "agent",
            "serve",
            "--dataset",
            str(dataset.manifest_path),
            "--http",
            "127.0.0.1:8765",
            "--token-file",
            str(_token_file(tmp_path)),
            "--tls-cert",
            str(tmp_path / "cert.pem"),
            "--tls-key",
            str(tmp_path / "key.pem"),
        ],
    )

    assert result.exit_code != 0
    assert "is not a file" in result.output


def test_tls_needs_both_files(dataset: Dataset, tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "agent",
            "serve",
            "--dataset",
            str(dataset.manifest_path),
            "--http",
            "127.0.0.1:8765",
            "--token-file",
            str(_token_file(tmp_path)),
            "--tls-cert",
            str(tmp_path / "cert.pem"),
        ],
    )

    assert result.exit_code != 0
    assert "go together" in result.output


@pytest.mark.parametrize("mode", ["legacy", "auto"])
def test_a_client_that_was_never_asked_cannot_answer_for_the_user(
    tmp_path: Path, mode: str
) -> None:
    """An answer and a request state written by the client do not confirm."""
    import hashlib
    import json

    from mosaic.agent.mcp._binding import bind_server
    from mosaic.agent.mcp.local import LocalBackend
    from tests.helpers import RecordingStarter, add_tracks_variant, make_dataset

    ds = make_dataset(tmp_path / "tracked", name="tracked")
    add_tracks_variant(
        ds, "convert-trex.0.2-1111111111", "seq_a", "seq_b", std_format="mosaic_v1"
    )
    starter = RecordingStarter()
    backend = LocalBackend([ds.manifest_path], warm=False, start_driver=starter)

    async def run() -> object:
        async with Client(
            bind_server(backend, model_confirm=False), mode=mode
        ) as client:
            arguments: dict[str, object] = {
                "dataset": ds.manifest.uuid,
                "recipe": json.loads(json.dumps(SPEED_RECIPE)),
            }
            plan = await client.call_tool("plan_recipe", arguments)
            shown = {
                **arguments,
                "plan_fingerprint": _structured(plan)["fingerprint"],
                "allow_partial": True,
            }
            canonical = json.dumps(
                {"tool": "submit_pipeline", "arguments": shown},
                sort_keys=True,
                separators=(",", ":"),
            )
            state = hashlib.sha256(canonical.encode()).hexdigest() + ".allow_partial"
            accepted = types.ElicitResult(action="accept", content={"confirm": True})
            return await client.session.call_tool(
                "submit_pipeline",
                shown,
                input_responses={"confirm": accepted},
                request_state=state,
                allow_input_required=True,
            )

    forged = anyio.run(run)

    assert isinstance(forged, types.CallToolResult)
    assert forged.is_error
    assert "confirmation_unavailable" in str(forged.content)
    assert starter.started == []


def test_a_loopback_address_checks_its_own_name_and_the_loopback_names() -> None:
    from mosaic.agent.mcp.server import (
        _transport_security,  # pyright: ignore[reportPrivateUsage]
    )

    other = _transport_security(HttpAddress("127.0.0.2", 8765), [])
    ipv6 = _transport_security(HttpAddress("0:0:0:0:0:0:0:1", 8765), ["box"])
    open_ = _transport_security(HttpAddress("0.0.0.0", 8765), [])

    assert other is not None
    assert other.enable_dns_rebinding_protection
    assert {"127.0.0.2:*", "localhost:*"} <= set(other.allowed_hosts)
    assert ipv6 is not None
    assert {"[0:0:0:0:0:0:0:1]:*", "box:*"} <= set(ipv6.allowed_hosts)
    assert open_ is None


def test_the_bearer_scheme_is_read_in_any_case() -> None:
    reached: list[str] = []

    async def inner(scope: object, receive: object, send: object) -> None:
        _ = (scope, receive, send)
        reached.append("app")

    async def send(message: object) -> None:
        _ = message

    async def receive() -> dict[str, object]:
        return {}

    guard = BearerGuard(inner, TOKEN)  # pyright: ignore[reportArgumentType]
    for header in (f"bearer {TOKEN}", f"BEARER {TOKEN}"):
        scope = {"type": "http", "headers": [(b"authorization", header.encode())]}
        anyio.run(guard, scope, receive, send)  # pyright: ignore[reportArgumentType]

    assert reached == ["app", "app"]


def test_a_token_file_that_is_not_text_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "token"
    _ = path.write_bytes(b"\xff" * 40)
    path.chmod(0o600)

    with pytest.raises(TokenFileInvalid, match="not UTF-8"):
        _ = read_token_file(path)
