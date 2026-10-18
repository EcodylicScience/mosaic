"""``mosaic agent serve`` as a client starts it: a process speaking JSON-RPC on stdio.

One test checks what the server imports before it answers ``initialize``. The
other writes requests to the server's standard input and reads every line of
its standard output. It validates, plans, submits with confirmation, polls the
request to its end through a real detached driver, and previews the output.
Validation prints, and starts a child that prints, both of which must reach
standard error. Each line that the server wrote to standard output must be one
JSON-RPC message.
"""

from __future__ import annotations

import json
import queue
import subprocess
import sys
import threading
import time
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Final

import pytest

pytest.importorskip("mcp")

import mcp_types as types

from mosaic.agent.mcp.spec import MAX_WAIT_SECONDS, TOOLS
from mosaic.core.dataset import Dataset
from mosaic.core.json_value import JsonValue
from mosaic.core.pipeline.graph import submit_request
from mosaic.runlog import JsonlRunLog, run_log_path
from tests.helpers import (
    CHAIN_RECIPE,
    add_tracks_variant,
    chain_recipe,
    make_dataset,
    wait_for,
)

VARIANT = "convert-trex.0.2-1111111111"

STORAGE = "speed-angvel__from__tracks"

RESPONSE_SECONDS: Final = 120.0
"""The longest one response takes, the first catalog build included."""

REQUEST_SECONDS: Final = 360.0
"""The longest the detached driver takes to run both steps of the recipe."""

PROTOCOL_VERSION: Final = "2025-11-25"
"""The protocol a client that opens with ``initialize`` asks for."""

INITIALIZE: Final[dict[str, JsonValue]] = {
    "protocolVersion": PROTOCOL_VERSION,
    "capabilities": {},
    "clientInfo": {"name": "test", "version": "0"},
}

LIBRARY_PRINT: Final = "a stray print from the library"

CHILD_PRINT: Final = "a stray print from a child process"

SERVE_NOISILY: Final = f"""
import subprocess, sys
from mosaic.agent.mcp.local import LocalBackend
from mosaic.cli import app

validate = LocalBackend.validate_recipe

def noisy(self, args):
    print({LIBRARY_PRINT!r}, flush=True)
    subprocess.run([sys.executable, "-c", "print({CHILD_PRINT!r})"], check=True)
    return validate(self, args)

LocalBackend.validate_recipe = noisy
app(["agent", "serve", "--dataset", sys.argv[1]], prog_name="mosaic")
"""
"""``mosaic agent serve``, with validation printing to standard output itself and
through a child."""

DEFERRED_MODULES: Final = (
    "av",
    "cv2",
    "mosaic.behavior",
    "mosaic.core.pipeline.graph.resolve",
    "mosaic.tracking",
    "numba",
    "pywt",
    "scipy",
    "sklearn",
    "torch",
)
"""Modules that the catalog build imports and the path to ``initialize`` must not."""

REPORTED: Final = "reported: "

SERVE_REPORTING_IMPORTS: Final = f"""
import json, sys, threading
from mosaic.agent.mcp import _binding
from mosaic.agent.mcp.local import LocalBackend
from mosaic.cli import app

construct = LocalBackend.__init__
bind = _binding.bind_server

def emit(key, value):
    print({REPORTED!r} + json.dumps({{key: value}}), file=sys.stderr, flush=True)

def recording(self, *args, **kwargs):
    loaded = sorted(name for name in {DEFERRED_MODULES!r} if name in sys.modules)
    emit("loaded", loaded)
    construct(self, *args, **kwargs)

async def report(ctx, call_next):
    result = await call_next(ctx)
    if ctx.method == "initialize":
        emit("warming", any(
            thread.name == "mosaic-mcp-warm" and thread.is_alive()
            for thread in threading.enumerate()
        ))
    return result

def reporting(backend):
    server = bind(backend)
    server.middleware.append(report)
    return server

LocalBackend.__init__ = recording
_binding.bind_server = reporting
app(["agent", "serve", "--dataset", sys.argv[1]], prog_name="mosaic")
"""
"""``mosaic agent serve``, writing to standard error which of the deferred modules
it holds when it builds the backend, and whether the catalog still warms when it
answers ``initialize``."""


class StdioSession:
    """A JSON-RPC client over a server process's standard streams.

    A thread reads every line that the server writes, and keeps each one for
    the check that the stream held only JSON-RPC messages.
    """

    def __init__(self, process: subprocess.Popen[str]) -> None:
        assert process.stdin is not None
        assert process.stdout is not None
        self._process: subprocess.Popen[str] = process
        self._input: IO[str] = process.stdin
        self._output: IO[str] = process.stdout
        self._next_id: int = 0
        self.lines: list[str] = []
        self._messages: queue.Queue[dict[str, JsonValue]] = queue.Queue()
        self._answered: dict[int, dict[str, JsonValue]] = {}
        self._reader: threading.Thread = threading.Thread(
            target=self._read, args=(process.stdout,), daemon=True
        )
        self._reader.start()

    def _read(self, stream: IO[str]) -> None:
        for line in stream:
            self.lines.append(line)
            try:
                parsed: JsonValue = json.loads(line)
            except ValueError:
                continue
            if isinstance(parsed, dict):
                self._messages.put(parsed)

    def notify(self, method: str, params: dict[str, JsonValue] | None = None) -> None:
        message: dict[str, JsonValue] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = params
        self._send(message)

    def begin(self, method: str, params: dict[str, JsonValue]) -> int:
        """Send one request without waiting, and return the id to ``finish`` it by."""
        self._next_id += 1
        sent = self._next_id
        self._send({"jsonrpc": "2.0", "id": sent, "method": method, "params": params})
        return sent

    def finish(self, sent: int) -> dict[str, JsonValue]:
        """Return the result of request *sent*, failing on an error response.

        A response to another request that arrives first is kept for its own
        ``finish``.
        """
        deadline = time.monotonic() + RESPONSE_SECONDS
        while sent not in self._answered:
            remaining = deadline - time.monotonic()
            assert remaining > 0, f"no response to {sent} within {RESPONSE_SECONDS} s"
            message = self._messages.get(timeout=remaining)
            answered = message.get("id")
            if isinstance(answered, int):
                self._answered[answered] = message
        message = self._answered.pop(sent)
        assert "error" not in message, message
        result = message["result"]
        assert isinstance(result, dict)
        return result

    def request(
        self, method: str, params: dict[str, JsonValue]
    ) -> dict[str, JsonValue]:
        """Send one request and return its result, failing on an error response."""
        return self.finish(self.begin(method, params))

    def call(self, tool: str, **arguments: JsonValue) -> dict[str, JsonValue]:
        """Call *tool* and return the whole result."""
        return self.request("tools/call", {"name": tool, "arguments": arguments})

    def _send(self, message: dict[str, JsonValue]) -> None:
        self._input.write(json.dumps(message) + "\n")
        self._input.flush()

    def close(self) -> int:
        """Close standard input, wait for the server to exit, and return its code."""
        if not self._input.closed:
            self._input.close()
        code = self._process.wait(timeout=RESPONSE_SECONDS)
        self._reader.join(timeout=RESPONSE_SECONDS)
        self._output.close()
        return code


def _structured(result: dict[str, JsonValue]) -> dict[str, JsonValue]:
    assert result.get("isError") is not True, result
    content = result["structuredContent"]
    assert isinstance(content, dict)
    return content


def _envelope(result: dict[str, JsonValue]) -> dict[str, JsonValue]:
    assert result.get("isError") is True, result
    blocks = result["content"]
    assert isinstance(blocks, list)
    envelope_block = blocks[1]
    assert isinstance(envelope_block, dict)
    text = envelope_block["text"]
    assert isinstance(text, str)
    envelope: dict[str, JsonValue] = json.loads(text)
    return envelope


@pytest.fixture
def dataset(tmp_path: Path) -> Dataset:
    """Two schema-valid track tables, and nothing computed."""
    ds = make_dataset(tmp_path / "tracked", name="tracked")
    add_tracks_variant(ds, VARIANT, "seq_a", "seq_b", std_format="mosaic_v1")
    return ds


@contextmanager
def _served(program: str, dataset: Dataset, errors: Path) -> Generator[StdioSession]:
    """Run *program* over *dataset*, its standard error written to *errors*."""
    with errors.open("w") as written:
        process = subprocess.Popen(
            [sys.executable, "-c", program, str(dataset.base_dir)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=written,
            text=True,
            encoding="utf-8",
        )
        served = StdioSession(process)
        try:
            yield served
        finally:
            if process.poll() is None:
                process.kill()
            _ = served.close()


@pytest.fixture
def session(dataset: Dataset, tmp_path: Path) -> Iterator[StdioSession]:
    with _served(SERVE_NOISILY, dataset, tmp_path / "server.stderr") as served:
        yield served


def test_the_path_to_initialize_defers_the_catalog_imports(
    dataset: Dataset, tmp_path: Path
) -> None:
    """``mosaic agent serve`` answers ``initialize`` before the catalog is built.

    What a client waits on is the interpreter, the command, the server and the
    handshake. The step-type catalog imports the feature library and the media
    readers on a thread of its own, and none of them is on that path.
    """
    errors = tmp_path / "server.stderr"
    with _served(SERVE_REPORTING_IMPORTS, dataset, errors) as served:
        initialized = served.request("initialize", INITIALIZE)
        reported: dict[str, JsonValue] = {}
        for line in errors.read_text(encoding="utf-8").splitlines():
            if line.startswith(REPORTED):
                reported.update(json.loads(line.removeprefix(REPORTED)))

    assert initialized["protocolVersion"] == PROTOCOL_VERSION
    assert reported == {"loaded": [], "warming": True}, reported


def test_a_client_plans_submits_polls_and_previews_over_clean_stdout(
    session: StdioSession, dataset: Dataset, tmp_path: Path
) -> None:
    initialized = session.request("initialize", INITIALIZE)
    session.notify("notifications/initialized")
    listed = session.request("tools/list", {})
    uuid = dataset.manifest.uuid
    assert uuid is not None
    recipe: dict[str, JsonValue] = json.loads(json.dumps(CHAIN_RECIPE))

    valid = _structured(session.call("validate_recipe", recipe=recipe))
    plan = _structured(session.call("plan_recipe", dataset=uuid, recipe=recipe))
    submission: dict[str, JsonValue] = {
        "dataset": uuid,
        "recipe": recipe,
        "plan_fingerprint": plan["fingerprint"],
        # Asks to confirm, and changes nothing over a plan that covers its scope.
        "allow_partial": True,
    }
    refused = _envelope(session.call("submit_pipeline", **submission))
    submitted = _structured(session.call("submit_pipeline", **submission, confirm=True))
    request_id = submitted["request_id"]

    def finished() -> bool:
        status = _structured(
            session.call("get_request_status", dataset=uuid, request_id=request_id)
        )
        return status["terminal"] is True

    wait_for(finished, REQUEST_SECONDS, "the detached driver never ended the request")
    status = _structured(
        session.call("get_request_status", dataset=uuid, request_id=request_id)
    )
    steps = status["steps"]
    assert isinstance(steps, list)
    speed = steps[0]
    assert isinstance(speed, dict)
    preview = _structured(
        session.call(
            "preview_output",
            dataset=uuid,
            ref=f"feature/{STORAGE}/{speed['run_id']}",
            group="",
            sequence="seq_a",
            limit=5,
        )
    )
    code = session.close()

    assert initialized["protocolVersion"] == PROTOCOL_VERSION
    tools = listed["tools"]
    assert isinstance(tools, list) and len(tools) == len(TOOLS)
    assert valid["valid"] is True
    assert (refused["reason"], refused["kind"]) == ("confirmation_required", "invalid")
    assert status["verdict"] == "finished", status
    rows = preview["rows"]
    assert isinstance(rows, list) and len(rows) == 5
    assert code == 0
    for line in session.lines:
        _ = types.jsonrpc_message_adapter.validate_json(line)
    errors = (tmp_path / "server.stderr").read_text(encoding="utf-8")
    assert LIBRARY_PRINT in errors
    assert CHILD_PRINT in errors


WAIT_SECONDS: Final = MAX_WAIT_SECONDS
"""The wait the stdio test asks for, ended long before by the step that ends."""


def test_a_wait_over_stdio_returns_when_a_step_ends_and_leaves_the_server_answering(
    session: StdioSession, dataset: Dataset
) -> None:
    _ = session.request("initialize", INITIALIZE)
    session.notify("notifications/initialized")
    uuid = dataset.manifest.uuid
    assert uuid is not None
    request = submit_request(dataset, chain_recipe()).request
    execution_id = request.execution_of("speed")
    path = run_log_path(dataset.base_dir, execution_id)
    with JsonlRunLog(path, execution_id) as log:
        log.started(kind="feature", target=STORAGE)
    # The first status read imports what a status read needs.
    _ = _structured(
        session.call("get_request_status", dataset=uuid, request_id=request.request_id)
    )

    began = time.monotonic()
    waiting = session.begin(
        "tools/call",
        {
            "name": "get_request_status",
            "arguments": {
                "dataset": uuid,
                "request_id": request.request_id,
                "wait_seconds": WAIT_SECONDS,
                "brief": True,
            },
        },
    )
    # Answered while the wait is pending, so the wait is not on the server's loop.
    listed = _structured(session.call("list_datasets"))
    time.sleep(0.5)
    with JsonlRunLog(path, execution_id) as log:
        log.finished()
    waited = _structured(session.finish(waiting))
    elapsed = time.monotonic() - began
    code = session.close()

    assert listed["total"] == 1
    assert elapsed < WAIT_SECONDS / 2
    steps = waited["steps"]
    assert isinstance(steps, list)
    ended = steps[0]
    assert isinstance(ended, dict)
    assert (ended["step"], ended["outcome"]) == ("speed", "finished")
    assert (waited["omitted_finished"], waited["omitted_pending"]) == (0, 1)
    assert waited["verdict"] == "running"
    assert code == 0
    for line in session.lines:
        _ = types.jsonrpc_message_adapter.validate_json(line)
