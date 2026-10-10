"""The runner side of the line protocol: its envelopes, and the loop that serves them.

Apart from the envelope round trips and the tests of the runner's helpers, each
test spawns a runner program that ``write_fake_runner`` writes and exchanges lines
with it on its standard streams, as mosaic does. The runner runs under this
interpreter, with the environment that an external tool receives.
"""

from __future__ import annotations

import importlib.metadata
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Final

import pytest
from pydantic import BaseModel, ValidationError

from mosaic.behavior.feature_library.external.runner_wire import (
    NOT_INSTALLED,
    REPLIES,
    Done,
    Emit,
    Event,
    Failed,
    Ready,
    Reply,
    Request,
    installed_version,
    serving,
)
from mosaic.core.pipeline.subprocess_util import LineChild, supervised_popen_kwargs
from mosaic.core.pipeline.toolenv import subprocess_env
from tests.helpers import (
    FAKE_HANDLERS,
    RUNNER_LINE_SECONDS,
    EventRecorder,
    LineChildren,
    import_runner_program,
    write_fake_runner,
)

HANDLERS: Final = (
    FAKE_HANDLERS
    + """

class Opaque(BaseModel):
    value: object


def chatter(payload, emit):
    print("printed by a library")
    os.write(1, b"written to descriptor 1\\n")
    subprocess.run([sys.executable, "-c", "print('printed by a child process')"])


def opaque(payload, emit):
    return Opaque(value=object())


def unmodeled(payload, emit):
    return {"text": "a mapping"}


DISPATCH.update(chatter=chatter, opaque=opaque, unmodeled=unmodeled)
"""
)


def _start(children: LineChildren, *, heartbeat_seconds: float = 30.0) -> LineChild:
    """Start a fake runner, and read its ``Ready``."""
    program = write_fake_runner(
        children.directory, HANDLERS, heartbeat_seconds=heartbeat_seconds
    )
    child = children.spawn([sys.executable, str(program)], env=subprocess_env())
    assert _reply(child) == Ready(tool="fake", version="1.0")
    return child


def _reply(child: LineChild) -> Reply:
    line = child.next_line(timeout=RUNNER_LINE_SECONDS)
    assert line is not None, "the runner's output ended"
    return REPLIES.validate_json(line)


def _send(child: LineChild, command: str, **payload: str | float) -> None:
    child.send(Request(command=command, payload=dict(payload)).model_dump_json())


# --- the envelopes ------------------------------------------------------------


@pytest.mark.parametrize(
    "reply",
    [
        Ready(tool="fake", version="1.0"),
        Done(),
        Done(result={"paths": ["a.npz", "b.npz"], "loss": 0.5, "missing": None}),
        Failed(kind="error", message="ValueError: no", traceback="Traceback ..."),
        Failed(kind="cancelled", message="KeyboardInterrupt"),
        Event(name="epoch", data={"epoch": 3, "losses": {"test": 1.25}}),
    ],
    ids=["ready", "done", "done-with-result", "error", "cancelled", "event"],
)
def test_each_reply_round_trips_through_one_line(reply: Reply) -> None:
    line = reply.model_dump_json()

    assert "\n" not in line
    assert REPLIES.validate_json(line) == reply


def test_a_request_round_trips_through_one_line() -> None:
    request = Request(command="fit", payload={"sessions": ["s000000.npz"], "seed": 42})

    assert Request.model_validate_json(request.model_dump_json()) == request


# --- the runner's helpers ----------------------------------------------------


class _Greeting(BaseModel):
    name: str


class _Greeted(BaseModel):
    text: str


def _greet(request: _Greeting, emit: Emit) -> _Greeted:
    emit("greeting", {"name": request.name})
    return _Greeted(text=f"hello {request.name}")


def test_a_served_handler_runs_on_its_payload_as_its_model() -> None:
    recorder = EventRecorder()

    result = serving(_Greeting, _greet)({"name": "ada"}, recorder)

    assert result == _Greeted(text="hello ada")
    assert recorder.events == [("greeting", {"name": "ada"})]


def test_a_served_handler_refuses_a_payload_that_its_model_refuses() -> None:
    recorder = EventRecorder()

    with pytest.raises(ValidationError, match="name"):
        _ = serving(_Greeting, _greet)({"title": "ada"}, recorder)

    assert recorder.events == []


def test_the_installed_version_is_read_from_the_distributions_metadata() -> None:
    assert installed_version("pydantic") == importlib.metadata.version("pydantic")


def test_a_distribution_that_is_not_installed_is_reported_as_such() -> None:
    assert installed_version("mosaic-no-such-distribution") == NOT_INSTALLED


# --- the loop -----------------------------------------------------------------


def test_a_handlers_result_is_returned_in_done(line_children: LineChildren) -> None:
    child = _start(line_children)

    _send(child, "echo", text="hello")

    assert _reply(child) == Done(result={"text": "hello"})


def test_events_precede_done_in_the_order_emitted(line_children: LineChildren) -> None:
    child = _start(line_children)

    _send(child, "count", up_to=4)

    replies = [_reply(child) for _ in range(5)]
    assert replies == [
        *(Event(name="counted", data={"number": number}) for number in range(4)),
        Done(),
    ]


@pytest.mark.parametrize(
    "line",
    ["this is not JSON", '{"payload": {}}', '{"command": "echo", "payload": []}'],
    ids=["not-json", "no-command", "payload-not-an-object"],
)
def test_a_malformed_line_is_failed_and_the_loop_goes_on(
    line_children: LineChildren, line: str
) -> None:
    child = _start(line_children)

    child.send(line)

    reply = _reply(child)
    assert isinstance(reply, Failed)
    assert reply.kind == "error"
    _send(child, "echo", text="still serving")
    assert _reply(child) == Done(result={"text": "still serving"})


def test_an_unknown_command_is_failed_naming_the_commands_served(
    line_children: LineChildren,
) -> None:
    child = _start(line_children)

    _send(child, "train")

    reply = _reply(child)
    assert isinstance(reply, Failed)
    assert "'train'" in reply.message
    assert "echo" in reply.message
    assert "shutdown" in reply.message


def test_a_handlers_exception_is_failed_with_its_traceback(
    line_children: LineChildren,
) -> None:
    child = _start(line_children)

    _send(child, "fail")

    reply = _reply(child)
    assert isinstance(reply, Failed)
    assert reply.kind == "error"
    assert reply.message == "ValueError: the handler failed"
    assert 'raise ValueError("the handler failed")' in reply.traceback
    _send(child, "echo", text="after the failure")
    assert _reply(child) == Done(result={"text": "after the failure"})


@pytest.mark.parametrize(
    ("command", "message"),
    [
        ("opaque", "PydanticSerializationError: Unable to serialize unknown type"),
        ("unmodeled", "AttributeError: 'dict' object"),
    ],
    ids=["unserializable", "not-a-model"],
)
def test_a_result_that_cannot_be_sent_is_failed(
    line_children: LineChildren, command: str, message: str
) -> None:
    child = _start(line_children)

    _send(child, command)

    reply = _reply(child)
    assert isinstance(reply, Failed)
    assert reply.kind == "error"
    assert reply.message.startswith(message)
    assert "model_dump" in reply.traceback, "the traceback ends at the serialization"
    _send(child, "echo", text="after the result")
    assert _reply(child) == Done(result={"text": "after the result"})


def test_a_keyboard_interrupt_is_failed_as_cancelled(
    line_children: LineChildren,
) -> None:
    child = _start(line_children)

    _send(child, "interrupt")

    reply = _reply(child)
    assert isinstance(reply, Failed)
    assert reply.kind == "cancelled"
    _send(child, "echo", text="after the interrupt")
    assert _reply(child) == Done(result={"text": "after the interrupt"})


def test_standard_output_of_a_command_is_written_to_standard_error(
    line_children: LineChildren,
) -> None:
    """A print, a write to descriptor 1 and a child process all write to the log."""
    child = _start(line_children)

    _send(child, "chatter")

    assert _reply(child) == Done()
    child.close(grace=10.0)
    log = child.log_path.read_text()
    assert "printed by a library" in log
    assert "written to descriptor 1" in log
    assert "printed by a child process" in log


def test_each_command_is_logged_to_standard_error(line_children: LineChildren) -> None:
    child = _start(line_children)

    _send(child, "echo", text="hello")

    assert _reply(child) == Done(result={"text": "hello"})
    child.close(grace=10.0)
    assert "[runner_wire] INFO: command: echo" in child.log_path.read_text()


def test_heartbeats_are_sent_while_a_command_runs(line_children: LineChildren) -> None:
    child = _start(line_children, heartbeat_seconds=0.1)

    _send(child, "nap", seconds=0.8)

    replies: list[Reply] = []
    while not replies or not isinstance(replies[-1], Done):
        replies.append(_reply(child))
    assert replies[0] == Event(name="napping", data={})
    assert replies.count(Event(name="heartbeat", data={})) >= 2
    _send(child, "echo", text="after the nap")
    assert _reply(child) == Done(result={"text": "after the nap"}), (
        "a heartbeat followed the command's Done"
    )


def test_shutdown_is_answered_and_ends_the_runner(line_children: LineChildren) -> None:
    child = _start(line_children)

    _send(child, "shutdown")

    assert _reply(child) == Done()
    assert child.next_line(timeout=RUNNER_LINE_SECONDS) is None
    child.close(grace=10.0)
    assert child.exit_code == 0


def test_the_end_of_input_between_commands_ends_the_runner(
    line_children: LineChildren,
) -> None:
    child = _start(line_children)

    child.close(grace=10.0)

    assert child.exit_code == 0


def test_the_end_of_input_during_a_command_ends_the_runner_within_seconds(
    line_children: LineChildren,
) -> None:
    """A runner whose parent is gone exits without finishing its command."""
    child = _start(line_children)
    _send(child, "nap", seconds=60)
    assert _reply(child) == Event(name="napping", data={})
    started = time.monotonic()

    child.close(grace=30.0)

    assert time.monotonic() - started < 10.0
    assert child.exit_code == 1


def test_a_runner_whose_reader_is_gone_exits(tmp_path: Path) -> None:
    """A reply that cannot be written ends the runner, although its input is open."""
    program = write_fake_runner(tmp_path, HANDLERS)
    with (tmp_path / "runner.log").open("ab") as log:
        process = subprocess.Popen(
            [sys.executable, str(program)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=log,
            env=subprocess_env(),
            **supervised_popen_kwargs(),
        )
    stdin, stdout = process.stdin, process.stdout
    assert stdin is not None and stdout is not None
    try:
        assert REPLIES.validate_json(stdout.readline()) == Ready(
            tool="fake", version="1.0"
        )
        stdout.close()

        _ = stdin.write(
            Request(command="echo", payload={"text": "x"}).model_dump_json()
        )
        _ = stdin.write("\n")
        stdin.flush()

        assert process.wait(timeout=RUNNER_LINE_SECONDS) == 1
    finally:
        if process.poll() is None:
            process.kill()
            _ = process.wait()
        stdin.close()


def test_importing_a_runner_has_no_side_effect(tmp_path: Path) -> None:
    """Importing a runner leaves standard output, descriptor 1 and the threads alone."""
    program = write_fake_runner(tmp_path, HANDLERS)
    stdout = sys.stdout
    descriptor = os.fstat(1)
    threads = threading.active_count()

    with import_runner_program(program, ("runner_wire",)) as module:
        assert getattr(module, "PROTOCOL") is None
        assert sys.stdout is stdout
        after = os.fstat(1)
        assert (after.st_dev, after.st_ino) == (descriptor.st_dev, descriptor.st_ino)
        assert threading.active_count() == threads

    assert program.stem not in sys.modules
    assert "runner_wire" not in sys.modules
    assert str(tmp_path) not in sys.path
