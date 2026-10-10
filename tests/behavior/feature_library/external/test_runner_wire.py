"""The runner side of the line protocol: its envelopes, and the loop that serves them.

Apart from the envelope round trips, each test spawns a runner program that
``write_fake_runner`` writes and exchanges lines with it on its standard streams,
as mosaic does. The runner runs under this interpreter, with the environment
that an external tool receives.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import pytest

from mosaic.behavior.feature_library.external.runner_wire import (
    REPLIES,
    Done,
    Event,
    Failed,
    Ready,
    Reply,
    Request,
)
from mosaic.core.pipeline.subprocess_util import LineChild, supervised_popen_kwargs
from mosaic.core.pipeline.toolenv import subprocess_env
from tests.helpers import import_runner_program, write_fake_runner

_WAIT: Final = 60.0
"""How long a test waits for a line, which covers a runner's start under load."""

HANDLERS: Final = """
from pydantic import BaseModel


class Echoed(BaseModel):
    text: str


class Opaque(BaseModel):
    value: object


def echo(payload, emit):
    return Echoed(text=str(payload["text"]))


def count(payload, emit):
    for number in range(int(payload["up_to"])):
        emit("counted", {"number": number})


def nap(payload, emit):
    emit("napping", {})
    time.sleep(float(payload["seconds"]))


def chatter(payload, emit):
    print("printed by a library")
    os.write(1, b"written to descriptor 1\\n")
    subprocess.run([sys.executable, "-c", "print('printed by a child process')"])


def fail(payload, emit):
    raise ValueError("the handler failed")


def interrupt(payload, emit):
    raise KeyboardInterrupt


def opaque(payload, emit):
    return Opaque(value=object())


def unmodeled(payload, emit):
    return {"text": "a mapping"}


DISPATCH = {
    "echo": echo,
    "count": count,
    "nap": nap,
    "chatter": chatter,
    "fail": fail,
    "interrupt": interrupt,
    "opaque": opaque,
    "unmodeled": unmodeled,
}
"""


class _Runners:
    """Starts the runners of one test, each logging into *directory*."""

    def __init__(self, directory: Path) -> None:
        self._directory = directory
        self._started: list[LineChild] = []

    @property
    def log(self) -> Path:
        """The file that the standard error of every runner is appended to."""
        return self._directory / "runner.log"

    def program(self, *, heartbeat_seconds: float = 30.0) -> Path:
        return write_fake_runner(
            self._directory, HANDLERS, heartbeat_seconds=heartbeat_seconds
        )

    def start(self, *, heartbeat_seconds: float = 30.0) -> LineChild:
        """Start a runner and read its ``Ready``."""
        program = self.program(heartbeat_seconds=heartbeat_seconds)
        child = LineChild(
            [sys.executable, str(program)], log_path=self.log, env=subprocess_env()
        )
        self._started.append(child)
        assert _reply(child) == Ready(tool="fake", version="1.0")
        return child

    def close_all(self) -> None:
        for child in self._started:
            child.close(grace=1.0)


@pytest.fixture
def runners(tmp_path: Path) -> Iterator[_Runners]:
    """Runners that are closed when the test ends, whatever its outcome."""
    started = _Runners(tmp_path)
    yield started
    started.close_all()


def _reply(child: LineChild) -> Reply:
    line = child.next_line(timeout=_WAIT)
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


# --- the loop -----------------------------------------------------------------


def test_a_handlers_result_is_returned_in_done(runners: _Runners) -> None:
    child = runners.start()

    _send(child, "echo", text="hello")

    assert _reply(child) == Done(result={"text": "hello"})


def test_events_precede_done_in_the_order_emitted(runners: _Runners) -> None:
    child = runners.start()

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
    runners: _Runners, line: str
) -> None:
    child = runners.start()

    child.send(line)

    reply = _reply(child)
    assert isinstance(reply, Failed)
    assert reply.kind == "error"
    _send(child, "echo", text="still serving")
    assert _reply(child) == Done(result={"text": "still serving"})


def test_an_unknown_command_is_failed_naming_the_commands_served(
    runners: _Runners,
) -> None:
    child = runners.start()

    _send(child, "train")

    reply = _reply(child)
    assert isinstance(reply, Failed)
    assert "'train'" in reply.message
    assert "echo" in reply.message
    assert "shutdown" in reply.message


def test_a_handlers_exception_is_failed_with_its_traceback(runners: _Runners) -> None:
    child = runners.start()

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
    runners: _Runners, command: str, message: str
) -> None:
    child = runners.start()

    _send(child, command)

    reply = _reply(child)
    assert isinstance(reply, Failed)
    assert reply.kind == "error"
    assert reply.message.startswith(message)
    assert "model_dump" in reply.traceback, "the traceback ends at the serialization"
    _send(child, "echo", text="after the result")
    assert _reply(child) == Done(result={"text": "after the result"})


def test_a_keyboard_interrupt_is_failed_as_cancelled(runners: _Runners) -> None:
    child = runners.start()

    _send(child, "interrupt")

    reply = _reply(child)
    assert isinstance(reply, Failed)
    assert reply.kind == "cancelled"
    _send(child, "echo", text="after the interrupt")
    assert _reply(child) == Done(result={"text": "after the interrupt"})


def test_standard_output_of_a_command_is_written_to_standard_error(
    runners: _Runners,
) -> None:
    """A print, a write to descriptor 1 and a child process all write to the log."""
    child = runners.start()

    _send(child, "chatter")

    assert _reply(child) == Done()
    child.close(grace=10.0)
    log = runners.log.read_text()
    assert "printed by a library" in log
    assert "written to descriptor 1" in log
    assert "printed by a child process" in log


def test_heartbeats_are_sent_while_a_command_runs(runners: _Runners) -> None:
    child = runners.start(heartbeat_seconds=0.1)

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


def test_shutdown_is_answered_and_ends_the_runner(runners: _Runners) -> None:
    child = runners.start()

    _send(child, "shutdown")

    assert _reply(child) == Done()
    assert child.next_line(timeout=_WAIT) is None
    child.close(grace=10.0)
    assert child.exit_code == 0


def test_the_end_of_input_between_commands_ends_the_runner(runners: _Runners) -> None:
    child = runners.start()

    child.close(grace=10.0)

    assert child.exit_code == 0


def test_the_end_of_input_during_a_command_ends_the_runner_within_seconds(
    runners: _Runners,
) -> None:
    """A runner whose parent is gone exits without finishing its command."""
    child = runners.start()
    _send(child, "nap", seconds=60)
    assert _reply(child) == Event(name="napping", data={})
    started = time.monotonic()

    child.close(grace=30.0)

    assert time.monotonic() - started < 10.0
    assert child.exit_code == 1


def test_a_runner_whose_reader_is_gone_exits(runners: _Runners) -> None:
    """A reply that cannot be written ends the runner, although its input is open."""
    program = runners.program()
    with runners.log.open("ab") as log:
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

        assert process.wait(timeout=_WAIT) == 1
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

    with import_runner_program(tmp_path, program.stem, ("runner_wire",)) as module:
        assert getattr(module, "PROTOCOL") is None
        assert sys.stdout is stdout
        after = os.fstat(1)
        assert (after.st_dev, after.st_ino) == (descriptor.st_dev, descriptor.st_ino)
        assert threading.active_count() == threads

    assert program.stem not in sys.modules
    assert "runner_wire" not in sys.modules
    assert str(tmp_path) not in sys.path
