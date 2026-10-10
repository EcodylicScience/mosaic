"""The runner side of the line protocol between mosaic and a tool's environment.

A runner is a program that runs in an external tool's environment, launched by
path. It reads requests on its standard input and writes replies and events on
its standard output, one JSON object per line. Arrays cross as files, which a
payload or a result names by path.

This module imports only the standard library and pydantic, which both
environments install. A runner imports it by its bare name from the runner's
directory, and mosaic imports it as
``mosaic.behavior.feature_library.external.runner_wire``. It does not import a
sibling module or ``mosaic``.

A runner's module scope imports the standard library, pydantic and its siblings,
and writes nothing. Its ``main()`` claims standard output first, then logs to
standard error and serves::

    def main() -> None:
        protocol = claim_stdout()
        log_to_stderr()
        serve(DISPATCH, "vame", installed_version("vame-py"), protocol=protocol)

:func:`serving` builds each handler of ``DISPATCH`` from a payload model. A
handler imports the tool's libraries (vame, torch, jax, keypoint_moseq) when it
runs. Their output at import is then written to standard error, and a test
imports a runner without side effects.
"""

from __future__ import annotations

import importlib.metadata
import logging
import os
import queue
import sys
import threading
import traceback
from collections.abc import Callable, Mapping
from typing import (
    Annotated,
    Final,
    Literal,
    NoReturn,
    Protocol,
    TextIO,
    TypeAlias,
    TypeVar,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    TypeAdapter,
    ValidationError,
)

__all__ = [
    "HEARTBEAT",
    "NOT_INSTALLED",
    "REPLIES",
    "SHUTDOWN",
    "Dispatch",
    "Done",
    "Emit",
    "Event",
    "Failed",
    "FailureKind",
    "Handler",
    "Ready",
    "Reply",
    "Request",
    "claim_stdout",
    "installed_version",
    "log_to_stderr",
    "serve",
    "serving",
]

SHUTDOWN: Final = "shutdown"
"""The command that ends the loop. :func:`serve` answers it before any handler."""

HEARTBEAT: Final = "heartbeat"
"""The name of the event that :func:`serve` writes while a command runs."""

NOT_INSTALLED: Final = "not installed"
"""The version that :func:`installed_version` reports for a missing distribution."""

_ABANDON_GRACE_SECONDS: Final = 2.0
"""How long a command may run after the end of input before the runner exits."""

_READ_SIZE: Final = 65536
"""The most bytes of standard input that one read takes."""

_M = TypeVar("_M", bound=BaseModel)

_log = logging.getLogger(__name__)


class _Envelope(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Request(_Envelope):
    """A command and its payload, which mosaic writes to the runner's standard input.

    Attributes:
        command: The handler to run, or :data:`SHUTDOWN`.
        payload: The handler's argument. The tool's protocol module declares its
            model, and the handler validates it.
    """

    command: str
    payload: Mapping[str, JsonValue] = Field(default_factory=dict[str, JsonValue])


class Ready(_Envelope):
    """The first line of a runner, written before it reads a request.

    Attributes:
        tool: The tool that the runner drives.
        version: The tool's installed version, for mosaic to compare with the
            version it requires.
    """

    reply: Literal["ready"] = "ready"
    tool: str
    version: str


class Done(_Envelope):
    """The reply to a command that returned.

    Attributes:
        result: The handler's result, as JSON. The tool's protocol module
            declares its model, and mosaic validates it.
    """

    reply: Literal["done"] = "done"
    result: JsonValue = None


FailureKind: TypeAlias = Literal["error", "cancelled"]
"""Why a command failed: it raised, or it was interrupted."""


class Failed(_Envelope):
    """The reply to a request that did not return a result.

    Attributes:
        kind: ``error`` for a malformed or unknown request and for a handler
            that raised, ``cancelled`` for a handler interrupted by
            ``KeyboardInterrupt``.
        message: The exception's type and text, or what was wrong with the
            request.
        traceback: The handler's traceback, empty for a request that ran no
            handler.
    """

    reply: Literal["failed"] = "failed"
    kind: FailureKind
    message: str
    traceback: str = ""


class Event(_Envelope):
    """A line written while a command runs, before its reply.

    Attributes:
        name: What happened, such as :data:`HEARTBEAT` or a tool's progress.
        data: The details, as a JSON object.
    """

    reply: Literal["event"] = "event"
    name: str
    data: Mapping[str, JsonValue] = Field(default_factory=dict[str, JsonValue])


Reply: TypeAlias = Annotated[
    Ready | Done | Failed | Event, Field(discriminator="reply")
]
"""Any line that a runner writes to standard output."""

REPLIES: Final[TypeAdapter[Reply]] = TypeAdapter(Reply)
"""Reads one line of a runner's standard output as a :data:`Reply`."""


class Emit(Protocol):
    """Writes an :class:`Event` while a handler runs."""

    def __call__(self, name: str, data: Mapping[str, JsonValue] | None = None) -> None:
        """Write the event *name* with *data*, an empty object when ``None``."""
        ...


Handler: TypeAlias = Callable[[Mapping[str, JsonValue], Emit], BaseModel | None]
"""Runs one command on its payload, and returns the result of :class:`Done`."""

Dispatch: TypeAlias = Mapping[str, Handler]
"""The handler of each command that a runner serves."""


def serving(model: type[_M], handle: Callable[[_M, Emit], BaseModel | None]) -> Handler:
    """Return the handler that validates its payload as *model* and runs *handle*.

    A payload that *model* refuses raises ``ValidationError``. :func:`serve`
    answers the error with ``Failed(error)``.
    """

    def handler(payload: Mapping[str, JsonValue], emit: Emit) -> BaseModel | None:
        return handle(model.model_validate(payload), emit)

    return handler


def installed_version(distribution: str) -> str:
    """Return the installed version of *distribution*, read without importing it.

    An environment without the distribution reports :data:`NOT_INSTALLED`. The
    runner's first command that imports it then fails.
    """
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return NOT_INSTALLED


def log_to_stderr() -> None:
    """Log records of INFO and above to standard error, each under its logger's name."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(name)s] %(levelname)s: %(message)s",
        stream=sys.stderr,
    )


def claim_stdout() -> TextIO:
    """Reserve standard output for the protocol, and return the protocol stream.

    Descriptor 1 is duplicated for the protocol. Then descriptor 1 and
    ``sys.stdout`` are pointed at standard error. A library's print, a write to
    descriptor 1 and the output of a child process are then written to standard
    error. Call it first in a runner's ``main()``.
    """
    sys.stdout.flush()
    protocol = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    return os.fdopen(protocol, "w", encoding="utf-8", newline="\n")


def serve(
    dispatch: Dispatch,
    tool: str,
    version: str,
    *,
    protocol: TextIO,
    heartbeat_seconds: float = 30.0,
) -> None:
    """Write :class:`Ready`, then answer requests until shutdown or the end of input.

    A thread reads standard input and queues its lines. The loop runs one command
    at a time and replies with :class:`Done` or :class:`Failed`:

    - a malformed line, or a request for a command without a handler, is
      answered with ``Failed(error)``, and the loop continues;
    - a handler's exception, or a result that is not a model or cannot be
      serialized as JSON, is answered with ``Failed(error)`` and its traceback,
      and ``KeyboardInterrupt`` with ``Failed(cancelled)``;
    - :data:`SHUTDOWN` is answered with ``Done`` and ends the loop.

    Each command that runs a handler is logged at INFO. While a handler runs, a
    :data:`HEARTBEAT` event is written every *heartbeat_seconds*.

    The end of input between commands ends the loop. The end of input during a
    command means that the parent has gone. The runner then exits with status 1
    unless the command finishes within 2 s. A reply that cannot be written, because
    standard output lacks a reader, also ends the runner with status 1. Standard
    error is flushed first in both cases.

    Args:
        dispatch: The handler of each command.
        tool: The tool name that :class:`Ready` reports.
        version: The tool version that :class:`Ready` reports.
        protocol: The stream that :func:`claim_stdout` returned.
        heartbeat_seconds: The interval between two heartbeats of a command.
    """
    writer = _ProtocolWriter(protocol)
    session = _Session()
    lines: queue.Queue[str | None] = queue.Queue()
    writer.write(Ready(tool=tool, version=version))
    threading.Thread(target=_read_lines, args=(lines, session), daemon=True).start()
    while True:
        line = lines.get()
        if line is None:
            return
        try:
            request = Request.model_validate_json(line)
        except ValidationError as error:
            writer.write(Failed(kind="error", message=f"malformed request: {error}"))
            continue
        if request.command == SHUTDOWN:
            writer.write(Done())
            return
        handler = dispatch.get(request.command)
        if handler is None:
            served = ", ".join(sorted([*dispatch, SHUTDOWN]))
            message = (
                f"unknown command {request.command!r}. This runner serves {served}."
            )
            writer.write(Failed(kind="error", message=message))
            continue
        if not session.begin():
            return
        _log.info("command: %s", request.command)
        try:
            reply = _run(handler, request.payload, writer, heartbeat_seconds)
        finally:
            session.end()
        writer.write(reply)


def _run(
    handler: Handler,
    payload: Mapping[str, JsonValue],
    writer: _ProtocolWriter,
    heartbeat_seconds: float,
) -> Done | Failed:
    """Run *handler* with heartbeats, and return its reply.

    The heartbeat thread is joined before the reply is returned. A heartbeat
    therefore precedes the reply or is not written.
    """
    finished = threading.Event()
    heartbeat = threading.Thread(
        target=_beat, args=(writer, finished, heartbeat_seconds), daemon=True
    )
    heartbeat.start()
    try:
        result = handler(payload, writer.event)
        return Done(result=None if result is None else result.model_dump(mode="json"))
    except KeyboardInterrupt:
        return Failed(
            kind="cancelled",
            message="KeyboardInterrupt",
            traceback=traceback.format_exc(),
        )
    except Exception as error:
        return Failed(
            kind="error",
            message=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
        )
    finally:
        finished.set()
        heartbeat.join()


def _beat(writer: _ProtocolWriter, finished: threading.Event, seconds: float) -> None:
    while not finished.wait(seconds):
        writer.write(Event(name=HEARTBEAT))


def _read_lines(lines: queue.Queue[str | None], session: _Session) -> None:
    """Queue each line of standard input and then ``None``.

    A command that outlives the input by 2 s ends the process.

    The thread reads descriptor 0 directly. A daemon thread blocked in a read of
    ``sys.stdin`` has acquired the lock of its buffer, and the interpreter aborts
    when that lock is still acquired at exit.
    """
    pending = bytearray()
    while chunk := os.read(0, _READ_SIZE):
        pending += chunk
        *complete, rest = pending.split(b"\n")
        for raw in complete:
            lines.put(raw.decode("utf-8", errors="replace").rstrip("\r"))
        pending = rest
    if pending:
        lines.put(pending.decode("utf-8", errors="replace"))
    lines.put(None)
    if session.input_ended():
        _abandon()


def _abandon() -> NoReturn:
    """End the process at once, with status 1, after flushing standard error."""
    sys.stderr.flush()
    os._exit(1)


class _ProtocolWriter:
    """Writes replies and events to the protocol stream, one line at a time."""

    def __init__(self, stream: TextIO) -> None:
        self._stream = stream
        self._lock = threading.Lock()

    def write(self, reply: Ready | Done | Failed | Event) -> None:
        """Write *reply* as one line, or end the process on a broken pipe."""
        line = reply.model_dump_json() + "\n"
        with self._lock:
            try:
                _ = self._stream.write(line)
                self._stream.flush()
            except BrokenPipeError:
                _abandon()

    def event(self, name: str, data: Mapping[str, JsonValue] | None = None) -> None:
        """Write the event *name* with *data*, an empty object when ``None``."""
        self.write(Event(name=name, data={} if data is None else data))


class _Session:
    """Whether a command runs and whether the input has ended, under one lock."""

    def __init__(self) -> None:
        self._changed = threading.Condition()
        self._running = False
        self._input_ended = False

    def begin(self) -> bool:
        """Mark a command as running, unless the input has ended.

        Returns:
            Whether the command may run.
        """
        with self._changed:
            if self._input_ended:
                return False
            self._running = True
            return True

    def end(self) -> None:
        """Mark the running command as finished."""
        with self._changed:
            self._running = False
            self._changed.notify_all()

    def input_ended(self) -> bool:
        """Record the end of input, and wait up to 2 s for a running command.

        Returns:
            Whether a command still runs after the wait.
        """
        with self._changed:
            self._input_ended = True
            return not self._changed.wait_for(
                lambda: not self._running, timeout=_ABANDON_GRACE_SECONDS
            )
