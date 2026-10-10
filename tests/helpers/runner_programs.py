"""Runner programs, imported into the test process or written for one test.

A runner program runs in an external tool's environment and is launched by path.
Its directory is then ``sys.path[0]``, and it imports its siblings by their bare
names. Mosaic imports the same sibling files as modules of mosaic's packages. A
test process can therefore import one file under two module names.
:func:`import_runner_program` adds the bare names for one import and removes them
afterwards.

:func:`write_fake_runner` writes a runner that serves handlers chosen by a test
through ``runner_wire``, as a tool's runner serves its commands, and
:data:`FAKE_HANDLERS` contains the handlers that the tests of the wire share. A
fake runner records its start and its commands in a journal with
:func:`journal_source`, which a test reads through :class:`RunnerJournal`.
:class:`EventRecorder` is the ``Emit`` of a handler called in the test process.
:func:`assert_ready_then_shutdown` checks that a started runner program answers
``Ready`` and ``shutdown``.
"""

from __future__ import annotations

import importlib
import shutil
import sys
import textwrap
from collections.abc import Generator, Iterable, Mapping
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Final

from pydantic import JsonValue, TypeAdapter

from tests.helpers.paths import RUNNER_WIRE, SOURCE_ROOT
from tests.helpers.processes import wait_for

if TYPE_CHECKING:
    from mosaic.behavior.feature_library.external.runner_wire import Ready
    from mosaic.core.pipeline.subprocess_util import LineChild

RUNNER_LINE_SECONDS: Final = 60.0
"""How long a test waits for a runner's line, enough for a start under load."""

FAKE_HANDLERS: Final = """
from pydantic import BaseModel


class Echoed(BaseModel):
    text: str


def echo(payload, emit):
    return Echoed(text=str(payload["text"]))


def count(payload, emit):
    for number in range(int(payload["up_to"])):
        emit("counted", {"number": number})


def nap(payload, emit):
    emit("napping")
    time.sleep(float(payload["seconds"]))


def fail(payload, emit):
    raise ValueError("the handler failed")


def interrupt(payload, emit):
    raise KeyboardInterrupt


DISPATCH = {
    "echo": echo,
    "count": count,
    "nap": nap,
    "fail": fail,
    "interrupt": interrupt,
}
"""
"""Handler source for :func:`write_fake_runner` that the tests of the wire share.

``echo`` returns its ``text``, ``count`` emits ``counted`` up to ``up_to``, and
``nap`` emits ``napping`` and sleeps for ``seconds``. ``fail`` raises ValueError,
and ``interrupt`` raises KeyboardInterrupt. A test appends further handlers, and
adds them to ``DISPATCH``.
"""

_FAKE_RUNNER: Final = '''\
"""A runner program written by a test."""

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import runner_wire

PROTOCOL = None

{handlers}


def main():
    global PROTOCOL
    PROTOCOL = runner_wire.claim_stdout()
    runner_wire.log_to_stderr()
    runner_wire.serve(
        DISPATCH,
        {tool!r},
        {version!r},
        protocol=PROTOCOL,
        heartbeat_seconds={heartbeat_seconds!r},
    )


if __name__ == "__main__":
    main()
'''

_JOURNAL_SOURCE: Final = '''\
import json

JOURNAL = Path({path!r})


def record(command, **details):
    """Append the command, its details and this process's id to the journal."""
    with JOURNAL.open("a") as journal:
        journal.write(json.dumps({{"pid": os.getpid(), "command": command, **details}}))
        journal.write("\\n")


record("start")
'''

_RECORDS: Final[TypeAdapter[dict[str, JsonValue]]] = TypeAdapter(dict[str, JsonValue])
"""Reads one line of a journal."""


@contextmanager
def import_runner_program(
    program: Path, siblings: Iterable[str] = ()
) -> Generator[ModuleType]:
    """Import the runner *program* under its bare name.

    The program's directory is inserted at the front of ``sys.path`` for the
    import, unless it is on the path already. On exit the inserted entry is
    removed, and the program and each of *siblings* are removed from
    ``sys.modules``. Left in place, the bare names would resolve for the rest of
    the session. Each sibling would also be in ``sys.modules`` twice, under its
    bare name and its package name, and each copy would define classes distinct
    from the other's.

    Args:
        program: The program's file.
        siblings: The bare names of the modules that the program imports from
            its directory.

    Yields:
        The imported program.
    """
    entry = str(program.parent)
    inserted = entry not in sys.path
    if inserted:
        sys.path.insert(0, entry)
    try:
        yield importlib.import_module(program.stem)
    finally:
        if inserted:
            sys.path.remove(entry)
        for name in (program.stem, *siblings):
            _ = sys.modules.pop(name, None)


def write_fake_runner(
    directory: Path,
    handlers_source: str,
    *,
    name: str = "fake_runner",
    tool: str = "fake",
    version: str = "1.0",
    heartbeat_seconds: float = 30.0,
    siblings: Iterable[Path] = (),
) -> Path:
    """Write a runner program that serves the handlers defined in *handlers_source*.

    The program imports ``runner_wire`` by its bare name, from a copy written
    beside it. Its ``main()`` claims standard output and assigns the protocol
    stream to the module global ``PROTOCOL``. It then logs to standard error and
    serves ``DISPATCH``.

    Args:
        directory: The directory to write the program and the copy into.
        handlers_source: Module-level source that defines ``DISPATCH``, the
            mapping from each command to its handler. It is dedented, and runs
            after imports of ``os``, ``subprocess``, ``sys``, ``threading``,
            ``time``, ``Path`` and ``runner_wire``.
        name: The program's module name.
        tool: The tool name that the program's ``Ready`` reports.
        version: The version that the program's ``Ready`` reports.
        heartbeat_seconds: The interval of the heartbeats during a command.
        siblings: Further modules that the handlers import by their bare names,
            such as a tool's protocol module. Each is copied beside the program.

    Returns:
        The program's path.
    """
    for source in (SOURCE_ROOT / RUNNER_WIRE, *siblings):
        _ = shutil.copyfile(source, directory / source.name)
    program = directory / f"{name}.py"
    _ = program.write_text(
        _FAKE_RUNNER.format(
            handlers=textwrap.dedent(handlers_source).strip(),
            tool=tool,
            version=version,
            heartbeat_seconds=heartbeat_seconds,
        )
    )
    return program


class EventRecorder:
    """Records each event that a handler emits, as ``runner_wire``'s ``Emit``.

    Attributes:
        events: Each event's name and data, in the order emitted.
    """

    def __init__(self) -> None:
        self.events: list[tuple[str, Mapping[str, JsonValue]]] = []

    def __call__(self, name: str, data: Mapping[str, JsonValue] | None = None) -> None:
        """Record the event *name* with *data*, an empty object when ``None``."""
        self.events.append((name, {} if data is None else data))


def journal_source(journal: Path) -> str:
    """Return handler source that records a fake runner's start and commands.

    The source defines ``JOURNAL``, the path *journal*, and ``record(command,
    **details)``, which appends one JSON line with ``pid``, ``command`` and
    *details*. It then records ``start``. Place it at the front of the source
    passed to :func:`write_fake_runner`, and read the journal with
    :class:`RunnerJournal`.
    """
    return _JOURNAL_SOURCE.format(path=str(journal))


class RunnerJournal:
    """The journal of the fake runners of one test, which :func:`journal_source` writes.

    Attributes:
        path: The journal file.
    """

    def __init__(self, path: Path) -> None:
        self.path = path

    def records(self) -> list[dict[str, JsonValue]]:
        """Each line of the journal, in the order written."""
        if not self.path.exists():
            return []
        return [
            _RECORDS.validate_json(line) for line in self.path.read_text().splitlines()
        ]

    def recorded(self, command: str) -> bool:
        """Whether any runner recorded *command*."""
        return any(record["command"] == command for record in self.records())

    def starts(self) -> list[int]:
        """The process id of each runner started so far."""
        pids: list[int] = []
        for record in self.records():
            pid = record["pid"]
            assert isinstance(pid, int), record
            if record["command"] == "start":
                pids.append(pid)
        return pids

    def commands(self, pid: int) -> list[str]:
        """The commands that the runner *pid* recorded, after its start."""
        return [
            str(record["command"])
            for record in self.records()
            if record["pid"] == pid and record["command"] != "start"
        ]

    def details(self, command: str, key: str) -> list[JsonValue]:
        """The detail *key* of each *command* that a runner recorded, in order."""
        return [
            record[key] for record in self.records() if record["command"] == command
        ]


def assert_ready_then_shutdown(child: LineChild) -> Ready:
    """Assert that the runner program *child* answers ``Ready``, then ``shutdown``.

    The runner then exits with status 0.

    Returns:
        The runner's ``Ready``.
    """
    # Importing the feature library loads every feature, which most of the
    # suite's users of these helpers do not need.
    from mosaic.behavior.feature_library.external.runner_wire import (
        REPLIES,
        SHUTDOWN,
        Done,
        Ready,
        Request,
    )

    ready = child.next_line(timeout=RUNNER_LINE_SECONDS)
    assert ready is not None, child.log_path.read_text()
    reply = REPLIES.validate_json(ready)
    assert isinstance(reply, Ready), reply

    child.send(Request(command=SHUTDOWN).model_dump_json())

    done = child.next_line(timeout=RUNNER_LINE_SECONDS)
    assert done is not None, child.log_path.read_text()
    assert REPLIES.validate_json(done) == Done()
    wait_for(lambda: not child.alive, RUNNER_LINE_SECONDS, "the runner did not exit")
    assert child.exit_code == 0
    return reply
