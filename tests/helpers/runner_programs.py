"""Runner programs, imported into the test process or written for one test.

A runner program runs in an external tool's environment and is launched by path.
Its directory is then ``sys.path[0]``, and it imports its siblings by their bare
names. Mosaic imports the same sibling files as modules of mosaic's packages. A
test process can therefore import one file under two module names.
:func:`import_runner_program` adds the bare names for one import and removes them
afterwards.

:func:`write_fake_runner` writes a runner that serves handlers chosen by a test
through ``runner_wire``, as a tool's runner serves its commands.
"""

from __future__ import annotations

import importlib
import shutil
import sys
import textwrap
from collections.abc import Generator, Iterable
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Final

from tests.helpers.paths import SOURCE_ROOT

_RUNNER_WIRE: Final = (
    SOURCE_ROOT / "behavior" / "feature_library" / "external" / "runner_wire.py"
)
"""The runner side of the line protocol, which a fake runner imports from a copy."""

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


@contextmanager
def import_runner_program(
    directory: Path, program: str, siblings: Iterable[str] = ()
) -> Generator[ModuleType]:
    """Import the runner *program* from *directory* under its bare name.

    *directory* is inserted at the front of ``sys.path`` for the import, unless
    it is on the path already. On exit the inserted entry is removed, and
    *program* and each of *siblings* are removed from ``sys.modules``. Left in
    place, the bare names would resolve for the rest of the session. Each
    sibling would also be in ``sys.modules`` twice, under its bare name and its
    package name, and each copy would define classes distinct from the other's.

    Args:
        directory: The directory that contains the program and its siblings.
        program: The program's module name, without ``.py``.
        siblings: The bare names of the modules that the program imports from
            *directory*.

    Yields:
        The imported program.
    """
    entry = str(directory)
    inserted = entry not in sys.path
    if inserted:
        sys.path.insert(0, entry)
    try:
        yield importlib.import_module(program)
    finally:
        if inserted:
            sys.path.remove(entry)
        for name in (program, *siblings):
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
    beside it. Its ``main()`` claims standard output, assigns the protocol stream
    to the module global ``PROTOCOL`` and serves ``DISPATCH``.

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
    for source in (_RUNNER_WIRE, *siblings):
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
