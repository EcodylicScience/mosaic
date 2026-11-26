"""The program that imports lmt-analysis inside its own environment, and never imports mosaic.

lmt-analysis is GPL-3.0. A mosaic that imports it is one work with it, so
lmt-analysis runs in an environment of its own, from a checkout the user
clones, and this program is what runs inside that environment. Mosaic and this
file exchange command-line arguments, a JSON report file, the exit status and
the output of the script, which is what keeps them two programs. Nothing here
may grow a ``mosaic`` import, and the standard library is the only other
module that this file imports at all. lmt-analysis and its script are reached
through ``importlib`` by name, and nothing of either is copied or reimplemented
here.

Invoked as::

    <env>/bin/python lmt_rebuild_runner.py probe --checkout DIR --commit SHA \\
        [--builder NAME]... --report FILE
    <env>/bin/python lmt_rebuild_runner.py rebuild --checkout DIR --commit SHA \\
        [--builder NAME]... --database FILE --report FILE

Both subcommands first verify the checkout: ``git -C DIR rev-parse HEAD`` equals
``SHA``, and ``git -C DIR status --porcelain --untracked-files=no`` is empty.
They then put ``DIR/LMT`` and ``DIR/LMT/scripts`` on ``sys.path``, set
``MPLBACKEND=Agg``, import ``lmtanalysis.AnimalType`` and the script
``Rebuild_All_Events``, and resolve the builders. Without ``--builder`` they are
the modules of the script's ``eventClassList``, in its order. ``--builder NAME``
names ``lmtanalysis.NAME``, which must define ``flush`` and ``reBuildEvent``.
Builders are reported by that name, the last component of the module's name.
The program calls ``setAnimalType(AnimalType.MOUSE)``, as the script's own
``__main__`` does, and ``setEventClassList`` with the named builders.

``probe`` stops there. ``rebuild`` reads ``MAX(FRAMENUMBER)`` of the database's
``FRAME`` table and sets the script's module global ``maxT`` to
``max(maxT, ceil((MAX(FRAMENUMBER) + 1) / windowT) * windowT)``, so that the
script's windows cover a recording longer than its own range, and then calls
``process(DATABASE)``. The script reads ``minT``, ``maxT`` and ``windowT`` when
``process`` runs. The program never lowers ``maxT`` and never changes
``windowT``.

Exit status:

==== ==========================================================================
0    Done, and the report is written.
2    Usage (argparse). No report is written.
3    The environment: ``git`` cannot run, the commit differs, tracked files are
     modified, ``LMT/scripts`` is missing, an import fails, or the script lacks
     a name this program reads.
4    The input: an unknown builder, or a database without a readable ``FRAME``
     table.
5    ``process`` raised.
==== ==========================================================================

The report is JSON, written to a temporary file beside ``--report`` and renamed,
for every exit but 2. It holds the keys known when the program stopped:
``runner_protocol`` (``1``), ``commit`` (the checkout's HEAD, once ``git
rev-parse HEAD`` ran), ``available_builders`` (the module names of the script's
``eventClassList``, once the script imported), ``builders``, ``script_max_frame``,
``window_frames``, ``max_frame`` (the ``maxT`` used), ``last_frame``, and an
``error`` sentence for exits 3, 4 and 5. A report of exit 0 also holds
``seconds`` (``rebuild`` only), ``python`` and ``packages``, a mapping of each
installed distribution's name to its version, sorted by name.

Standard output and standard error carry the script's own output and
tracebacks, and this program writes no line of its own to standard output.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import sqlite3
import subprocess
import sys
import tempfile
import time
import traceback
from collections.abc import Callable, Sequence
from pathlib import Path
from types import ModuleType
from typing import Final, TypeGuard

RUNNER_PROTOCOL: Final = 1

EXIT_ENVIRONMENT: Final = 3
EXIT_INPUT: Final = 4
EXIT_PROCESS: Final = 5

_MAX_FRAME_NAME: Final = "maxT"
"""The script's module global that bounds the windows ``process`` loops over."""

_Report = dict[str, object]


class _Arguments(argparse.Namespace):
    """The parsed command line, typed."""

    command: str
    checkout: str
    commit: str
    builder: list[str]
    database: str
    report: str


class _Stop(Exception):
    """The run ends with *status*, and the report says why in one sentence."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status: int = status
        self.message: str = message


def _sentence(text: str) -> str:
    """Returns *text* on one line, ending in a full stop."""
    line = " ".join(text.split())
    return line if line.endswith(".") else f"{line}."


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lmt_rebuild_runner.py",
        description="Rebuild the events of a Live Mouse Tracker database.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("probe", "rebuild"):
        command = commands.add_parser(name)
        _ = command.add_argument("--checkout", required=True)
        _ = command.add_argument("--commit", required=True)
        _ = command.add_argument("--builder", action="append", default=[])
        _ = command.add_argument("--report", required=True)
        if name == "rebuild":
            _ = command.add_argument("--database", required=True)
    return parser


def _git(checkout: Path, *arguments: str) -> str:
    """Runs git in *checkout* and returns its output without surrounding space."""
    try:
        completed = subprocess.run(
            ["git", "-C", str(checkout), *arguments],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as failure:
        raise _Stop(
            EXIT_ENVIRONMENT, _sentence(f"git could not run: {failure}")
        ) from None
    if completed.returncode != 0:
        raise _Stop(
            EXIT_ENVIRONMENT,
            _sentence(
                f"git {' '.join(arguments)} failed in {checkout}: {completed.stderr}"
            ),
        )
    return completed.stdout.strip()


def _verify_checkout(checkout: Path, commit: str, report: _Report) -> None:
    head = _git(checkout, "rev-parse", "HEAD")
    report["commit"] = head
    if head != commit:
        raise _Stop(
            EXIT_ENVIRONMENT,
            f"The checkout is at {head}, and the run asks for {commit}.",
        )
    modified = _git(checkout, "status", "--porcelain", "--untracked-files=no")
    if modified:
        files = ", ".join(line.strip() for line in modified.splitlines())
        raise _Stop(
            EXIT_ENVIRONMENT,
            _sentence(f"The checkout has modified tracked files: {files}"),
        )
    if not (checkout / "LMT" / "scripts").is_dir():
        raise _Stop(
            EXIT_ENVIRONMENT,
            f"The checkout at {checkout} has no LMT/scripts directory.",
        )


def _import_lmt_analysis() -> tuple[ModuleType, ModuleType]:
    """Imports ``lmtanalysis.AnimalType`` and the script, and returns both."""
    try:
        animal_types = importlib.import_module("lmtanalysis.AnimalType")
        script = importlib.import_module("Rebuild_All_Events")
    except Exception as failure:
        # The script imports plotting and window-system modules, and any of them
        # can fail in more ways than an ImportError.
        traceback.print_exc()
        raise _Stop(
            EXIT_ENVIRONMENT,
            _sentence(
                f"Importing lmt-analysis failed: {type(failure).__name__}: {failure}"
            ),
        ) from failure
    return animal_types, script


def _attribute(owner: object, name: str, where: str) -> object:
    """Returns the attribute *name* of *owner*, or stops for a missing one."""
    found: object = getattr(owner, name, None)
    if found is None:
        raise _Stop(EXIT_ENVIRONMENT, f"{where} defines no {name}.")
    return found


def _function(owner: object, name: str, where: str) -> Callable[..., object]:
    found = _attribute(owner, name, where)
    if not callable(found):
        raise _Stop(EXIT_ENVIRONMENT, f"{where} defines {name}, which is not callable.")
    return found


def _integer(owner: object, name: str, where: str) -> int:
    found = _attribute(owner, name, where)
    if not isinstance(found, int) or isinstance(found, bool):
        raise _Stop(
            EXIT_ENVIRONMENT, f"{where} defines {name}, which is not an integer."
        )
    return found


def _is_object_sequence(value: object) -> TypeGuard[Sequence[object]]:
    return isinstance(value, (list, tuple))


def _event_classes(script: ModuleType) -> list[ModuleType]:
    found = _attribute(script, "eventClassList", script.__name__)
    if not _is_object_sequence(found):
        raise _Stop(
            EXIT_ENVIRONMENT,
            f"{script.__name__} defines eventClassList, which is not a list.",
        )
    modules = [item for item in found if isinstance(item, ModuleType)]
    if len(modules) != len(found):
        raise _Stop(
            EXIT_ENVIRONMENT,
            f"{script.__name__} defines eventClassList with an item that is not a module.",
        )
    return modules


def _builder_name(module: ModuleType) -> str:
    """Returns the last component of *module*'s name, as ``--builder`` spells it."""
    return module.__name__.rpartition(".")[2]


def _named_builder(name: str, available: Sequence[str]) -> ModuleType:
    """Returns ``lmtanalysis.<name>``, which must define ``flush`` and ``reBuildEvent``."""
    unknown = _Stop(
        EXIT_INPUT,
        _sentence(
            f"{name} is not a builder of lmtanalysis. The script's builders are "
            f"{', '.join(available)}"
        ),
    )
    if not name.isidentifier():
        raise unknown
    module_name = f"lmtanalysis.{name}"
    try:
        module = importlib.import_module(module_name)
    except Exception as failure:
        if isinstance(failure, ModuleNotFoundError) and failure.name == module_name:
            raise unknown from None
        traceback.print_exc()
        raise _Stop(
            EXIT_ENVIRONMENT,
            _sentence(
                f"Importing {module_name} failed: {type(failure).__name__}: {failure}"
            ),
        ) from failure
    if not all(
        callable(getattr(module, entry, None)) for entry in ("flush", "reBuildEvent")
    ):
        raise _Stop(
            EXIT_INPUT,
            f"{module_name} does not define both flush and reBuildEvent, so it is not a builder.",
        )
    return module


def _last_frame(database: Path) -> int:
    """Returns ``MAX(FRAMENUMBER)`` of the database's ``FRAME`` table."""
    unreadable = f"The database {database} has no readable FRAME table"
    try:
        connection = sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True)
        try:
            row: tuple[object, ...] | None = connection.execute(
                "SELECT MAX(FRAMENUMBER) FROM FRAME"
            ).fetchone()
        finally:
            connection.close()
    except sqlite3.Error as failure:
        raise _Stop(EXIT_INPUT, _sentence(f"{unreadable}: {failure}")) from failure
    last = None if row is None else row[0]
    if not isinstance(last, int):
        raise _Stop(EXIT_INPUT, _sentence(f"{unreadable}: it holds no frame"))
    return last


def _packages() -> dict[str, str]:
    """Returns each installed distribution's version by name, sorted by name."""
    versions = {
        distribution.name: distribution.version
        for distribution in importlib.metadata.distributions()
    }
    return dict(sorted(versions.items()))


def _write_report(path: Path, report: _Report) -> None:
    """Writes *report* to *path* through a temporary file beside it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(report, handle)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _execute(arguments: _Arguments, report: _Report) -> None:
    """Runs the subcommand and fills *report*, or raises :class:`_Stop`."""
    checkout = Path(arguments.checkout)
    _verify_checkout(checkout, arguments.commit, report)

    sys.path[:0] = [str(checkout / "LMT"), str(checkout / "LMT" / "scripts")]
    os.environ["MPLBACKEND"] = "Agg"
    animal_types, script = _import_lmt_analysis()

    available = [_builder_name(module) for module in _event_classes(script)]
    report["available_builders"] = available

    animal_type = _attribute(animal_types, "AnimalType", animal_types.__name__)
    mouse = _attribute(animal_type, "MOUSE", "AnimalType")
    set_animal_type = _function(script, "setAnimalType", script.__name__)
    set_event_class_list = _function(script, "setEventClassList", script.__name__)
    process = _function(script, "process", script.__name__)
    # The script's ``process`` reads all three, so a script without one is not
    # the script this program was written for.
    _ = _integer(script, "minT", script.__name__)
    script_max_frame = _integer(script, _MAX_FRAME_NAME, script.__name__)
    window_frames = _integer(script, "windowT", script.__name__)
    if window_frames <= 0:
        raise _Stop(EXIT_ENVIRONMENT, f"{script.__name__} defines a windowT below 1.")

    named = [_named_builder(name, available) for name in arguments.builder]
    _ = set_animal_type(mouse)
    if named:
        _ = set_event_class_list(named)
    report["builders"] = (
        [_builder_name(module) for module in named] if named else available
    )
    report["script_max_frame"] = script_max_frame
    report["window_frames"] = window_frames
    report["max_frame"] = script_max_frame

    if arguments.command == "rebuild":
        database = Path(arguments.database)
        last_frame = _last_frame(database)
        report["last_frame"] = last_frame
        covering = -(-(last_frame + 1) // window_frames) * window_frames
        max_frame = max(script_max_frame, covering)
        setattr(script, _MAX_FRAME_NAME, max_frame)
        report["max_frame"] = max_frame
        started = time.monotonic()
        try:
            _ = process(str(database))
        except Exception as failure:
            traceback.print_exc()
            raise _Stop(
                EXIT_PROCESS,
                _sentence(f"process raised {type(failure).__name__}: {failure}"),
            ) from failure
        report["seconds"] = time.monotonic() - started

    report["python"] = platform.python_version()
    report["packages"] = _packages()


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv, namespace=_Arguments())
    report: _Report = {"runner_protocol": RUNNER_PROTOCOL}
    status = 0
    try:
        _execute(arguments, report)
    except _Stop as stop:
        report["error"] = stop.message
        status = stop.status
    _write_report(Path(arguments.report), report)
    return status


if __name__ == "__main__":
    sys.exit(main())
