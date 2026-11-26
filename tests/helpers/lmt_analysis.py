"""A stand-in for the lmt-analysis checkout that the rebuild runner imports.

:func:`write_fake_lmt_analysis` writes a git repository with the names the
runner reads from lmt-analysis: the ``lmtanalysis`` package with
``AnimalType`` and two builder modules, and ``LMT/scripts/Rebuild_All_Events.py``
with ``eventClassList``, ``process``, ``setAnimalType``, ``setEventClassList``,
``minT``, ``maxT`` and ``windowT``. The files are this suite's own and are
written from those names alone. They share nothing else with lmt-analysis,
which is GPL-3.0 and is not part of this repository.

The stand-in's ``process`` loops over windows of ``windowT`` frames from
``minT`` while below the module's ``maxT``, which it reads when it is called.
Each builder deletes its events in ``flush``, and ``reBuildEvent`` inserts one
event for each window that has detections, ending at the window's last
detected frame. ``process`` then writes one ``LOG`` row. Its ``maxT`` is three
windows of 100 frames, so a test database reaches past it.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Final, NamedTuple, Protocol

import pytest

import mosaic.tracking.ops.rebuild_lmt_events as rebuild_module
from mosaic.tracking.common.lmt_env import (
    LMT_ANALYSIS_DIR_VAR,
    LMT_ENV,
    LMT_PYTHON_VAR,
    LmtRebuildError,
    LmtRunnerReport,
)
from tests.helpers.environment import isolate_tool_location

__all__ = [
    "InstalledLmtAnalysis",
    "LmtAnalysisInstaller",
    "fail_lmt_rebuilds",
    "git_in_checkout",
    "lmt_analysis_installer",
    "write_fake_lmt_analysis",
]

_ANIMAL_TYPE: Final = '''"""The animal types of the stand-in."""

from enum import Enum


class AnimalType(Enum):
    MOUSE = 1
    RAT = 2
'''

_BUILDER: Final = '''"""A stand-in event builder: one event for each window that has detections."""

EVENT_NAME = {event_name!r}


def flush(connection):
    connection.execute("DELETE FROM EVENT WHERE NAME = ?", (EVENT_NAME,))


def reBuildEvent(connection, tmin, tmax):
    first, last = connection.execute(
        "SELECT MIN(FRAMENUMBER), MAX(FRAMENUMBER) FROM DETECTION "
        "WHERE FRAMENUMBER >= ? AND FRAMENUMBER < ?",
        (tmin, tmax),
    ).fetchone()
    if last is None:
        return
    connection.execute(
        "INSERT INTO EVENT (NAME, DESCRIPTION, STARTFRAME, ENDFRAME) "
        "VALUES (?, ?, ?, ?)",
        (EVENT_NAME, EVENT_NAME, first, last),
    )
'''

_SCRIPT: Final = '''"""A stand-in for the script that rebuilds the events of a database."""

import os
import sqlite3
import time
from datetime import datetime

from lmtanalysis import BuildEventDetection, BuildEventFlickering

RAISES = {raises!r}
WAIT_FOR = {wait_for!r}

eventClassList = [BuildEventDetection, BuildEventFlickering]
minT = 0
windowT = 100
maxT = 3 * windowT
animalType = None


def setAnimalType(aType):
    global animalType
    animalType = aType


def setEventClassList(classList):
    global eventClassList
    eventClassList = classList


def process(file):
    connection = sqlite3.connect(file)
    try:
        if WAIT_FOR is not None:
            print("waiting for " + WAIT_FOR, flush=True)
            while not os.path.exists(WAIT_FOR):
                time.sleep(0.05)
        for builder in eventClassList:
            builder.flush(connection)
        currentT = minT
        while currentT < maxT:
            for builder in eventClassList:
                builder.reBuildEvent(connection, currentT, currentT + windowT)
            currentT += windowT
        connection.execute(
            "CREATE TABLE IF NOT EXISTS LOG ( id INTEGER PRIMARY KEY "
            "AUTOINCREMENT UNIQUE, process TEXT, version TEXT, date TEXT, "
            "tmin INTEGER, tmax INTEGER)"
        )
        connection.execute(
            "INSERT INTO LOG (process, version, date, tmin, tmax) "
            "VALUES (?, ?, ?, ?, ?)",
            ("Stand-in rebuild", "0", datetime.now().isoformat(), minT, maxT),
        )
        connection.commit()
    finally:
        connection.close()
    if RAISES:
        raise RuntimeError("the stand-in rebuild was asked to fail")
'''

_BUILDERS: Final = (
    ("BuildEventDetection", "Fake Detection"),
    ("BuildEventFlickering", "Fake Flickering"),
)
_COMMITTER: Final = ("-c", "user.name=mosaic-tests", "-c", "user.email=tests@invalid")


def git_in_checkout(root: Path, *arguments: str) -> str:
    """Run git in the checkout *root* as the tests' committer and return its output.

    Raises:
        subprocess.CalledProcessError: git exits with a non-zero status.
    """
    completed = subprocess.run(
        ["git", "-C", str(root), *_COMMITTER, "-c", "commit.gpgsign=false", *arguments],
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def write_fake_lmt_analysis(
    root: Path, *, raises: bool = False, wait_for: Path | None = None
) -> str:
    """Write the stand-in checkout to *root*, commit it, and return its commit.

    Args:
        root: The checkout's directory. It is created if it is missing.
        raises: Whether the script's ``process`` raises after it rebuilt the
            events and wrote its ``LOG`` row.
        wait_for: A file that ``process`` waits for before its first window.
            It prints a line and then waits until the file exists.

    Returns:
        The hash that ``git rev-parse HEAD`` prints in the checkout.
    """
    package = root / "LMT" / "lmtanalysis"
    scripts = root / "LMT" / "scripts"
    package.mkdir(parents=True)
    scripts.mkdir(parents=True)
    (package / "__init__.py").write_text('"""The stand-in package."""\n')
    (package / "AnimalType.py").write_text(_ANIMAL_TYPE)
    for module, event_name in _BUILDERS:
        (package / f"{module}.py").write_text(_BUILDER.format(event_name=event_name))
    (scripts / "Rebuild_All_Events.py").write_text(
        _SCRIPT.format(
            raises=raises, wait_for=None if wait_for is None else str(wait_for)
        )
    )
    git_in_checkout(root, "init", "-q")
    git_in_checkout(root, "add", "-A")
    git_in_checkout(root, "commit", "-q", "-m", "Write the stand-in")
    return git_in_checkout(root, "rev-parse", "HEAD")


class InstalledLmtAnalysis(NamedTuple):
    """A stand-in checkout that the environment variables name."""

    root: Path
    commit: str


class LmtAnalysisInstaller(Protocol):
    """Writes a stand-in checkout and points the location ladder at it."""

    def __call__(
        self, *, raises: bool = False, wait_for: Path | None = None
    ) -> InstalledLmtAnalysis: ...


def lmt_analysis_installer(
    monkeypatch: pytest.MonkeyPatch, directory: Path
) -> LmtAnalysisInstaller:
    """Point the location ladder at this Python and return the stand-in's writer.

    Each call of the returned writer commits a new stand-in checkout under
    *directory* and names it in the environment, so the last one written is the
    checkout that a run finds.
    """
    isolate_tool_location(monkeypatch, LMT_ENV, found={})
    monkeypatch.setenv(LMT_PYTHON_VAR, sys.executable)
    count = [0]

    def installed(
        *, raises: bool = False, wait_for: Path | None = None
    ) -> InstalledLmtAnalysis:
        count[0] += 1
        root = directory / f"lmt-analysis-{count[0]}"
        commit = write_fake_lmt_analysis(root, raises=raises, wait_for=wait_for)
        monkeypatch.setenv(LMT_ANALYSIS_DIR_VAR, str(root))
        return InstalledLmtAnalysis(root, commit)

    return installed


def fail_lmt_rebuilds(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the rebuild op's runner exit with status 5 after it wrote its logs.

    The failed attempt leaves ``rebuild.log`` and the report in the entry's
    directory, as a runner whose ``process`` raised does.
    """

    def exiting_with_five(
        database: Path,
        commit: str,
        builders: Sequence[str] | None,
        *,
        report_path: Path,
        log_path: Path,
        cancel_check: Callable[[], bool] | None,
        on_line: Callable[[str], None],
    ) -> LmtRunnerReport:
        _ = log_path.write_text("the failed attempt's output\n")
        _ = report_path.write_text('{"runner_protocol": 1, "error": "failed"}')
        raise LmtRebuildError(["runner"], 5, "", "", reason="the attempt failed.")

    monkeypatch.setattr(rebuild_module, "run_lmt_rebuild", exiting_with_five)
