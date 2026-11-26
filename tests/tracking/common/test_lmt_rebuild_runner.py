"""The lmt-analysis runner program, run as a subprocess on a stand-in checkout.

The runner is the one file that imports lmt-analysis, and it imports no mosaic
module. Each test runs it under the suite's interpreter on a stand-in checkout
that :func:`~tests.helpers.write_fake_lmt_analysis` commits, and on a database
that :func:`~tests.helpers.write_lmt_sqlite` writes. CI needs no lmt-analysis.
"""

from __future__ import annotations

import importlib.metadata
import platform
import sqlite3
import sys
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final

import pytest
from pydantic import JsonValue, TypeAdapter

from tests.helpers import (
    LMT_REBUILD_RUNNER,
    SOURCE_ROOT,
    LmtRecording,
    computed_dynamic_imports,
    default_lmt_recording,
    git_in_checkout,
    imports_of,
    imports_reaching,
    run_child,
    source_tree,
    write_fake_lmt_analysis,
    write_lmt_sqlite,
)

_RUNNER: Final = SOURCE_ROOT / LMT_REBUILD_RUNNER
_REPORT: Final = TypeAdapter(dict[str, JsonValue])
_BUILDERS: Final = ["BuildEventDetection", "BuildEventFlickering"]
_STAND_IN_MAX_FRAME: Final = 300
_WINDOW_FRAMES: Final = 100
_FIRST_UNCOVERED_MULTIPLE: Final = 500
"""The multiple of the window above the last frame, 401, of the default recording."""


@dataclass(frozen=True, slots=True)
class _Checkout:
    """A committed stand-in checkout and its HEAD."""

    root: Path
    commit: str


@dataclass(frozen=True, slots=True)
class _Outcome:
    """What one run of the runner left: its exit status, output and report."""

    returncode: int
    stderr: str
    report: dict[str, JsonValue]


@pytest.fixture
def checkout(tmp_path: Path) -> _Checkout:
    """A stand-in checkout under *tmp_path*."""
    root = tmp_path / "lmt-analysis"
    return _Checkout(root, write_fake_lmt_analysis(root))


def _database(tmp_path: Path, recording: LmtRecording | None = None) -> Path:
    """Write *recording*, the default one when ``None``, and return its path."""
    chosen = recording or default_lmt_recording()
    return write_lmt_sqlite(
        tmp_path / "recording.sqlite",
        animals=chosen.animals,
        frames=chosen.frames,
        events=chosen.events,
        detections=chosen.detections,
        log=chosen.log,
    )


def _run(
    subcommand: str,
    checkout: _Checkout,
    tmp_path: Path,
    *,
    commit: str | None = None,
    builders: Sequence[str] = (),
    database: Path | None = None,
    path: str | None = None,
) -> _Outcome:
    """Run the runner's *subcommand* and read the report it writes."""
    report = tmp_path / "report.json"
    argv = [
        sys.executable,
        str(_RUNNER),
        subcommand,
        "--checkout",
        str(checkout.root),
        "--commit",
        commit or checkout.commit,
        "--report",
        str(report),
    ]
    for builder in builders:
        argv += ["--builder", builder]
    if database is not None:
        argv += ["--database", str(database)]
    env = None
    if path is not None:
        env = {"PATH": path}
    completed = run_child(argv, env=env)
    written = _REPORT.validate_json(report.read_text()) if report.exists() else {}
    return _Outcome(completed.returncode, completed.stderr, written)


def _query(database: Path, statement: str) -> list[tuple[object, ...]]:
    connection = sqlite3.connect(database)
    try:
        return [tuple(row) for row in connection.execute(statement).fetchall()]
    finally:
        connection.close()


def _fake_event_names(database: Path) -> set[str]:
    """The names of the events that the stand-in's builders wrote."""
    rows = _query(database, "SELECT NAME FROM EVENT WHERE NAME LIKE 'Fake %'")
    names = {name for (name,) in rows if isinstance(name, str)}
    assert len(names) <= len(rows)
    return names


def _fake_event_ends(database: Path) -> list[int]:
    """The ``ENDFRAME`` of each event that the stand-in's builders wrote."""
    rows = _query(database, "SELECT ENDFRAME FROM EVENT WHERE NAME LIKE 'Fake %'")
    ends = [end for (end,) in rows if isinstance(end, int)]
    assert len(ends) == len(rows)
    return ends


def _log_rows(database: Path) -> int:
    (row,) = _query(database, "SELECT COUNT(*) FROM LOG")
    count = row[0]
    assert isinstance(count, int)
    return count


def _last_frame(database: Path) -> int:
    (row,) = _query(database, "SELECT MAX(FRAMENUMBER) FROM FRAME")
    last = row[0]
    assert isinstance(last, int)
    return last


def test_probe_reports_what_the_script_holds(
    checkout: _Checkout, tmp_path: Path
) -> None:
    outcome = _run("probe", checkout, tmp_path)

    assert outcome.returncode == 0, outcome.stderr
    report = outcome.report
    assert report["runner_protocol"] == 1
    assert report["commit"] == checkout.commit
    assert report["builders"] == _BUILDERS
    assert report["max_frame"] == report["script_max_frame"] == _STAND_IN_MAX_FRAME
    assert report["window_frames"] == _WINDOW_FRAMES
    assert report["python"] == platform.python_version()
    packages = report["packages"]
    assert isinstance(packages, dict)
    assert list(packages) == sorted(packages)
    assert packages["pytest"] == importlib.metadata.version("pytest")
    assert "error" not in report


def test_probe_resolves_a_named_builder(checkout: _Checkout, tmp_path: Path) -> None:
    outcome = _run("probe", checkout, tmp_path, builders=["BuildEventFlickering"])

    assert outcome.returncode == 0, outcome.stderr
    assert outcome.report["builders"] == ["BuildEventFlickering"]


def test_the_report_of_another_commit_names_the_checkouts_head(
    checkout: _Checkout, tmp_path: Path
) -> None:
    git_in_checkout(checkout.root, "commit", "--allow-empty", "-m", "Move the head")
    head = git_in_checkout(checkout.root, "rev-parse", "HEAD")
    assert head != checkout.commit

    outcome = _run("probe", checkout, tmp_path)

    assert outcome.returncode == 3
    assert outcome.report["commit"] == head
    assert outcome.report["runner_protocol"] == 1
    error = outcome.report["error"]
    assert isinstance(error, str)
    assert head in error
    assert checkout.commit in error
    assert "available_builders" not in outcome.report


def test_a_modified_tracked_file_is_refused(
    checkout: _Checkout, tmp_path: Path
) -> None:
    script = checkout.root / "LMT" / "scripts" / "Rebuild_All_Events.py"
    script.write_text(script.read_text() + "\n# edited\n")

    outcome = _run("rebuild", checkout, tmp_path, database=_database(tmp_path))

    assert outcome.returncode == 3
    error = outcome.report["error"]
    assert isinstance(error, str)
    assert "Rebuild_All_Events.py" in error
    assert outcome.report["commit"] == checkout.commit
    assert "available_builders" not in outcome.report


def test_a_checkout_without_the_scripts_directory_is_refused(
    checkout: _Checkout, tmp_path: Path
) -> None:
    git_in_checkout(checkout.root, "rm", "-r", "-q", "LMT/scripts")
    git_in_checkout(checkout.root, "commit", "-q", "-m", "Remove the scripts")
    head = git_in_checkout(checkout.root, "rev-parse", "HEAD")

    outcome = _run("probe", checkout, tmp_path, commit=head)

    assert outcome.returncode == 3
    error = outcome.report["error"]
    assert isinstance(error, str)
    assert "LMT/scripts" in error
    assert outcome.report["commit"] == head
    assert "available_builders" not in outcome.report


def test_an_unknown_builder_is_refused_with_the_known_ones(
    checkout: _Checkout, tmp_path: Path
) -> None:
    outcome = _run("probe", checkout, tmp_path, builders=["BuildEventNowhere"])

    assert outcome.returncode == 4
    error = outcome.report["error"]
    assert isinstance(error, str)
    assert "BuildEventNowhere" in error
    assert outcome.report["available_builders"] == _BUILDERS
    assert outcome.report["commit"] == checkout.commit


def test_a_missing_git_is_refused_with_neither_commit_nor_builders(
    checkout: _Checkout, tmp_path: Path
) -> None:
    empty = tmp_path / "empty-path"
    empty.mkdir()

    outcome = _run("probe", checkout, tmp_path, path=str(empty))

    assert outcome.returncode == 3
    error = outcome.report["error"]
    assert isinstance(error, str)
    assert "git" in error
    assert outcome.report["runner_protocol"] == 1
    assert "commit" not in outcome.report
    assert "available_builders" not in outcome.report


def test_a_usage_error_exits_2() -> None:
    completed = run_child([sys.executable, str(_RUNNER), "rebuild"])

    assert completed.returncode == 2
    assert "usage:" in completed.stderr


def test_rebuild_inserts_the_events_and_a_log_row(
    checkout: _Checkout, tmp_path: Path
) -> None:
    database = _database(tmp_path)
    logged = _log_rows(database)

    outcome = _run("rebuild", checkout, tmp_path, database=database)

    assert outcome.returncode == 0, outcome.stderr
    assert _log_rows(database) == logged + 1
    names = _fake_event_names(database)
    assert names == {"Fake Detection", "Fake Flickering"}
    report = outcome.report
    assert report["builders"] == _BUILDERS
    assert report["last_frame"] == _last_frame(database)
    seconds = report["seconds"]
    assert isinstance(seconds, float)
    assert seconds >= 0


def test_a_database_past_the_scripts_frame_range_gets_whole_windows_more(
    checkout: _Checkout, tmp_path: Path
) -> None:
    database = _database(tmp_path)
    last = _last_frame(database)
    assert last > _STAND_IN_MAX_FRAME

    outcome = _run("rebuild", checkout, tmp_path, database=database)

    assert outcome.returncode == 0, outcome.stderr
    assert outcome.report["max_frame"] == _FIRST_UNCOVERED_MULTIPLE
    assert outcome.report["script_max_frame"] == _STAND_IN_MAX_FRAME
    ends = _fake_event_ends(database)
    assert max(ends) == last
    assert all(end <= last for end in ends)


def test_a_shorter_database_runs_the_scripts_own_range(
    checkout: _Checkout, tmp_path: Path
) -> None:
    """The frames up to 180 are covered by windows to 200, below the script's 300.

    The runner then keeps the script's value and never lowers it.
    """
    recording = default_lmt_recording()
    short = replace(
        recording,
        frames=[frame for frame in recording.frames if frame.number <= 180],
        detections=[d for d in recording.detections if d.frame <= 180],
    )
    database = _database(tmp_path, short)

    outcome = _run("rebuild", checkout, tmp_path, database=database)

    assert outcome.returncode == 0, outcome.stderr
    assert outcome.report["max_frame"] == outcome.report["script_max_frame"]
    assert outcome.report["max_frame"] == _STAND_IN_MAX_FRAME
    last = _last_frame(database)
    ends = _fake_event_ends(database)
    assert ends
    assert all(end <= last for end in ends)


def test_a_named_builder_rebuilds_only_its_events(
    checkout: _Checkout, tmp_path: Path
) -> None:
    database = _database(tmp_path)

    outcome = _run(
        "rebuild",
        checkout,
        tmp_path,
        builders=["BuildEventFlickering"],
        database=database,
    )

    assert outcome.returncode == 0, outcome.stderr
    assert outcome.report["builders"] == ["BuildEventFlickering"]
    names = _fake_event_names(database)
    assert names == {"Fake Flickering"}


def test_a_database_without_frames_is_refused(
    checkout: _Checkout, tmp_path: Path
) -> None:
    database = tmp_path / "no-frames.sqlite"
    connection = sqlite3.connect(database)
    try:
        connection.execute("CREATE TABLE EVENT (ID INTEGER PRIMARY KEY)")
        connection.commit()
    finally:
        connection.close()

    outcome = _run("rebuild", checkout, tmp_path, database=database)

    assert outcome.returncode == 4
    error = outcome.report["error"]
    assert isinstance(error, str)
    assert "FRAME" in error
    assert "last_frame" not in outcome.report


def test_a_raising_rebuild_exits_5_and_names_the_exception(tmp_path: Path) -> None:
    root = tmp_path / "raising"
    raising = _Checkout(root, write_fake_lmt_analysis(root, raises=True))
    database = _database(tmp_path)

    outcome = _run("rebuild", raising, tmp_path, database=database)

    assert outcome.returncode == 5
    error = outcome.report["error"]
    assert isinstance(error, str)
    assert "RuntimeError" in error
    assert outcome.report["commit"] == raising.commit
    assert outcome.report["available_builders"] == _BUILDERS
    assert "Traceback" in outcome.stderr


def test_the_runner_imports_only_the_standard_library() -> None:
    """The program imports no third-party module, no mosaic, and no lmt-analysis.

    It reaches lmt-analysis through ``importlib`` with the two names the script
    needs, and a builder module whose name the command line gives.
    """
    imported = {
        imported.module.partition(".")[0]
        for imported in imports_of(_RUNNER, scope="anywhere", relative=False)
    }
    assert imported <= set(sys.stdlib_module_names), sorted(
        imported - set(sys.stdlib_module_names)
    )

    tree = source_tree(_RUNNER)
    literals = {
        module
        for reached in imports_reaching(tree, "mosaic.tracking.external.runner")
        if not reached.statement.startswith(("import ", "from "))
        for module in reached.reaches
    }
    assert literals == {"lmtanalysis.AnimalType", "Rebuild_All_Events"}
    # The one other import names the builder that the command line gives.
    assert computed_dynamic_imports(tree) == ["module_name"]
