"""Mosaic's side of the lmt-analysis runner: locating it, probing it, running it.

The runner runs under the suite's interpreter on a stand-in checkout that
:func:`~tests.helpers.write_fake_lmt_analysis` commits. A test that needs a
report the runner would never write places a shell script in the interpreter's
position, which writes the report it is given and exits with the status it is
given. Every refusal is raised before a copy exists, so these tests need no
dataset.
"""

from __future__ import annotations

import stat
from collections.abc import Callable
from pathlib import Path
from typing import Final

import pytest
from pydantic import ValidationError

from mosaic.core.pipeline.refusal import ToolUnavailable
from mosaic.core.pipeline.subprocess_util import ProcessCancelled
from mosaic.tracking.common.lmt_env import (
    LMT_ANALYSIS_DIR_VAR,
    LMT_ENV_DIR,
    LMT_PYTHON_VAR,
    LmtRebuildError,
    LmtRunnerDefect,
    LmtRunnerReport,
    lmt_analysis_checkout,
    probe_lmt_analysis,
    run_lmt_rebuild,
    runner_script,
)
from tests.helpers import (
    InstalledLmtAnalysis,
    LmtAnalysisInstaller,
    default_lmt_recording,
    git_in_checkout,
    write_lmt_sqlite,
)

_README: Final = "src/mosaic/tracking/external/lmt-env/README.md"
"""The README as a refusal spells it: the repository path, never an absolute one."""
_BUILDERS: Final = ("BuildEventDetection", "BuildEventFlickering")
_COMPLETE_KEYS: Final = (
    '"builders": ["BuildEventDetection", "BuildEventFlickering"], '
    '"max_frame": 300, "script_max_frame": 300, "window_frames": 100, '
    '"python": "3.12.0", "packages": {"pytest": "9"}'
)


def _never() -> bool:
    return False


def _always() -> bool:
    return True


def _scripted_interpreter(
    monkeypatch: pytest.MonkeyPatch, directory: Path, report: str, status: int
) -> None:
    """Place a script in the interpreter's position.

    The script writes *report* to the path after ``--report``, prints nothing
    and exits with *status*.
    """
    body = directory / "report.json"
    _ = body.write_text(report)
    script = directory / "scripted-python"
    _ = script.write_text(
        "#!/bin/sh\n"
        'while [ "$#" -gt 0 ]; do\n'
        '  if [ "$1" = "--report" ]; then shift; cat "' + str(body) + '" > "$1"; fi\n'
        "  shift\n"
        "done\n"
        f"exit {status}\n"
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv(LMT_PYTHON_VAR, str(script))


def _silent_interpreter(monkeypatch: pytest.MonkeyPatch, directory: Path) -> None:
    """Place a script in the interpreter's position that exits 0 and writes no report."""
    script = directory / "silent-python"
    _ = script.write_text("#!/bin/sh\nexit 0\n")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv(LMT_PYTHON_VAR, str(script))


def _database(tmp_path: Path) -> Path:
    recording = default_lmt_recording()
    return write_lmt_sqlite(
        tmp_path / "recording.sqlite",
        animals=recording.animals,
        frames=recording.frames,
        events=recording.events,
        detections=recording.detections,
        log=recording.log,
    )


def _probe(
    checkout: InstalledLmtAnalysis,
    tmp_path: Path,
    *,
    commit: str | None = None,
    builders: tuple[str, ...] | None = None,
    cancel_check: Callable[[], bool] = _never,
) -> LmtRunnerReport:
    return probe_lmt_analysis(
        commit or checkout.commit,
        builders,
        tmp_path / "probe-report.json",
        cancel_check=cancel_check,
    )


def _rebuild(
    checkout: InstalledLmtAnalysis,
    tmp_path: Path,
    database: Path,
    *,
    commit: str | None = None,
    builders: tuple[str, ...] | None = None,
    lines: list[str] | None = None,
    cancel_check: Callable[[], bool] = _never,
) -> LmtRunnerReport:
    seen = [] if lines is None else lines
    return run_lmt_rebuild(
        database,
        commit or checkout.commit,
        builders,
        report_path=tmp_path / "rebuild-report.json",
        log_path=tmp_path / "rebuild.log",
        cancel_check=cancel_check,
        on_line=seen.append,
    )


def test_the_probe_reports_the_stand_ins_commit_and_builders(
    install_lmt: LmtAnalysisInstaller, tmp_path: Path
) -> None:
    checkout = install_lmt()
    report = _probe(checkout, tmp_path)

    assert report.runner_protocol == 1
    assert report.error is None
    assert report.commit == checkout.commit
    assert report.builders == _BUILDERS
    assert report.max_frame == report.script_max_frame == 300
    assert report.window_frames == 100
    assert report.python is not None
    assert report.packages is not None
    assert "pytest" in report.packages


def test_the_probe_of_named_builders_reports_them(
    install_lmt: LmtAnalysisInstaller, tmp_path: Path
) -> None:
    checkout = install_lmt()
    report = _probe(checkout, tmp_path, builders=("BuildEventFlickering",))

    assert report.builders == ("BuildEventFlickering",)


def test_a_rebuild_report_holds_the_keys_of_a_rebuild(
    install_lmt: LmtAnalysisInstaller, tmp_path: Path
) -> None:
    checkout = install_lmt()
    report = _rebuild(checkout, tmp_path, _database(tmp_path))

    assert report.last_frame is not None
    assert report.seconds is not None
    assert report.max_frame == 500
    assert report.script_max_frame == 300


def test_a_report_is_frozen() -> None:
    report = LmtRunnerReport(runner_protocol=1)

    with pytest.raises(ValidationError):
        report.error = "changed"


_SUBSTITUTED = {
    "an unknown key": '{"runner_protocol": 1, "surprise": true}',
    "protocol 2": '{"runner_protocol": 2}',
    "invalid JSON": '{"runner_protocol": 1',
    "a list": "[]",
    "an unreadable value": '{"runner_protocol": 1, "max_frame": "far"}',
}


@pytest.mark.parametrize("report", _SUBSTITUTED.values(), ids=_SUBSTITUTED)
def test_an_invalid_report_is_a_defect_from_both_functions(
    install_lmt: LmtAnalysisInstaller,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    report: str,
) -> None:
    checkout = install_lmt()
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    _scripted_interpreter(monkeypatch, scripts, report, 0)

    with pytest.raises(LmtRunnerDefect) as probed:
        _ = _probe(checkout, tmp_path)
    with pytest.raises(LmtRunnerDefect) as rebuilt:
        _ = _rebuild(checkout, tmp_path, _database(tmp_path))

    for refusal in (probed.value, rebuilt.value):
        assert refusal.returncode == 0
        assert "rebuild" in refusal.cmd or "probe" in refusal.cmd
        assert "report" in str(refusal)


def test_a_missing_report_is_a_defect(
    install_lmt: LmtAnalysisInstaller, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = install_lmt()
    _silent_interpreter(monkeypatch, tmp_path)

    with pytest.raises(LmtRunnerDefect, match="no report"):
        _ = _probe(checkout, tmp_path)


def test_a_stale_report_is_not_read_when_the_runner_writes_none(
    install_lmt: LmtAnalysisInstaller, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = install_lmt()
    report = _probe(checkout, tmp_path)
    assert report.runner_protocol == 1
    _silent_interpreter(monkeypatch, tmp_path)

    with pytest.raises(LmtRunnerDefect, match="no report"):
        _ = _probe(checkout, tmp_path)


@pytest.mark.parametrize(
    "missing",
    [
        "builders",
        "max_frame",
        "script_max_frame",
        "window_frames",
        "python",
        "packages",
    ],
)
def test_an_exit_zero_report_without_a_key_is_a_defect(
    install_lmt: LmtAnalysisInstaller,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    missing: str,
) -> None:
    checkout = install_lmt()
    keys = {
        "builders": '"builders": ["BuildEventDetection"]',
        "max_frame": '"max_frame": 300',
        "script_max_frame": '"script_max_frame": 300',
        "window_frames": '"window_frames": 100',
        "python": '"python": "3.12.0"',
        "packages": '"packages": {}',
    }
    del keys[missing]
    report = '{"runner_protocol": 1, ' + ", ".join(keys.values()) + "}"
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    _scripted_interpreter(monkeypatch, scripts, report, 0)

    with pytest.raises(LmtRunnerDefect, match=missing):
        _ = _probe(checkout, tmp_path)


def test_an_exit_zero_report_of_another_commit_is_a_defect(
    install_lmt: LmtAnalysisInstaller, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = install_lmt()
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    report = (
        '{"runner_protocol": 1, "commit": "' + "0" * 40 + '", ' + _COMPLETE_KEYS + "}"
    )
    _scripted_interpreter(monkeypatch, scripts, report, 0)

    with pytest.raises(LmtRunnerDefect, match="commit"):
        _ = _probe(checkout, tmp_path)


def test_an_exit_zero_report_of_other_builders_is_a_defect(
    install_lmt: LmtAnalysisInstaller, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = install_lmt()
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    report = (
        '{"runner_protocol": 1, "commit": "'
        + checkout.commit
        + '", '
        + _COMPLETE_KEYS
        + "}"
    )
    _scripted_interpreter(monkeypatch, scripts, report, 0)

    with pytest.raises(LmtRunnerDefect, match="builders"):
        _ = _probe(checkout, tmp_path, builders=("BuildEventFlickering",))


@pytest.mark.parametrize("status", [2, 7])
def test_a_usage_exit_and_an_undefined_exit_are_defects(
    install_lmt: LmtAnalysisInstaller,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status: int,
) -> None:
    checkout = install_lmt()
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    _scripted_interpreter(monkeypatch, scripts, '{"runner_protocol": 1}', status)

    with pytest.raises(LmtRunnerDefect) as probed:
        _ = _probe(checkout, tmp_path)
    with pytest.raises(LmtRunnerDefect) as rebuilt:
        _ = _rebuild(checkout, tmp_path, _database(tmp_path))

    assert probed.value.returncode == rebuilt.value.returncode == status


def test_an_environment_exit_without_an_error_is_a_defect(
    install_lmt: LmtAnalysisInstaller, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = install_lmt()
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    _scripted_interpreter(monkeypatch, scripts, '{"runner_protocol": 1}', 3)

    with pytest.raises(LmtRunnerDefect):
        _ = _probe(checkout, tmp_path)


def test_an_interpreter_that_is_not_a_file_is_refused(
    install_lmt: LmtAnalysisInstaller, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = install_lmt()
    missing = tmp_path / "no-environment" / "bin" / "python"
    monkeypatch.setenv(LMT_PYTHON_VAR, str(missing))

    with pytest.raises(ToolUnavailable) as refused:
        _ = _probe(checkout, tmp_path)

    assert refused.value.cause == "environment"
    assert refused.value.tool == "lmt-analysis"
    assert str(missing) in str(refused.value)
    assert f"Follow {_README}," in str(refused.value)


def test_an_interpreter_that_does_not_start_is_refused(
    install_lmt: LmtAnalysisInstaller, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = install_lmt()
    unrunnable = tmp_path / "python"
    _ = unrunnable.write_text("not a program")
    monkeypatch.setenv(LMT_PYTHON_VAR, str(unrunnable))

    with pytest.raises(ToolUnavailable) as refused:
        _ = _probe(checkout, tmp_path)

    assert refused.value.cause == "environment"
    assert "could not be started" in str(refused.value)
    assert f"Follow {_README}," in str(refused.value)


def test_a_ladder_that_locates_nothing_is_refused(
    install_lmt: LmtAnalysisInstaller, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = install_lmt()
    monkeypatch.delenv(LMT_PYTHON_VAR)

    with pytest.raises(ToolUnavailable) as refused:
        _ = _probe(checkout, tmp_path)

    assert refused.value.cause == "environment"
    assert f"Follow {_README}," in str(refused.value)


def test_a_missing_checkout_directory_is_refused_by_name(
    install_lmt: LmtAnalysisInstaller, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = install_lmt()
    absent = tmp_path / "no-checkout"
    monkeypatch.setenv(LMT_ANALYSIS_DIR_VAR, str(absent))

    with pytest.raises(ToolUnavailable) as refused:
        _ = _probe(checkout, tmp_path)

    message = str(refused.value)
    assert refused.value.cause == "environment"
    assert str(absent) in message
    assert LMT_ANALYSIS_DIR_VAR in message
    assert f"as {_README} describes" in message
    assert "clone" in message.lower()


def test_the_interpreter_is_refused_before_the_checkout(
    install_lmt: LmtAnalysisInstaller, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = install_lmt()
    monkeypatch.setenv(LMT_PYTHON_VAR, str(tmp_path / "no-python"))
    monkeypatch.setenv(LMT_ANALYSIS_DIR_VAR, str(tmp_path / "no-checkout"))

    with pytest.raises(ToolUnavailable) as refused:
        _ = _probe(checkout, tmp_path)

    assert "interpreter" in str(refused.value)


def test_a_checkout_at_another_commit_names_both_and_the_fix(
    install_lmt: LmtAnalysisInstaller, tmp_path: Path
) -> None:
    checkout = install_lmt()
    git_in_checkout(checkout.root, "commit", "--allow-empty", "-m", "Move the head")
    head = git_in_checkout(checkout.root, "rev-parse", "HEAD")

    with pytest.raises(ToolUnavailable) as refused:
        _ = _probe(checkout, tmp_path)

    message = str(refused.value)
    assert refused.value.cause == "environment"
    assert head in message
    assert checkout.commit in message
    assert f"git -C {checkout.root} checkout {checkout.commit}" in message


def test_a_modified_tracked_file_is_refused_by_name(
    install_lmt: LmtAnalysisInstaller, tmp_path: Path
) -> None:
    checkout = install_lmt()
    script = checkout.root / "LMT" / "scripts" / "Rebuild_All_Events.py"
    _ = script.write_text(script.read_text() + "\n# edited\n")

    with pytest.raises(ToolUnavailable) as refused:
        _ = _probe(checkout, tmp_path)

    assert refused.value.cause == "environment"
    message = str(refused.value)
    assert "Rebuild_All_Events.py" in message
    assert f"git -C {checkout.root} checkout {checkout.commit} -- ." in message
    assert f"See {_README}." in message


def test_a_checkout_that_cannot_import_is_refused(
    install_lmt: LmtAnalysisInstaller, tmp_path: Path
) -> None:
    checkout = install_lmt()
    git_in_checkout(checkout.root, "rm", "-r", "-q", "LMT/scripts")
    git_in_checkout(checkout.root, "commit", "-q", "-m", "Drop the script")
    head = git_in_checkout(checkout.root, "rev-parse", "HEAD")

    with pytest.raises(ToolUnavailable) as refused:
        _ = _probe(checkout, tmp_path, commit=head)

    assert refused.value.cause == "environment"
    assert "LMT/scripts" in str(refused.value)
    assert f"See {_README}." in str(refused.value)


def test_an_unknown_builder_is_a_value_error_naming_the_builders(
    install_lmt: LmtAnalysisInstaller, tmp_path: Path
) -> None:
    checkout = install_lmt()
    with pytest.raises(ValueError, match="BuildEventNoSuch") as refused:
        _ = _probe(checkout, tmp_path, builders=("BuildEventNoSuch",))

    assert not isinstance(refused.value, ToolUnavailable)
    for builder in _BUILDERS:
        assert builder in str(refused.value)


def test_a_rebuild_writes_the_output_to_the_log_and_calls_on_line(
    install_lmt: LmtAnalysisInstaller, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate = tmp_path / "gate"
    gate.touch()
    waiting = install_lmt(wait_for=gate)
    lines: list[str] = []

    report = _rebuild(waiting, tmp_path, _database(tmp_path), lines=lines)

    assert report.commit == waiting.commit
    assert lines == [f"waiting for {gate}\n"]
    assert (tmp_path / "rebuild.log").read_text() == f"waiting for {gate}\n"


def test_a_raising_process_is_a_rebuild_error(
    tmp_path: Path, install_lmt: LmtAnalysisInstaller
) -> None:
    raising = install_lmt(raises=True)
    lines: list[str] = []

    with pytest.raises(LmtRebuildError) as failed:
        _ = _rebuild(raising, tmp_path, _database(tmp_path), lines=lines)

    assert failed.value.returncode == 5
    assert "the stand-in rebuild was asked to fail" in str(failed.value)
    assert not isinstance(failed.value, LmtRunnerDefect)
    log = (tmp_path / "rebuild.log").read_text()
    assert "RuntimeError" in log
    assert "".join(lines) == log


def test_a_checkout_moved_since_the_probe_fails_the_rebuild(
    install_lmt: LmtAnalysisInstaller, tmp_path: Path
) -> None:
    checkout = install_lmt()
    git_in_checkout(checkout.root, "commit", "--allow-empty", "-m", "Move the head")

    with pytest.raises(LmtRebuildError) as failed:
        _ = _rebuild(checkout, tmp_path, _database(tmp_path))

    assert failed.value.returncode == 3
    assert checkout.commit in str(failed.value)


def test_a_database_without_frames_fails_the_rebuild(
    install_lmt: LmtAnalysisInstaller, tmp_path: Path
) -> None:
    checkout = install_lmt()
    empty = tmp_path / "empty.sqlite"
    empty.touch()

    with pytest.raises(LmtRebuildError) as failed:
        _ = _rebuild(checkout, tmp_path, empty)

    assert failed.value.returncode == 4
    assert "FRAME" in str(failed.value)


def test_a_cancel_stops_the_rebuild(
    tmp_path: Path, install_lmt: LmtAnalysisInstaller
) -> None:
    waiting = install_lmt(wait_for=tmp_path / "never-created")

    with pytest.raises(ProcessCancelled):
        _ = _rebuild(waiting, tmp_path, _database(tmp_path), cancel_check=_always)


def test_the_checkout_is_the_variable_and_otherwise_the_directory_beside_the_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(LMT_ANALYSIS_DIR_VAR, raising=False)
    assert lmt_analysis_checkout() == LMT_ENV_DIR / "lmt-analysis"

    monkeypatch.setenv(LMT_ANALYSIS_DIR_VAR, str(tmp_path / "elsewhere"))
    assert lmt_analysis_checkout() == tmp_path / "elsewhere"


def test_the_runner_script_is_the_program_beside_the_other_runners() -> None:
    script = runner_script()

    assert script.name == "lmt_rebuild_runner.py"
    assert script.is_file()
    assert script.parent.name == "runner"
