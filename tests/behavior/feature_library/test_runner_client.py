"""Mosaic's side of the runner protocol: one runner process, started when first asked.

Each test drives a runner program that ``write_fake_runner`` writes, under this
interpreter, placed through the binary rung of the tool's location ladder. The
program records its start in a journal, which counts the spawns.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import BaseModel

from mosaic.behavior.feature_library.external.runner_wire import Event
from mosaic.behavior.feature_library.runner_client import (
    OnStart,
    RunnerCommandError,
    RunnerProcess,
    RunnerVersionError,
)
from mosaic.core.pipeline.job import CancelCheck, Cancelled
from mosaic.core.pipeline.toolenv import ToolEnv, ToolExitError, ToolNotFoundError
from tests.helpers import (
    FAKE_HANDLERS,
    RunnerJournal,
    appender_program,
    assert_process_exited,
    assert_stopped_writing,
    journal_source,
    wait_for,
    write_fake_runner,
)


class _FakeNotFound(ToolNotFoundError):
    default_message = "Build the fake tool's environment."


class _FakeExit(ToolExitError):
    tool_name = "The fake runner"


_ENV = ToolEnv(
    tool="fake",
    conda_env_var="MOSAIC_FAKE_RUNNER_CONDA_ENV",
    bin_var="MOSAIC_FAKE_RUNNER_PYTHON",
    bin_mode="direct",
    not_found=_FakeNotFound,
)

HANDLERS = (
    FAKE_HANDLERS
    + """

def note(payload, emit):
    sys.stderr.write(str(payload["text"]) + "\\n")
    sys.stderr.flush()


def crash(payload, emit):
    sys.stderr.write("the runner is crashing\\n")
    sys.stderr.flush()
    os._exit(3)


def garble(payload, emit):
    PROTOCOL.write("this line is not an envelope\\n")
    PROTOCOL.flush()


def linger(payload, emit):
    subprocess.Popen([sys.executable, "-c", payload["program"]])
    if payload["outlives_shutdown"]:
        threading.Thread(target=time.sleep, args=(3600,)).start()


DISPATCH.update(note=note, crash=crash, garble=garble, linger=linger)
"""
)
"""The handlers of :data:`FAKE_HANDLERS`, and ``note``, ``crash``, ``garble`` and
``linger``."""

_NOISE = 'print("activating the environment", flush=True)'
"""Prints a line before ``Ready``, as an activation script run by ``conda run`` may."""

_FIRST_RUNNER_EXITS_AFTER_READY = """
_serve = runner_wire.serve


def _serve_unless_first(dispatch, tool, version, *, protocol, heartbeat_seconds):
    records = [json.loads(line) for line in JOURNAL.read_text().splitlines()]
    if sum(record["command"] == "start" for record in records) > 1:
        _serve(
            dispatch,
            tool,
            version,
            protocol=protocol,
            heartbeat_seconds=heartbeat_seconds,
        )
        return
    protocol.write(runner_wire.Ready(tool=tool, version=version).model_dump_json())
    protocol.write("\\n")
    protocol.flush()
    os._exit(0)


runner_wire.serve = _serve_unless_first
"""
"""Makes the first runner of a test exit right after its ``Ready``."""


class _Text(BaseModel):
    text: str


class _Count(BaseModel):
    up_to: int


class _Nap(BaseModel):
    seconds: float


class _Linger(BaseModel):
    program: str
    outlives_shutdown: bool


class _Runners:
    """Builds the runner processes of one test from programs in *directory*."""

    def __init__(self, directory: Path) -> None:
        self._directory = directory
        self.journal = RunnerJournal(directory / "journal.jsonl")
        self._made: list[RunnerProcess] = []

    def program(self, *, version: str = "1.0", preamble: str = "") -> Path:
        """Write the runner program, with *preamble* run after it records its start."""
        handlers = journal_source(self.journal.path) + preamble + "\n" + HANDLERS
        return write_fake_runner(self._directory, handlers, version=version)

    def make(
        self,
        *,
        version: str = "1.0",
        required: str | None = "1.0",
        preamble: str = "",
        on_start: OnStart | None = None,
        interpreter: Path = Path(sys.executable),
        scratch_dir: Path | None = None,
    ) -> RunnerProcess:
        runner = RunnerProcess(
            _ENV.placed(bin_path=interpreter),
            self.program(version=version, preamble=preamble),
            failure=_FakeExit,
            version=required,
            on_start=on_start,
            scratch_dir=scratch_dir,
        )
        self._made.append(runner)
        return runner

    def close_all(self) -> None:
        for runner in self._made:
            _ = runner.close(grace=1.0)


@pytest.fixture
def runners(tmp_path: Path) -> Iterator[_Runners]:
    """Runner processes that are closed when the test ends, whatever its outcome."""
    made = _Runners(tmp_path)
    yield made
    made.close_all()


# --- starting -----------------------------------------------------------------


def test_one_runner_serves_successive_requests(runners: _Runners) -> None:
    runner = runners.make()

    assert runner.request("echo", _Text(text="hello")) == {"text": "hello"}
    assert runner.request("echo", _Text(text="again")) == {"text": "again"}

    assert len(runners.journal.starts()) == 1, "the second request reused the runner"


def test_a_missing_interpreter_is_refused_at_the_first_request(
    runners: _Runners, tmp_path: Path
) -> None:
    """Construction does not start the interpreter."""
    missing = tmp_path / "no-environment" / "bin" / "python"
    runner = runners.make(interpreter=missing)

    with pytest.raises(_FakeNotFound) as refused:
        _ = runner.request("echo", _Text(text="hello"))

    message = str(refused.value)
    assert f"'{missing}'" in message
    assert "the binary this run was placed with" in message
    assert "could not be started" in message
    assert "Build the fake tool's environment." in message


def test_locate_resolves_the_interpreter_and_starts_no_runner(
    runners: _Runners,
) -> None:
    runner = runners.make()

    located = runner.locate()

    assert located.argv == [sys.executable]
    assert located.source == "the binary this run was placed with"
    assert runners.journal.starts() == []


def test_a_runner_of_another_version_is_refused(runners: _Runners) -> None:
    runner = runners.make(version="2.0", required="1.0")

    with pytest.raises(RunnerVersionError) as refused:
        _ = runner.request("echo", _Text(text="hello"))

    message = str(refused.value)
    assert "fake 2.0" in message
    assert "requires 1.0" in message
    assert "Build the fake tool's environment." in message
    (pid,) = runners.journal.starts()
    assert_process_exited(pid)


def test_a_runner_without_a_required_version_is_accepted_at_any(
    runners: _Runners,
) -> None:
    runner = runners.make(version="2.0", required=None)

    assert runner.request("echo", _Text(text="hello")) == {"text": "hello"}


def test_a_line_before_ready_that_is_not_an_envelope_is_skipped(
    runners: _Runners,
) -> None:
    """An activation script run by ``conda run`` can print before the runner starts."""
    runner = runners.make(preamble=_NOISE)

    assert runner.request("echo", _Text(text="hello")) == {"text": "hello"}


# --- requests -----------------------------------------------------------------


def test_events_are_passed_on_in_the_order_written(runners: _Runners) -> None:
    runner = runners.make()
    events: list[Event] = []

    result = runner.request("count", _Count(up_to=5), on_event=events.append)

    assert result is None
    assert events == [Event(name="counted", data={"number": n}) for n in range(5)]


def test_a_line_after_ready_that_is_not_an_envelope_is_refused(
    runners: _Runners,
) -> None:
    runner = runners.make()

    with pytest.raises(_FakeExit) as refused:
        _ = runner.request("garble")

    message = str(refused.value)
    assert "this line is not an envelope" in message
    assert "breaks the runner protocol" in message, "the exit code reads as the cause"


def test_a_command_that_raised_is_raised_with_the_runners_traceback(
    runners: _Runners,
) -> None:
    runner = runners.make()

    with pytest.raises(RunnerCommandError) as failed:
        _ = runner.request("fail")

    assert failed.value.command == "fail"
    assert failed.value.message == "ValueError: the handler failed"
    assert 'raise ValueError("the handler failed")' in failed.value.traceback
    assert failed.value.traceback in str(failed.value)
    assert runner.request("echo", _Text(text="after")) == {"text": "after"}
    assert len(runners.journal.starts()) == 1, "the runner served on after the failure"


def test_a_command_interrupted_in_the_runner_is_cancelled(runners: _Runners) -> None:
    runner = runners.make()

    with pytest.raises(Cancelled):
        _ = runner.request("interrupt")


def test_a_crash_is_raised_with_its_exit_code_and_the_logs_tail(
    runners: _Runners,
) -> None:
    runner = runners.make()

    with pytest.raises(_FakeExit) as crashed:
        _ = runner.request("crash")

    assert crashed.value.returncode == 3
    assert "the runner is crashing" in str(crashed.value)


def test_a_cancel_kills_a_quiet_command_within_about_a_second(
    runners: _Runners,
) -> None:
    runner = runners.make()
    napping: list[float] = []

    def note_the_nap(event: Event) -> None:
        if event.name == "napping":
            napping.append(time.monotonic())

    with pytest.raises(Cancelled):
        _ = runner.request(
            "nap", _Nap(seconds=60), on_event=note_the_nap, cancel=lambda: bool(napping)
        )

    assert time.monotonic() - napping[0] < 2.5
    (pid,) = runners.journal.starts()
    assert_process_exited(pid)


def test_a_raising_event_callback_stops_the_runner(runners: _Runners) -> None:
    """A runner left in the middle of a command would answer the next request
    with that command's replies."""
    runner = runners.make()

    def refuse(event: Event) -> None:
        raise RuntimeError(f"refused {event.name}")

    with pytest.raises(RuntimeError, match="refused napping"):
        _ = runner.request("nap", _Nap(seconds=60), on_event=refuse)

    assert runner.request("echo", _Text(text="next")) == {"text": "next"}
    first, _second = runners.journal.starts()
    assert_process_exited(first)


# --- respawning and closing ---------------------------------------------------


def _loads_on_start(started: list[RunnerProcess]) -> OnStart:
    """An ``on_start`` that records each call and sends a request."""

    def on_start(runner: RunnerProcess, cancel: CancelCheck | None) -> None:
        started.append(runner)
        _ = runner.request("note", _Text(text="started"), cancel=cancel)

    return on_start


def test_a_crashed_runner_is_started_again_with_on_start(runners: _Runners) -> None:
    started: list[RunnerProcess] = []
    runner = runners.make(on_start=_loads_on_start(started))
    assert runner.request("echo", _Text(text="first")) == {"text": "first"}
    with pytest.raises(_FakeExit):
        _ = runner.request("crash")

    assert runner.request("echo", _Text(text="second")) == {"text": "second"}

    assert started == [runner, runner]
    assert len(runners.journal.starts()) == 2
    assert runner.log().count("started\n") == 2


def test_a_closed_runner_is_started_again_with_on_start(runners: _Runners) -> None:
    started: list[RunnerProcess] = []
    runner = runners.make(on_start=_loads_on_start(started))
    assert runner.request("echo", _Text(text="first")) == {"text": "first"}
    _ = runner.close()

    assert runner.request("echo", _Text(text="second")) == {"text": "second"}

    assert started == [runner, runner]
    assert len(runners.journal.starts()) == 2


def test_a_runner_that_exits_before_the_request_of_on_start_fails(
    runners: _Runners,
) -> None:
    """The request of ``on_start`` fails, and the next request starts a runner.

    A request of ``on_start`` does not start a runner, because each runner that
    starts runs ``on_start``.
    """
    started: list[RunnerProcess] = []

    def on_start(runner: RunnerProcess, cancel: CancelCheck | None) -> None:
        started.append(runner)
        # The runner exits during the wait, or else during the request. The
        # request fails in both cases.
        time.sleep(0.5)
        _ = runner.request("note", _Text(text="started"), cancel=cancel)

    runner = runners.make(preamble=_FIRST_RUNNER_EXITS_AFTER_READY, on_start=on_start)

    with pytest.raises(_FakeExit) as failed:
        _ = runner.request("echo", _Text(text="first"))

    assert failed.value.returncode == 0
    assert started == [runner]
    assert len(runners.journal.starts()) == 1
    assert runner.request("echo", _Text(text="second")) == {"text": "second"}
    assert started == [runner, runner]
    assert len(runners.journal.starts()) == 2


@pytest.mark.parametrize(
    "requests_again", [True, False], ids=["requests-again", "returns"]
)
def test_on_start_that_catches_the_crash_of_its_runner_fails_the_request(
    runners: _Runners, requests_again: bool
) -> None:
    """The request fails with the crash of the only runner started."""
    started: list[RunnerProcess] = []

    def on_start(runner: RunnerProcess, cancel: CancelCheck | None) -> None:
        started.append(runner)
        with pytest.raises(_FakeExit):
            _ = runner.request("crash", cancel=cancel)
        if requests_again:
            _ = runner.request("note", _Text(text="after the crash"), cancel=cancel)

    runner = runners.make(on_start=on_start)

    with pytest.raises(_FakeExit) as failed:
        _ = runner.request("echo", _Text(text="first"))

    assert failed.value.returncode == 3
    assert started == [runner]
    assert len(runners.journal.starts()) == 1


def test_a_cancel_reaches_the_request_of_on_start(runners: _Runners) -> None:
    napping: list[float] = []

    def note_the_nap(event: Event) -> None:
        if event.name == "napping":
            napping.append(time.monotonic())

    def on_start(runner: RunnerProcess, cancel: CancelCheck | None) -> None:
        _ = runner.request(
            "nap", _Nap(seconds=60), on_event=note_the_nap, cancel=cancel
        )

    runner = runners.make(on_start=on_start)

    with pytest.raises(Cancelled):
        _ = runner.request("echo", _Text(text="hello"), cancel=lambda: bool(napping))

    assert time.monotonic() - napping[0] < 2.5
    (pid,) = runners.journal.starts()
    assert_process_exited(pid)


def test_close_stops_the_runner_and_returns_its_log(runners: _Runners) -> None:
    runner = runners.make()
    _ = runner.request("note", _Text(text="a line for the log"))
    assert "a line for the log" in runner.log()

    log = runner.close()

    assert "a line for the log" in log
    (pid,) = runners.journal.starts()
    assert_process_exited(pid)
    assert runner.log() == "", "close removed the log"


def test_a_kept_log_receives_the_log_once_and_the_rest_at_close(
    runners: _Runners, tmp_path: Path
) -> None:
    """The kept file's earlier text stays, and each line of the log is appended once."""
    kept = tmp_path / "run" / "runner.log"
    kept.parent.mkdir()
    _ = kept.write_text("an earlier attempt\n")
    runner = runners.make()
    _ = runner.request("note", _Text(text="before keep_log"))

    runner.keep_log(kept)
    runner.keep_log(kept)

    text = kept.read_text()
    assert text.startswith("an earlier attempt\n")
    assert text.count("before keep_log") == 1
    _ = runner.request("note", _Text(text="after keep_log"))
    assert "after keep_log" not in kept.read_text()

    _ = runner.close()

    text = kept.read_text()
    assert text.count("before keep_log") == 1
    assert text.count("after keep_log") == 1
    assert text.index("before keep_log") < text.index("after keep_log")
    assert sorted(path.name for path in kept.parent.iterdir()) == ["runner.log"]
    _ = runner.request("note", _Text(text="after close"))
    _ = runner.close()
    assert "after close" not in kept.read_text(), "close forgot the kept file"


@pytest.mark.parametrize(
    "outlives_shutdown", [True, False], ids=["runner-outlives-shutdown", "runner-exits"]
)
def test_close_kills_the_children_that_a_runner_started(
    runners: _Runners, tmp_path: Path, outlives_shutdown: bool
) -> None:
    """Close kills the children of a runner, whether or not the runner exits."""
    ticks = tmp_path / "ticks.txt"
    runner = runners.make()
    _ = runner.request(
        "linger",
        _Linger(program=appender_program(ticks), outlives_shutdown=outlives_shutdown),
    )
    wait_for(lambda: ticks.exists() and ticks.stat().st_size > 0, 30)
    started = time.monotonic()

    _ = runner.close(grace=0.5)

    assert time.monotonic() - started < 5.0
    (pid,) = runners.journal.starts()
    assert_process_exited(pid)
    assert_stopped_writing(ticks, "the runner's child went on writing after close()")


def test_a_second_close_does_nothing(runners: _Runners) -> None:
    runner = runners.make()
    _ = runner.request("note", _Text(text="logged once"))
    assert "logged once" in runner.close()

    assert runner.close() == ""
    assert len(runners.journal.starts()) == 1


def test_the_runners_until_close_share_one_exchange_directory(
    runners: _Runners, tmp_path: Path
) -> None:
    """A runner started after a crash finds the files that the first was given."""
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    runner = runners.make(scratch_dir=scratch)
    exchange = runner.exchange_dir()
    _ = (exchange / "track.npz").write_bytes(b"tracks")
    with pytest.raises(_FakeExit):
        _ = runner.request("crash")
    assert runner.request("echo", _Text(text="again")) == {"text": "again"}

    assert runner.exchange_dir() == exchange
    assert exchange.parent == scratch
    assert (exchange / "track.npz").read_bytes() == b"tracks"
    _ = runner.close()

    assert list(scratch.iterdir()) == []
    assert runner.exchange_dir().is_dir(), "the first call after close creates one"
