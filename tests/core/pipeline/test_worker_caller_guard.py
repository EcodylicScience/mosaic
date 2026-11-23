"""A process worker ends when the process that built its pool is killed.

A caller that dies by SIGKILL (the out-of-memory killer, a node under memory pressure,
an expired grace period) runs no cleanup, and a worker holds both ends of the
executor's queue pipes, so it never reads end of file. These tests kill a real caller
under each start method and watch its workers.
"""

from __future__ import annotations

import multiprocessing as mp
import signal
import subprocess
import sys
import textwrap
import time
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import chain
from pathlib import Path

import psutil
import pytest

from tests.helpers import process_alive

POLL_DEADLINE_SECONDS = 15.0

CALLER_PRELUDE = textwrap.dedent(
    """
    import os
    import signal
    import time
    from pathlib import Path

    import psutil


    def record_pid_then_wait(pid_file, seconds):
        Path(pid_file).write_text(str(os.getpid()))
        time.sleep(seconds)


    def wait_until(condition, seconds=30):
        deadline = time.monotonic() + seconds
        while not condition() and time.monotonic() < deadline:
            time.sleep(0.1)
        return condition()


    def record_the_family_then_kill_the_caller(directory):
        family = psutil.Process().children(recursive=True)
        Path(directory, "processes.txt").write_text(
            " ".join(str(process.pid) for process in family)
        )
        os.kill(os.getpid(), signal.SIGKILL)
    """
)

PARENT_SCRIPT = CALLER_PRELUDE + textwrap.dedent(
    """
    import multiprocessing as mp
    import sys

    from mosaic.core.pipeline.process_pool import caller_guarded_process_pool

    if __name__ == "__main__":
        method, state, directory = sys.argv[1:4]
        if method == "default-forkserver":
            mp.set_start_method("forkserver")
            context = None
        else:
            context = mp.get_context(method)
        pool = caller_guarded_process_pool(2, context)
        pause = 600 if state == "busy" else 0
        futures = [
            pool.submit(record_pid_then_wait, f"{directory}/worker-{n}", pause)
            for n in range(2)
        ]
        if state == "idle":
            for future in futures:
                future.result()
        expected = 2 if state == "busy" else 1
        wait_until(lambda: len(list(Path(directory).glob("worker-*"))) >= expected)
        record_the_family_then_kill_the_caller(directory)
    """
)

POOL_PARENT_SCRIPT = CALLER_PRELUDE + textwrap.dedent(
    """
    import sys

    from mosaic.core.pipeline.entry_pool import EntryPool
    from mosaic.core.pipeline.memory_watch import MemoryWatch


    def ignore(_tag, _outcome):
        pass


    def sighted_both(watch, directory):
        sample = watch.sample()
        started = len(list(Path(directory).glob("worker-*")))
        return sample is not None and sample.spawned_workers == 2 and started == 2


    if __name__ == "__main__":
        directory = sys.argv[1]
        watch = MemoryWatch()
        watch.begin()
        pool = EntryPool(
            workers=2, mode="process", on_result=ignore, on_error=ignore, watch=watch
        )
        for n in range(2):
            pool.submit(n, record_pid_then_wait, f"{directory}/worker-{n}", 600)
        if wait_until(lambda: sighted_both(watch, directory)):
            Path(directory, "sighted").touch()
        record_the_family_then_kill_the_caller(directory)
    """
)

START_SCRIPT = textwrap.dedent(
    """
    import multiprocessing as mp
    import sys
    import time

    from mosaic.core.pipeline.process_pool import caller_guarded_process_pool

    if __name__ == "__main__":
        method = sys.argv[1]
        if method == "default-forkserver":
            mp.set_start_method("forkserver")
            context = None
        else:
            context = mp.get_context(method)
        with caller_guarded_process_pool(2, context) as pool:
            futures = [pool.submit(time.sleep, 3) for _ in range(2)]
            assert [future.result() for future in futures] == [None, None]
    """
)

START_METHODS = ["spawn", "fork", "forkserver", "default-forkserver"]


def _skip_unless_available(method: str) -> None:
    if method.removeprefix("default-") not in mp.get_all_start_methods():
        pytest.skip(f"the {method} start method is unavailable on this platform")


@dataclass(frozen=True)
class KilledCaller:
    """What a caller script left behind after it killed itself."""

    returncode: int
    pid_directory: Path
    workers: list[int]
    family: list[int]
    errors: str


def _end(pids: Iterable[int]) -> None:
    for pid in set(pids):
        if process_alive(pid):
            psutil.Process(pid).kill()


def _read_pids(pid_directory: Path) -> tuple[list[int], list[int]]:
    """The pids that the caller's tasks recorded, and every process it started.

    The second list holds the forkserver and a worker that no task recorded, so a
    failure leaves none of them running.
    """
    workers = [int(path.read_text()) for path in pid_directory.glob("worker-*")]
    family_file = pid_directory / "processes.txt"
    family = (
        [int(pid) for pid in family_file.read_text().split()]
        if family_file.exists()
        else []
    )
    return workers, family


def _kill_caller(
    script_text: str, arguments: list[str], tmp_path: Path
) -> KilledCaller:
    """Run a caller script that kills itself and read what it recorded.

    *arguments* follow the script and the directory it records in. A caller that does
    not end in time is killed with every process it recorded, and the wait fails.
    """
    script = tmp_path / "caller.py"
    script.write_text(script_text)
    pid_directory = tmp_path / "pids"
    pid_directory.mkdir()
    errors_file = tmp_path / "errors.txt"
    # The caller's output goes to files, because a surviving worker would hold a pipe
    # open and keep this test waiting for the end of file.
    with errors_file.open("wb") as errors:
        caller = subprocess.Popen(
            [sys.executable, str(script), *arguments, str(pid_directory)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=errors,
        )
        try:
            returncode = caller.wait(timeout=60)
        except BaseException:
            caller.kill()
            _ = caller.wait()
            _end(chain.from_iterable(_read_pids(pid_directory)))
            raise
    workers, family = _read_pids(pid_directory)
    return KilledCaller(
        returncode, pid_directory, workers, family, errors_file.read_text()
    )


def _assert_workers_end(killed: KilledCaller, message: str, *, count: int = 1) -> None:
    """Assert that the caller was killed and then every worker it recorded ends.

    At least *count* workers must have been recorded. Every process the caller
    started is killed when the assertion ends.
    """
    try:
        assert killed.returncode == -signal.SIGKILL, killed.errors
        assert len(killed.workers) >= count, killed.errors
        deadline = time.monotonic() + POLL_DEADLINE_SECONDS
        while (
            any(process_alive(pid) for pid in killed.workers)
            and time.monotonic() < deadline
        ):
            time.sleep(0.2)
        survivors = [pid for pid in killed.workers if process_alive(pid)]
        assert survivors == [], message
    finally:
        _end([*killed.workers, *killed.family])


@pytest.mark.parametrize("method", START_METHODS)
@pytest.mark.parametrize("state", ["idle", "busy"])
def test_process_workers_end_when_the_caller_is_killed(
    method: str, state: str, tmp_path: Path
) -> None:
    _skip_unless_available(method)
    killed = _kill_caller(PARENT_SCRIPT, [method, state], tmp_path)
    _assert_workers_end(killed, f"{method} {state} workers outlived the caller")


@pytest.mark.parametrize("method", START_METHODS)
def test_workers_keep_running_while_the_caller_lives(method: str) -> None:
    """A task of three seconds finishes, whichever process is the worker's parent."""
    _skip_unless_available(method)
    finished = subprocess.run(
        [sys.executable, "-c", START_SCRIPT, method],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert finished.returncode == 0, finished.stderr


def test_pool_workers_end_when_the_caller_with_a_begun_watch_is_killed(
    tmp_path: Path,
) -> None:
    killed = _kill_caller(POOL_PARENT_SCRIPT, [], tmp_path)
    try:
        assert (killed.pid_directory / "sighted").exists(), (
            f"the watch never counted both workers: {killed.errors}"
        )
    except AssertionError:
        _end([*killed.workers, *killed.family])
        raise
    _assert_workers_end(killed, "pool workers outlived the caller", count=2)
