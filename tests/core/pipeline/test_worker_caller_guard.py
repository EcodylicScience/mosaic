"""A process worker ends when the process that built its pool is killed, or the pool breaks.

A caller that dies by SIGKILL (the out-of-memory killer, a node under memory pressure,
an expired grace period) runs no cleanup, and a worker holds both ends of the
executor's queue pipes, so it never reads end of file. These tests kill a real caller
under each start method and watch its workers.

A worker that ignores SIGTERM survives the executor's end of a broken pool, so the
pool kills its live workers itself. The tests of that kill run a caller whose
workers ignore the stop signals.
"""

from __future__ import annotations

import gc
import multiprocessing as mp
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time
import weakref
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from itertools import chain
from multiprocessing.context import SpawnContext, SpawnProcess
from pathlib import Path
from typing import override

import pytest

from mosaic.core.pipeline.process_pool import (
    CallerGuardedPool,
    caller_guarded_process_pool,
    ignore_stop_signals,
)
from tests.helpers import (
    assert_marked_child_ends,
    kill_running,
    listed_pids,
    pool_call,
    process_alive,
    run_child_to_files,
    wait_for,
)

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

INITIALIZED_PARENT_SCRIPT = CALLER_PRELUDE + textwrap.dedent(
    """
    import multiprocessing as mp
    import sys
    import threading

    from mosaic.core.pipeline.process_pool import caller_guarded_process_pool


    def mark_the_guard(directory):
        guarded = any(thread.name == "caller-guard" for thread in threading.enumerate())
        Path(directory, f"initialized-{os.getpid()}").write_text(str(guarded))


    if __name__ == "__main__":
        method, directory = sys.argv[1:3]
        if method == "default-forkserver":
            mp.set_start_method("forkserver")
            context = None
        else:
            context = mp.get_context(method)
        pool = caller_guarded_process_pool(
            2, context, initializer=mark_the_guard, initargs=(directory,)
        )
        futures = [
            pool.submit(record_pid_then_wait, f"{directory}/worker-{n}", 600)
            for n in range(2)
        ]
        wait_until(lambda: len(list(Path(directory).glob("worker-*"))) >= 2)
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
    try:
        returncode = run_child_to_files(
            [sys.executable, str(script), *arguments, str(pid_directory)],
            errors=errors_file,
            timeout=60,
        )
    except AssertionError:
        kill_running(chain.from_iterable(_read_pids(pid_directory)))
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
        kill_running([*killed.workers, *killed.family])


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
        kill_running([*killed.workers, *killed.family])
        raise
    _assert_workers_end(killed, "pool workers outlived the caller", count=2)


@pytest.mark.parametrize("method", START_METHODS)
def test_a_workers_initializer_runs_after_the_guard_and_the_guard_still_ends_it(
    method: str, tmp_path: Path
) -> None:
    _skip_unless_available(method)
    killed = _kill_caller(INITIALIZED_PARENT_SCRIPT, [method], tmp_path)
    marks = {
        path.name: path.read_text()
        for path in killed.pid_directory.glob("initialized-*")
    }
    expected = {f"initialized-{pid}": "True" for pid in killed.workers}
    try:
        assert marks == expected, killed.errors
    except AssertionError:
        kill_running([*killed.workers, *killed.family])
        raise
    _assert_workers_end(killed, f"{method} initialized workers outlived the caller")


WATCHER_NAME = "caller-guarded-pool-watcher"


def _spawn_ignoring_pool() -> CallerGuardedPool:
    return caller_guarded_process_pool(
        2, mp.get_context("spawn"), initializer=ignore_stop_signals
    )


def _held_workers(pool: CallerGuardedPool, directory: Path) -> list[int]:
    """Submit two calls that hold a worker each, and return the workers' pids."""
    for tag in (1, 2):
        _ = pool.submit(pool_call, str(directory), tag, 2, "hold")
    wait_for(
        lambda: len(listed_pids(directory)) == 2,
        POLL_DEADLINE_SECONDS,
        "the workers never began their calls",
    )
    return list(listed_pids(directory).values())


@pytest.mark.parametrize("shut_down", [False, True], ids=["running", "shut-down"])
def test_killing_a_pools_workers_ends_every_worker_it_started(
    shut_down: bool, tmp_path: Path
) -> None:
    pool = _spawn_ignoring_pool()
    pids: list[int] = []
    try:
        pids = _held_workers(pool, tmp_path)
        if shut_down:
            pool.shutdown(wait=False, cancel_futures=True)
        pool.kill_workers()
        wait_for(
            lambda: not any(process_alive(pid) for pid in pids),
            POLL_DEADLINE_SECONDS,
            "a worker outlived the kill",
        )
    finally:
        kill_running(pids)
        pool.shutdown(wait=False, cancel_futures=True)


def _new_watcher(before: set[threading.Thread]) -> threading.Thread:
    """The one pool watcher thread that started since *before* was listed."""
    watchers = [
        thread
        for thread in threading.enumerate()
        if thread not in before and thread.name == WATCHER_NAME
    ]
    assert len(watchers) == 1
    return watchers[0]


@pytest.mark.parametrize("ending", ["waited", "not-waited", "dropped"])
def test_a_pool_ended_without_a_break_ends_its_watcher_once_its_workers_exit(
    ending: str,
) -> None:
    """A pool shut down, or dropped without a shutdown, ends its watcher thread."""
    before = set(threading.enumerate())
    pool = caller_guarded_process_pool(2, mp.get_context("spawn"))
    futures = [pool.submit(abs, -n) for n in range(2)]
    assert [future.result() for future in futures] == [0, 1]
    watchers = [_new_watcher(before)]

    if ending == "dropped":
        del pool, futures
        _ = gc.collect()
    else:
        pool.shutdown(wait=ending == "waited")
    watchers[0].join(timeout=POLL_DEADLINE_SECONDS)

    assert not watchers[0].is_alive()


@pytest.mark.parametrize("ending", ["broken", "shut-down-twice"])
def test_a_pool_shut_down_after_its_watcher_returned_is_collected(
    ending: str, tmp_path: Path
) -> None:
    """A shutdown that follows the watcher's return keeps nothing of the pool alive."""
    before = set(threading.enumerate())
    pool = caller_guarded_process_pool(2, mp.get_context("spawn"))
    if ending == "broken":
        future = pool.submit(pool_call, str(tmp_path), 1, 1, "die")
        with pytest.raises(BrokenProcessPool):
            _ = future.result()
    else:
        future = pool.submit(abs, -1)
        assert future.result() == 1
    watcher = _new_watcher(before)
    if ending == "shut-down-twice":
        pool.shutdown()
    watcher.join(timeout=POLL_DEADLINE_SECONDS)
    assert not watcher.is_alive()

    pool.shutdown(wait=False, cancel_futures=True)
    reference = weakref.ref(pool)
    del pool, future
    _ = gc.collect()

    assert reference() is None


def _child_command(function: str, directory: Path) -> list[str]:
    """Run *function* of this module in a child, with *directory* as its argument."""
    program = (
        f"import sys\nfrom {__name__} import {function}\n{function}(sys.argv[1])\n"
    )
    return [sys.executable, "-c", program, str(directory)]


def break_during_shutdown(directory: str) -> None:
    """Shut a pool down without waiting, then end one of its two workers."""
    marks = Path(directory)
    pool = _spawn_ignoring_pool()
    _ = pool.submit(pool_call, directory, 1, 2, "hold")
    _ = pool.submit(pool_call, directory, 2, 2, "die-on-go")
    wait_for(lambda: len(listed_pids(marks)) == 2, POLL_DEADLINE_SECONDS)
    pool.shutdown(wait=False)
    (marks / "go").touch()


def test_a_pool_that_breaks_during_its_shutdown_still_kills_its_survivors(
    tmp_path: Path,
) -> None:
    marks = tmp_path / "marks"
    marks.mkdir()
    assert_marked_child_ends(_child_command("break_during_shutdown", marks), marks)


START_DELAY_SECONDS = 1.0
"""How long a worker of :class:`SlowStartContext` waits before it starts."""


class SlowStartProcess(SpawnProcess):
    """A spawned process that starts :data:`START_DELAY_SECONDS` late."""

    @override
    def start(self) -> None:
        time.sleep(START_DELAY_SECONDS)
        super().start()


class SlowStartContext(SpawnContext):
    """The ``spawn`` context, with each worker started late.

    The executor's ``submit`` wakes its manager thread, then starts a worker. The
    delay lets the manager list the workers it waits on before the new one exists,
    as a loaded host may.
    """

    Process = SlowStartProcess


def a_worker_dies_alone(directory: str) -> None:
    """End the worker that the last submission started, with no call after it.

    The first call holds the first worker. The second starts a second worker and
    ends it, and the caller waits on that call alone.
    """
    pool = caller_guarded_process_pool(
        2, SlowStartContext(), initializer=ignore_stop_signals
    )
    _ = pool.submit(pool_call, directory, 1, 1, "hold")
    dying = pool.submit(pool_call, directory, 2, 2, "die")
    with pytest.raises(BrokenProcessPool):
        _ = dying.result()
    pool.shutdown(wait=False)


def test_a_worker_started_by_the_last_submission_breaks_the_pool_when_it_dies(
    tmp_path: Path,
) -> None:
    marks = tmp_path / "marks"
    marks.mkdir()
    assert_marked_child_ends(_child_command("a_worker_dies_alone", marks), marks)


def count_threads_at_each_fork(directory: str) -> None:
    """Record the threads of this process at each fork of a ``fork`` pool's workers."""
    counts: list[int] = []
    assert threading.active_count() == 1
    os.register_at_fork(before=lambda: counts.append(threading.active_count()))
    pool = caller_guarded_process_pool(2, mp.get_context("fork"))
    futures = [pool.submit(abs, -n) for n in range(2)]
    assert [future.result() for future in futures] == [0, 1]
    pool.shutdown()
    _ = Path(directory, "counts").write_text(" ".join(map(str, counts)))


@pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="the fork pools of TRex run on Linux"
)
def test_a_fork_pool_forks_its_workers_while_the_caller_has_one_thread(
    tmp_path: Path,
) -> None:
    errors = tmp_path / "errors.txt"
    status = run_child_to_files(
        _child_command("count_threads_at_each_fork", tmp_path),
        errors=errors,
        timeout=POLL_DEADLINE_SECONDS * 2,
    )
    assert status == 0, errors.read_text()
    counts = (tmp_path / "counts").read_text().split()

    assert counts, "the pool forked no worker"
    assert set(counts) == {"1"}
