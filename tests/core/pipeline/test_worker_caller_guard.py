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
from pathlib import Path

import psutil
import pytest

POLL_DEADLINE_SECONDS = 15.0

PARENT_SCRIPT = textwrap.dedent(
    """
    import multiprocessing as mp
    import os
    import signal
    import sys
    import time
    from pathlib import Path

    import psutil

    from mosaic.core.pipeline.process_pool import caller_guarded_process_pool


    def record_pid_then_wait(pid_file, seconds):
        Path(pid_file).write_text(str(os.getpid()))
        time.sleep(seconds)


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
        deadline = time.monotonic() + 30
        expected = 2 if state == "busy" else 1
        while (
            len(list(Path(directory).glob("worker-*"))) < expected
            and time.monotonic() < deadline
        ):
            time.sleep(0.1)
        family = psutil.Process().children(recursive=True)
        Path(directory, "processes.txt").write_text(
            " ".join(str(process.pid) for process in family)
        )
        os.kill(os.getpid(), signal.SIGKILL)
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


def _alive(pid: int) -> bool:
    try:
        process = psutil.Process(pid)
        return process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def _skip_unless_available(method: str) -> None:
    if method.removeprefix("default-") not in mp.get_all_start_methods():
        pytest.skip(f"the {method} start method is unavailable on this platform")


@pytest.mark.parametrize("method", START_METHODS)
@pytest.mark.parametrize("state", ["idle", "busy"])
def test_process_workers_end_when_the_caller_is_killed(
    method: str, state: str, tmp_path: Path
) -> None:
    _skip_unless_available(method)
    script = tmp_path / "caller.py"
    script.write_text(PARENT_SCRIPT)
    pid_directory = tmp_path / "pids"
    pid_directory.mkdir()
    errors_file = tmp_path / "errors.txt"
    # The caller's output goes to files, because a surviving worker would hold a pipe
    # open and keep this test waiting for the end of file.
    with errors_file.open("wb") as errors:
        caller = subprocess.Popen(
            [sys.executable, str(script), method, state, str(pid_directory)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=errors,
        )
        try:
            returncode = caller.wait(timeout=60)
        finally:
            if caller.poll() is None:
                caller.kill()
    workers = [int(path.read_text()) for path in pid_directory.glob("worker-*")]
    # Every process the caller started, the forkserver and a worker that no task
    # recorded included, so a failure leaves none of them running.
    family_file = pid_directory / "processes.txt"
    family = (
        [int(pid) for pid in family_file.read_text().split()]
        if family_file.exists()
        else []
    )
    try:
        assert returncode == -signal.SIGKILL, errors_file.read_text()
        assert workers, errors_file.read_text()
        deadline = time.monotonic() + POLL_DEADLINE_SECONDS
        while any(_alive(pid) for pid in workers) and time.monotonic() < deadline:
            time.sleep(0.2)
        survivors = [pid for pid in workers if _alive(pid)]
        assert survivors == [], f"{method} {state} workers outlived the caller"
    finally:
        for pid in {*workers, *family}:
            if _alive(pid):
                psutil.Process(pid).kill()


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
