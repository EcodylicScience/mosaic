"""Map program names to the first-time setup that `install-tool --setup`
runs."""

import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path

from lock_runner_contract import FALLBACK_VARIABLE, WAIT_TIMEOUT_STATUS, wait_timeout_line
from lossless_text import print_error, write_error
from program_messages import say
from prettier_pin import (
    PRETTIER_VERSION,
    prettier_fault,
    prettier_prefix,
    prettier_program,
)
from program_start import run_program
from tool_installers import installer
from tool_interpreter import (
    first_dotted_number,
    first_line,
    program_head,
    required_minimum,
    shebang_interpreter,
    version_at_least,
    version_report,
)

PROGRAM = "install-tool"


def heavy_runs(command: str | Path) -> bool:
    """Return whether `<command> true` exits 0 under a scratch lock
    directory."""
    with tempfile.TemporaryDirectory() as lock_directory:
        status, _ = probe([command, "true"], {**os.environ, "HEAVY_TASK_DIR": lock_directory})
    return status == 0


def smoke_run(name: str, target: Path, runs: Callable[[str | Path], bool], invocation: str) -> bool:
    """Smoke-run `name` where PATH resolves it, else the installed `target`."""
    on_path = shutil.which(name) is not None
    if not runs(name if on_path else target):
        if on_path:
            say(PROGRAM, f"FAILED -- '{name}' is on PATH but a smoke run did not exit 0.")
        else:
            say(PROGRAM, "FAILED -- the installed program did not exit 0 on a smoke run.")
        return False
    if on_path:
        print(f"install-tool: verified -- '{invocation}' exits 0; installation is working.")
    else:
        print("install-tool: verified -- the installed program runs (exit 0).")
        print(f"install-tool: it is not yet resolvable as '{name}'; add {target.parent} to PATH (above).")
    return True


def smoke_run_heavy(target: Path) -> bool:
    return smoke_run("heavy", target, heavy_runs, "heavy true")


def probe(
    arguments: list[str | Path], environment: dict[str, str], directory: Path | None = None
) -> tuple[int, bytes]:
    """Return the status of a program run once, and its standard error
    output."""
    completed = run_program(
        arguments, directory=directory, environment=environment, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
    )
    return completed.returncode, completed.stderr.rstrip(b"\n")


def replay(said: bytes) -> None:
    if said:
        say(PROGRAM, "it said:")
        write_error(said + b"\n")


def smoke_run_remote(target: Path) -> bool:
    """Run the installed `remote --wait` on a log that no run writes, with a
    zero timeout. The lock runner's wait times out at once, and the probe
    needs neither configuration nor a connection.

    The install directory leads PATH. The runner installed there then resolves
    as it will once the PATH line is in place.
    """
    environment = {
        **os.environ,
        "PATH": f"{target.parent}:{os.environ.get('PATH', '')}",
        "REMOTE_TASK_TIMEOUT": "0",
    }
    with tempfile.TemporaryDirectory() as directory:
        status, said = probe([target, "--wait", Path(directory) / "never-written.log"], environment)
    if status == WAIT_TIMEOUT_STATUS and wait_timeout_line("remote-task").encode() in said:
        print(
            f"install-tool: verified -- 'remote --wait' on a log no run writes exits {WAIT_TIMEOUT_STATUS},"
            + " the lock runner's wait timeout; installation is working."
        )
        return True
    say(
        PROGRAM,
        f"FAILED -- 'remote --wait' on a log no run writes should exit {WAIT_TIMEOUT_STATUS}"
        + f" with the lock runner's wait timeout line, but exited {status}.",
    )
    replay(said)
    return False


def smoke_run_mutate(target: Path) -> bool:
    """Run the installed `mutate` with no argument, outside any `remote` run.

    It must exit 2 with the refusal line. It runs from the install directory
    without the fallback variable. Neither a task clone around the caller nor
    the caller's fallback can then let it start.
    """
    environment = {name: value for name, value in os.environ.items() if name != FALLBACK_VARIABLE}
    status, said = probe([target], environment, target.parent)
    if status == 2 and b"mutate: runs only when remote starts it;" in said:
        print("install-tool: verified -- 'mutate' with no argument exits 2 and refuses to start outside 'remote'; installation is working.")
        return True
    say(PROGRAM, f"FAILED -- 'mutate' with no argument should exit 2 and say it runs only when remote starts it, but exited {status}.")
    replay(said)
    return False


def prose_check_runs(command: str | Path) -> bool:
    return probe([command, "--list"], dict(os.environ))[0] == 0


def smoke_run_prose_check(target: Path) -> bool:
    return smoke_run("prose-check", target, prose_check_runs, "prose-check --list")


def installed(name: str, bin_directory: Path, reason: str, installer_path: str) -> bool:
    """Return whether `name` is on PATH or in the install directory, where an
    install not yet on PATH leaves it.

    Report how to install it when it is neither.
    """
    if shutil.which(name) is not None or os.access(bin_directory / name, os.X_OK):
        return True
    say(PROGRAM, f"'{name}' is not installed, and {reason}.")
    say(PROGRAM, "install it first:")
    print_error(f"    {installer_path}")
    return False


def heavy_installed(_source: Path, bin_directory: Path) -> bool:
    return installed("heavy", bin_directory, "'remote' delegates its lock to it", installer("heavy"))


def interpreter_meets_minimum(source: Path) -> bool:
    """Return whether the interpreter named on the #! line of the program
    reports at least the version that the program and its modules declare."""
    interpreter = shebang_interpreter(first_line(program_head(source)))
    try:
        minimum = required_minimum(source)
    except OSError:
        minimum = None
    if interpreter is None or minimum is None:
        say(PROGRAM, f"cannot read the interpreter and minimum version {source.name} and its modules declare.")
        return False
    if not minimum:
        return True
    report = version_report(interpreter, merge_error=True)
    found = first_dotted_number(report)
    if found and version_at_least(found, minimum):
        return True
    say(PROGRAM, f"{interpreter} reports '{report}'; '{source.name}' needs {interpreter} {minimum} or newer.")
    return False


def remote_installed(_source: Path, bin_directory: Path) -> bool:
    return installed("remote", bin_directory, "'mutate' runs only under it", installer("remote"))


def npm_available(_source: Path, _bin_directory: Path) -> bool:
    if shutil.which("npm") is not None:
        return True
    say(PROGRAM, "npm is required for prettier, which rewrap-md runs.")
    return False


def install_prettier(_target: Path) -> bool:
    """Install the pinned prettier under a dedicated prefix.

    Global installs move whenever projects or node version managers change the
    prettier that they manage.
    """
    prefix = prettier_prefix()
    try:
        prefix.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        say(PROGRAM, f"FAILED -- {error}")
        return False
    status = run_program(
        ["npm", "install", "--prefix", prefix, "--no-audit", "--no-fund", "--silent", f"prettier@{PRETTIER_VERSION}"]
    ).returncode
    if status != 0:
        say(PROGRAM, f"FAILED -- npm install exited {status}.")
        return False
    # npm can exit 0 without leaving a usable program.
    fault = prettier_fault(prettier_program(prefix))
    if fault is not None:
        say(PROGRAM, f"FAILED -- npm exited 0, but {fault}.")
        return False
    print(f"install-tool: installed prettier@{PRETTIER_VERSION} to {prefix}")
    return True


# Checked before anything is copied, given the program and the install
# directory.
INSTALL_PRECONDITIONS: dict[str, Callable[[Path, Path], bool]] = {
    "remote": heavy_installed,
    "mutate": remote_installed,
    "rewrap-md": npm_available,
}

INSTALL_STEPS: dict[str, Callable[[Path], bool]] = {
    "heavy": smoke_run_heavy,
    "remote": smoke_run_remote,
    "mutate": smoke_run_mutate,
    "prose-check": smoke_run_prose_check,
    "rewrap-md": install_prettier,
}
