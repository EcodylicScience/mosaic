"""Start programs as a shell starts them, and report a program that cannot
start."""

import errno
import os
import shutil
import signal
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import IO

from lossless_text import flush_output, write_error
from quiet_paths import present

# Where a child's standard stream goes: a descriptor, one of subprocess's
# constants, an open file, or this process's own stream for None.
Stream = int | IO[bytes] | None


def program_file(command: str) -> Path | None:
    """Return the file that `command` names, searched on PATH unless the name
    contains a slash, as execvp does."""
    if "/" in command:
        return Path(command)
    found = shutil.which(command)
    return None if found is None else Path(found)


def binary_file(program: Path) -> bool:
    """Return whether `program` reads as a binary, with a NUL byte before its
    first newline, as bash checks."""
    try:
        with program.open("rb") as handle:
            sample = handle.read(80)
    except OSError:
        return False
    return b"\0" in sample.split(b"\n", 1)[0]


def shell_script_command(arguments: list[str], error: OSError) -> list[str] | None:
    """Return the /bin/sh command that runs `arguments` after starting them
    failed with `error`.

    None unless the program is a script without a `#!` line, which execvp hands
    to /bin/sh.
    """
    program = program_file(arguments[0])
    if error.errno != errno.ENOEXEC or program is None or binary_file(program):
        return None
    return ["/bin/sh", str(program), *arguments[1:]]


def start_program(
    arguments: list[str],
    *,
    directory: Path | None = None,
    stdin: Stream = None,
    stdout: Stream = None,
    stderr: Stream = None,
    env: dict[str, str] | None = None,
) -> subprocess.Popen[bytes]:
    """Start `arguments` in `directory`, by default this process's, with every
    inherited descriptor, as a shell starts a command.

    Raises the OSError of a program that cannot start.
    """
    try:
        return subprocess.Popen(
            arguments, cwd=directory, stdin=stdin, stdout=stdout, stderr=stderr, env=env, close_fds=False
        )
    except OSError as error:
        command = shell_script_command(arguments, error)
        if command is None:
            raise
    return subprocess.Popen(
        command, cwd=directory, stdin=stdin, stdout=stdout, stderr=stderr, env=env, close_fds=False
    )


def start_failure(command: str, error: OSError) -> tuple[str, int]:
    """Return why a command could not start, and the status that a shell gives
    it."""
    if error.errno == errno.ENOEXEC:
        return "cannot execute binary file: Exec format error", 126
    if not isinstance(error, FileNotFoundError):
        return error.strerror or str(error), 126
    program = program_file(command)
    if program is None:
        return "command not found", 127
    if present(program.exists):
        # The file is there. The interpreter that its `#!` line names is not.
        return "cannot execute: required file not found", 127
    return error.strerror or str(error), 127


def start_failure_diagnostic(name: str, command: str, error: OSError) -> tuple[bytes, int]:
    """Return the line that says why `command` could not start, under the
    program name `name`, and the status that a shell gives it."""
    reason, status = start_failure(command, error)
    return os.fsencode(f"{name}: {command}: {reason}\n"), status


def report_start_failure(name: str, command: str, error: OSError, stderr: Stream) -> int:
    """Print why `command` could not start, under the program name `name`,
    and return the status that a shell gives it.

    The diagnostic goes where the command's standard error would have gone:
    this process's, a file, or nowhere for a pipe or the null device.
    """
    diagnostic, status = start_failure_diagnostic(name, command, error)
    if stderr is None:
        write_error(diagnostic)
    elif not isinstance(stderr, int):
        _ = stderr.write(diagnostic)
    return status


def shell_status(returncode: int) -> int:
    """Return the return code of a process as a shell reports it: 128+N for a
    death by signal N."""
    return returncode if returncode >= 0 else 128 - returncode


def exec_program(arguments: list[str]) -> OSError:
    """Replace this process with `arguments`, or return the error that stopped
    it.

    Python ignores SIGPIPE and SIGXFSZ, and an ignored signal stays ignored
    across exec. Both are restored to the default first, as a shell hands them
    over.
    """
    _ = signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    _ = signal.signal(signal.SIGXFSZ, signal.SIG_DFL)
    flush_output()
    try:
        os.execvp(arguments[0], arguments)
    except OSError as error:
        command = shell_script_command(arguments, error)
        if command is None:
            return error
    try:
        os.execv(command[0], command)
    except OSError as error:
        return error


def run_program(
    arguments: Sequence[str | Path],
    *,
    name: str | None = None,
    directory: Path | None = None,
    environment: dict[str, str] | None = None,
    stdin: Stream = None,
    input_data: bytes | None = None,
    stdout: Stream = None,
    stderr: Stream = None,
) -> subprocess.CompletedProcess[bytes]:
    """Run a command to its end in `directory`, with `input_data` as its
    standard input when given, and return it with the status that a shell
    gives it.

    Commands that cannot start get a shell's status and diagnostic, under the
    program name `name`, by default this program's file name. The diagnostic
    goes where the command's standard error would have gone, captured with it
    for PIPE.
    """
    command = [str(argument) for argument in arguments]
    if input_data is not None:
        stdin = subprocess.PIPE
    try:
        process = start_program(
            command, directory=directory, stdin=stdin, stdout=stdout, stderr=stderr, env=environment
        )
    except OSError as error:
        program = Path(sys.argv[0]).name if name is None else name
        if stderr == subprocess.PIPE:
            diagnostic, status = start_failure_diagnostic(program, command[0], error)
            return subprocess.CompletedProcess(command, status, b"", diagnostic)
        status = report_start_failure(program, command[0], error, stderr)
        return subprocess.CompletedProcess(command, status, b"", b"")
    output, errors = process.communicate(input_data)
    return subprocess.CompletedProcess(command, shell_status(process.returncode), output or b"", errors or b"")
