"""Stand in for a tool environment's interpreter in the tests of the decode probe.

A tracker that declares a decode probe runs it through the ``python`` of the tool's
environment, found by the tool's location ladder.
:func:`install_fake_tool_python` writes a ``python`` shell script into a temporary
directory and points the tool's ``MOSAIC_<TOOL>_BIN`` at that directory. The script
records the arguments of each run. It prints the output and exits with the code that
the test chose. A test drives every outcome of the probe on a machine without any of
the tools.
"""

from __future__ import annotations

import shlex
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from mosaic.tracking.common.toolenv import ToolEnv


@dataclass(frozen=True, slots=True)
class FakeToolPython:
    """Name the files of a fake tool interpreter, which a test reads back after a run.

    Attributes:
        directory: The directory that contains the ``python`` script.
        log: One line per run of the script, containing its first and third
            arguments separated by a tab.
        program: The second argument of the latest run, which is the program
            passed with ``-c``.
    """

    directory: Path
    log: Path
    program: Path

    @property
    def interpreter(self) -> Path:
        """Return the path of the ``python`` script."""
        return self.directory / "python"

    def calls(self) -> list[tuple[str, str]]:
        """Return the flag and the file path of each run, in order."""
        if not self.log.is_file():
            return []
        lines = self.log.read_text().splitlines()
        return [(flag, path) for flag, path in (line.split("\t") for line in lines)]

    def last_program(self) -> str:
        """Return the program that the latest run was given."""
        return self.program.read_text()


def install_fake_tool_python(
    monkeypatch: pytest.MonkeyPatch,
    env: ToolEnv,
    directory: Path,
    *,
    exit_code: int = 0,
    output: str = "",
    seconds: int = 0,
    startable: bool = True,
    imports: Path | None = None,
) -> FakeToolPython:
    """Make *env*'s location ladder find a fake ``python`` in *directory*.

    The script sleeps for *seconds*, prints *output* to standard error, and exits
    with *exit_code*. With *imports* set, it runs the program instead. The tool's
    conda variable is cleared, because the ladder reads it before the bin
    variable. ``MOSAIC_ALLOW_TOOL_CODECS`` is cleared too, because a codec that it
    lists is allowed before any probe runs.

    Args:
        monkeypatch: The test's monkeypatch, which restores every variable set.
        env: The placement declaration of the tool whose ladder is redirected.
        directory: Where the script and its records are written.
        exit_code: The exit status of every run.
        output: The text written to standard error by every run.
        seconds: How long each run sleeps before it prints its output.
        startable: False leaves the ``python`` script unwritten. The
            interpreter then cannot start.
        imports: A directory of stand-in modules. The script then runs its
            arguments with this test's interpreter, with *imports* first on the
            module search path, in place of *seconds*, *output* and *exit_code*.

    Returns:
        The records of the script.
    """
    directory.mkdir(parents=True, exist_ok=True)
    fake = FakeToolPython(
        directory=directory,
        log=directory / "calls.log",
        program=directory / "program.py",
    )
    if startable:
        answer = directory / "output.txt"
        _ = answer.write_text(output)
        pause = f"sleep {seconds}\n" if seconds else ""
        answering = (
            f"{pause}cat {shlex.quote(str(answer))} >&2\nexit {exit_code}\n"
            if imports is None
            else f"PYTHONPATH={shlex.quote(str(imports))} "
            f'exec {shlex.quote(sys.executable)} "$@"\n'
        )
        script = (
            "#!/bin/sh\n"
            f'printf \'%s\\t%s\\n\' "$1" "$3" >> {shlex.quote(str(fake.log))}\n'
            f"printf '%s' \"$2\" > {shlex.quote(str(fake.program))}\n"
            f"{answering}"
        )
        _ = fake.interpreter.write_text(script)
        fake.interpreter.chmod(0o755)
    entry = "python" if env.bin_mode == "direct" else env.locator or "python"
    monkeypatch.delenv(env.conda_env_var, raising=False)
    monkeypatch.delenv("MOSAIC_ALLOW_TOOL_CODECS", raising=False)
    monkeypatch.setenv(env.bin_var, str(directory / entry))
    return fake
