"""Read the programs a Bash command runs, for the mosaic hooks.

A command is cut into top-level steps by heavy-hook's `top_level_segments`,
and each step is read as its environment assignments, the launchers and
wrappers in front of the program, the program, and its arguments.
"""

import importlib.util
import os
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

from shell_command import joined_continuations, program_name

# Wrappers run the program that follows them. Each maps to the options that
# take a separate value, which are skipped with their value.
WRAPPER_VALUE_OPTIONS: dict[str, frozenset[str]] = {
    "heavy": frozenset({"--log", "--wait"}),
    "remote": frozenset({"--log", "--wait", "--directory", "--depends"}),
    "env": frozenset({"-u", "--unset", "-C", "--chdir", "-S", "--split-string"}),
    "nice": frozenset({"-n", "--adjustment"}),
    "timeout": frozenset({"-k", "--kill-after", "-s", "--signal"}),
    "stdbuf": frozenset({"-i", "-o", "-e", "--input", "--output", "--error"}),
    "time": frozenset(),
    "nohup": frozenset(),
    "caffeinate": frozenset(),
    "command": frozenset(),
    "exec": frozenset(),
}

# Wrappers that take one positional argument before the program.
WRAPPER_POSITIONALS = {"timeout": 1}

ASSIGNMENT = re.compile(r"[A-Za-z_]\w*=")
PYTHON = re.compile(r"python(?:3(?:\.\d+)?)?")

# Interpreter options that take a separate value.
PYTHON_VALUE_OPTIONS = frozenset({"-X", "-W", "--check-hash-based-pycs"})


@dataclass
class Step:
    """One top-level step of a command."""

    raw: str
    assignments: dict[str, str] = field(default_factory=dict)
    wrappers: list[str] = field(default_factory=list)
    program: str = ""
    program_token: str = ""
    arguments: list[str] = field(default_factory=list)

    def launched_by(self, name: str) -> bool:
        """True when `name` wraps the program, as `heavy` or `remote` does."""
        return name in self.wrappers


def load_tool_module(file_name: str) -> ModuleType:
    """Import a sibling program whose file name is not a module name, such as
    `heavy-hook.py`."""
    path = Path(__file__).with_name(file_name)
    spec = importlib.util.spec_from_file_location(path.stem.replace("-", "_"), path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HEAVY_HOOK = load_tool_module("heavy-hook.py")


def _tokens(raw: str) -> list[str]:
    try:
        return shlex.split(raw, comments=True)
    except ValueError:
        return raw.split()


def _python_module(arguments: list[str]) -> tuple[str, list[str]] | None:
    """Return the module and its arguments of `python ... -m module ...`."""
    index = 0
    while index < len(arguments):
        token = arguments[index]
        if token == "-m" and index + 1 < len(arguments):
            return arguments[index + 1], arguments[index + 2 :]
        if token.startswith("-m") and len(token) > 2:
            return token[2:], arguments[index + 1 :]
        if token in PYTHON_VALUE_OPTIONS:
            index += 2
            continue
        if not token.startswith("-"):
            return None
        index += 1
    return None


def parse_step(raw: str) -> Step:
    """Read one step's assignments, wrappers, program and arguments.

    `python -m pytest` reads as the program `pytest`. The program is named
    without its directory, and `program_token` keeps it as written.
    """
    step = Step(raw=raw)
    tokens = _tokens(raw)
    index = 0
    wrapper: str | None = None
    positionals = 0
    while index < len(tokens):
        token = tokens[index]
        if ASSIGNMENT.match(token):
            name, _, value = token.partition("=")
            step.assignments[name] = value
            index += 1
            continue
        if wrapper is not None and token.startswith("-"):
            name = token.split("=", 1)[0]
            skip = (
                2 if "=" not in token and name in WRAPPER_VALUE_OPTIONS[wrapper] else 1
            )
            index += skip
            continue
        if wrapper is not None and positionals > 0:
            positionals -= 1
            index += 1
            continue
        name = program_name(token)
        if name in WRAPPER_VALUE_OPTIONS:
            step.wrappers.append(name)
            wrapper = name
            positionals = WRAPPER_POSITIONALS.get(name, 0)
            index += 1
            continue
        step.program = name
        step.program_token = token
        step.arguments = tokens[index + 1 :]
        break
    if PYTHON.fullmatch(step.program):
        module = _python_module(step.arguments)
        if module is not None:
            step.program, step.arguments = module
    return step


def steps(command: str) -> list[Step]:
    """Return the top-level steps of a Bash command."""
    segments = HEAVY_HOOK.top_level_segments(joined_continuations(command))
    return [parse_step(raw) for raw, _ in segments]


def target_directory(cwd: Path, command_steps: list[Step]) -> Path:
    """Return the directory the command's runs start in: the target of a
    leading `cd`, or the working directory."""
    if (
        command_steps
        and command_steps[0].program == "cd"
        and command_steps[0].arguments
    ):
        target = command_steps[0].arguments[0]
        if "$" not in target:
            return cwd / Path(os.path.expanduser(target))
    return cwd
