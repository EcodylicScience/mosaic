#!/usr/bin/env -S python3 -B
"""Deny a full mosaic test suite run on this machine, as a PreToolUse hook.

The full suite runs on the task machine through `remote`, which runs
`mosaic-suite` there. A pytest step that names no test path, names only the
whole `tests` directory, or selects across the tree with a marker expression
other than one CI job's marker, is a full suite. The hook denies it here, with
or without `heavy`, and names the command to run instead.

Only a run from a mosaic checkout's toplevel or its `tests` directory counts:
the working directory, or the target of a leading `cd`, decides it. A step
passes when `remote` launches it, when it narrows the run (a path or node id
below `tests`, `-k`, `--lf`, `--collect-only`), or when `MOSAIC_LOCAL_SUITE=1`
is set on the step or in the session's environment. That override is the
user's to give. Every other command, and a pytest run in another repository,
passes untouched.
"""

import os
from pathlib import Path

from hook_payload import read_payload
from mosaic_commands import Step, steps, target_directory
from mosaic_repository import mosaic_toplevel
from pre_tool_use_hook import bash_command, deny

OVERRIDE = "MOSAIC_LOCAL_SUITE"

SUITE_COMMAND = (
    "remote --log=<fresh FILE.log> .claude/skills/tools/mosaic-suite [pytest args]"
)

# pytest options that take a separate value.
VALUE_OPTIONS = frozenset(
    {
        "-m",
        "-k",
        "-n",
        "-p",
        "-c",
        "-o",
        "-W",
        "-r",
        "--dist",
        "--rootdir",
        "--tb",
        "--durations",
        "--durations-min",
        "--maxfail",
        "--deselect",
        "--ignore",
        "--ignore-glob",
        "--basetemp",
        "--log-level",
        "--junitxml",
        "--confcutdir",
        "--override-ini",
        "--numprocesses",
        "--maxprocesses",
        "--import-mode",
        "--log-file",
        "--log-cli-level",
        "--log-file-level",
        "--log-format",
        "--log-date-format",
        "--log-cli-format",
        "--log-cli-date-format",
        "--log-file-format",
        "--log-file-date-format",
        "--log-disable",
        "--color",
        "--capture",
        "--timeout",
        "--report-log",
        "--cov",
        "--cov-report",
        "--max-worker-restart",
    }
)

# Options that make a run select a small part of the suite or run nothing.
NARROWING_OPTIONS = frozenset(
    {
        "-k",
        "--lf",
        "--last-failed",
        "--sw",
        "--stepwise",
        "--co",
        "--collect-only",
        "--fixtures",
        "--markers",
        "--help",
        "-h",
        "--version",
        "--setup-plan",
    }
)

# Markers that select one CI job, each a small part of the suite.
JOB_MARKERS = frozenset({"tracker", "identity", "feral", "lightning_action"})


def whole_suite_paths(root: Path) -> frozenset[Path]:
    """The directories whose run is the whole suite of the checkout at `root`."""
    resolved = root.resolve()
    return frozenset({resolved, resolved / "tests"})


def selection(arguments: list[str]) -> tuple[bool, list[str]]:
    """Read pytest `arguments` as whether an option narrows the run, and the
    paths and node ids they name."""
    paths: list[str] = []
    index = 0
    while index < len(arguments):
        token = arguments[index]
        name, has_value, attached = token.partition("=")
        if not token.startswith("-"):
            paths.append(token)
            index += 1
            continue
        if name in NARROWING_OPTIONS or (token.startswith("-k") and len(token) > 2):
            return True, paths
        value = (
            attached
            if has_value
            else (arguments[index + 1] if index + 1 < len(arguments) else "")
        )
        if name == "-m" and value.strip() in JOB_MARKERS:
            return True, paths
        index += 1 if has_value or name not in VALUE_OPTIONS else 2
    return False, paths


def full_run(arguments: list[str], directory: Path, whole: frozenset[Path]) -> bool:
    """True when pytest `arguments`, run from `directory`, select the whole
    suite: no option narrows them, and the paths they name, or the directory
    when they name none, are the toplevel or `tests`."""
    narrowed, paths = selection(arguments)
    if narrowed:
        return False
    if not paths:
        return directory.resolve() in whole
    return all(
        "::" not in path and (directory / os.path.expanduser(path)).resolve() in whole
        for path in paths
    )


def overridden(step: Step) -> bool:
    return step.assignments.get(OVERRIDE) == "1" or os.environ.get(OVERRIDE) == "1"


def denial(step: Step, directory: Path, whole: frozenset[Path]) -> str | None:
    """Return the reason to deny `step`, run from `directory`, None to let it
    run."""
    if step.launched_by("remote") or overridden(step):
        return None
    whole_suite = full_run(step.arguments, directory, whole)
    if step.program == "pytest" and whole_suite:
        return (
            "A full mosaic suite runs on the task machine, not here. Run "
            f"`{SUITE_COMMAND}` in the background and read the log, or narrow this "
            "run to test paths or node ids (`locate-tests <changed files>` picks "
            f"them). Set {OVERRIDE}=1 on the command only when the user has "
            "allowed a full local run."
        )
    if step.program == "mosaic-suite":
        return (
            f"`mosaic-suite` runs the full suite. Launch it through remote: `{SUITE_COMMAND}`. "
            f"Set {OVERRIDE}=1 only when the user has allowed a full local run."
        )
    return None


def main() -> None:
    payload = read_payload()
    if payload is None:
        return
    command = bash_command(payload)
    cwd = payload.get("cwd")
    if command is None or not isinstance(cwd, str):
        return
    command_steps = steps(command)
    if not any(step.program in ("pytest", "mosaic-suite") for step in command_steps):
        return
    directory = target_directory(Path(cwd), command_steps)
    root = mosaic_toplevel(directory)
    if root is None:
        return
    whole = whole_suite_paths(root)
    for step in command_steps:
        reason = denial(step, directory, whole)
        if reason is not None:
            deny(reason)
            return


if __name__ == "__main__":
    main()
