"""Parse the command line that heavy and remote share: --log, --wait, and the
wrapped command."""

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import TypeVar

from command_line_parser import EXIT_USAGE, CommandLineParser, nonempty_value, program_parser
from lexical_paths import appended_to_name
from lock_runner_contract import EXIT_MARKER_SUFFIX, NOT_RUN_MARKER_SUFFIX, RunnerName, not_run_status_line
from quiet_paths import present

LOG_SUFFIX = ".log"


class LockRunnerArguments(argparse.Namespace):
    def __init__(self) -> None:
        super().__init__()
        self.log: str | None = None
        self.wait: str | None = None
        self.command: list[str] = []

    def beside_wait(self) -> bool:
        """Return whether an argument that --wait refuses was given: --log or
        a command."""
        return self.log is not None or bool(self.command)


Arguments = TypeVar("Arguments", bound=LockRunnerArguments)


def log_file(text: str) -> str:
    """Return a --log value, refusing one that does not end in .log.

    A --log written bare in front of the command would otherwise take the
    command's name as its file.
    """
    if not text.endswith(LOG_SUFFIX):
        raise argparse.ArgumentTypeError(f'takes a file name ending in {LOG_SUFFIX}, got "{text}"')
    return text


def used_log_file(log: Path) -> Path | None:
    """Return the first of `log` and its result markers that exists. Return
    None when all are absent.

    Dangling symbolic links count as existing, because the capture writes to
    their targets.
    """
    markers = (appended_to_name(log, suffix) for suffix in (EXIT_MARKER_SUFFIX, NOT_RUN_MARKER_SUFFIX))
    for path in (log, *markers):
        if present(lambda: path.exists(follow_symlinks=False)):
            return path
    return None


def lock_runner_parser(name: RunnerName, description: str, example: str = "") -> CommandLineParser:
    """Return the parser for --log, --wait, and the wrapped command. Its usage
    errors close with `name`'s exit-status line. Its help shows a given
    `example` ahead of the epilog."""
    epilog = (
        "Run it with nothing after it: no pipe, no `;`. Read the log in a separate command.\n"
        "If a run is moved to the background, run `%(prog)s --wait FILE` until it prints exit-status=<N>."
    )
    if example:
        epilog = f"Example: {example}\n{epilog}"
    parser = program_parser(description, epilog, closing_line=not_run_status_line(name, EXIT_USAGE, "usage"))
    _ = parser.add_argument("--log", metavar="FILE.log", type=log_file, help="capture the output to FILE.log")
    _ = parser.add_argument(
        "--wait", metavar="FILE", type=nonempty_value, help="wait for the run that logs to FILE, and report its result"
    )
    _ = parser.add_argument("command", nargs=argparse.REMAINDER, help="the command and its arguments")
    return parser


def parse_lock_runner_arguments(
    parser: CommandLineParser, words: Sequence[str], arguments: Arguments
) -> Arguments:
    """Parse `words` into `arguments`, the wrapped command without the `--`
    that argparse leaves ahead of it, and refuse --wait beside any other
    argument. Refuse a --log path that an earlier run used.

    The refusal comes before the lock wait.
    """
    _ = parser.parse_args(words, namespace=arguments)
    if arguments.command[:1] == ["--"]:
        arguments.command = arguments.command[1:]
    if arguments.wait is not None and arguments.beside_wait():
        parser.error("--wait takes no other arguments")
    if arguments.log is not None:
        used = used_log_file(Path(arguments.log))
        if used is not None:
            parser.error(f"--log {arguments.log}: name a fresh log, because {used} already exists")
    return arguments
