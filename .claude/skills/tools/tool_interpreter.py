"""Read the `#!` interpreter of a program, the minimum version that it and
its modules declare, and the version that a program reports."""

import re
import subprocess
from pathlib import Path

from library_imports import library_modules
from lossless_text import substitution_text
from program_start import run_program
from shell_whitespace import SHELL_WHITESPACE_CLASS, blank_separated_words

MINIMUM_MARKER = "remote-tool-minimum-version:"
# The declaration is looked for on this many of a program's first lines.
HEAD_LINES = 10


def program_head(path: Path) -> list[bytes]:
    """Return the program's first lines, empty when it cannot be read."""
    try:
        with path.open("rb") as handle:
            return [handle.readline() for _ in range(HEAD_LINES)]
    except OSError:
        return []


def first_line(head: list[bytes]) -> str:
    return substitution_text(head[0]) if head else ""


def shebang_interpreter(line: str) -> str | None:
    """Return the interpreter of `#!<absolute path>`, `#!/usr/bin/env <name>`,
    or `#!/usr/bin/env -S <name> [flags]`, None for any other line."""
    if not line.startswith("#!"):
        return None
    words = blank_separated_words(line[2:])
    if len(words) == 1 and words[0].startswith("/"):
        return words[0]
    if len(words) == 2 and words[0] == "/usr/bin/env" and not words[1].startswith("-"):
        return words[1]
    if len(words) > 2 and words[:2] == ["/usr/bin/env", "-S"] and plain_word(words[2]):
        return words[2]
    return None


def plain_word(word: str) -> bool:
    """Return whether `env -S` passes `word` on as written: not an option, an
    assignment, a comment, or a word it unquotes or expands."""
    return not word.startswith(("-", "#")) and not any(character in word for character in "='\"\\$")


def declared_minimum(head: list[bytes]) -> str | None:
    """Return the minimum version that a program's head declares.

    Empty without a declaration, and None for a declaration without a version.
    """
    marker = MINIMUM_MARKER.encode()
    declaration = next((line for line in head if marker in line), None)
    if declaration is None:
        return ""
    space = SHELL_WHITESPACE_CLASS.encode()
    version = re.search(re.escape(marker) + space + rb"*([0-9]+(?:\.[0-9]+)*)", declaration)
    return None if version is None else version.group(1).decode()


def required_minimum(program: Path) -> str | None:
    """Return the highest minimum version that the program and the sibling
    modules it imports declare.

    Empty without a declaration, and None when a declaration has no version.
    Raises OSError when a module cannot be read.
    """
    highest = ""
    for path in [program, *library_modules(program)]:
        minimum = declared_minimum(program_head(path))
        if minimum is None:
            return None
        if minimum and not (highest and version_at_least(highest, minimum)):
            highest = minimum
    return highest


def version_report(program: str | Path, *, merge_error: bool) -> str:
    """Return the output of `program --version`, empty when it cannot run.

    `merge_error` keeps standard error in the report, where old interpreters
    print their version.
    """
    completed = run_program(
        [program, "--version"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT if merge_error else subprocess.DEVNULL
    )
    return substitution_text(completed.stdout)


def first_dotted_number(text: str) -> str:
    found = re.search(r"[0-9]+(?:\.[0-9]+)+", text)
    return found.group(0) if found else ""


def version_at_least(version: str, minimum: str) -> bool:
    """Compare versions component by component as numbers, a missing component
    counting as zero: 3.10 ranks above 3.9, and 3.10 equals 3.10.0."""
    found = [int(part) for part in version.split(".")]
    wanted = [int(part) for part in minimum.split(".")]
    width = max(len(found), len(wanted))
    found += [0] * (width - len(found))
    wanted += [0] * (width - len(wanted))
    return found >= wanted
