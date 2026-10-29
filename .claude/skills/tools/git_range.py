"""Report the lines that a git range adds to each file."""

import os
import subprocess
from pathlib import Path
from typing import NamedTuple

from git_in_directory import repository_toplevel, run_git

# Rename detection needs both names of a renamed file in the pathspec. Filtered
# to the new name alone, every line of it reads as added.
DIFF_ARGUMENTS = ["diff", "-M", "--find-renames=40%"]


class RangeError(Exception):
    pass


class Target(NamedTuple):
    """Name a file to check.

    `name` spells it as findings and permitted records do: relative to the
    repository root for a file that a range names, as given otherwise. `path`
    reads it from the working directory. `lines` are the lines that the range
    adds, None for a file that the caller named. A file that a range names may
    be gone from the working tree, and one that the caller named must exist.
    """

    name: str
    path: Path
    lines: set[int] | None


def hunk_lines(diff: bytes) -> set[int]:
    """Return the new-side line numbers that a `--unified=0` diff adds."""
    numbers: set[int] = set()
    for line in diff.split(b"\n"):
        if line.startswith(b"@@ "):
            span = line.split()[2].removeprefix(b"+").split(b",")
            start = int(span[0])
            count = int(span[1]) if len(span) > 1 else 1
            numbers.update(range(start, start + count))
    return numbers


def range_files(range_text: str, pathspecs: list[str]) -> tuple[Path, dict[str, set[int]]]:
    """Return the repository root, and the lines that the range adds to each
    file, keyed by the path relative to the root."""
    root = repository_toplevel(Path())
    if root is None:
        raise RangeError("--range needs a git repository")
    # For a range that git cannot resolve, the listing comes back empty with
    # exit 0, and a gate reads that as convergence.
    if run_git(Path(), ["rev-list", "-1", range_text], stdout=subprocess.DEVNULL).returncode:
        raise RangeError(f"git cannot read the range '{range_text}'")
    listing = run_git(
        Path(), [*DIFF_ARGUMENTS, "--name-status", "-z", range_text, "--", *pathspecs], stdout=subprocess.PIPE
    )
    if listing.returncode != 0:
        raise RangeError(f"git diff exited {listing.returncode}")
    fields = listing.stdout.split(b"\0")[:-1]
    added: dict[str, set[int]] = {}
    index = 0
    while index < len(fields):
        status_field = fields[index]
        names = 2 if status_field[:1] in (b"R", b"C") else 1
        if index + names >= len(fields):
            break
        old_name = os.fsdecode(fields[index + 1]) if names == 2 else ""
        name = os.fsdecode(fields[index + names])
        index += names + 1
        if not name:
            continue
        literal = [f":(top,literal){side}" for side in (old_name, name) if side]
        diff = run_git(
            Path(), [*DIFF_ARGUMENTS, "--unified=0", "--no-color", range_text, "--", *literal], stdout=subprocess.PIPE
        )
        if diff.returncode != 0:
            raise RangeError(f"git diff exited {diff.returncode}")
        numbers = hunk_lines(diff.stdout)
        if numbers:
            added[name] = numbers
    return root, added


def range_targets(range_text: str | None, operands: list[str]) -> list[Target]:
    """Return the files to check: under `range_text`, the files that it adds
    lines to within the pathspecs `operands`, else the files `operands`."""
    if range_text is None:
        return [Target(operand, Path(operand), None) for operand in operands]
    root, added = range_files(range_text, operands)
    return [Target(name, root / name, lines) for name, lines in added.items()]
