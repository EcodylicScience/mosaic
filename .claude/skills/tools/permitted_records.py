"""Read the `path:line:rule: keyword: reason` records of a --permitted file."""

import re
from collections.abc import Callable, Hashable, Iterable, Iterator
from functools import cache
from pathlib import Path
from typing import Generic, NamedTuple, TypeVar

from lossless_text import read_text, split_lines
from program_messages import say
from quiet_paths import present
from shell_whitespace import SHELL_WHITESPACE, SHELL_WHITESPACE_CLASS

Key = TypeVar("Key", bound=Hashable)


class Record(NamedTuple):
    path: str
    number: str
    rule: str
    keyword: str


class PermittedFileError(Exception):
    """A --permitted file that is missing or cannot be read."""


class PermittedRecords(Generic[Key]):
    """Track the findings that a --permitted file clears, keyed as its caller
    keys a finding, and the records that cleared one."""

    def __init__(self, keys: Iterable[Key]) -> None:
        self.records: dict[Key, None] = dict.fromkeys(keys)
        self.used: set[Key] = set()

    def clears(self, key: Key) -> bool:
        """Return whether a record permits this finding, and mark the record
        used."""
        if key not in self.records:
            return False
        self.used.add(key)
        return True

    def unused(self) -> list[Key]:
        """Return the records that did not clear a finding, in file order."""
        return [key for key in self.records if key not in self.used]


def numbered_records(text: str) -> Iterator[tuple[int, str]]:
    """Yield each non-blank line of a permitted file, numbered and
    unindented."""
    for number, line in enumerate(split_lines(text), start=1):
        line = line.lstrip(SHELL_WHITESPACE)
        if line:
            yield number, line


@cache
def record_form(keywords: tuple[str, ...]) -> re.Pattern[str]:
    # The path is matched greedily and keeps any colon in it. The keyword is
    # part of the pattern. Reasons that contain `:12:` then cannot move the
    # split. The rule field is required, and a record clears only the finding
    # of its rule at its line.
    alternatives = "|".join(re.escape(keyword) for keyword in keywords)
    return re.compile(f"(.*):([0-9]+):([^:]+):{SHELL_WHITESPACE_CLASS}*({alternatives}):{SHELL_WHITESPACE_CLASS}*\\S.*")


def parse_record(line: str, keywords: tuple[str, ...]) -> Record | None:
    """Return the record in `line`, or None when the line is not of that
    form."""
    match = record_form(keywords).fullmatch(line)
    if match is None:
        return None
    return Record(match[1], match[2], match[3], match[4])


def unreadable(_line: str) -> str:
    """Return why a line is not a permitted record, whatever the line."""
    return "cannot read this permitted record"


def read_records(
    path: Path, keywords: tuple[str, ...], program: str, explain: Callable[[str], str] = unreadable
) -> list[tuple[str, Record]]:
    """Return the records of the --permitted file at `path` in file order,
    each after the `path:line` that locates it.

    Lines that are not records are reported under `program` with the reason
    that `explain` gives, and skipped. Raises PermittedFileError when the file
    is missing or cannot be read.
    """
    if not present(path.is_file):
        raise PermittedFileError(f"no permitted file at {path}")
    try:
        text = read_text(path)
    except OSError as error:
        raise PermittedFileError(f"cannot read {path}") from error
    records: list[tuple[str, Record]] = []
    for number, line in numbered_records(text):
        where = f"{path}:{number}"
        record = parse_record(line, keywords)
        if record is None:
            say(program, f"{where}: {explain(line)}")
        else:
            records.append((where, record))
    return records


def report_unused(program: str, keys: Iterable[str]) -> None:
    """Report each `path:line:rule` record key that cleared no finding.

    The line that such a record names moved, or its finding is fixed.
    """
    for key in keys:
        say(program, f"{key}: this record cleared no finding")
