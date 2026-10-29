"""Scan the comment runs of a source file and the prose lines in them.

wrap-check and rewrap-comment read this one scan. They agree on every line and
every bound, and each finding of the check is one that the fill clears.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

from lossless_text import split_lines

TRIPLE_DOUBLE = '"""'
TRIPLE_SINGLE = "'''"
# A triple quote behind any string prefix that Python accepts, in either case.
DOCSTRING_OPENER = re.compile(f"(?i:u|[fb]?r|r?[fb])?({TRIPLE_DOUBLE}|{TRIPLE_SINGLE})")

# By file name, in order. Names with any other extension keep every shape.
EXTENSION_LANGUAGES = (
    ((".tmpl",), "text"),
    ((".sh", ".bash"), "sh"),
    ((".awk",), "awk"),
    ((".py",), "py"),
    ((".ts", ".tsx", ".js", ".jsx", ".mjs"), "js"),
    ((".css",), "css"),
)

LEADING_SPACE = re.compile(r"[ \t]*")
HASH_PREFIX = re.compile(r"[ \t]*# ")
SLASH_PREFIX = re.compile(r"[ \t]*// ")
STAR_PREFIX = re.compile(r"[ \t]*\* ")
HASH_MARKER = re.compile(r"[ \t]*#")
SLASH_MARKER = re.compile(r"[ \t]*//")
PROSE = re.compile(r"[A-Za-z][A-Za-z]+[^A-Za-z]+[A-Za-z]")
DIRECTIVE = re.compile(
    r"eslint-|biome-ignore|stylelint-|prettier-ignore|@ts-|type:|noqa|ruff:|pylint:|fmt: (on|off)|shellcheck |-\*-"
)
UNIT_MARKER = re.compile(r"([-*+=#!]|//|[0-9]+[.)]|@[a-z]+)([ \t]|$)|-{2,}")
# Matches key-value metadata. Docstring header blocks (Alembic "Revision ID:")
# and Google-style section headings keep one item per line.
UNIT_HEADING = re.compile(r"[A-Z][A-Za-z ]{1,24}:([ \t]|$)")
UNIT_END = re.compile(r"([.:!?;][)\"]?|-{2,}|={2,})$")
CODE_KEYWORD = re.compile(r"(if|elif|else|for|while|try|except|with|class|def|async)[ (:]")
# The arrow excludes the two-way arrow `<->` of prose.
CODE_SHAPE = re.compile(r"__|\(\)|[a-z_]\(|;$| = |[A-Za-z0-9_]=[^ ]|(?<!<)->|=>")
# Matches a whole Python import statement, a line that CODE_SHAPE misses.
PYTHON_IMPORT = re.compile(r"(from [\w.]+ )?import [\w.]+( as \w+)?(, [\w.]+( as \w+)?)*")
ORPHAN = re.compile(r"[A-Za-z][A-Za-z'-]*[.,;:!?]?")
# All-capital words are markers such as TODO.
CAPITALS = re.compile(r"[A-Z]+[.,;:!?]?")


def language_for(path: Path, text: str) -> str:
    """Return the language that decides the comment shapes of `path`.

    Extensions name it, and files without one are read by their `#!` line. The
    empty language keeps every shape, the widest reading. Templates are read as
    plain prose from top to bottom.
    """
    name = path.name
    for extensions, language in EXTENSION_LANGUAGES:
        if name.endswith(extensions):
            return language
    if "." in name:
        return ""
    first = text.split("\n", 1)[0]
    if first.startswith("#!") and "python" in first[2:]:
        return "py"
    if first.startswith("#!") and "sh" in first[2:]:
        return "sh"
    return ""


class Shapes(NamedTuple):
    """List the comment shapes that a language has.

    Shell and awk lack block comments. Globs and search strings ending in a
    star then do not open one.
    """

    hash: bool
    slash: bool
    block: bool
    docstring: bool
    plain: bool


def shapes_of(language: str) -> Shapes:
    return Shapes(
        hash=language in ("sh", "py", "awk", ""),
        slash=language in ("js", ""),
        block=language in ("js", "css", ""),
        docstring=language in ("py", ""),
        plain=language == "text",
    )


def leading_space(line: str) -> str:
    match = LEADING_SPACE.match(line)
    return match.group() if match else ""


def line_prefix(line: str, shapes: Shapes) -> str:
    """Return the marker of a line comment and the one space after it, or
    empty."""
    for present, pattern in ((shapes.hash, HASH_PREFIX), (shapes.slash, SLASH_PREFIX)):
        match = pattern.match(line) if present else None
        if match:
            return match.group()
    return ""


def comment_marker(line: str, shapes: Shapes) -> str:
    """Return the marker opening a comment line, with its indentation and
    without the space after it.

    Blank comment lines have one. Paragraph gaps then do not break the run.
    Comments at another indentation have another, and take another prefix.
    """
    for present, pattern in ((shapes.hash, HASH_MARKER), (shapes.slash, SLASH_MARKER)):
        match = pattern.match(line) if present else None
        if match:
            return match.group()
    return ""


def span_prefix(line: str, starred: bool, base: str) -> str:
    """Return the prefix of a line inside a block comment or docstring.

    The prefix is the opening indentation of the span, plus the star that a
    block comment continues a line with. In a docstring a star opening a line
    is a bullet.
    Indentation past the base stays in the text, where `starts_unit` reads an
    indented sample.
    """
    if starred:
        match = STAR_PREFIX.match(line)
        if match:
            return match.group()
    own = leading_space(line)
    if len(own) >= len(base) and line.startswith(base):
        return base
    return own


def is_prose(text: str) -> bool:
    return PROSE.search(text) is not None


def is_directive(text: str) -> bool:
    return DIRECTIVE.match(text) is not None


def starts_unit(text: str) -> bool:
    """Return whether `text` opens an item kept on its own line.

    That is a list item, a marker, an indented sample, or a heading. The
    line-comment marker counts because a block comment can contain one.
    """
    return UNIT_MARKER.match(text) is not None or text[:1] in (" ", "\t") or UNIT_HEADING.match(text) is not None


def ends_unit(text: str) -> bool:
    return UNIT_END.search(text) is not None


def looks_like_code(text: str) -> bool:
    """Return whether `text` reads as code, which the fill keeps on its line.

    A code shape or a whole Python import statement makes a line code. Prose
    opens with the same words as code, so a line opening with a keyword is code
    only when it also ends with a colon.
    """
    if CODE_SHAPE.search(text) is not None or PYTHON_IMPORT.fullmatch(text) is not None:
        return True
    return CODE_KEYWORD.match(text) is not None and text.endswith(":")


@dataclass
class Scan:
    """Record per-line results, indexed by line number from 1.

    Index 0 and the index past the last line contain the defaults. Neighbor
    lookups then do not need a bounds check.
    """

    count: int
    lines: list[str]
    endings: list[str]
    prose: list[bool]
    prefix: list[str]
    run: list[int]
    # Marks each line where a span delimiter was counted.
    delimiter: list[bool]
    run_start: list[int] = field(default_factory=lambda: [0])
    run_end: list[int] = field(default_factory=lambda: [0])

    @property
    def runs(self) -> int:
        return len(self.run_start) - 1

    def text(self, number: int, prefix: str | None = None) -> str:
        """Return the line without its prefix, or without `prefix` when
        given."""
        cut = self.prefix[number] if prefix is None else prefix
        return self.lines[number][len(cut) :]

    def is_orphan(self, number: int) -> bool:
        """Return whether the line is a single word alone, beside prose with
        the same prefix.

        It fails `is_prose`, and the fill exists to reclaim it.
        """
        prefix = self.prefix[number]
        text = self.text(number)
        if not ORPHAN.fullmatch(text) or CAPITALS.fullmatch(text):
            return False
        return any(
            self.prose[other]
            and self.prefix[other] == prefix
            and is_prose(self.text(other, prefix))
            for other in (number - 1, number + 1)
        )

    def fillable(self, number: int) -> bool:
        """Return whether the line is prose that the fill refills, the only
        joins that the check asks for."""
        if not self.prose[number] or self.delimiter[number]:
            return False
        text = self.text(number)
        if not is_prose(text) and not self.is_orphan(number):
            return False
        return not (is_directive(text) or starts_unit(text) or looks_like_code(text))


def scan(text: str, language: str) -> Scan:
    shapes = shapes_of(language)
    records = split_lines(text)
    count = len(records)
    result = Scan(
        count=count,
        lines=[""] * (count + 2),
        endings=["\n"] * (count + 2),
        prose=[False] * (count + 2),
        prefix=[""] * (count + 2),
        run=[0] * (count + 2),
        delimiter=[False] * (count + 2),
    )
    current_key = ""

    def note_run(number: int, key: str) -> None:
        # Runs break where the marker changes and where a line lacks one.
        nonlocal current_key
        if key and key != current_key:
            result.run_start.append(number)
            result.run_end.append(number)
        current_key = key
        if key:
            result.run[number] = result.runs
            result.run_end[-1] = number

    in_plain_run = False
    plain_base = ""
    in_block = False
    in_docstring = False
    span_base = ""
    span_number = 0

    for number, record in enumerate(records, start=1):
        # Each line keeps its terminator, because a file may mix the two.
        line = record
        if line.endswith("\r"):
            line = line[:-1]
            result.endings[number] = "\r\n"
        result.lines[number] = line

        # Shebangs are not prose, and their marker does not open a line-comment
        # run.
        if number == 1 and line.startswith("#!"):
            note_run(number, "")
            continue

        if shapes.plain:
            # The first line of a run sets the indentation that the rest fill
            # against.
            if line.strip(" \t"):
                result.prose[number] = True
                if not in_plain_run:
                    plain_base = leading_space(line)
                    in_plain_run = True
                result.prefix[number] = span_prefix(line, False, plain_base)
                note_run(number, "text")
            else:
                in_plain_run = False
                note_run(number, "")
            continue

        head = line.lstrip(" \t")

        # Lines already inside a span are prose before any delimiter on them
        # counts. The block state is kept as it stood here, and the docstring
        # reading below then does not take a block's closing line for an
        # opener.
        span_line = False
        was_in_block = in_block
        if in_block or in_docstring:
            result.prose[number] = True
            result.prefix[number] = span_prefix(line, in_block, span_base)
            span_line = True

        if shapes.block and not in_docstring:
            opens = line.count("/*")
            closes = line.count("*/")
            # Block comments open only where the delimiter opens the line.
            # Behind code the two characters are a string literal or a glob.
            if not in_block and opens > 0 and head.startswith("/*"):
                result.prose[number] = True
                span_base = leading_space(line)
                result.prefix[number] = span_prefix(line, True, span_base)
                span_number += 1
                span_line = True
                result.delimiter[number] = True
                in_block = opens > closes
            elif in_block and closes > 0:
                in_block = False
                result.delimiter[number] = True

        if shapes.docstring and not in_block and not was_in_block and not comment_marker(line, shapes):
            quotes = line.count(TRIPLE_DOUBLE) + line.count(TRIPLE_SINGLE)
            # Docstrings open only where the quote, or its string prefix, opens
            # the line. Behind an assignment the quotes belong to a string
            # whose body is not prose.
            opens_docstring = DOCSTRING_OPENER.match(head) is not None
            if quotes > 0 and (in_docstring or opens_docstring):
                if not span_line:
                    span_number += 1
                result.prose[number] = True
                if not in_docstring:
                    span_base = leading_space(line)
                result.prefix[number] = span_prefix(line, False, span_base)
                span_line = True
                result.delimiter[number] = True
                # Odd counts open or close. Pairs on one line do both.
                if quotes % 2 == 1:
                    in_docstring = not in_docstring

        if span_line:
            note_run(number, f"span{span_number}")
            continue

        prefix = line_prefix(line, shapes)
        if prefix:
            result.prose[number] = True
            result.prefix[number] = prefix
        note_run(number, comment_marker(line, shapes))

    # Spans still open at the end of the file did not open on a comment. The
    # lines under them are code.
    if (in_block or in_docstring) and result.runs > 0:
        for number in range(result.run_start[-1], result.run_end[-1] + 1):
            result.prose[number] = False
            result.delimiter[number] = False
            result.run[number] = 0
        _ = result.run_start.pop()
        _ = result.run_end.pop()
    return result
