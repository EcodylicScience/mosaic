"""Refill one comment run to the column, for rewrap-comment."""

from dataclasses import dataclass, field

from prose_scan import Scan, ends_unit
from shell_whitespace import blank_separated_words

EXIT_NOT_A_RUN = 1
EXIT_NOT_IN_FILE = 2


@dataclass
class Fill:
    """Record the filled file as lines with their terminators, or a refusal.

    `first` and `last` bound the filled run among `lines`, counted from 1.
    """

    status: int = 0
    refusal: str = ""
    lines: list[str] = field(default_factory=list)
    endings: list[str] = field(default_factory=list)
    first: int = 0
    last: int = 0
    notes: list[str] = field(default_factory=list)

    def text(self, run_only: bool) -> str:
        start, stop = (self.first - 1, self.last) if run_only else (0, len(self.lines))
        return "".join(line + ending for line, ending in zip(self.lines[start:stop], self.endings[start:stop], strict=True))


def refusal(scan: Scan, start: int, end: int) -> tuple[int, str]:
    """Return the status and message refusing `start`-`end`, or (0, "")."""
    if start < 1 or start > end or end > scan.count:
        return EXIT_NOT_IN_FILE, f"{start}-{end} is not a range in a file of {scan.count} lines"
    run = scan.run[start]
    if run == 0 or scan.run[end] != run:
        return EXIT_NOT_A_RUN, f"{start}-{end} is not one comment run"
    # Fills reflow whole runs. Every line of a run then wraps against the moved
    # text.
    if scan.run_start[run] != start or scan.run_end[run] != end:
        return EXIT_NOT_A_RUN, (
            f"{start}-{end} is part of the run {scan.run_start[run]}-{scan.run_end[run]}, the range to pass"
        )
    return 0, ""


def filled_lines(prefix: str, words: str, width: int) -> list[str]:
    """Fill greedily, adding each word to the line while it fits.

    Words wider than the column go out alone and over the column.
    """
    lines: list[str] = []
    current = ""
    for word in blank_separated_words(words):
        if not current:
            current = prefix + word
        elif len(current) + 1 + len(word) <= width:
            current = f"{current} {word}"
        else:
            lines.append(current)
            current = prefix + word
    if current:
        lines.append(current)
    return lines


def fill(scan: Scan, width: int, start: int, end: int) -> Fill:
    status, message = refusal(scan, start, end)
    if status:
        return Fill(status=status, refusal=message)
    result = Fill(lines=scan.lines[1:start], endings=scan.endings[1:start])
    # Filled lines take the terminator that the run opened with.
    run_ending = scan.endings[start]
    kept: set[int] = set()
    segment_prefix = ""
    segment_words = ""

    def flush() -> None:
        for line in filled_lines(segment_prefix, segment_words, width):
            result.lines.append(line)
            result.endings.append(run_ending)

    for number in range(start, end + 1):
        if not scan.fillable(number):
            flush()
            segment_words = ""
            result.lines.append(scan.lines[number])
            result.endings.append(scan.endings[number])
            kept.add(len(result.lines))
            continue
        text = scan.text(number)
        if segment_words and scan.prefix[number] != segment_prefix:
            flush()
            segment_words = ""
        if not segment_words:
            segment_prefix = scan.prefix[number]
        segment_words = f"{segment_words} {text}" if segment_words else text
        # Lines ending a unit close the segment. Joining past one undoes a
        # break that the author put there.
        if ends_unit(text):
            flush()
            segment_words = ""
    flush()
    result.first = start
    result.last = len(result.lines)
    result.lines.extend(scan.lines[end + 1 : scan.count + 1])
    result.endings.extend(scan.endings[end + 1 : scan.count + 1])

    # Lines that the fill wrote contain words too wide to break. They take
    # permitted records. Lines that the fill passed over have their words moved
    # by hand.
    for number in range(result.first, result.last + 1):
        if len(result.lines[number - 1]) <= width:
            continue
        if number in kept:
            result.notes.append(
                f"line {number} is over the column and is not prose to fill; move its words yourself"
            )
        else:
            result.notes.append(
                f"line {number} does not contain a break and stays over the column; record it as unbreakable"
            )
    return result
