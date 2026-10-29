"""Find the comment runs that wrap-check reports.

Runs are reported for a prose line over the column, or for one short enough
that the next word would still have fit.
"""

from dataclasses import dataclass

from prose_scan import Scan, ends_unit, is_directive, is_prose
from shell_whitespace import BLANKS

OVER_COLUMN = "over column"
SHORT_LINE = "short line"
RULES = (OVER_COLUMN, SHORT_LINE)


@dataclass(frozen=True)
class Finding:
    """Record one rule broken by a run, and the lines inside it that break
    the rule.

    Findings name a run because one fill repairs the whole run.
    """

    start: int
    end: int
    rule: str
    lines: list[int]
    width: int

    def report(self, name: str) -> str:
        """Format `name:start-end:rule: line 7 over 79`, or `lines 1,3`.

        Reviewers copy a number out of the list to key a permitted record. The
        list therefore omits spaces.
        """
        listed = ",".join(str(number) for number in self.lines)
        label = f"lines {listed}" if len(self.lines) > 1 else f"line {listed}"
        relation = "over" if self.rule == OVER_COLUMN else "short of"
        return f"{name}:{self.start}-{self.end}:{self.rule}: {label} {relation} {self.width}"


def findings(scan: Scan, width: int) -> list[Finding]:
    over: dict[int, list[int]] = {}
    short: dict[int, list[int]] = {}
    for number in range(1, scan.count + 1):
        if not scan.prose[number]:
            continue
        prefix = scan.prefix[number]
        text = scan.text(number)
        # Orphans fail is_prose, and the short-line rule reclaims them.
        if not is_prose(text) and not scan.is_orphan(number):
            continue
        if is_directive(text):
            continue
        run = scan.run[number]
        length = len(scan.lines[number])
        if length > width:
            over.setdefault(run, []).append(number)
            continue

        # Short by a wide margin, and the next line continues the same prose
        # with a word that would have fit. Both lines are read as the fill
        # reads them, and each reported join is one that the fill makes.
        if length >= width * 0.75:
            continue
        if not scan.fillable(number) or ends_unit(text):
            continue
        if not scan.fillable(number + 1) or scan.prefix[number + 1] != prefix:
            continue
        next_word = BLANKS.split(scan.text(number + 1, prefix))[0]
        if next_word and length + 1 + len(next_word) <= width:
            short.setdefault(run, []).append(number)

    found: list[Finding] = []
    for run in range(1, scan.runs + 1):
        for rule, lines in zip(RULES, (over, short), strict=True):
            if run in lines:
                found.append(Finding(scan.run_start[run], scan.run_end[run], rule, lines[run], width))
    return found
