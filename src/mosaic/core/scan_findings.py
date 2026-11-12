"""What a source scan drops or cannot place, as findings.

A scan that finds nothing in a source is indistinguishable from a scan of an
empty source unless it says why. :class:`ScanFinding` names the source and the
problem. :class:`FindingLog` prints each finding on stderr as it is made and
keeps it, so ``mosaic scan --json`` and the MCP's ``manage_dataset`` return what
stderr alone would not carry to an agent.

Imports pydantic, :mod:`mosaic.core.strict_model` and the standard library only, so
the agent tool contract can declare a field over :class:`ScanFinding`. The kind is
spelled here once, because :mod:`mosaic.core.manifest` loads pandas.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .strict_model import StrictModel

__all__ = [
    "FindingKind",
    "FindingLog",
    "ScanFinding",
    "ScanProblem",
    "merged",
    "missing_path_finding",
]

FindingKind = Literal["media", "tracks", "labels"]
"""The raw root a source feeds, as ``mosaic.core.manifest.SourceKind`` spells it."""

ScanProblem = Literal[
    "path_missing",
    "listed_file_missing",
    "listed_file_skipped",
    "group_pattern_unmatched",
    "group_path_pattern_unmatched",
]
"""Why a source's files were dropped or left ungrouped.

``path_missing``: the source's directory does not exist, or is a link to one that
does not. ``listed_file_missing``: a file source lists a file that is not there.
``listed_file_skipped``: a file source lists a file that a rule skips, such as a
resource fork. ``group_pattern_unmatched`` and
``group_path_pattern_unmatched``: the pattern matched none of the files the
source indexed, so each has an empty group.
"""


class ScanFinding(StrictModel):
    """One thing a scan dropped or could not place.

    Attributes:
        kind: Which raw root the source feeds.
        source: The source's id.
        problem: What happened.
        message: The problem, for a person.
        path: The path it concerns, when it concerns one.
        pattern: The pattern it concerns, when it concerns one.
    """

    kind: FindingKind
    source: str
    problem: ScanProblem
    message: str
    path: str | None = None
    pattern: str | None = None


def missing_path_finding(
    kind: FindingKind, source: str, resolved: Path
) -> ScanFinding | None:
    """The finding for a source whose path is *resolved*, or ``None`` when it exists.

    A link whose target is gone is missing: ``exists`` follows it.
    """
    if resolved.exists():
        return None
    what = (
        "is a link to a path that does not exist"
        if resolved.is_symlink()
        else "does not exist"
    )
    return ScanFinding(
        kind=kind,
        source=source,
        problem="path_missing",
        message=(
            f"{resolved} {what}, so the source scans no files. Its indexed rows "
            "are kept until the path returns or `mosaic scan --prune-unsourced` "
            "drops them"
        ),
        path=str(resolved),
    )


@dataclass
class FindingLog:
    """The findings of one source's scan, printed on stderr as they are made.

    Attributes:
        kind: Which raw root the source feeds.
        source: The source's id, or ``""`` for a scan of no declared source.
        findings: What has been reported so far, in order.
    """

    kind: FindingKind
    source: str
    findings: list[ScanFinding] = field(default_factory=list[ScanFinding])

    def add(self, finding: ScanFinding) -> None:
        """Record *finding* and print it."""
        self.findings.append(finding)
        subject = (
            f"{finding.kind} source {finding.source!r}"
            if finding.source
            else f"{finding.kind} scan"
        )
        print(f"[WARN] {subject}: {finding.message}", file=sys.stderr)

    def report(
        self,
        problem: ScanProblem,
        message: str,
        *,
        path: Path | None = None,
        pattern: str | None = None,
    ) -> None:
        """Make a finding of this source's, and record and print it."""
        self.add(
            ScanFinding(
                kind=self.kind,
                source=self.source,
                problem=problem,
                message=message,
                path=None if path is None else str(path),
                pattern=pattern,
            )
        )


def merged(logs: Sequence[FindingLog]) -> tuple[ScanFinding, ...]:
    """The findings of every log in *logs*, in order."""
    return tuple(finding for log in logs for finding in log.findings)
