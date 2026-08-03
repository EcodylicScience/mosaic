"""What a scan claims, and therefore what it is allowed to replace.

A scan writes an index by replacing everything it claims and preserving
everything it does not. Which rows those are used to be one question with one
answer -- "is this file under one of the directories I was given" -- and a
:class:`ScanClaim` is that question generalized, because a source may instead
name an explicit list of files.

The distinction is not cosmetic. Importing twelve of a folder's two hundred
videos and later rescanning that import must not delete rows for the other
hundred and eighty-eight, nor for a second import batch beside it. A directory
claim covers a subtree; a file claim covers exactly what it lists.

Deliberately free of dataset and manifest concepts: it takes resolved absolute
paths and answers about resolved absolute paths. Turning a declared source into
one of these needs to know where the dataset is, and that stays with the dataset.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

__all__ = ["ScanClaim"]


@dataclass(frozen=True)
class ScanClaim:
    """The set of files a scan is responsible for.

    Attributes:
        directories: Subtrees claimed whole. A file at or under one is claimed.
        files: Individual files claimed, and nothing beside them.
    """

    directories: frozenset[Path] = frozenset()
    files: frozenset[Path] = frozenset()

    @classmethod
    def over_directories(cls, directories: Iterable[Path]) -> ScanClaim:
        """A claim over whole subtrees."""
        return cls(directories=frozenset(Path(d).resolve() for d in directories))

    @classmethod
    def over_files(cls, files: Iterable[Path]) -> ScanClaim:
        """A claim over exactly these files."""
        return cls(files=frozenset(Path(f).resolve() for f in files))

    def __or__(self, other: ScanClaim) -> ScanClaim:
        """The union of two claims, so several sources scan as one pass."""
        return ScanClaim(
            directories=self.directories | other.directories,
            files=self.files | other.files,
        )

    def __bool__(self) -> bool:
        return bool(self.directories or self.files)

    def claims(self, resolved: Path) -> bool:
        """Whether *resolved* -- an absolute path -- falls inside this claim.

        Args:
            resolved: An absolute, resolved path to test.

        Returns:
            True if the path is one of the claimed files, or lies at or under
            one of the claimed directories.
        """
        if resolved in self.files:
            return True
        for directory in self.directories:
            if resolved == directory or directory in resolved.parents:
                return True
        return False

    def describe(self) -> str:
        """A short human-facing summary, for a message that has to name a claim."""
        parts: list[str] = []
        if self.directories:
            listed = ", ".join(sorted(str(d) for d in self.directories))
            parts.append(f"under {listed}")
        if self.files:
            parts.append(f"{len(self.files)} listed file(s)")
        return "; ".join(parts) if parts else "nothing"
