"""What a scan claims, and therefore what it is allowed to replace.

A scan writes an index by replacing everything it claims and preserving
everything it does not. Which rows those are used to be one question with one
answer -- "is this file under one of the directories I was given" -- and a
:class:`ScanClaim` is that question generalized, because a source may instead
name an explicit list of files, or share its folder with another source.

The distinction is not cosmetic. Importing twelve of a folder's two hundred
videos and later rescanning that import must not delete rows for the other
hundred and eighty-eight, nor for a second import batch beside it. A directory
claim covers a subtree; a file claim covers exactly what it lists.

A directory may carry a filter. A filtered directory covers only the paths at
or under it that the filter accepts, which is how two sources with disjoint
patterns over one folder each claim their own files. :class:`Unclaimed` is the
filter of a directory claimed less what other sources claim. A directory without a
filter covers its whole subtree. The filter judges a path by its own name and
place, so a path that only a symlink brings into a source, such as a target
with another name or outside the directory, is claimed as a file beside it.

Deliberately free of dataset and manifest concepts: it takes resolved absolute
paths and answers about resolved absolute paths. A filter is anything with an
``accepts`` method (:class:`PathFilter`). Turning a declared source into one of
these needs to know where the dataset is, and that stays with the dataset.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

__all__ = ["AnyFilter", "PathFilter", "ScanClaim", "Unclaimed"]


class PathFilter(Protocol):
    """Which paths at or under a claimed directory the claim covers."""

    def accepts(self, resolved: Path) -> bool:
        """Whether the claim covers *resolved*, an absolute, resolved path."""
        ...


@dataclass(frozen=True)
class AnyFilter:
    """Accepts a path that any of its filters accepts.

    Attributes:
        filters: The filters joined, none of them an :class:`AnyFilter`.
    """

    filters: tuple[PathFilter, ...]

    def accepts(self, resolved: Path) -> bool:
        """Whether any of :attr:`filters` accepts *resolved*."""
        return any(f.accepts(resolved) for f in self.filters)


class Unclaimed:
    """Accepts a path that another claim does not cover, or whose file is gone.

    A path with no file on disk is accepted without asking the other claim: a
    file that was removed is no other claim's to keep. The other claim is built
    by *build* on the first call that asks about a file on disk, and kept, so a
    claim asked only about removed files, or about none, never builds it.
    """

    def __init__(self, build: Callable[[], ScanClaim]) -> None:
        self._build = build
        self._claim: ScanClaim | None = None

    def claim(self) -> ScanClaim:
        """The claim whose paths this filter rejects, built once."""
        if self._claim is None:
            self._claim = self._build()
        return self._claim

    def accepts(self, resolved: Path) -> bool:
        """Whether *resolved* has no file, or the other claim leaves it out."""
        if not resolved.exists():
            return True
        return not self.claim().claims(resolved)


def _members(path_filter: PathFilter) -> tuple[PathFilter, ...]:
    if isinstance(path_filter, AnyFilter):
        return path_filter.filters
    return (path_filter,)


def _either(left: PathFilter | None, right: PathFilter | None) -> PathFilter | None:
    """The filter accepting what *left* or *right* accepts. ``None`` accepts all."""
    if left is None or right is None:
        return None
    joined: list[PathFilter] = []
    for member in (*_members(left), *_members(right)):
        if member not in joined:
            joined.append(member)
    return joined[0] if len(joined) == 1 else AnyFilter(tuple(joined))


def _no_directories() -> dict[Path, PathFilter | None]:
    return {}


@dataclass(frozen=True)
class ScanClaim:
    """The set of files a scan is responsible for.

    Attributes:
        directories: Claimed directories, each with the filter of the paths at
            or under it that the claim covers. ``None`` claims the whole subtree.
        files: Individual files claimed, and nothing beside them.
    """

    directories: Mapping[Path, PathFilter | None] = field(
        default_factory=_no_directories
    )
    files: frozenset[Path] = frozenset()

    @classmethod
    def over_directories(cls, directories: Iterable[Path]) -> ScanClaim:
        """A claim over whole subtrees."""
        return cls(directories={Path(d).resolve(): None for d in directories})

    @classmethod
    def over_filtered(
        cls, directory: Path, path_filter: PathFilter | None
    ) -> ScanClaim:
        """A claim over the paths at or under *directory* that *path_filter* accepts.

        Args:
            directory: The directory claimed. It is resolved here.
            path_filter: Which paths it covers, or ``None`` for the subtree.
        """
        return cls(directories={Path(directory).resolve(): path_filter})

    @classmethod
    def over_files(cls, files: Iterable[Path]) -> ScanClaim:
        """A claim over exactly these files."""
        return cls(files=frozenset(Path(f).resolve() for f in files))

    def __or__(self, other: ScanClaim) -> ScanClaim:
        """The union of two claims, so several sources scan as one pass.

        Two filters over one directory join into one that accepts what either
        accepts, and a directory claimed whole by either side stays whole.
        """
        directories = dict(self.directories)
        for directory, path_filter in other.directories.items():
            directories[directory] = (
                _either(directories[directory], path_filter)
                if directory in directories
                else path_filter
            )
        return ScanClaim(directories=directories, files=self.files | other.files)

    def __bool__(self) -> bool:
        return bool(self.directories or self.files)

    def claims(self, resolved: Path) -> bool:
        """Whether *resolved* -- an absolute path -- falls inside this claim.

        Args:
            resolved: An absolute, resolved path to test.

        Returns:
            True if the path is one of the claimed files, or lies at or under
            one of the claimed directories and that directory's filter, if any,
            accepts it.
        """
        if resolved in self.files:
            return True
        for directory, path_filter in self.directories.items():
            if resolved == directory or directory in resolved.parents:
                if path_filter is None or path_filter.accepts(resolved):
                    return True
        return False

    def describe(self) -> str:
        """A short human-facing summary, for a message that has to name a claim."""
        parts: list[str] = []
        if self.directories:
            listed = ", ".join(
                f"{d} (filtered)" if f is not None else str(d)
                for d, f in sorted(self.directories.items(), key=lambda item: item[0])
            )
            parts.append(f"under {listed}")
        if self.files:
            parts.append(f"{len(self.files)} file(s)")
        return "; ".join(parts) if parts else "nothing"
