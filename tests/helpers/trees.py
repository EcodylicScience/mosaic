"""Snapshots of a directory tree, for the tests that assert a call wrote nothing."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

__all__ = ["PathState", "assert_tree_unchanged", "tree_snapshot"]


class PathState(NamedTuple):
    """One path of a tree as ``lstat`` reports it.

    A directory records its modification time, which moves when an entry is
    created or removed inside it, even one removed again before the snapshot.
    """

    is_dir: bool
    size: int
    mtime_ns: int


def tree_snapshot(root: Path) -> dict[str, PathState]:
    """Every path under *root*, keyed by its POSIX path relative to *root*.

    *root* itself is recorded under ``"."``. Symbolic links are recorded and not
    followed.
    """
    snapshot: dict[str, PathState] = {}
    for path in (root, *sorted(root.rglob("*"))):
        stat = path.lstat()
        is_dir = path.is_dir() and not path.is_symlink()
        snapshot[path.relative_to(root).as_posix()] = PathState(
            is_dir=is_dir,
            size=0 if is_dir else stat.st_size,
            mtime_ns=stat.st_mtime_ns,
        )
    return snapshot


def assert_tree_unchanged(root: Path, before: dict[str, PathState]) -> None:
    """Assert that the tree under *root* matches *before*, naming each difference."""
    after = tree_snapshot(root)
    added = sorted(after.keys() - before.keys())
    removed = sorted(before.keys() - after.keys())
    changed = sorted(
        path for path in before.keys() & after.keys() if before[path] != after[path]
    )
    assert not (added or removed or changed), (
        f"{root} changed. added={added} removed={removed} changed={changed}"
    )
