"""Resolving the paths an index CSV or a frame-extraction manifest stores.

It also remaps a moved tree.

An index stores a file's location root-relative when it lives inside the dataset
tree and absolute otherwise, so reading one back is a two-case rule rather than a
plain ``Path()``. That rule is shared by the dataset layer and by the media
tooling underneath it, and a second copy of it would be free to drift. A
frame-extraction manifest's ``files`` records are read by one rule too
(:func:`manifest_file_path`), shared by the frames readers and the dataset's
repair.

Deliberately dependency-free and free of the ``Dataset`` object: it holds the
*mechanism*, while which remapping applies and when it is established stays with
the dataset that owns the manifest. ``core.helpers`` is not a home for this -- it
pulls numpy and pandas, and a caller that needs one path rule should not pay for
those. Its one import, ``mosaic.user_paths``, is a leaf of the same shape.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from pathlib import Path, PurePosixPath

from mosaic.user_paths import user_path


def remap_single_path(path: Path, mapping: Sequence[tuple[Path, Path]]) -> Path | None:
    """Rewrite *path* under the first matching prefix pair, or ``None``.

    *mapping* is ``(source_prefix, destination_prefix)`` pairs; the caller orders
    them, longest prefix first, so the most specific match wins.
    """
    for source, destination in mapping:
        try:
            relative = path.relative_to(source)
        except ValueError:
            continue
        return destination / relative
    return None


def resolve_stored_path(
    stored: str | Path,
    anchor: Path,
    *,
    path_map: Sequence[tuple[Path, Path]] | None = None,
) -> Path:
    """Resolve a stored index path cell to an absolute path.

    A relative cell is resolved against *anchor* and returned immediately: it
    names a location inside the tree, which is true whether or not the file
    exists yet, so it is never a candidate for remapping. *anchor* is read only
    in that case.

    An absolute cell is returned as it stands when it exists. When it does not,
    it is a path recorded on another machine or before a move, so *path_map* --
    when the caller has one -- gets to rewrite it; an unmapped path is returned
    unchanged rather than raising, leaving the caller to report the miss.

    A ``~`` is expanded before either test. ``Path("~/x").is_absolute()`` is
    False, so an unexpanded cell would take the relative branch and be anchored
    into ``<anchor>/~/x`` -- a plausible path naming nothing. Cells mosaic writes
    never begin with ``~``; ones reaching this through an op's params can.
    """
    path = user_path(stored)
    if not path.is_absolute():
        return (anchor / path).resolve()
    if path.exists():
        return path
    if path_map is None:
        return path
    remapped = remap_single_path(path, path_map)
    return path if remapped is None else remapped


def resolve_stored_paths(
    stored: Iterable[str | Path],
    anchor: Callable[[], Path],
    *,
    path_map: Sequence[tuple[Path, Path]] | None = None,
) -> list[Path]:
    """Resolve each cell of *stored* as :func:`resolve_stored_path` does.

    The answers are those of the one-at-a-time call. The difference is cost for a
    list of thousands of cells: *anchor* is called once, and each directory is
    resolved once, where the call resolves every component of every path. A cell
    that is absolute, or that holds ``..``, goes through the one-at-a-time call.

    Args:
        stored: The cells to resolve.
        anchor: Gives the directory that a relative cell is resolved against. It is
            called when the first relative cell needs it, and not at all when none
            does.
        path_map: As for :func:`resolve_stored_path`.
    """
    base: Path | None = None
    directories: dict[Path, Path] = {}
    resolved: list[Path] = []
    for cell in stored:
        path = user_path(cell)
        if path.is_absolute():
            resolved.append(resolve_stored_path(path, path, path_map=path_map))
            continue
        if base is None:
            base = anchor()
        if ".." in path.parts or not path.parts:
            resolved.append(resolve_stored_path(path, base, path_map=path_map))
            continue
        joined = base / path
        parent = directories.get(joined.parent)
        if parent is None:
            parent = directories[joined.parent] = joined.parent.resolve()
        target = parent / joined.name
        resolved.append(target.resolve() if target.is_symlink() else target)
    return resolved


def manifest_file_path(recorded: str) -> str:
    """The path of a frame-extraction manifest's ``files`` record, from its directory.

    A manifest written before relative paths names each file by the absolute
    path of the machine that extracted it. The file sits beside the manifest,
    so only that path's file name is kept. A relative record is returned as
    recorded.

    Args:
        recorded: The record's ``path`` as the manifest stores it.

    Returns:
        The file's path relative to the manifest's directory.
    """
    path = PurePosixPath(recorded)
    return path.name if path.is_absolute() else recorded
