"""Storing and resolving the paths an index CSV or a frame-extraction manifest stores.

An index stores a file's location root-relative when it lives inside the dataset
tree and absolute otherwise. :func:`stored_form` applies that two-case rule to
store a path. :func:`resolve_stored_path` reverses it to read a path back, and
remaps a path recorded before a tree moved. The dataset layer and the media
tooling underneath it share both functions, so the write rule and the read rule
cannot drift apart. A frame-extraction manifest's ``files`` records are read by
one rule too (:func:`manifest_file_path`), shared by the frames readers and the
dataset's repair.

Deliberately dependency-free and free of the ``Dataset`` object: it holds the
*mechanism*, while which remapping applies and when it is established stays with
the dataset that owns the manifest. ``core.helpers`` is not a home for this -- it
pulls numpy and pandas, and a caller that needs one path rule should not pay for
those. Its one import, ``mosaic.user_paths``, is a leaf of the same shape.

A dataset that adopts another's rows through a relative link stores a cell
relative to itself when the file lies in the link's tree and no symlink lies on
its path (:func:`link_tree`, :func:`tree_relative_cell`).
"""

from __future__ import annotations

import os
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


def stored_form(resolved: Path, base: Path) -> str:
    """The cell that an index stores for *resolved*.

    The cell is root-relative inside *base* and absolute outside it, which
    :func:`resolve_stored_path` reverses. Neither argument is resolved here and
    the filesystem is not touched, so a caller that stores many paths under one
    directory resolves *base* and that directory once.

    Args:
        resolved: A resolved absolute path.
        base: The resolved dataset directory that a relative cell is anchored to.

    Returns:
        *resolved* relative to *base* when it lies under it, else *resolved*.
    """
    try:
        return str(resolved.relative_to(base))
    except ValueError:
        return str(resolved)


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


def link_tree(base: Path, target: Path) -> Path | None:
    """The directory that a relative link's ``..`` steps lead to, or ``None``.

    The tree is the common directory of *base* and the link's target, both taken
    lexically: neither path is resolved. A link that crosses no symlink reaches
    its target and any sibling of it inside the tree by a relative path,
    wherever the tree is mounted.

    Args:
        base: The resolved directory of the dataset that declares the link.
        target: The linked dataset's directory, the link's relative path joined
            to *base* and normalized without resolving it.

    Returns:
        The tree, or ``None`` when it is the file-system root (its own anchor)
        or when ``os.path.commonpath`` refuses the two paths, as it does on
        Windows for paths on two drives.
    """
    try:
        tree = Path(os.path.commonpath([base, target]))
    except ValueError:
        return None
    return None if tree == Path(tree.anchor) else tree


def tree_relative_cell(
    lexical: Path, resolved: Path, anchor: Path, tree: Path | None, target: Path
) -> str | None:
    """The cell that names a file from inside a link's tree, or ``None``.

    The cell is relative only when the file lies under *tree* and neither the
    link's target nor the file is reached through a symlink: *target* and
    *lexical* each resolve to themselves, and *lexical* is the file that
    *resolved* names. A cell that a path map remapped then differs from its
    lexical form, and a file behind a symlink would move apart from the tree, so
    each is stored absolute by the caller.

    Args:
        lexical: The file's normalized path, built from the link's target
            without resolving it.
        resolved: The file as the linked dataset resolves its stored cell,
            with that dataset's path map applied.
        anchor: The resolved directory that the relative cell is read against.
        tree: The link's tree, from :func:`link_tree`.
        target: The link's normalized target, built without resolving it.

    Returns:
        *lexical* relative to *anchor* in POSIX form, or ``None`` when the cell
        is stored absolute, including when ``os.path.relpath`` refuses the two
        paths.
    """
    if tree is None or not lexical.is_relative_to(tree):
        return None
    if target.resolve() != target or lexical.resolve() != lexical:
        return None
    if lexical != resolved.resolve():
        return None
    try:
        return Path(os.path.relpath(lexical, anchor)).as_posix()
    except ValueError:
        return None
