"""Resolving the path cells an index CSV stores, and remapping a moved tree.

An index stores a file's location root-relative when it lives inside the dataset
tree and absolute otherwise, so reading one back is a two-case rule rather than a
plain ``Path()``. That rule is shared by the dataset layer and by the media
tooling underneath it, and a second copy of it would be free to drift.

Deliberately stdlib-only and free of dataset concepts: it holds the *mechanism*,
while which remapping applies and when it is established stays with the dataset
that owns the manifest. ``core.helpers`` is not a home for this -- it pulls numpy
and pandas, and a caller that needs one path rule should not pay for those.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path


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
    """
    path = Path(str(stored).strip())
    if not path.is_absolute():
        return (anchor / path).resolve()
    if path.exists():
        return path
    if path_map is None:
        return path
    remapped = remap_single_path(path, path_map)
    return path if remapped is None else remapped
