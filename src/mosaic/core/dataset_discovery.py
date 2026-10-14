"""The datasets below a directory, found by their manifests.

:func:`discover_datasets` walks down from a directory and returns each dataset
whose manifest it finds, named by its path, ``uuid`` and ``name``. It reads
each manifest and writes nothing.
"""

from __future__ import annotations

import os
from collections import deque
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from typing import Final, NamedTuple

import yaml

from mosaic.core.manifest import (
    ManifestVersionError,
    default_roots,
    is_mosaic_manifest,
    migrate_to_current,
    parse_manifest,
    resolve_manifest_path,
)
from mosaic.core.strict_model import StrictModel, terse

__all__ = [
    "DEFAULT_DISCOVERY_DEPTH",
    "SKIPPED_DIRECTORIES",
    "DiscoveredDataset",
    "discover_datasets",
]

DEFAULT_DISCOVERY_DEPTH: Final = 3
"""How many directory levels below its root a walk descends by default."""


def _top_directories(roots: Iterable[str]) -> frozenset[str]:
    """Return the first component of each relative root path in *roots*.

    An absolute path and a path that leaves the dataset name no directory of it.
    """
    paths = [PurePosixPath(root) for root in roots]
    return frozenset(
        path.parts[0]
        for path in paths
        if path.parts and not path.is_absolute() and path.parts[0] != ".."
    )


SKIPPED_DIRECTORIES: Final = _top_directories(default_roots.values())
"""The top directory of each default root, which a walk does not enter inside a dataset.

A dataset's roots contain its media, tracks, features and models, which can
number thousands of files, and do not contain other datasets. Inside a dataset
a walk skips the top directories of the roots that its manifest declares, and
these names when the manifest does not read. Outside a dataset these names are
ordinary folder names, and a walk enters them.
"""


class DiscoveredDataset(StrictModel):
    """One dataset a walk found.

    Attributes:
        manifest: The absolute path of the manifest file.
        uuid: The manifest's ``uuid``. ``None`` for a manifest without one or
            one that does not read.
        name: The manifest's ``name``. ``None`` for a manifest that does not
            read.
        error: Why the manifest does not read, such as a newer format version.
            ``None`` for one that reads.
    """

    manifest: str
    uuid: str | None = None
    name: str | None = None
    error: str | None = None


def discover_datasets(
    root: Path, *, max_depth: int = DEFAULT_DISCOVERY_DEPTH
) -> list[DiscoveredDataset]:
    """Find the datasets in *root* and in the directories below it.

    A directory is a dataset when it holds a manifest file with a key of
    :data:`~mosaic.core.manifest.MANIFEST_MARKER_KEYS`, or a manifest file that
    does not parse. The walk enters a dataset to find others inside it. It does
    not enter a hidden directory, the top directory of a root that a dataset
    declares, or a directory it has entered through another path. It follows
    symbolic links, and of two paths to one dataset reports the first by depth,
    then by name. A directory that cannot be listed or probed for a manifest is
    passed over.

    Args:
        root: The directory to search.
        max_depth: How many levels below *root* to search. 0 checks *root*
            alone.

    Returns:
        Every dataset found, ordered by manifest path.

    Raises:
        NotADirectoryError: If *root* is not a directory.
        ValueError: If *max_depth* is negative.
    """
    if max_depth < 0:
        msg = f"max_depth must be 0 or more, got {max_depth}"
        raise ValueError(msg)
    if not root.is_dir():
        raise NotADirectoryError(root)
    found: list[DiscoveredDataset] = []
    entered: set[Path] = set()
    # Breadth first, so a directory reached by two paths is entered at the
    # shallower depth and its subtree searched as deep as that allows.
    pending = deque([(root.absolute(), 0)])
    while pending:
        directory, depth = pending.popleft()
        real = directory.resolve()
        if real in entered:
            continue
        entered.add(real)
        dataset = _dataset_at(directory)
        skipped: frozenset[str] = frozenset()
        if dataset is not None:
            found.append(dataset.found)
            skipped = dataset.root_directories
        if depth < max_depth:
            pending.extend(
                (child, depth + 1)
                for child in _subdirectories(directory, skipped=skipped)
            )
    return sorted(found, key=lambda dataset: dataset.manifest)


class _DatasetHere(NamedTuple):
    """A dataset in a directory, with the root directories a walk skips in it."""

    found: DiscoveredDataset
    root_directories: frozenset[str]


def _dataset_at(directory: Path) -> _DatasetHere | None:
    """Return the dataset whose manifest is in *directory*.

    Returns ``None`` when *directory* lacks a manifest, cannot be probed for
    one, such as a directory without read permission, or holds a file under a
    manifest name that another tool wrote.
    """
    try:
        path = resolve_manifest_path(directory)
    except OSError:
        return None
    try:
        data = parse_manifest(path)
    except (OSError, ValueError, yaml.YAMLError) as error:
        return _unreadable(path, error)
    if not is_mosaic_manifest(data):
        return None
    try:
        manifest = migrate_to_current(data, source=str(path))
    except (ValueError, ManifestVersionError) as error:
        return _unreadable(path, error)
    return _DatasetHere(
        DiscoveredDataset(manifest=str(path), uuid=manifest.uuid, name=manifest.name),
        _top_directories(manifest.roots.values()),
    )


def _unreadable(path: Path, error: Exception) -> _DatasetHere:
    """Return the dataset of the manifest at *path*, which raised *error* on reading."""
    return _DatasetHere(
        DiscoveredDataset(manifest=str(path), error=terse(error)), SKIPPED_DIRECTORIES
    )


def _subdirectories(directory: Path, *, skipped: frozenset[str]) -> list[Path]:
    """Return the directories of *directory* that a walk enters, empty when it cannot be listed.

    A hidden directory and a directory named in *skipped* are not entered.
    """
    try:
        with os.scandir(directory) as listing:
            entries = list(listing)
    except OSError:
        return []
    return [
        Path(entry.path)
        for entry in sorted(entries, key=lambda entry: entry.name)
        if not entry.name.startswith(".")
        and entry.name not in skipped
        and _is_directory(entry)
    ]


def _is_directory(entry: os.DirEntry[str]) -> bool:
    """Return whether *entry* is a directory, following links, False when it cannot be stat'ed."""
    try:
        return entry.is_dir()
    except OSError:
        return False
