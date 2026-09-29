"""Locate media variants in a dataset, and exclude them from media scans.

Every variant is stored in one kind directory under the media root,
``media/preprocess/``: a run directory per variant, with one file per entry and
camera and the recipe that the variant was made from, and one index beside the
run directories. The media root is organized by artifact kind. The variants
directory is therefore a sibling of ``transcode/`` and ``frames/``.

A variant is generated media, and a recursive media scan over the media root
filters on extension alone and matches a variant's file. A media scan therefore
skips the dataset's variants directory, through the predicate that
:func:`media_variant_filter` builds. The predicate compares resolved paths against
:func:`media_variants_root` and does not match a directory name, because a scan
source may point outside the dataset at a folder that happens to be called
``media/preprocess``, and a media root may be set to a directory with another
name.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Final

from mosaic.core.helpers import entry_camera_path

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset

__all__ = [
    "MEDIA_ROOT_KEY",
    "PREPROCESS_KIND",
    "media_variant_filter",
    "media_variant_index_path",
    "media_variant_path",
    "media_variant_recipe_path",
    "media_variant_run_root",
    "media_variant_work_root",
    "media_variants_root",
]

MEDIA_ROOT_KEY: Final = "media"
"""The dataset root that contains the variants' kind directory."""

PREPROCESS_KIND: Final = "preprocess"
"""The op kind, which leads every variant's run identifier and names its directory."""

_WORK_DIRECTORY: Final = ".work"
"""The directory inside a run directory that contains each entry's claim."""


def media_variants_root(ds: Dataset) -> Path:
    """Return the kind directory under *ds*'s media root that contains every variant."""
    return ds.get_root(MEDIA_ROOT_KEY) / PREPROCESS_KIND


def media_variant_filter(ds: Dataset) -> Callable[[Path], bool]:
    """Return a predicate that is true for a path inside *ds*'s variants directory.

    The directory is resolved once, here, and each path is resolved when tested. A
    symlink into the directory is therefore caught, and a folder of the same name
    elsewhere is not. A dataset without a media root cannot contain a variant, and
    its predicate is false for every path.
    """
    if not ds.has_root(MEDIA_ROOT_KEY):
        return lambda _path: False
    root = media_variants_root(ds).resolve()

    def is_media_variant(path: Path) -> bool:
        return path.resolve().is_relative_to(root)

    return is_media_variant


def media_variant_run_root(ds: Dataset, run_id: str) -> Path:
    """Return the directory that contains the files of variant *run_id*."""
    return media_variants_root(ds) / run_id


def media_variant_recipe_path(ds: Dataset, run_id: str) -> Path:
    """Return the path of the recipe that variant *run_id* was made from.

    The file is in the form that ``mosaic run --params`` reads. Every run writes
    its validated parameters here. The command that rewrites an entry's variant can
    therefore name the recipe instead of asking for it. The file contains only the
    op's parameters, and mosaic modules do not read it.
    """
    return media_variant_run_root(ds, run_id) / "recipe.json"


def media_variant_work_root(ds: Dataset, run_id: str) -> Path:
    """Return the directory with one claim directory per entry of variant *run_id*.

    An entry's claim directory is ``<work_root>/<entry_key>``, keyed on the entry
    alone. A run encodes one camera per entry, the camera that a tracker reads. The
    entry key therefore keeps every claim apart. The claim directory also contains
    the entry's encode in progress. A partial file is therefore never among the
    variant files.
    """
    return media_variant_run_root(ds, run_id) / _WORK_DIRECTORY


def media_variant_path(
    ds: Dataset, run_id: str, group: str, sequence: str, camera: str
) -> Path:
    """Return the file of variant *run_id* for one entry and camera.

    The path is ``<run_root>/<entry_key>.mp4``, or
    ``<run_root>/<entry_key>/<camera>.mp4`` when *camera* names one, the layout
    that frame extraction gives its runs.
    """
    entry = entry_camera_path(
        media_variant_run_root(ds, run_id), group, sequence, camera
    )
    return entry.with_name(f"{entry.name}.mp4")


def media_variant_index_path(ds: Dataset) -> Path:
    """Return the one index that every variant of *ds* is recorded in."""
    return media_variants_root(ds) / "index.csv"
