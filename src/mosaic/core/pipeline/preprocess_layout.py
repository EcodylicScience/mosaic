"""Where media variants sit in a dataset, and the scan exclusion that keeps them out.

Every variant lives in one kind directory under the media root,
``media/preprocess/``: a run directory per variant, one file per entry and camera
inside it, and one index beside the run directories. The media root is organized
by artifact kind, so the variants sit beside ``transcode/`` and ``frames/``.

A variant is generated media, and a recursive media scan over the media root
filters on extension alone, so it would index a variant as an original. A media
scan therefore steps over the dataset's own variants directory, compared by
resolved path against :func:`media_variants_root`. Directory names are not
matched: a scan source may point outside the dataset at a folder that happens to
be called ``media/preprocess``, and a media root may be set to a directory with
another name.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final

from mosaic.core.helpers import entry_camera_path

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset

__all__ = [
    "MEDIA_ROOT_KEY",
    "PREPROCESS_KIND_DIRECTORY",
    "media_variant_index_path",
    "media_variant_path",
    "media_variant_run_root",
    "media_variant_work_root",
    "media_variants_root",
]

MEDIA_ROOT_KEY: Final = "media"
"""The dataset root the variants' kind directory sits under."""

PREPROCESS_KIND_DIRECTORY: Final = "preprocess"
"""The kind directory under the media root that holds every variant."""

_WORK_DIRECTORY: Final = ".work"
"""The directory inside a run directory that holds each entry's claim."""


def media_variants_root(ds: Dataset) -> Path:
    """The kind directory under *ds*'s media root that holds every variant."""
    return ds.get_root(MEDIA_ROOT_KEY) / PREPROCESS_KIND_DIRECTORY


def media_variant_run_root(ds: Dataset, run_id: str) -> Path:
    """The directory holding the files of variant *run_id*."""
    return media_variants_root(ds) / run_id


def media_variant_work_root(ds: Dataset, run_id: str) -> Path:
    """The directory holding one claim directory per entry of variant *run_id*.

    An entry's claim directory is ``<work_root>/<entry_key>``, keyed on the entry
    alone. A run encodes one camera per entry, the camera a tracker reads, so the
    entry key keeps every claim apart. The claim directory also holds the entry's
    encode in flight, so a partial file never sits among the variant files.
    """
    return media_variant_run_root(ds, run_id) / _WORK_DIRECTORY


def media_variant_path(
    ds: Dataset, run_id: str, group: str, sequence: str, camera: str
) -> Path:
    """The file of variant *run_id* for one entry and camera.

    ``<run_root>/<entry_key>.mp4``, or ``<run_root>/<entry_key>/<camera>.mp4`` when
    *camera* names one: the layout frame extraction gives its runs.
    """
    entry = entry_camera_path(
        media_variant_run_root(ds, run_id), group, sequence, camera
    )
    return entry.with_name(f"{entry.name}.mp4")


def media_variant_index_path(ds: Dataset) -> Path:
    """The one index every variant of *ds* is recorded in."""
    return media_variants_root(ds) / "index.csv"
