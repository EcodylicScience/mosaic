"""The example datasets, each pinned to a commit of its Hugging Face repository.

A catalog entry names the repository, the commit the archives are read at, and
the archives to fetch in order. Checksums are not copied here: the repository's
``SHA256SUMS`` at the pinned commit owns them. Sizes are copied so a listing needs
no network and a download has a total to report progress against.

Pinning a commit rather than ``main`` means a release fetches the same bytes for
as long as Hugging Face serves the commit. A republished example reaches users
through a change to this table and a release.

This module imports the standard library only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

__all__ = ["CATALOG", "Example", "ExampleArchive"]

_REPOSITORY_PREFIX: Final = "EcodylicScience/mosaic-example-"


@dataclass(frozen=True)
class ExampleArchive:
    """One archive of an example, unpacked under one top-level directory.

    Attributes:
        file: The file name in the repository.
        unpacks_to: The one top-level directory every member lies under.
        download_bytes: The file's size at the pinned revision.
        dataset: Whether ``unpacks_to`` is a mosaic dataset.
    """

    file: str
    unpacks_to: str
    download_bytes: int
    dataset: bool


@dataclass(frozen=True)
class Example:
    """One example: a Hugging Face dataset repository pinned to a commit.

    Attributes:
        name: The CLI name, lowercase with hyphens.
        repository: The Hugging Face dataset repository.
        revision: The repository commit, 40 hexadecimal characters.
        license: The SPDX-style identifier from the repository card.
        summary: One sentence for a listing.
        notebook: The file in ``notebooks/`` that uses the example.
        archives: The archives, in fetch order.
        raw_needs_unpickling: Whether converting the raw files needs the user to
            trust pickle, because they load only by unpickling.
    """

    name: str
    repository: str
    revision: str
    license: str
    summary: str
    notebook: str
    archives: tuple[ExampleArchive, ...]
    raw_needs_unpickling: bool = False

    @property
    def dataset_archive(self) -> ExampleArchive:
        """The one archive whose directory is a mosaic dataset."""
        (archive,) = (archive for archive in self.archives if archive.dataset)
        return archive

    @property
    def download_bytes(self) -> int:
        """The total size of every archive, which is what fetching the example downloads."""
        return sum(archive.download_bytes for archive in self.archives)

    @property
    def manifest_relative(self) -> str:
        """The dataset's manifest, relative to the directory the archives unpack into."""
        return f"{self.dataset_archive.unpacks_to}/dataset.yaml"


CATALOG: Final[tuple[Example, ...]] = (
    Example(
        name="calms21",
        repository=f"{_REPOSITORY_PREFIX}calms21",
        revision="ee1eacd1febed2e2c89a332237d668548f16836a",
        license="cc-by-4.0",
        summary=("CalMS21 mouse pairs, raw tracks and behavior labels to convert."),
        notebook="calms21-template.ipynb",
        archives=(
            ExampleArchive(
                file="calms21-tracks-and-labels.tar.gz",
                unpacks_to="calms21",
                download_bytes=12_613_112,
                dataset=True,
            ),
        ),
        raw_needs_unpickling=True,
    ),
    Example(
        name="shiners",
        repository=f"{_REPOSITORY_PREFIX}shiners",
        revision="e05ba682a9a8403243e206f040e33601ba9ac125",
        license="cc0-1.0",
        summary=(
            "10 shiner school recordings with converted SchoolTracker tracks, "
            "plus a raw sample."
        ),
        notebook="collective-motion-shiners.ipynb",
        archives=(
            ExampleArchive(
                file="shiners-tracks.tar.gz",
                unpacks_to="shiners-tracks",
                download_bytes=349_818_205,
                dataset=True,
            ),
            ExampleArchive(
                file="shiners-raw-sample.tar.gz",
                unpacks_to="shiners-raw-sample",
                download_bytes=8_942_902,
                dataset=False,
            ),
        ),
    ),
    Example(
        name="zebrafish",
        repository=f"{_REPOSITORY_PREFIX}zebrafish",
        revision="3ff7f840a32a1e749694f90e12990ae41fdf93ad",
        license="cc0-1.0",
        summary=("33 zebrafish trials with converted Ctrax tracks, plus a raw sample."),
        notebook="collective-motion-zebrafish.ipynb",
        archives=(
            ExampleArchive(
                file="zebrafish-wt-tracks.tar.gz",
                unpacks_to="zebrafish-wt-tracks",
                download_bytes=767_079_825,
                dataset=True,
            ),
            ExampleArchive(
                file="zebrafish-raw-sample.tar.gz",
                unpacks_to="zebrafish-raw-sample",
                download_bytes=78_553_937,
                dataset=False,
            ),
        ),
    ),
    Example(
        name="calms21-pose",
        repository=f"{_REPOSITORY_PREFIX}calms21-pose",
        revision="2180fbc40148f4712efa97144eeab07eb0f24749",
        license="cc-by-4.0",
        summary="CalMS21 videos and tracks for pose training.",
        notebook="calms21-pose-training-and-tracking.ipynb",
        archives=(
            ExampleArchive(
                file="calms21-pose.tar.gz",
                unpacks_to="calms21-pose",
                download_bytes=1_940_249_616,
                dataset=True,
            ),
        ),
    ),
    Example(
        name="shiners-polo",
        repository=f"{_REPOSITORY_PREFIX}shiners-polo",
        revision="136fb5fe5763fee5fcad0bb473b968585035c536",
        license="cc0-1.0",
        summary="Shiner videos and tracks for POLO and TRex.",
        notebook="shiners-polo-tracking.ipynb",
        archives=(
            ExampleArchive(
                file="shiners-polo.tar.gz",
                unpacks_to="shiners-polo",
                download_bytes=1_090_708_630,
                dataset=True,
            ),
        ),
    ),
)
