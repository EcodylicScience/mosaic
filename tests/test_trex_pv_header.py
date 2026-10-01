"""The frame count of a TREx conversion, read from its ``.pv`` header.

The headers are written by :func:`~tests.helpers.trex.write_pv_header`, which
follows TREx's own writer for each version. The current layout was checked
against a ``.pv`` that TREx wrote from a 60-frame clip, whose header recorded 58.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mosaic.tracking.trex.pv import pv_frame_count
from tests.helpers.trex import write_pv_header


@pytest.mark.parametrize("version", [15, 14, 13, 12, 11, 3, 2, 1])
def test_the_count_is_read_from_every_version_of_the_header(
    tmp_path: Path, version: int
) -> None:
    """Each version adds fields before the count, so each is read to its own layout."""
    path = tmp_path / "clip.pv"
    write_pv_header(path, 1798, version=version)

    assert pv_frame_count(path) == 1798


def test_an_unfinished_conversion_reads_as_zero(tmp_path: Path) -> None:
    """TREx writes the count when it closes the file."""
    path = tmp_path / "clip.pv"
    write_pv_header(path, 0)

    assert pv_frame_count(path) == 0


@pytest.mark.parametrize(
    "content",
    [b"pv", b"PV15\0gray\0\x40\x00", b"x" * 70_000, b""],
    ids=["no-header", "cut-short", "no-string-end", "empty"],
)
def test_a_file_that_is_not_a_header_reads_as_unknown(
    tmp_path: Path, content: bytes
) -> None:
    path = tmp_path / "clip.pv"
    _ = path.write_bytes(content)

    assert pv_frame_count(path) is None


def test_a_newer_header_reads_as_unknown(tmp_path: Path) -> None:
    """A version that TREx did not write when this was written may move the count."""
    path = tmp_path / "clip.pv"
    write_pv_header(path, 1798, version=16)

    assert pv_frame_count(path) is None


def test_a_missing_file_reads_as_unknown(tmp_path: Path) -> None:
    assert pv_frame_count(tmp_path / "absent.pv") is None
