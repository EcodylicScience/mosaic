"""What a TREx conversion read, from its ``.pv`` header: a frame count and its sources.

The headers are written by :func:`~tests.helpers.trex.write_pv_header`, which
follows TREx's own writer for each version. The current layout was checked
against a ``.pv`` that TREx wrote from a 60-frame clip, whose header recorded 58
frames and the clip's path as its source, and against one that it wrote from two
60-frame clips, whose header is held below.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest

from mosaic.core.pipeline.tracks_axis import tail_loss_of_files
from mosaic.core.pipeline.tracks_index import frame_axis_verdict
from mosaic.tracking.trex.pv import PvHeader, read_pv_header

from tests.helpers import write_pv_header

_TWO_CLIPS: Final = b"".join(
    [
        b"PV15\0rgb8\0",
        bytes.fromhex("a000 7800"),  # 160 by 120
        bytes.fromhex("0000 0000 401f 401f"),  # crop offsets 0, 0, 8000, 8000
        b"\xff" * 16,  # no conversion range
        b"[/data/clips/x264-60.mp4,/data/clips/bf0-60.mp4]\0",
        bytes.fromhex("04 7600 0000"),  # line size 4, 118 frames
        bytes.fromhex("1cf5 1000 0000 0000 99d8 a96f b95c 0600"),
        b"multi\0",
    ]
)
"""The header that TREx build ``4b48601`` wrote from two 60-frame H.264 clips.

Byte for byte what it wrote up to the name, with the clips' directory shortened
to ``/data/clips``. The first clip has B-frames and the second none.
"""


@pytest.mark.parametrize("version", [15, 14, 13, 12, 11, 3, 2, 1])
def test_the_count_is_read_from_every_version_of_the_header(
    tmp_path: Path, version: int
) -> None:
    """Each version adds fields before the count, so each is read to its own layout."""
    path = tmp_path / "clip.pv"
    write_pv_header(path, 1798, version=version)

    header = read_pv_header(path)
    assert header is not None
    assert header.frames == 1798


def test_an_unfinished_conversion_reads_as_zero(tmp_path: Path) -> None:
    """TREx writes the count when it closes the file."""
    path = tmp_path / "clip.pv"
    write_pv_header(path, 0)

    header = read_pv_header(path)
    assert header is not None
    assert header.frames == 0


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

    assert read_pv_header(path) is None


def test_a_newer_header_reads_as_unknown(tmp_path: Path) -> None:
    """A version that TREx did not write when this was written may move the count."""
    path = tmp_path / "clip.pv"
    write_pv_header(path, 1798, version=16)

    assert read_pv_header(path) is None


def test_a_missing_file_reads_as_unknown(tmp_path: Path) -> None:
    assert read_pv_header(tmp_path / "absent.pv") is None


def test_one_source_is_its_path(tmp_path: Path) -> None:
    path = tmp_path / "clip.pv"
    write_pv_header(path, 58, sources=["/data/clips/a b.mp4"])

    assert read_pv_header(path) == PvHeader(frames=58, sources=("/data/clips/a b.mp4",))


def test_several_sources_are_each_path(tmp_path: Path) -> None:
    """TREx writes several sources between brackets, unquoted."""
    path = tmp_path / "clip.pv"
    write_pv_header(path, 118, sources=["/data/a.mp4", "/data/b.mp4"])

    header = read_pv_header(path)
    assert header is not None
    assert header.sources == ("/data/a.mp4", "/data/b.mp4")


def test_a_header_trex_wrote_from_two_clips_is_read_as_two(tmp_path: Path) -> None:
    """Two clips are read as two, and TREx reading them two short is a mismatch.

    Its 118 frames of 120 are two short, which the first clip's B-frames would
    explain alone, but TREx loses frames at each boundary of several files, so
    none is allowed.
    """
    path = tmp_path / "multi.pv"
    _ = path.write_bytes(_TWO_CLIPS)

    header = read_pv_header(path)

    assert header is not None
    assert header == PvHeader(
        frames=118,
        sources=("/data/clips/x264-60.mp4", "/data/clips/bf0-60.mp4"),
    )
    loss = tail_loss_of_files("trex", header.sources)
    assert loss == 0
    verdict = frame_axis_verdict("trex", read=118, media=120, known_tail_loss=loss)
    assert verdict == "mismatch"


@pytest.mark.parametrize("version", [14, 3, 1])
def test_a_header_before_version_15_records_no_source(
    tmp_path: Path, version: int
) -> None:
    path = tmp_path / "clip.pv"
    write_pv_header(path, 58, sources=["/data/a.mp4"], version=version)

    assert read_pv_header(path) == PvHeader(frames=58, sources=())
