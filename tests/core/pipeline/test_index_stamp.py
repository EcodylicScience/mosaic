"""Test the stamp of an index file."""

from __future__ import annotations

from pathlib import Path

from mosaic.core.pipeline.index_stamp import IndexStamp


def test_an_absent_file_still_stamps(tmp_path: Path) -> None:
    """Its appearing later is a change, and recording nothing hides that."""
    stamp = IndexStamp.of(tmp_path / "absent.csv")

    assert not stamp.exists
    assert stamp.size == 0
