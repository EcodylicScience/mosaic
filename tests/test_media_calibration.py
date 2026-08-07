"""A calibration is user knowledge, so no scan may overwrite it with silence.

``cm_per_pixel`` says how far apart two adjacent pixels are in the world. It
lives on the media row because that is what it is a property of -- the camera and
the rig -- rather than on a tracks table derived from it: a dataset may mix rigs,
and a reconversion must not have to be told the scale again.

Nothing measures it. ``ffprobe`` cannot report it and no scan can propose one, so
unlike ``(group, sequence)`` it has no guessed counterpart that a rescan might
legitimately re-derive. That is why it survives a rescan unconditionally, rather
than only when the row's identity was assigned.

Empty means *uncalibrated*, and the column is text so it can. As a number the
empty cell would arrive as ``0.0`` or NaN, and the first of those is a scale
factor rather than an absence -- exactly the confusion between "does not say" and
"says one" that the TRex conversion refuses for the same reason.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mosaic.core.dataset import Dataset, new_dataset_manifest
from mosaic.core.media.facts_columns import MEDIA_INDEX_COLUMNS

from .conftest import add_media_sequence


def _dataset_with_media(tmp_path: Path) -> tuple[Dataset, str, str]:
    """A dataset holding one indexed sequence, plus its (group, sequence)."""
    manifest = new_dataset_manifest(name="calib", base_dir=tmp_path / "dataset")
    ds = Dataset(manifest_path=manifest).load(ensure_roots=True)
    add_media_sequence(ds, "seq_a", videos=("a.mp4",), frames=3)
    return ds, "", "seq_a"


def test_the_column_is_part_of_the_media_schema() -> None:
    assert "cm_per_pixel" in MEDIA_INDEX_COLUMNS


def test_an_uncalibrated_sequence_reports_none_rather_than_one(tmp_path: Path) -> None:
    """ "Nobody said" must not read as "one centimetre per pixel"."""
    ds, group, sequence = _dataset_with_media(tmp_path)
    assert ds.media_calibration(group, sequence) is None


def test_a_recorded_calibration_reads_back(tmp_path: Path) -> None:
    ds, group, sequence = _dataset_with_media(tmp_path)

    assert ds.set_media_calibration(0.0412) == 1
    assert ds.media_calibration(group, sequence) == pytest.approx(0.0412)


def test_clearing_returns_it_to_uncalibrated(tmp_path: Path) -> None:
    ds, group, sequence = _dataset_with_media(tmp_path)

    _ = ds.set_media_calibration(0.5)
    _ = ds.set_media_calibration(None)
    assert ds.media_calibration(group, sequence) is None


def test_a_non_positive_scale_refuses(tmp_path: Path) -> None:
    ds, _group, _sequence = _dataset_with_media(tmp_path)
    with pytest.raises(ValueError, match="not a usable scale"):
        _ = ds.set_media_calibration(0.0)


def test_a_rescan_does_not_clear_a_calibration(tmp_path: Path) -> None:
    """The invariant this column exists to keep.

    A rescan refreshes what it measured -- size, mtime, probe -- and a
    calibration is none of those. Losing it here would be silent: the column
    would simply be empty again, and a later run would refuse to produce
    centimetres with no indication that somebody had already said what they were.
    """
    ds, group, sequence = _dataset_with_media(tmp_path)
    _ = ds.set_media_calibration(0.25)

    _ = ds.index_media([ds.get_root("media_raw")], extensions=(".mp4",))

    assert ds.media_calibration(group, sequence) == pytest.approx(0.25)
