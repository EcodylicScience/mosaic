"""The reader of a readback ``.npz`` of tracked SLEAP instances."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mosaic.core.track_library.sleap import (
    AppearanceReadback,
    read_appearance_readback,
)
from tests.helpers import ReadbackRecord, write_appearance_readback


def _arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as npz:
        return {name: npz[name] for name in npz.files}


def _rewrite(path: Path, **replaced: np.ndarray) -> Path:
    arrays = {**_arrays(path), **replaced}
    with path.open("wb") as handle:
        np.savez(
            handle,
            frame_idx=arrays["frame_idx"],
            track_name=arrays["track_name"],
            is_user=arrays["is_user"],
            has_vector=arrays["has_vector"],
            vectors=arrays["vectors"],
        )
    return path


def test_records_with_and_without_vectors_round_trip(tmp_path: Path) -> None:
    records = [
        ReadbackRecord(3, "track_0", is_user=True, vector=(1.0, 2.0)),
        ReadbackRecord(3, "track_1"),
        ReadbackRecord(7, "track_0", vector=(-0.5, 0.25)),
    ]
    path = write_appearance_readback(tmp_path / "e.embeddings.npz", records, dim=2)

    readback = read_appearance_readback(path)

    assert isinstance(readback, AppearanceReadback)
    assert readback.frame_idx.tolist() == [3, 3, 7]
    assert readback.frame_idx.dtype == np.int64
    assert readback.track_name.tolist() == ["track_0", "track_1", "track_0"]
    assert readback.is_user.tolist() == [True, False, False]
    assert readback.has_vector.tolist() == [True, False, True]
    assert readback.vectors.dtype == np.float32
    assert readback.vectors.tolist() == [[1.0, 2.0], [0.0, 0.0], [-0.5, 0.25]]


def test_a_file_without_vectors_has_zero_columns(tmp_path: Path) -> None:
    records = [ReadbackRecord(0, "a"), ReadbackRecord(1, "b", is_user=True)]
    path = write_appearance_readback(tmp_path / "e.embeddings.npz", records, dim=0)

    readback = read_appearance_readback(path)

    assert readback.vectors.shape == (2, 0)
    assert not readback.has_vector.any()


def test_an_empty_file_reads_as_no_records(tmp_path: Path) -> None:
    path = write_appearance_readback(tmp_path / "e.embeddings.npz", [], dim=0)

    readback = read_appearance_readback(path)

    assert readback.frame_idx.shape == (0,)
    assert readback.vectors.shape == (0, 0)


def test_mismatched_lengths_are_refused(tmp_path: Path) -> None:
    records = [ReadbackRecord(0, "a"), ReadbackRecord(1, "b")]
    path = write_appearance_readback(tmp_path / "e.embeddings.npz", records, dim=2)
    _ = _rewrite(path, is_user=np.asarray([False], dtype=np.bool_))

    with pytest.raises(ValueError, match="is_user"):
        _ = read_appearance_readback(path)


def test_vectors_of_another_row_count_are_refused(tmp_path: Path) -> None:
    records = [ReadbackRecord(0, "a"), ReadbackRecord(1, "b")]
    path = write_appearance_readback(tmp_path / "e.embeddings.npz", records, dim=2)
    _ = _rewrite(path, vectors=np.zeros((3, 2), dtype=np.float32))

    with pytest.raises(ValueError, match="vectors"):
        _ = read_appearance_readback(path)


def test_a_vector_flag_without_vector_columns_is_refused(tmp_path: Path) -> None:
    records = [ReadbackRecord(0, "a")]
    path = write_appearance_readback(tmp_path / "e.embeddings.npz", records, dim=0)
    _ = _rewrite(path, has_vector=np.asarray([True], dtype=np.bool_))

    with pytest.raises(ValueError, match="has_vector"):
        _ = read_appearance_readback(path)


@pytest.mark.parametrize(
    ("name", "array"),
    [
        ("frame_idx", np.asarray([0.0], dtype=np.float64)),
        ("track_name", np.asarray([0], dtype=np.int64)),
        ("is_user", np.asarray([0], dtype=np.int64)),
        ("has_vector", np.asarray([0], dtype=np.int64)),
        ("vectors", np.zeros((1, 2), dtype=np.float64)),
        ("vectors", np.zeros(1, dtype=np.float32)),
    ],
)
def test_an_array_of_another_dtype_or_rank_is_refused(
    tmp_path: Path, name: str, array: np.ndarray
) -> None:
    records = [ReadbackRecord(0, "a", vector=(1.0, 2.0))]
    path = write_appearance_readback(tmp_path / "e.embeddings.npz", records, dim=2)
    _ = _rewrite(path, **{name: array})

    with pytest.raises(ValueError, match=name):
        _ = read_appearance_readback(path)


def test_a_missing_array_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "e.embeddings.npz"
    with path.open("wb") as handle:
        np.savez(handle, frame_idx=np.asarray([0], dtype=np.int64))

    with pytest.raises(ValueError, match="track_name"):
        _ = read_appearance_readback(path)
