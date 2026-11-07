"""The SLEAP analysis HDF5 reader refuses a file it cannot read safely."""

from __future__ import annotations

import json
from pathlib import Path
import time

import h5py
import numpy as np
import pytest

from mosaic.core.track_converter import EntryHints
from mosaic.core.track_library.safe_load import (
    LIMIT_VARIABLE,
    TRUST_VARIABLE,
    UnsafeRawFileError,
)
from mosaic.core.track_library.sleap import (
    SleapAnalysisH5Converter,
    SleapConvertParams,
    read_analysis_track_names,
)
from tests.helpers import write_sleap_analysis_h5


@pytest.fixture(autouse=True)
def untrusted(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run each case without the trust flag that the suite sets."""
    monkeypatch.delenv(TRUST_VARIABLE, raising=False)


def _convert(path: Path) -> None:
    _ = SleapAnalysisH5Converter().convert(
        path, SleapConvertParams(fps=30.0), EntryHints(group="g", sequence="s")
    )


def _tracks(n_frames: int = 4, n_tracks: int = 2, n_nodes: int = 3) -> np.ndarray:
    return np.random.default_rng(0).random((n_frames, n_tracks, n_nodes, 2))


def test_an_oversized_tracks_dataset_is_refused_unread(tmp_path: Path) -> None:
    path = tmp_path / "huge.analysis.h5"
    with h5py.File(path, "w") as handle:
        _ = handle.create_dataset(
            "tracks", shape=(2**20, 2, 3, 2**20), dtype="<f8", chunks=(1, 2, 3, 1024)
        )

    with pytest.raises(UnsafeRawFileError, match=LIMIT_VARIABLE):
        _convert(path)


def test_a_tracks_dataset_in_external_storage_is_refused(tmp_path: Path) -> None:
    raw = tmp_path / "raw.bin"
    raw.write_bytes(b"\0" * 8 * 2 * 2 * 3 * 4)
    path = tmp_path / "external.analysis.h5"
    with h5py.File(path, "w") as handle:
        _ = handle.create_dataset(
            "tracks",
            shape=(2, 2, 3, 4),
            dtype="<f8",
            external=[(str(raw), 0, 8 * 2 * 2 * 3 * 4)],
        )

    with pytest.raises(UnsafeRawFileError, match="external") as caught:
        _convert(path)

    assert caught.value.member == "tracks"


def test_a_tracks_link_to_another_file_is_refused(tmp_path: Path) -> None:
    other = tmp_path / "other.h5"
    write_sleap_analysis_h5(other, _tracks())
    path = tmp_path / "linked.analysis.h5"
    with h5py.File(path, "w") as handle:
        handle["tracks"] = h5py.ExternalLink(str(other), "/tracks")

    with pytest.raises(UnsafeRawFileError, match="link") as caught:
        _convert(path)

    assert caught.value.member == "tracks"


def test_float32_tracks_count_at_the_float64_they_are_read_as(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tracks = _tracks(n_frames=100)
    path = tmp_path / "float32.analysis.h5"
    write_sleap_analysis_h5(path, tracks)
    with h5py.File(path, "a") as handle:
        stored = np.asarray(handle["tracks"], dtype=np.float32)
        dims = handle["tracks"].attrs["dims"]
        del handle["tracks"]
        handle["tracks"] = stored
        handle["tracks"].attrs["dims"] = dims
    monkeypatch.setenv(LIMIT_VARIABLE, str(tracks.size * 4 + 8))

    with pytest.raises(UnsafeRawFileError, match=LIMIT_VARIABLE):
        _convert(path)


def test_tracks_and_scores_past_the_limit_together_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tracks = _tracks(n_frames=100)
    scores = np.ones(tracks.shape[:3])
    path = tmp_path / "scored.analysis.h5"
    write_sleap_analysis_h5(path, tracks, scores)
    monkeypatch.setenv(LIMIT_VARIABLE, str(tracks.size * 8 + 8))

    with pytest.raises(UnsafeRawFileError, match=LIMIT_VARIABLE):
        _convert(path)


def test_a_file_within_the_limit_converts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tracks = _tracks(n_frames=100)
    scores = np.ones(tracks.shape[:3])
    path = tmp_path / "scored.analysis.h5"
    write_sleap_analysis_h5(path, tracks, scores)
    monkeypatch.setenv(LIMIT_VARIABLE, str((tracks.size + scores.size) * 8))

    _convert(path)


def test_a_dims_attribute_of_many_variable_length_strings_is_refused(
    tmp_path: Path,
) -> None:
    path = tmp_path / "dims.analysis.h5"
    write_sleap_analysis_h5(path, _tracks())
    with h5py.File(path, "a") as handle:
        tracks = handle["tracks"]
        del tracks.attrs["dims"]
        tracks.attrs.create(
            "dims", ["track", "xy", "node", "frame"], dtype=h5py.string_dtype()
        )

    with pytest.raises(UnsafeRawFileError, match="attribute dims") as caught:
        _convert(path)

    assert caught.value.member == "tracks"


def test_an_oversized_dims_attribute_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "dims.analysis.h5"
    # The latest format stores an attribute past 64 KiB outside the header.
    with h5py.File(path, "w", libver="latest") as handle:
        handle["tracks"] = _tracks()
        handle["tracks"].attrs["dims"] = np.bytes_(b"x" * 100_000)

    with pytest.raises(UnsafeRawFileError, match="attribute dims"):
        _convert(path)


def test_variable_length_track_names_are_read(tmp_path: Path) -> None:
    path = tmp_path / "names.analysis.h5"
    write_sleap_analysis_h5(path, _tracks())
    with h5py.File(path, "a") as handle:
        _ = handle.create_dataset(
            "track_names", data=["left", "right"], dtype=h5py.string_dtype()
        )

    assert read_analysis_track_names(path) == ("left", "right")


def test_variable_length_track_names_past_the_limit_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "names.analysis.h5"
    write_sleap_analysis_h5(path, _tracks())
    with h5py.File(path, "a") as handle:
        _ = handle.create_dataset(
            "track_names", data=["x" * 5000] * 4, dtype=h5py.string_dtype()
        )
        names = handle["track_names"]
        assert isinstance(names, h5py.Dataset)
        assert names.dtype.itemsize * 4 < 1000
    monkeypatch.setenv(LIMIT_VARIABLE, "10000")

    with pytest.raises(UnsafeRawFileError, match=LIMIT_VARIABLE) as caught:
        _ = read_analysis_track_names(path)

    assert caught.value.member == "track_names"


def test_track_names_in_external_storage_are_refused(tmp_path: Path) -> None:
    raw = tmp_path / "raw.bin"
    raw.write_bytes(b"a" * 16)
    path = tmp_path / "names.analysis.h5"
    write_sleap_analysis_h5(path, _tracks())
    with h5py.File(path, "a") as handle:
        _ = handle.create_dataset(
            "track_names", shape=(2,), dtype="S8", external=[(str(raw), 0, 16)]
        )

    with pytest.raises(UnsafeRawFileError, match="external"):
        _ = read_analysis_track_names(path)


def test_a_tracks_chunk_past_the_limit_is_refused_unread(tmp_path: Path) -> None:
    path = tmp_path / "chunked.analysis.h5"
    with h5py.File(path, "w") as handle:
        # HDF5 inflates a whole chunk to read any element of it.
        _ = handle.create_dataset(
            "tracks",
            shape=(1, 2, 2, 1),
            maxshape=(None, 2, 2, None),
            chunks=(4096, 2, 2, 32768),
            dtype="<f8",
            compression="gzip",
        )

    with pytest.raises(UnsafeRawFileError, match=LIMIT_VARIABLE):
        _convert(path)


def test_a_tracks_group_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "group.analysis.h5"
    with h5py.File(path, "w") as handle:
        handle.create_group("tracks")["0"] = np.zeros(3)

    with pytest.raises(UnsafeRawFileError, match="not a dataset") as caught:
        _convert(path)

    assert caught.value.member == "tracks"


def test_a_dims_string_past_the_attribute_cap_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "dims.analysis.h5"
    write_sleap_analysis_h5(path, _tracks())
    with h5py.File(path, "a") as handle:
        tracks = handle["tracks"]
        del tracks.attrs["dims"]
        tracks.attrs["dims"] = "[" + "[]," * 50_000 + "[]]"

    with pytest.raises(UnsafeRawFileError, match="attribute dims") as caught:
        _convert(path)

    assert caught.value.member == "tracks"


def test_a_refusal_of_dims_shows_them_briefly(tmp_path: Path) -> None:
    path = tmp_path / "dims.analysis.h5"
    write_sleap_analysis_h5(path, _tracks())
    with h5py.File(path, "a") as handle:
        handle["tracks"].attrs["dims"] = json.dumps(["axis"] * 5000)

    with pytest.raises(ValueError, match="dims") as caught:
        _convert(path)

    assert len(str(caught.value)) < 1000


def test_track_names_in_one_oversized_chunk_are_refused_quickly(
    tmp_path: Path,
) -> None:
    path = tmp_path / "names.analysis.h5"
    write_sleap_analysis_h5(path, _tracks())
    with h5py.File(path, "a") as handle:
        names = handle.create_dataset(
            "track_names",
            shape=(200,),
            maxshape=(None,),
            chunks=(1 << 17,),
            dtype=h5py.string_dtype("ascii"),
            compression="gzip",
        )
        names[...] = [b"t"] * 200

    started = time.process_time()
    with pytest.raises(UnsafeRawFileError, match="chunk") as caught:
        _ = read_analysis_track_names(path)

    assert time.process_time() - started < 1.0
    assert caught.value.member == "track_names"


def test_more_variable_length_names_than_the_cap_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "names.analysis.h5"
    write_sleap_analysis_h5(path, _tracks())
    with h5py.File(path, "a") as handle:
        _ = handle.create_dataset(
            "track_names", data=["a", "b", "c"], dtype=h5py.string_dtype()
        )
    monkeypatch.setattr("mosaic.core.track_library.safe_load._MAX_VARIABLE_STRINGS", 2)

    with pytest.raises(UnsafeRawFileError, match="more than 2"):
        _ = read_analysis_track_names(path)


def test_the_empty_track_names_of_an_export_without_tracks_read_as_none(
    tmp_path: Path,
) -> None:
    path = tmp_path / "untracked.analysis.h5"
    write_sleap_analysis_h5(path, _tracks(n_tracks=0))
    with h5py.File(path, "a") as handle:
        # sleap-io writes ``np.array([])`` of encoded names, float64 when empty.
        handle["track_names"] = np.array([])

    assert read_analysis_track_names(path) == ()


def test_numeric_track_names_are_refused(tmp_path: Path) -> None:
    path = tmp_path / "numeric.analysis.h5"
    write_sleap_analysis_h5(path, _tracks())
    with h5py.File(path, "a") as handle:
        handle["track_names"] = np.array([1.0, 2.0])

    with pytest.raises(UnsafeRawFileError, match="not strings"):
        _ = read_analysis_track_names(path)
