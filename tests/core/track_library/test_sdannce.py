"""The ``sdannce_mat`` converter: a social or lone s-DANNCE file to world meters."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import h5py
import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from pydantic import ValidationError
from scipy.io import savemat

from mosaic.core.pipeline.tracks_index import read_tracks_index
from mosaic.core.schema import ensure_track_schema
from mosaic.core.track_converter import (
    TRACK_CONVERTERS,
    EntryHints,
    FrameRateUnknownError,
)
from mosaic.core.track_library.sdannce import (
    RAT23_KEYPOINT_NAMES,
    SdannceMatConverter,
    SdannceMatParams,
)
from tests.helpers import make_dataset, write_sdannce_mat

FRAMES = 12
FPS = 50.0
FOCAL = 301
PARTNER = 701


def _convert(
    path: Path,
    *,
    hints: EntryHints | None = None,
    **params: object,
) -> pd.DataFrame:
    overrides: dict[str, object] = {"fps": FPS, **params}
    return SdannceMatConverter().convert(
        path,
        SdannceMatParams.from_overrides(overrides),
        hints or EntryHints(group="cage1", sequence="day1"),
    )


def _write(
    tmp_path: Path,
    *,
    social: bool = True,
    blank: Sequence[tuple[int, int, int | None]] = (),
    frames: int = FRAMES,
) -> tuple[Path, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    path = tmp_path / "LONGEVANS_M0_20221128_0493_S.mat"
    m1, m2 = write_sdannce_mat(
        path, frames, focal=FOCAL, partner=PARTNER, social=social, blank=blank
    )
    return path, m1, m2


def _save(
    path: Path,
    m1: object,
    m2: object,
    *,
    ratint: object = float(FOCAL),
    ratp_int: object = float(PARTNER),
    issoc: object = 1.0,
) -> Path:
    """Write a v5 file of the given fields, which may be outside the format."""
    record = {
        "m1": m1,
        "m2": m2,
        "ratint": ratint,
        "ratp_int": ratp_int,
        "issoc": issoc,
    }
    savemat(path, {"sdannce": record})
    return path


def _pose(table: pd.DataFrame, animal: int, axis: str) -> npt.NDArray[np.float64]:
    rows = table[table["id"] == animal]
    return np.column_stack(
        [rows[f"pose{axis}{k}"].to_numpy(dtype=np.float64) for k in range(23)]
    )


def _expected(array: npt.NDArray[np.float64], axis: int) -> npt.NDArray[np.float64]:
    """Meters of one source axis, as ``(frame, keypoint)``."""
    return array[:, axis, :] * 0.001


# --- registration and the table -----------------------------------------------


def test_the_converter_declares_the_world_schema_and_tracklets() -> None:
    converter = TRACK_CONVERTERS["sdannce_mat"]

    assert converter is SdannceMatConverter
    assert converter.output_schema == "mosaic_world_v1"
    assert converter.carries_tracklets is True
    assert not converter.enumerable
    assert not converter.merges_per_sequence


def test_a_social_table_validates_and_has_23_keypoints(tmp_path: Path) -> None:
    path, _, _ = _write(tmp_path)

    table = _convert(path)
    _, report = ensure_track_schema(
        table, "mosaic_world_v1", strict=True, source=str(path)
    )

    assert report["missing_required"] == []
    assert len(table) == FRAMES * 2
    assert "poseX0" in table.columns
    assert "poseZ22" in table.columns
    assert "poseX23" not in table.columns
    assert len(RAT23_KEYPOINT_NAMES) == 23
    assert RAT23_KEYPOINT_NAMES[6] == "TailBase"


def test_ids_are_the_animals_and_each_holds_its_own_array(tmp_path: Path) -> None:
    path, m1, m2 = _write(tmp_path)

    table = _convert(path)

    assert set(table["id"]) == {FOCAL, PARTNER}
    for axis_index, axis in enumerate("XYZ"):
        np.testing.assert_allclose(_pose(table, FOCAL, axis), _expected(m1, axis_index))
        np.testing.assert_allclose(
            _pose(table, PARTNER, axis), _expected(m2, axis_index)
        )


def test_a_lone_file_has_one_animal(tmp_path: Path) -> None:
    path, m1, _ = _write(tmp_path, social=False)

    table = _convert(path)

    assert set(table["id"]) == {FOCAL}
    assert len(table) == FRAMES
    np.testing.assert_allclose(_pose(table, FOCAL, "X"), _expected(m1, 0))


def test_an_uncompressed_file_converts_alike(tmp_path: Path) -> None:
    path, _, _ = _write(tmp_path)
    plain = tmp_path / "plain.mat"
    _ = write_sdannce_mat(plain, FRAMES, focal=FOCAL, partner=PARTNER, compress=False)

    assert_frame_equal(_convert(path), _convert(plain))


def test_ids_are_int64_and_rows_sort_by_frame_then_id(tmp_path: Path) -> None:
    path = tmp_path / "high_first.mat"
    _ = write_sdannce_mat(path, FRAMES, focal=900, partner=100)

    table = _convert(path)

    assert table["id"].dtype == np.int64
    assert list(table["frame"]) == [frame for frame in range(FRAMES) for _ in (1, 2)]
    assert list(table["id"]) == [100, 900] * FRAMES


def test_the_hints_name_the_group_and_sequence(tmp_path: Path) -> None:
    path, _, _ = _write(tmp_path)

    table = _convert(path)

    assert set(table["group"]) == {"cage1"}
    assert set(table["sequence"]) == {"day1"}


def test_the_sequence_falls_back_to_the_file_stem(tmp_path: Path) -> None:
    path, _, _ = _write(tmp_path)

    table = _convert(path, hints=EntryHints())

    assert set(table["sequence"]) == {"LONGEVANS_M0_20221128_0493_S"}
    assert set(table["group"]) == {""}


def test_a_one_frame_file_converts_to_one_row_per_animal(tmp_path: Path) -> None:
    path, m1, m2 = _write(tmp_path, frames=1)

    table = _convert(path)

    assert len(table) == 2
    np.testing.assert_allclose(_pose(table, FOCAL, "Y"), _expected(m1, 1))
    np.testing.assert_allclose(_pose(table, PARTNER, "Z"), _expected(m2, 2))


# --- units and axes -----------------------------------------------------------


def test_a_unit_factor_rescales_every_value(tmp_path: Path) -> None:
    path, m1, _ = _write(tmp_path)

    table = _convert(path, meters_per_unit=0.01)

    np.testing.assert_allclose(_pose(table, FOCAL, "X"), m1[:, 0, :] * 0.01)


def test_axes_map_source_axes_to_world_axes(tmp_path: Path) -> None:
    path, m1, _ = _write(tmp_path)

    table = _convert(path, axes=("x", "-z", "y"))

    np.testing.assert_allclose(_pose(table, FOCAL, "X"), _expected(m1, 0))
    np.testing.assert_allclose(_pose(table, FOCAL, "Y"), -_expected(m1, 2))
    np.testing.assert_allclose(_pose(table, FOCAL, "Z"), _expected(m1, 1))


def test_a_reflection_is_refused_at_construction() -> None:
    with pytest.raises(ValidationError, match="reflection"):
        _ = SdannceMatParams(axes=("x", "y", "-z"))


def test_a_unit_factor_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        _ = SdannceMatParams(meters_per_unit=0.0)


# --- gaps ---------------------------------------------------------------------


def test_a_nan_keypoint_stays_nan_and_leaves_the_center(tmp_path: Path) -> None:
    path, m1, _ = _write(tmp_path, blank=[(1, 4, 6)])

    table = _convert(path)

    row = table[(table["id"] == FOCAL) & (table["frame"] == 4)].iloc[0]
    assert np.isnan(row["poseX6"])
    assert np.isnan(row["poseZ6"])
    assert not np.isnan(row["poseX5"])
    kept = [k for k in range(23) if k != 6]
    expected = m1[4][:, kept].mean(axis=1) * 0.001
    np.testing.assert_allclose(
        row[["X", "Y", "Z"]].to_numpy(dtype=np.float64), expected
    )


def test_an_all_nan_animal_frame_keeps_its_row_with_nan(tmp_path: Path) -> None:
    path, _, _ = _write(tmp_path, blank=[(2, 3, None)])

    table = _convert(path)

    assert len(table) == FRAMES * 2
    row = table[(table["id"] == PARTNER) & (table["frame"] == 3)].iloc[0]
    assert np.isnan(row[["X", "Y", "Z"]].to_numpy(dtype=np.float64)).all()
    pose = row[[c for c in table.columns if c.startswith("pose")]]
    assert np.isnan(pose.to_numpy(dtype=np.float64)).all()
    other = table[(table["id"] == FOCAL) & (table["frame"] == 3)].iloc[0]
    assert not np.isnan(other["X"])


def test_a_file_of_only_nan_gives_nan_rows_that_validate(tmp_path: Path) -> None:
    path = tmp_path / "empty.mat"
    nothing = np.full((FRAMES, 3, 23), np.nan)
    _ = _save(path, nothing, nothing.copy())

    table = _convert(path)
    _, report = ensure_track_schema(
        table, "mosaic_world_v1", strict=True, source=str(path)
    )

    assert report["missing_required"] == []
    assert len(table) == FRAMES * 2
    assert np.isnan(table[["X", "Y", "Z", "poseX0"]].to_numpy(dtype=np.float64)).all()


# --- time ---------------------------------------------------------------------


def test_time_is_frame_over_the_explicit_rate(tmp_path: Path) -> None:
    path, _, _ = _write(tmp_path)

    table = _convert(path)

    np.testing.assert_allclose(
        table["time"].to_numpy(dtype=np.float64),
        table["frame"].to_numpy(dtype=np.float64) / 50.0,
    )


def test_a_conversion_with_no_rate_is_refused(tmp_path: Path) -> None:
    path, _, _ = _write(tmp_path)

    with pytest.raises(FrameRateUnknownError, match="fps_default"):
        _ = _convert(path, fps=None)


# --- refusals -----------------------------------------------------------------


def test_a_v73_file_is_refused_with_the_remedy(tmp_path: Path) -> None:
    """A v7.3 file is HDF5 behind a 512-byte user block that holds a MAT header."""
    bare = tmp_path / "bare.h5"
    with h5py.File(bare, "w") as handle:
        _ = handle.create_dataset("sdannce", data=np.zeros(3))
    text = b"MATLAB 7.3 MAT-file".ljust(116, b" ")
    header = text + b"\x00" * 8 + b"\x00\x02" + b"IM"
    path = tmp_path / "v73.mat"
    _ = path.write_bytes(header.ljust(512, b"\x00") + bare.read_bytes())

    with pytest.raises(ValueError, match=r"v7\.3.*-v7"):
        _ = _convert(path)


def test_a_bare_hdf5_file_is_refused_as_not_a_mat_file(tmp_path: Path) -> None:
    path = tmp_path / "bare.mat"
    with h5py.File(path, "w") as handle:
        _ = handle.create_dataset("sdannce", data=np.zeros(3))

    with pytest.raises(ValueError, match=r"not a MATLAB v5/v7 MAT-file"):
        _ = _convert(path)


def test_a_file_without_the_variable_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "other.mat"
    savemat(path, {"other": np.zeros(3)})

    with pytest.raises(ValueError, match="sdannce"):
        _ = _convert(path)


def test_a_variable_that_is_not_a_struct_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "array.mat"
    savemat(path, {"sdannce": np.zeros((4, 3, 23))})

    with pytest.raises(ValueError, match="sdannce.*struct"):
        _ = _convert(path)


@pytest.mark.parametrize(
    "shape",
    [(5, 3, 22), (5, 23, 3), (5, 3), (0, 3, 23)],
    ids=["22-keypoints", "transposed", "two-dimensional", "no-frames"],
)
def test_an_m1_outside_the_layout_is_refused(
    tmp_path: Path, shape: tuple[int, ...]
) -> None:
    path = _save(tmp_path / "bad.mat", np.zeros(shape), np.zeros((0, 0)), issoc=0.0)

    with pytest.raises(ValueError, match=r"m1.*expected \(frames, 3, 23\)"):
        _ = _convert(path)


def test_an_integer_m1_is_refused(tmp_path: Path) -> None:
    path = _save(
        tmp_path / "ints.mat",
        np.zeros((4, 3, 23), dtype=np.int32),
        np.zeros((0, 0)),
        issoc=0.0,
    )

    with pytest.raises(ValueError, match="m1"):
        _ = _convert(path)


def test_an_m2_with_another_frame_count_is_refused(tmp_path: Path) -> None:
    path = _save(tmp_path / "bad.mat", np.zeros((5, 3, 23)), np.zeros((6, 3, 23)))

    with pytest.raises(ValueError, match=r"m2.*\(6, 3, 23\)"):
        _ = _convert(path)


@pytest.mark.parametrize("ratint", [301.5, np.nan, np.inf, "301"])
def test_a_ratint_that_is_not_an_integer_is_refused(
    tmp_path: Path, ratint: object
) -> None:
    path = _save(
        tmp_path / "bad.mat",
        np.zeros((4, 3, 23)),
        np.zeros((4, 3, 23)),
        ratint=ratint,
    )

    with pytest.raises(ValueError, match="ratint"):
        _ = _convert(path)


def test_a_ratp_int_that_is_not_an_integer_is_refused(tmp_path: Path) -> None:
    path = _save(
        tmp_path / "bad.mat",
        np.zeros((4, 3, 23)),
        np.zeros((4, 3, 23)),
        ratp_int=701.25,
    )

    with pytest.raises(ValueError, match="ratp_int"):
        _ = _convert(path)


@pytest.mark.parametrize(
    ("social", "issoc"), [(True, 0.0), (False, 1.0)], ids=["full-m2", "empty-m2"]
)
def test_an_issoc_that_disagrees_with_m2_is_refused(
    tmp_path: Path, social: bool, issoc: float
) -> None:
    m2 = np.zeros((4, 3, 23)) if social else np.zeros((0, 0))
    path = _save(tmp_path / "bad.mat", np.zeros((4, 3, 23)), m2, issoc=issoc)

    with pytest.raises(ValueError, match="issoc"):
        _ = _convert(path)


def test_a_full_m2_with_an_empty_ratp_int_is_refused(tmp_path: Path) -> None:
    path = _save(
        tmp_path / "bad.mat",
        np.zeros((4, 3, 23)),
        np.zeros((4, 3, 23)),
        ratp_int=np.zeros((0, 0)),
    )

    with pytest.raises(ValueError, match="ratp_int"):
        _ = _convert(path)


def test_equal_ratint_and_ratp_int_are_refused(tmp_path: Path) -> None:
    path = _save(
        tmp_path / "bad.mat",
        np.zeros((4, 3, 23)),
        np.zeros((4, 3, 23)),
        ratp_int=float(FOCAL),
    )

    with pytest.raises(ValueError, match="ratint and ratp_int are both"):
        _ = _convert(path)


# --- determinism and identity -------------------------------------------------


def test_one_file_converts_to_equal_frames_twice(tmp_path: Path) -> None:
    path, _, _ = _write(tmp_path, blank=[(1, 2, None), (2, 7, 3)])

    assert_frame_equal(_convert(path), _convert(path))


def test_the_frame_and_unit_enter_the_variant_identity() -> None:
    plain = SdannceMatParams().identity_dump()
    turned = SdannceMatParams(axes=("x", "-z", "y")).identity_dump()
    scaled = SdannceMatParams(meters_per_unit=0.01).identity_dump()

    assert plain["axes"] != turned["axes"]
    assert plain["meters_per_unit"] == 0.001
    assert plain["meters_per_unit"] != scaled["meters_per_unit"]
    assert "fps" not in plain


# --- a dataset ----------------------------------------------------------------


def test_a_dataset_converts_a_social_and_a_lone_file(tmp_path: Path) -> None:
    dataset = make_dataset((tmp_path / "ds").resolve(), roots=("tracks_raw", "tracks"))
    dataset.meta["fps_default"] = FPS
    raw = dataset.base_dir / "raw"
    _ = write_sdannce_mat(raw / "social_S.mat", FRAMES, focal=800, partner=802)
    _ = write_sdannce_mat(raw / "lone_L.mat", FRAMES, focal=800, social=False)
    _ = dataset.index_tracks_raw([raw], patterns=["*.mat"], src_format="sdannce_mat")

    _ = dataset.convert_all_tracks()

    index = read_tracks_index(dataset)
    assert len(index) == 2
    assert set(index["std_format"]) == {"mosaic_world_v1"}
    assert set(index["sequence"]) == {"social_S", "lone_L"}
    social = dataset.load_tracks("", "social_S")
    lone = dataset.load_tracks("", "lone_L")
    assert len(social) == FRAMES * 2
    assert len(lone) == FRAMES
    np.testing.assert_allclose(
        social["time"].to_numpy(dtype=np.float64),
        social["frame"].to_numpy(dtype=np.float64) / FPS,
    )
