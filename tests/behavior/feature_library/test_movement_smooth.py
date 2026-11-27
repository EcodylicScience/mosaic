"""The zero-padded rolling mean of ``movement-smooth`` and the arrays behind it.

``movement-smooth`` reads a movement dataset through ``movement_arrays`` and writes
positions back through ``write_positions``. Neither imports the movement library, so
these tests run without it, and the one apply path that needs no library
(``rolling_edges: zero_padded``) runs here with the import blocked.
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray
from pydantic import ValidationError

from mosaic.behavior.feature_library.movement.convert import (
    from_movement_dataset,
    movement_arrays,
    to_movement_dataset,
    write_positions,
)
from mosaic.behavior.feature_library.movement.smooth import MovementSmooth

_ZERO_PADDED: dict[str, object] = {
    "method": "rolling",
    "statistic": "mean",
    "window": 5,
    "rolling_edges": "zero_padded",
}

_POSITION_COLUMNS = ("X", "Y", "poseX0", "poseY0", "poseX1", "poseY1")


@pytest.fixture(autouse=True)
def _no_movement_library(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``import movement`` fail, so a test of ``apply`` proves it never imports."""
    monkeypatch.setitem(sys.modules, "movement", None)


def _tracks(frames_by_id: dict[int, list[int]], seed: int = 0) -> pd.DataFrame:
    """Random positions for each id on its own frames, in frame order."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for ident, frames in frames_by_id.items():
        for frame in frames:
            row: dict[str, object] = {"id": ident, "frame": frame}
            row.update({name: float(rng.normal()) for name in _POSITION_COLUMNS})
            rows.append(row)
    return pd.DataFrame(rows)


def _zero_padded(**overrides: object) -> MovementSmooth:
    return MovementSmooth(params={**_ZERO_PADDED, **overrides})


def _series(df: pd.DataFrame, ident: int, column: str) -> np.ndarray:
    return df.loc[df["id"] == ident, column].to_numpy(dtype=np.float64)


# --- the zero-padded mean ---------------------------------------------------------


def test_each_series_is_the_numpy_same_convolution_of_its_window() -> None:
    df = _tracks({0: list(range(12)), 1: list(range(12))})

    out = _zero_padded().apply(df)

    for ident in (0, 1):
        for column in _POSITION_COLUMNS:
            expected = np.convolve(
                _series(df, ident, column), np.ones(5) / 5, mode="same"
            )
            np.testing.assert_array_equal(_series(out, ident, column), expected)


def test_a_window_of_three_is_the_same_convolution_with_three_taps() -> None:
    df = _tracks({0: list(range(9))}, seed=3)

    out = _zero_padded(window=3).apply(df)

    for column in _POSITION_COLUMNS:
        expected = np.convolve(_series(df, 0, column), np.ones(3) / 3, mode="same")
        np.testing.assert_array_equal(_series(out, 0, column), expected)


def test_excluding_the_centroid_leaves_x_and_y_as_they_were() -> None:
    df = _tracks({0: list(range(10))})

    out = _zero_padded(include_centroid=False).apply(df)

    for column in ("X", "Y"):
        np.testing.assert_array_equal(_series(out, 0, column), _series(df, 0, column))
    for column in ("poseX0", "poseY0", "poseX1", "poseY1"):
        expected = np.convolve(_series(df, 0, column), np.ones(5) / 5, mode="same")
        np.testing.assert_array_equal(_series(out, 0, column), expected)


def test_the_other_columns_and_the_row_order_are_kept() -> None:
    df = _tracks({4: list(range(8)), 2: list(range(8))})
    df["note"] = "kept"

    out = _zero_padded().apply(df)

    assert list(out.columns) == list(df.columns)
    pd.testing.assert_series_equal(out["id"], df["id"])
    pd.testing.assert_series_equal(out["frame"], df["frame"])
    pd.testing.assert_series_equal(out["note"], df["note"])


def test_the_computation_is_float64_and_the_input_is_unchanged() -> None:
    df = _tracks({0: list(range(8))})
    before = df.copy()

    out = _zero_padded().apply(df)

    pd.testing.assert_frame_equal(df, before)
    assert all(out[column].dtype == np.float64 for column in _POSITION_COLUMNS)


def test_a_nan_sample_makes_nan_the_window_outputs_that_cover_it() -> None:
    df = _tracks({0: list(range(12))})
    df.loc[6, "poseX0"] = np.nan

    out = _zero_padded().apply(df)

    smoothed = _series(out, 0, "poseX0")
    expected = np.convolve(_series(df, 0, "poseX0"), np.ones(5) / 5, mode="same")
    np.testing.assert_array_equal(smoothed, expected)
    assert np.flatnonzero(np.isnan(smoothed)).tolist() == [4, 5, 6, 7, 8]
    np.testing.assert_array_equal(
        _series(out, 0, "poseY0"),
        np.convolve(_series(df, 0, "poseY0"), np.ones(5) / 5, mode="same"),
    )


def test_an_infinite_sample_is_read_as_nan() -> None:
    df = _tracks({0: list(range(12))})
    df.loc[3, "X"] = np.inf

    out = _zero_padded().apply(df)

    smoothed = _series(out, 0, "X")
    assert np.flatnonzero(np.isnan(smoothed)).tolist() == [1, 2, 3, 4, 5]


def test_a_series_shorter_than_the_window_is_nan_throughout() -> None:
    df = _tracks({0: [0, 1, 2], 1: [0, 1, 2]})

    out = _zero_padded().apply(df)

    for column in _POSITION_COLUMNS:
        assert np.isnan(out[column].to_numpy(dtype=np.float64)).all()


def test_a_series_as_long_as_the_window_is_smoothed() -> None:
    df = _tracks({0: [0, 1, 2, 3, 4]})

    out = _zero_padded().apply(df)

    expected = np.convolve(_series(df, 0, "X"), np.ones(5) / 5, mode="same")
    np.testing.assert_array_equal(_series(out, 0, "X"), expected)


def test_a_frame_one_id_lacks_is_a_nan_sample_on_the_union_axis() -> None:
    df = _tracks({0: list(range(10)), 1: [0, 1, 2, 4, 5, 6, 7, 8, 9]})

    out = _zero_padded().apply(df)

    present = _series(df, 1, "Y")
    on_the_axis = np.full(10, np.nan)
    on_the_axis[[0, 1, 2, 4, 5, 6, 7, 8, 9]] = present
    expected = np.convolve(on_the_axis, np.ones(5) / 5, mode="same")
    np.testing.assert_array_equal(
        _series(out, 1, "Y"), expected[[0, 1, 2, 4, 5, 6, 7, 8, 9]]
    )
    np.testing.assert_array_equal(
        _series(out, 0, "Y"),
        np.convolve(_series(df, 0, "Y"), np.ones(5) / 5, mode="same"),
    )


def test_the_poses_confidence_columns_are_not_rewritten() -> None:
    df = _tracks({0: list(range(8))})
    df["poseP0"] = np.linspace(0.1, 0.9, 8)
    df["poseP1"] = np.linspace(0.9, 0.1, 8)

    out = _zero_padded().apply(df)

    pd.testing.assert_series_equal(out["poseP0"], df["poseP0"])
    pd.testing.assert_series_equal(out["poseP1"], df["poseP1"])


def test_an_empty_table_gives_an_empty_frame() -> None:
    assert _zero_padded().apply(pd.DataFrame()).empty


# --- the params -------------------------------------------------------------------


def test_the_edges_default_to_min_periods() -> None:
    assert MovementSmooth.Params().rolling_edges == "min_periods"


def test_the_edges_enter_the_identity_only_when_zero_padded() -> None:
    assert "rolling_edges" not in MovementSmooth.Params().identity_dump()
    explicit = MovementSmooth.Params.from_overrides({"rolling_edges": "min_periods"})
    assert "rolling_edges" not in explicit.identity_dump()
    assert (
        MovementSmooth.Params.from_overrides(_ZERO_PADDED).identity_dump()[
            "rolling_edges"
        ]
        == "zero_padded"
    )


def test_zero_padded_with_rolling_mean_and_an_odd_window_is_accepted() -> None:
    for window in (1, 3, 5, 21):
        params = MovementSmooth.Params.from_overrides(
            {**_ZERO_PADDED, "window": window}
        )
        assert params.window == window


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"method": "savgol"}, "method"),
        ({"statistic": "median"}, "mean"),
        ({"statistic": "max"}, "mean"),
        ({"min_periods": 3}, "min_periods"),
        ({"fps": 25.0}, "fps"),
        ({"polyorder": 3}, "polyorder"),
        ({"savgol_mode": "mirror"}, "savgol_mode"),
        ({"window": 4}, "odd"),
        ({"window": 10}, "odd"),
        ({"window": 0}, "positive"),
        ({"window": -3}, "positive"),
    ],
    ids=[
        "savgol",
        "median",
        "max",
        "min-periods",
        "fps",
        "polyorder",
        "savgol-mode",
        "even-4",
        "even-10",
        "zero",
        "negative",
    ],
)
def test_zero_padded_refuses_what_it_does_not_compute(
    override: dict[str, object], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        _ = MovementSmooth.Params.from_overrides({**_ZERO_PADDED, **override})


def test_the_default_statistic_is_not_a_mean() -> None:
    without_statistic = {
        key: value for key, value in _ZERO_PADDED.items() if key != "statistic"
    }
    with pytest.raises(ValidationError, match="mean"):
        _ = MovementSmooth.Params.from_overrides(without_statistic)


def test_min_periods_edges_leave_every_other_combination_accepted() -> None:
    overrides: tuple[dict[str, object], ...] = (
        {},
        {"method": "rolling", "statistic": "max", "window": 4},
        {"method": "rolling", "min_periods": 2},
        {"method": "savgol", "window": 4},
        {"method": "rolling", "fps": 25.0, "polyorder": 3, "savgol_mode": "mirror"},
    )
    for override in overrides:
        assert MovementSmooth.Params.from_overrides(override).rolling_edges == (
            "min_periods"
        )


# --- movement_arrays --------------------------------------------------------------


def _two_animals() -> pd.DataFrame:
    """Ids 7 and 3 on frames 11, 12, 14 and 10, 11, 12, with one inf and one NaN."""
    return pd.DataFrame(
        {
            "id": [7, 7, 7, 3, 3, 3],
            "frame": [11, 12, 14, 10, 11, 12],
            "X": [1.0, 2.0, np.inf, 4.0, 5.0, 6.0],
            "Y": [10.0, 20.0, 30.0, 40.0, np.nan, 60.0],
            "poseX0": [0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
            "poseY0": [5.0, 15.0, 25.0, 35.0, 45.0, 55.0],
            "poseP0": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
            "poseX1": [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0],
            "poseY1": [-10.0, -20.0, -30.0, -40.0, -50.0, -60.0],
        }
    )


def test_the_arrays_are_float64_on_the_union_of_frames_and_the_sorted_ids() -> None:
    position, confidence, metadata = movement_arrays(
        _two_animals(), ["head", "tail"], True
    )

    assert position.dtype == np.float64
    assert confidence.dtype == np.float64
    assert position.shape == (4, 2, 3, 2)
    assert confidence.shape == (4, 3, 2)
    assert metadata["individual_ids"] == [3, 7]
    assert metadata["frame_index"] == [10, 11, 12, 14]
    assert metadata["include_centroid"] is True
    assert metadata["pose_pairs"] == [("poseX0", "poseY0"), ("poseX1", "poseY1")]
    assert metadata["has_confidence"] is True
    assert metadata["conf_cols"] == ["poseP0", None]
    assert metadata["keypoint_names"] == ["centroid", "head", "tail"]


def _expected_position() -> np.ndarray:
    """The positions of ``_two_animals`` on its frame axis, in float64."""
    expected = np.full((4, 2, 3, 2), np.nan)
    # id 3 is index 0 on frames 10, 11, 12 (axis 0 to 2); id 7 is index 1 on 11, 12, 14
    expected[[0, 1, 2], 0, 0, 0] = [4.0, 5.0, 6.0]
    expected[[0, 1, 2], 1, 0, 0] = [40.0, np.nan, 60.0]
    expected[[1, 2, 3], 0, 0, 1] = [1.0, 2.0, np.nan]
    expected[[1, 2, 3], 1, 0, 1] = [10.0, 20.0, 30.0]
    expected[[0, 1, 2], 0, 1, 0] = [3.5, 4.5, 5.5]
    expected[[0, 1, 2], 1, 1, 0] = [35.0, 45.0, 55.0]
    expected[[1, 2, 3], 0, 1, 1] = [0.5, 1.5, 2.5]
    expected[[1, 2, 3], 1, 1, 1] = [5.0, 15.0, 25.0]
    expected[[0, 1, 2], 0, 2, 0] = [-4.0, -5.0, -6.0]
    expected[[0, 1, 2], 1, 2, 0] = [-40.0, -50.0, -60.0]
    expected[[1, 2, 3], 0, 2, 1] = [-1.0, -2.0, -3.0]
    expected[[1, 2, 3], 1, 2, 1] = [-10.0, -20.0, -30.0]
    return expected


def _expected_confidence() -> np.ndarray:
    """The scores of ``_two_animals`` on its frame axis, in float64."""
    expected = np.full((4, 3, 2), np.nan)
    # the centroid scores 1 where X and Y are finite: id 3 has a NaN Y at frame 11,
    # and id 7 an infinite X (read as NaN) at frame 14
    expected[[0, 1, 2], 0, 0] = [1.0, np.nan, 1.0]
    expected[[1, 2, 3], 0, 1] = [1.0, 1.0, np.nan]
    expected[[0, 1, 2], 1, 0] = [0.6, 0.5, 0.4]
    expected[[1, 2, 3], 1, 1] = [0.9, 0.8, 0.7]
    expected[[0, 1, 2], 2, 0] = [1.0, 1.0, 1.0]
    expected[[1, 2, 3], 2, 1] = [1.0, 1.0, 1.0]
    return expected


def test_each_value_is_on_its_frame_axis_keypoint_and_individual() -> None:
    position, _, _ = movement_arrays(_two_animals(), ["head", "tail"], True)

    np.testing.assert_array_equal(position, _expected_position())


def test_a_score_column_is_read_and_a_valid_point_without_one_scores_one() -> None:
    _, confidence, _ = movement_arrays(_two_animals(), ["head", "tail"], True)

    np.testing.assert_array_equal(confidence, _expected_confidence())


def test_the_movement_dataset_is_built_from_float32_arrays_and_the_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arrays: dict[str, NDArray[np.generic]] = {}
    received: dict[str, object] = {}

    def from_numpy(
        *,
        position_array: NDArray[np.generic],
        confidence_array: NDArray[np.generic],
        individual_names: list[str],
        keypoint_names: list[str],
        fps: float | None,
        source_software: str,
    ) -> str:
        arrays["position"] = position_array
        arrays["confidence"] = confidence_array
        received.update(
            individual_names=individual_names,
            keypoint_names=keypoint_names,
            fps=fps,
            source_software=source_software,
        )
        return "dataset"

    load_poses = ModuleType("movement.io.load_poses")
    load_poses.__dict__["from_numpy"] = from_numpy
    monkeypatch.setitem(sys.modules, "movement", ModuleType("movement"))
    monkeypatch.setitem(sys.modules, "movement.io", ModuleType("movement.io"))
    monkeypatch.setitem(sys.modules, "movement.io.load_poses", load_poses)

    dataset, metadata = to_movement_dataset(
        _two_animals(), fps=25.0, keypoint_names=["head", "tail"]
    )

    assert dataset == "dataset"
    assert arrays["position"].dtype == np.float32
    assert arrays["confidence"].dtype == np.float32
    np.testing.assert_array_equal(
        arrays["position"], _expected_position().astype(np.float32)
    )
    np.testing.assert_array_equal(
        arrays["confidence"], _expected_confidence().astype(np.float32)
    )
    assert received == {
        "individual_names": ["3", "7"],
        "keypoint_names": ["centroid", "head", "tail"],
        "fps": 25.0,
        "source_software": "mosaic",
    }
    assert metadata["keypoint_names"] == ["centroid", "head", "tail"]


def test_without_the_centroid_the_keypoints_start_at_the_first_pose_pair() -> None:
    position, _, metadata = movement_arrays(_two_animals(), None, False)

    assert metadata["include_centroid"] is False
    assert metadata["keypoint_names"] == ["keypoint_0", "keypoint_1"]
    assert position.shape == (4, 2, 2, 2)
    np.testing.assert_array_equal(position[[0, 1, 2], 0, 0, 0], [3.5, 4.5, 5.5])


def test_a_table_without_pose_columns_has_the_centroid_alone() -> None:
    df = _two_animals()[["id", "frame", "X", "Y"]]

    position, _, metadata = movement_arrays(df, None, True)

    assert position.shape == (4, 2, 1, 2)
    assert metadata["keypoint_names"] == ["centroid"]
    assert metadata["has_confidence"] is False
    assert metadata["conf_cols"] == []


def test_a_wrong_count_of_keypoint_names_is_refused() -> None:
    with pytest.raises(ValueError, match="3 entries but found 2 pose column pairs"):
        _ = movement_arrays(_two_animals(), ["a", "b", "c"], True)


def test_a_table_with_no_keypoint_to_convert_is_refused() -> None:
    df = _two_animals()[["id", "frame", "X", "Y"]]

    with pytest.raises(ValueError, match="No keypoints to convert"):
        _ = movement_arrays(df, None, False)


# --- write_positions and from_movement_dataset ------------------------------------


def _rows_to_rewrite() -> pd.DataFrame:
    """Ids 7 and 3 in a shuffled row order and index, with poseP0 and no poseP1."""
    df = _two_animals().drop(columns=["poseX1", "poseY1"])
    df["poseX1"] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    df["poseY1"] = [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
    df = df.iloc[[4, 0, 5, 2, 3, 1]]
    df.index = pd.Index([40, 10, 50, 20, 30, 0])
    return df


def _coded_positions() -> np.ndarray:
    """A position array whose value says its frame, axis, keypoint and individual."""
    frames, keypoints, individuals = 4, 3, 2
    position = np.empty((frames, 2, keypoints, individuals), dtype=np.float64)
    for f in range(frames):
        for a in range(2):
            for k in range(keypoints):
                for i in range(individuals):
                    position[f, a, k, i] = 1000.0 * f + 100.0 * a + 10.0 * k + i
    return position


def _coded_confidence() -> np.ndarray:
    frames, keypoints, individuals = 4, 3, 2
    confidence = np.empty((frames, keypoints, individuals), dtype=np.float64)
    for f in range(frames):
        for k in range(keypoints):
            for i in range(individuals):
                confidence[f, k, i] = 0.5 + 10.0 * f + 1.0 * k + 0.25 * i
    return confidence


def _axes_of(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """The frame axis index and the individual index of each row of ``df``.

    The frames 10, 11, 12 and 14 are 0 to 3 on the axis, and the ids 3 and 7 are 0
    and 1.
    """
    frame_axis = df["frame"].map({10: 0, 11: 1, 12: 2, 14: 3}).to_numpy(dtype=float)
    individual = df["id"].map({3: 0, 7: 1}).to_numpy(dtype=float)
    return frame_axis, individual


def _coded(
    df: pd.DataFrame, axis: int, keypoint: int, dtype: type[np.floating]
) -> np.ndarray:
    """What ``_coded_positions`` holds at each row of ``df``."""
    frame_axis, individual = _axes_of(df)
    return (1000.0 * frame_axis + 100.0 * axis + 10.0 * keypoint + individual).astype(
        dtype
    )


def _coded_score(
    df: pd.DataFrame, keypoint: int, dtype: type[np.floating]
) -> np.ndarray:
    """What ``_coded_confidence`` holds at each row of ``df``."""
    frame_axis, individual = _axes_of(df)
    return (0.5 + 10.0 * frame_axis + 1.0 * keypoint + 0.25 * individual).astype(dtype)


def test_positions_are_written_at_each_rows_frame_and_individual() -> None:
    df = _rows_to_rewrite()
    metadata = movement_arrays(df, None, True)[2]

    out = write_positions(df, _coded_positions(), metadata)

    for column, axis, keypoint in (
        ("X", 0, 0),
        ("Y", 1, 0),
        ("poseX0", 0, 1),
        ("poseY0", 1, 1),
        ("poseX1", 0, 2),
        ("poseY1", 1, 2),
    ):
        np.testing.assert_array_equal(
            out[column].to_numpy(), _coded(df, axis, keypoint, np.float64)
        )
    assert list(out.index) == list(df.index)


def test_positions_leave_the_input_and_the_scores_as_they_were() -> None:
    df = _rows_to_rewrite()
    before = df.copy()
    metadata = movement_arrays(df, None, True)[2]

    out = write_positions(df, _coded_positions(), metadata)

    pd.testing.assert_frame_equal(df, before)
    pd.testing.assert_series_equal(out["poseP0"], df["poseP0"])
    assert "poseP1" not in out.columns


def test_a_centroid_the_metadata_leaves_out_keeps_its_columns() -> None:
    df = _rows_to_rewrite()
    metadata = movement_arrays(df, None, False)[2]
    position = _coded_positions()[:, :, :2, :]

    out = write_positions(df, position, metadata)

    pd.testing.assert_series_equal(out["X"], df["X"])
    pd.testing.assert_series_equal(out["Y"], df["Y"])
    np.testing.assert_array_equal(
        out["poseX0"].to_numpy(), _coded(df, 0, 0, np.float64)
    )
    np.testing.assert_array_equal(
        out["poseY1"].to_numpy(), _coded(df, 1, 1, np.float64)
    )


def test_a_score_array_rewrites_the_score_columns_the_metadata_names() -> None:
    df = _rows_to_rewrite()
    metadata = movement_arrays(df, None, True)[2]

    out = write_positions(df, _coded_positions(), metadata, _coded_confidence())

    np.testing.assert_array_equal(
        out["poseP0"].to_numpy(), _coded_score(df, 1, np.float64)
    )
    assert "poseP1" not in out.columns


def _movement_dataset() -> dict[str, SimpleNamespace]:
    """What ``from_movement_dataset`` reads of an xarray dataset: ``.values``."""
    return {
        "position": SimpleNamespace(values=_coded_positions().astype(np.float32)),
        "confidence": SimpleNamespace(values=_coded_confidence().astype(np.float32)),
    }


def test_from_movement_dataset_writes_positions_and_leaves_scores_by_default() -> None:
    df = _rows_to_rewrite()
    metadata = movement_arrays(df, None, True)[2]

    out = from_movement_dataset(_movement_dataset(), df, metadata)

    np.testing.assert_array_equal(out["X"].to_numpy(), _coded(df, 0, 0, np.float32))
    np.testing.assert_array_equal(
        out["poseY1"].to_numpy(), _coded(df, 1, 2, np.float32)
    )
    pd.testing.assert_series_equal(out["poseP0"], df["poseP0"])


def test_from_movement_dataset_rewrites_scores_when_asked() -> None:
    df = _rows_to_rewrite()
    metadata = movement_arrays(df, None, True)[2]

    out = from_movement_dataset(
        _movement_dataset(), df, metadata, update_confidence=True
    )

    np.testing.assert_array_equal(
        out["poseP0"].to_numpy(), _coded_score(df, 1, np.float32)
    )
    assert "poseP1" not in out.columns
