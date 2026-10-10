"""VAME's sessions, built from a track table, and its latents placed back on the rows.

VAME reads a session as a run of contiguous frames of one individual. Each id is
reindexed onto contiguous frames and split at every gap longer than
``max_gap_frames``. A session that cannot yield latents is excluded with its
reason, and the rows outside every session and exclusion are counted. Latent
``i`` of a session starting at frame ``f0`` is placed at frame
``f0 + i + time_window // 2``, the center of its window.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from mosaic.behavior.feature_library.vame_sessions import (
    EntrySessions,
    SessionPurpose,
    SessionSettings,
    build_sessions,
    latent_columns,
    min_session_frames,
    place_latents,
    vame_keypoints,
    write_session_tracks,
)
from mosaic.core.pipeline.types import PoseConfig
from tests.helpers import keypoint_rows

_KEYPOINTS = 4
_TIME_WINDOW = 4
_ZDIMS = 3

IdValue = int | str


def _rows(
    frames: Iterable[int],
    individual: IdValue | None = None,
    *,
    confidence: float = 0.9,
) -> pd.DataFrame:
    """One individual's rows, keypoint ``k`` at frame ``f`` at ``(f + k / 10, -f - k / 10)``."""
    return keypoint_rows(
        frames, individual, n_keypoints=_KEYPOINTS, confidence=confidence
    )


def _table(*individuals: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(individuals, ignore_index=True)


def _settings(**changes: object) -> SessionSettings:
    base = SessionSettings(
        pose=PoseConfig(pose_n=_KEYPOINTS),
        centered_reference_keypoint=0,
        orientation_reference_keypoint=1,
        run_lowconf_cleaning=True,
        pose_confidence=0.5,
        max_gap_frames=3,
        time_window=_TIME_WINDOW,
    )
    return dataclasses.replace(base, **changes)


def _spans(entry: EntrySessions) -> list[tuple[object, int, int]]:
    """Each session's id, first frame and last frame, in order."""
    return [
        (session.individual, session.first_frame, session.last_frame)
        for session in entry.sessions
    ]


def _set_cells(
    df: pd.DataFrame, frames: Sequence[int], columns: Sequence[str], value: float
) -> None:
    df.loc[df["frame"].isin(frames), list(columns)] = value


# --- Refusals ---


def test_a_table_without_keypoints_is_refused() -> None:
    df = _rows(range(10))
    df = df.drop(columns=df.filter(like="pose").columns)

    with pytest.raises(ValueError, match=r"vame: .*lacks keypoint columns"):
        _ = build_sessions(df, _settings(), "embed")


def test_a_selected_keypoint_without_its_columns_is_refused() -> None:
    df = _rows(range(10))

    with pytest.raises(ValueError, match=r"vame: .*poseX4"):
        _ = build_sessions(df, _settings(pose=PoseConfig(pose_n=5)), "embed")


def test_an_entry_of_two_cameras_is_refused() -> None:
    df = _table(_rows(range(10), 0), _rows(range(10), 1))
    df["camera"] = ["left"] * 10 + ["right"] * 10

    with pytest.raises(ValueError, match=r"vame: .*'left', 'right'"):
        _ = build_sessions(df, _settings(), "embed")


def test_an_entry_of_one_camera_is_read() -> None:
    df = _rows(range(10), 0)
    df["camera"] = "left"

    assert _spans(build_sessions(df, _settings(), "embed")) == [(0, 0, 9)]


def test_two_rows_of_one_frame_and_id_are_refused() -> None:
    df = _table(_rows(range(10), 0), _rows([4], 0))

    with pytest.raises(ValueError, match=r"vame: .*frame 4.*id 0"):
        _ = build_sessions(df, _settings(), "embed")


def test_two_rows_of_one_frame_without_an_id_column_are_refused() -> None:
    df = _table(_rows(range(10)), _rows([4]))

    with pytest.raises(ValueError, match=r"vame: .*frame 4"):
        _ = build_sessions(df, _settings(), "embed")


def test_a_row_without_an_id_is_refused() -> None:
    df = _table(_rows(range(10), 0), _rows(range(10), 1))
    df["id"] = df["id"].astype(float)
    df.loc[3, "id"] = np.nan

    with pytest.raises(ValueError, match=r"vame: 1 row lacks an id"):
        _ = build_sessions(df, _settings(), "embed")


def test_a_decimated_id_is_refused() -> None:
    """A table subsampled to every other frame would be half interpolated."""
    df = _table(_rows(range(0, 40, 2), 3), _rows(range(20), 4))

    with pytest.raises(ValueError, match=r"vame: id 3 .*median of 2 frames"):
        _ = build_sessions(df, _settings(), "embed")


def test_an_id_with_occasional_missing_frames_is_read() -> None:
    df = _rows([0, 1, 2, 4, 5, 6, 8, 9, 10], 3)

    assert _spans(build_sessions(df, _settings(), "embed")) == [(3, 0, 10)]


def test_a_table_without_an_id_column_is_one_individual() -> None:
    entry = build_sessions(_rows(range(10)), _settings(), "embed")

    assert _spans(entry) == [(None, 0, 9)]


# --- Keypoints and references ---


def test_references_resolve_by_name_and_by_index() -> None:
    pose = PoseConfig(pose_n=4, keypoint_names=["nose", "neck", "tail", "paw"])

    keypoints = vame_keypoints(
        _settings(
            pose=pose,
            centered_reference_keypoint="neck",
            orientation_reference_keypoint=0,
        )
    )

    assert keypoints.names == ("nose", "neck", "tail", "paw")
    assert keypoints.suffixes == ("0", "1", "2", "3")
    assert (keypoints.centered, keypoints.orientation) == ("neck", "nose")


def test_keypoints_follow_the_selection_in_its_order() -> None:
    pose = PoseConfig(
        pose_n=4, pose_indices=[3, 1, 2], keypoint_names=["nose", "neck", "tail", "paw"]
    )

    keypoints = vame_keypoints(
        _settings(
            pose=pose,
            centered_reference_keypoint=3,
            orientation_reference_keypoint="tail",
        )
    )

    assert keypoints.names == ("paw", "neck", "tail")
    assert keypoints.suffixes == ("3", "1", "2")
    assert (keypoints.centered, keypoints.orientation) == ("paw", "tail")


def test_unnamed_keypoints_are_named_by_their_index() -> None:
    pose = PoseConfig(pose_n=4, pose_indices=[2, 0])

    keypoints = vame_keypoints(
        _settings(
            pose=pose, centered_reference_keypoint=0, orientation_reference_keypoint=2
        )
    )

    assert keypoints.names == ("kp2", "kp0")
    assert (keypoints.centered, keypoints.orientation) == ("kp0", "kp2")


def test_a_reference_outside_the_selection_is_refused() -> None:
    pose = PoseConfig(pose_n=4, pose_indices=[1, 2])

    with pytest.raises(ValueError, match=r"centered_reference_keypoint.*\[0\]"):
        _ = vame_keypoints(_settings(pose=pose))


def test_references_to_one_keypoint_are_refused() -> None:
    pose = PoseConfig(pose_n=4, keypoint_names=["nose", "neck", "tail", "paw"])

    with pytest.raises(ValueError, match=r"vame: .*both 'nose'"):
        _ = vame_keypoints(
            _settings(
                pose=pose,
                centered_reference_keypoint="nose",
                orientation_reference_keypoint=0,
            )
        )


def test_a_selected_keypoint_without_a_name_is_refused() -> None:
    pose = PoseConfig(
        pose_n=4, pose_indices=[0, 1, 5], keypoint_names=["nose", "neck", "tail", "paw"]
    )

    with pytest.raises(ValueError, match=r"vame: .*\[5\]"):
        _ = vame_keypoints(_settings(pose=pose))


def test_a_keypoint_selected_twice_is_refused() -> None:
    pose = PoseConfig(pose_n=4, pose_indices=[0, 1, 1])

    with pytest.raises(ValueError, match=r"vame: .*'kp1'"):
        _ = vame_keypoints(_settings(pose=pose))


def test_a_session_holds_the_selected_keypoints_in_order() -> None:
    pose = PoseConfig(pose_n=4, pose_indices=[3, 0, 1])

    [session] = build_sessions(_rows(range(10)), _settings(pose=pose), "embed").sessions

    assert session.track.position.shape == (10, 3, 2)
    np.testing.assert_allclose(session.track.position[0, :, 0], [0.3, 0.0, 0.1])


# --- Reindexing and gaps ---


def test_a_gap_of_max_gap_frames_stays_inside_its_session() -> None:
    entry = build_sessions(_rows([*range(10), *range(13, 23)], 0), _settings(), "embed")

    [session] = entry.sessions
    assert session.track.frames.tolist() == list(range(23))
    assert np.isnan(session.track.position[10:13]).all()
    assert (session.track.confidence[10:13] == 0).all()
    assert np.isfinite(session.track.position[[9, 13]]).all()


def test_a_gap_longer_than_max_gap_frames_splits_the_session() -> None:
    entry = build_sessions(_rows([*range(10), *range(14, 24)], 0), _settings(), "embed")

    assert _spans(entry) == [(0, 0, 9), (0, 14, 23)]
    assert entry.excluded == ()


def test_rows_without_a_kept_point_count_as_a_gap() -> None:
    """A row whose every keypoint is missing is interpolated as a missing frame is."""
    df = _rows(range(20), 0)
    _set_cells(df, range(8, 12), [c for c in df.columns if "pose" in c], np.nan)

    assert _spans(build_sessions(df, _settings(), "embed")) == [(0, 0, 7), (0, 12, 19)]


def test_rows_of_low_confidence_count_as_a_gap() -> None:
    df = _rows(range(20), 0)
    _set_cells(df, range(8, 12), [f"poseP{k}" for k in range(_KEYPOINTS)], 0.1)

    assert _spans(build_sessions(df, _settings(), "embed")) == [(0, 0, 7), (0, 12, 19)]


def test_a_nan_score_at_a_finite_point_is_kept_while_cleaning_runs() -> None:
    """VAME compares a NaN confidence with pose_confidence, and NaN is not below it."""
    df = _rows(range(20), 0)
    _set_cells(df, range(8, 12), [f"poseP{k}" for k in range(_KEYPOINTS)], np.nan)

    entry = build_sessions(df, _settings(), "embed")

    assert _spans(entry) == [(0, 0, 19)]
    assert entry.excluded == ()


@pytest.mark.parametrize(("score", "kept"), [(0.75, True), (0.7, False)])
def test_a_score_is_compared_in_float64_after_rounding_to_float32(
    score: float, kept: bool
) -> None:
    """A session's confidence is float32, and the runner gives VAME float64.

    float32 represents 0.75 without rounding, and a score of 0.75 is kept at
    ``pose_confidence=0.75``. 0.7 rounds to the float32 0.699999988, which is below
    ``pose_confidence=0.7``.
    """
    df = _rows(range(10), 0, confidence=score)

    entry = build_sessions(df, _settings(pose_confidence=score), "embed")

    assert (len(entry.sessions), len(entry.excluded)) == ((1, 0) if kept else (0, 1))


_MISSING_POINT = pytest.mark.parametrize(
    ("column", "value"), [("poseX", np.nan), ("poseP", 0.1)], ids=["nan", "low_score"]
)
_REFERENCE = pytest.mark.parametrize(
    "reference", [0, 1], ids=["centered", "orientation"]
)


@_MISSING_POINT
def test_a_missing_keypoint_other_than_a_reference_does_not_split(
    column: str, value: float
) -> None:
    """Keypoint 2 is not a reference, and is missing longer than max_gap_frames."""
    df = _rows(range(20), 0)
    _set_cells(df, range(8, 16), [f"{column}2"], value)

    entry = build_sessions(df, _settings(), "embed")

    assert _spans(entry) == [(0, 0, 19)]
    assert entry.excluded == ()


@_MISSING_POINT
@_REFERENCE
def test_a_reference_missing_longer_than_max_gap_frames_splits_the_session(
    reference: int, column: str, value: float
) -> None:
    df = _rows(range(20), 0)
    _set_cells(df, range(8, 12), [f"{column}{reference}"], value)

    entry = build_sessions(df, _settings(), "embed")

    assert _spans(entry) == [(0, 0, 7), (0, 12, 19)]
    assert entry.excluded == ()


@_MISSING_POINT
@_REFERENCE
def test_a_reference_missing_for_max_gap_frames_stays_inside_its_session(
    reference: int, column: str, value: float
) -> None:
    df = _rows(range(20), 0)
    _set_cells(df, range(8, 11), [f"{column}{reference}"], value)

    entry = build_sessions(df, _settings(), "embed")

    assert _spans(entry) == [(0, 0, 19)]
    assert entry.excluded == ()


def test_references_are_found_on_the_keypoint_axis_of_the_selection() -> None:
    """The keypoint axis is kp3, kp1, kp2, and the references are kp3 and kp2."""
    settings = _settings(
        pose=PoseConfig(pose_n=_KEYPOINTS, pose_indices=[3, 1, 2]),
        centered_reference_keypoint=3,
        orientation_reference_keypoint=2,
    )
    other = _rows(range(20), 0)
    _set_cells(other, range(8, 12), ["poseX1"], np.nan)
    reference = _rows(range(20), 0)
    _set_cells(reference, range(8, 12), ["poseX3"], np.nan)

    assert _spans(build_sessions(other, settings, "embed")) == [(0, 0, 19)]
    assert _spans(build_sessions(reference, settings, "embed")) == [
        (0, 0, 7),
        (0, 12, 19),
    ]


def test_an_id_whose_reference_is_never_kept_is_excluded_whole() -> None:
    df = _table(_rows(range(10), 0), _rows(range(10), 1))
    df.loc[df["id"] == 1, "poseP1"] = 0.2

    entry = build_sessions(df, _settings(), "embed")

    assert _spans(entry) == [(0, 0, 9)]
    [excluded] = entry.excluded
    assert (excluded.individual, excluded.first_frame, excluded.last_frame) == (1, 0, 9)
    assert excluded.reason == (
        "at every frame, reference keypoint 'kp0' or 'kp1' lacks a finite point "
        "with confidence at or above pose_confidence=0.5"
    )
    assert entry.gap_rows.count == 0


@pytest.mark.parametrize(
    ("columns", "value"),
    [
        ([f"pose{axis}{k}" for axis in "XYP" for k in range(_KEYPOINTS)], np.nan),
        (["poseX1"], np.nan),
        (["poseP0"], 0.1),
    ],
    ids=["every_keypoint", "orientation_position", "centered_score"],
)
def test_frames_without_a_kept_reference_point_at_either_end_are_trimmed(
    columns: list[str], value: float
) -> None:
    df = _rows(range(20), 0)
    _set_cells(df, [0, 1, 19], columns, value)

    entry = build_sessions(df, _settings(), "embed")

    assert _spans(entry) == [(0, 2, 18)]
    assert entry.gap_rows.count == 3


def test_rows_out_of_frame_order_are_reindexed_in_order() -> None:
    df = _rows([*range(10), *range(13, 23)], 0)
    shuffled = df.sample(frac=1.0, random_state=0)

    [session] = build_sessions(shuffled, _settings(), "embed").sessions
    [expected] = build_sessions(df, _settings(), "embed").sessions

    np.testing.assert_array_equal(session.track.frames, expected.track.frames)
    np.testing.assert_array_equal(session.track.position, expected.track.position)


# --- Rows outside every session ---


def test_rows_in_a_reference_gap_longer_than_max_gap_frames_are_counted() -> None:
    df = _table(_rows(range(20), 0), _rows(range(20), 1))
    df.loc[(df["id"] == 0) & df["frame"].between(8, 11), "poseP0"] = 0.1
    df.loc[(df["id"] == 1) & df["frame"].between(4, 8), "poseX1"] = np.nan

    entry = build_sessions(df, _settings(), "embed")

    assert _spans(entry) == [(0, 0, 7), (0, 12, 19), (1, 0, 3), (1, 9, 19)]
    assert entry.gap_rows.count == 9
    assert entry.gap_rows.describe() == (
        "9 rows at frames where reference keypoint 'kp0' or 'kp1' lacks a finite "
        "point with confidence at or above pose_confidence=0.5, in a run longer "
        "than max_gap_frames=3 or at either end of an id's frames"
    )


def test_rows_in_a_gap_that_vame_interpolates_are_not_counted() -> None:
    """The references are missing for max_gap_frames, and keypoint 2 for longer."""
    df = _rows(range(20), 0)
    _set_cells(df, range(8, 11), ["poseP0"], 0.1)
    _set_cells(df, range(2, 18), ["poseX2"], np.nan)

    entry = build_sessions(df, _settings(), "embed")

    assert _spans(entry) == [(0, 0, 19)]
    assert entry.gap_rows.count == 0


def test_a_frame_without_a_row_is_not_counted() -> None:
    """The gap is frames 10 to 13, and frames 12 and 13 lack a row."""
    df = _rows([*range(12), *range(14, 24)], 0)
    _set_cells(df, [10, 11], ["poseP1"], 0.1)

    entry = build_sessions(df, _settings(), "embed")

    assert _spans(entry) == [(0, 0, 9), (0, 14, 23)]
    assert entry.gap_rows.count == 2


def test_the_rows_of_an_exclusion_are_not_counted() -> None:
    """Frames 0 to 2 are fewer than time_window, and their exclusion reports them."""
    df = _rows(range(20), 0)
    _set_cells(df, range(3, 8), ["poseP1"], 0.1)

    entry = build_sessions(df, _settings(), "embed")

    assert _spans(entry) == [(0, 8, 19)]
    [excluded] = entry.excluded
    assert (excluded.first_frame, excluded.last_frame) == (0, 2)
    assert entry.gap_rows.count == 5


def test_one_row_outside_every_session_is_counted_in_the_singular() -> None:
    df = _rows(range(10), 0)
    _set_cells(df, [0], ["poseP0"], 0.1)

    gap_rows = build_sessions(df, _settings(), "embed").gap_rows

    assert gap_rows.describe().startswith("1 row at frames where ")


# --- Splitting with low-confidence cleaning off ---


def test_without_cleaning_one_non_finite_keypoint_splits_the_session() -> None:
    df = _rows(range(20), 0)
    _set_cells(df, [9, 10], ["poseX2"], np.nan)
    settings = _settings(run_lowconf_cleaning=False, max_gap_frames=0)

    assert _spans(build_sessions(df, settings, "embed")) == [(0, 0, 8), (0, 11, 19)]


@pytest.mark.parametrize("max_gap_frames", [0, 3])
def test_without_cleaning_a_missing_frame_splits_the_session(
    max_gap_frames: int,
) -> None:
    """``max_gap_frames`` is not read without cleaning."""
    df = _rows([*range(10), *range(11, 20)], 0)
    settings = _settings(run_lowconf_cleaning=False, max_gap_frames=max_gap_frames)

    assert _spans(build_sessions(df, settings, "embed")) == [(0, 0, 9), (0, 11, 19)]


def test_without_cleaning_every_session_is_finite() -> None:
    df = _rows(range(30), 0)
    _set_cells(df, [5], ["poseY0"], np.inf)
    _set_cells(df, [17, 18], ["poseX3"], np.nan)
    settings = _settings(run_lowconf_cleaning=False, max_gap_frames=0)

    entry = build_sessions(df, settings, "embed")

    assert _spans(entry) == [(0, 0, 4), (0, 6, 16), (0, 19, 29)]
    for session in entry.sessions:
        assert np.isfinite(session.track.position).all()


def test_without_cleaning_every_row_with_a_missing_point_is_counted() -> None:
    df = _rows(range(30), 0)
    _set_cells(df, [5], ["poseY0"], np.inf)
    _set_cells(df, [17, 18], ["poseX3"], np.nan)
    settings = _settings(run_lowconf_cleaning=False, max_gap_frames=0)

    gap_rows = build_sessions(df, settings, "embed").gap_rows

    assert gap_rows.count == 3
    assert gap_rows.describe() == (
        "3 rows at frames where a selected keypoint lacks a finite point, and VAME "
        "does not interpolate without run_lowconf_cleaning"
    )


def test_without_cleaning_an_id_without_a_finite_frame_is_excluded_whole() -> None:
    df = _table(_rows(range(10), 0), _rows(range(10), 1))
    df.loc[df["id"] == 1, "poseY2"] = np.nan
    settings = _settings(run_lowconf_cleaning=False, max_gap_frames=0)

    entry = build_sessions(df, settings, "embed")

    assert _spans(entry) == [(0, 0, 9)]
    [excluded] = entry.excluded
    assert (excluded.individual, excluded.first_frame, excluded.last_frame) == (1, 0, 9)
    assert excluded.reason == (
        "every frame lacks a finite point of at least one selected keypoint, and "
        "VAME does not interpolate without run_lowconf_cleaning"
    )


def test_without_cleaning_low_confidence_neither_splits_nor_excludes() -> None:
    """Confidence decides nothing when cleaning does not read it."""
    df = _rows(range(20), 0, confidence=0.1)
    settings = _settings(run_lowconf_cleaning=False, max_gap_frames=0)

    entry = build_sessions(df, settings, "embed")

    assert _spans(entry) == [(0, 0, 19)]
    assert entry.excluded == ()


# --- Exclusions ---


def test_minimum_session_frames() -> None:
    assert min_session_frames(30, "embed") == 30
    assert min_session_frames(30, "fit") == 61


def test_a_session_shorter_than_time_window_is_excluded_from_an_embedding() -> None:
    df = _rows([*range(_TIME_WINDOW - 1), *range(10, 10 + _TIME_WINDOW)], 0)

    entry = build_sessions(df, _settings(), "embed")

    assert _spans(entry) == [(0, 10, 13)]
    [excluded] = entry.excluded
    assert (excluded.individual, excluded.first_frame, excluded.last_frame) == (0, 0, 2)
    assert "3 frames" in excluded.reason
    assert f"time_window={_TIME_WINDOW}" in excluded.reason


@pytest.mark.parametrize(
    ("n_frames", "kept"),
    [(_TIME_WINDOW, False), (2 * _TIME_WINDOW, False), (2 * _TIME_WINDOW + 1, True)],
)
def test_a_fit_needs_two_time_windows_and_one_frame(n_frames: int, kept: bool) -> None:
    entry = build_sessions(_rows(range(n_frames), 0), _settings(), "fit")

    assert (len(entry.sessions), len(entry.excluded)) == ((1, 0) if kept else (0, 1))
    if not kept:
        assert "2 * time_window + 1 = 9" in entry.excluded[0].reason


def test_a_session_whose_keypoint_never_reaches_pose_confidence_is_excluded() -> None:
    df = _table(_rows(range(10), 0), _rows(range(10), 1))
    df.loc[df["id"] == 1, "poseP2"] = 0.4

    entry = build_sessions(df, _settings(), "embed")

    assert _spans(entry) == [(0, 0, 9)]
    [excluded] = entry.excluded
    assert excluded.individual == 1
    assert "'kp2'" in excluded.reason
    assert "pose_confidence=0.5" in excluded.reason


def test_a_session_whose_keypoint_is_never_finite_is_excluded() -> None:
    df = _rows(range(10), 0)
    df["poseY3"] = np.nan

    [excluded] = build_sessions(df, _settings(), "embed").excluded

    assert "'kp3'" in excluded.reason


def test_an_id_without_a_kept_frame_is_excluded_whole() -> None:
    df = _table(_rows(range(10), 0), _rows(range(5, 15), 1, confidence=0.2))

    entry = build_sessions(df, _settings(), "embed")

    assert _spans(entry) == [(0, 0, 9)]
    [excluded] = entry.excluded
    assert (excluded.individual, excluded.first_frame, excluded.last_frame) == (
        1,
        5,
        14,
    )
    assert "pose_confidence=0.5" in excluded.reason


def test_a_one_frame_session_is_counted_in_the_singular() -> None:
    [excluded] = build_sessions(_rows([5], 0), _settings(), "embed").excluded

    assert excluded.reason == f"1 frame, fewer than time_window={_TIME_WINDOW}"


@pytest.mark.parametrize(
    ("individual", "frames"), [(7, "frames 0 to 1 of id 7"), (None, "frames 0 to 1")]
)
def test_an_exclusion_describes_its_frames_id_and_reason(
    individual: IdValue | None, frames: str
) -> None:
    entry = build_sessions(_rows(range(2), individual), _settings(), "embed")

    [excluded] = entry.excluded
    assert excluded.describe() == f"{frames}: {excluded.reason}"


# --- Confidence ---


def test_confidence_comes_from_the_score_columns() -> None:
    df = _rows(range(10), 0)
    df["poseP1"] = np.linspace(0.6, 0.9, 10)
    _set_cells(df, [4], ["poseX1"], np.nan)

    [session] = build_sessions(df, _settings(), "embed").sessions

    expected = np.linspace(0.6, 0.9, 10).astype(np.float32)
    expected[4] = 0.0
    np.testing.assert_array_equal(session.track.confidence[:, 1], expected)
    np.testing.assert_allclose(session.track.confidence[:, 0], 0.9)


def test_confidence_without_score_columns_is_one_where_the_point_is_finite() -> None:
    df = _rows(range(10), 0)
    df = df.drop(columns=[f"poseP{k}" for k in range(_KEYPOINTS)])
    _set_cells(df, [4], ["poseX1"], np.nan)

    [session] = build_sessions(df, _settings(), "embed").sessions

    expected = np.ones((10, _KEYPOINTS), dtype=np.float32)
    expected[4, 1] = 0.0
    np.testing.assert_array_equal(session.track.confidence, expected)


def test_confidence_is_one_unless_every_selected_keypoint_has_a_score() -> None:
    df = _rows(range(10), 0, confidence=0.6).drop(columns=["poseP3"])

    [session] = build_sessions(df, _settings(), "embed").sessions

    np.testing.assert_array_equal(session.track.confidence, 1.0)


# --- Session order ---


def _mixed_ids() -> pd.DataFrame:
    """Ids of two types, each with two sessions where the frames allow."""
    split = [*range(10), *range(20, 30)]
    return _table(
        _rows(split, 10), _rows(range(10), "b"), _rows(split, 2), _rows(range(10), "a")
    )


def test_sessions_are_ordered_by_the_text_of_the_id_then_the_first_frame() -> None:
    entry = build_sessions(_mixed_ids(), _settings(), "embed")

    assert _spans(entry) == [
        (10, 0, 9),
        (10, 20, 29),
        (2, 0, 9),
        (2, 20, 29),
        ("a", 0, 9),
        ("b", 0, 9),
    ]


def test_session_order_does_not_depend_on_row_order() -> None:
    df = _mixed_ids()

    shuffled = build_sessions(df.sample(frac=1.0, random_state=1), _settings(), "embed")

    assert _spans(shuffled) == _spans(build_sessions(df, _settings(), "embed"))


def test_exclusions_are_ordered_as_sessions_are() -> None:
    df = _table(_rows(range(2), "b"), _rows([*range(2), *range(10, 12)], 10))

    entry = build_sessions(df, _settings(), "embed")

    assert [(e.individual, e.first_frame) for e in entry.excluded] == [
        (10, 0),
        (10, 10),
        ("b", 0),
    ]


# --- Placement ---


def _latents(entry: EntrySessions) -> list[npt.NDArray[np.float64]]:
    """Latent ``i`` of session ``s`` is ``100 s + i`` in every dimension."""
    return [
        np.repeat(
            100.0 * index
            + np.arange(session.n_frames - _TIME_WINDOW + 1, dtype=np.float64)[:, None],
            _ZDIMS,
            axis=1,
        )
        for index, session in enumerate(entry.sessions)
    ]


def _place(df: pd.DataFrame, purpose: SessionPurpose = "embed") -> pd.DataFrame:
    entry = build_sessions(df, _settings(), purpose)
    return place_latents(df, entry, _latents(entry), time_window=_TIME_WINDOW)


def test_latent_columns() -> None:
    assert latent_columns(3) == ["vame_z0", "vame_z1", "vame_z2"]


def test_a_latent_is_placed_at_the_center_of_its_window() -> None:
    placed = _place(_rows(range(3, 13), 0))

    assert placed["frame"].tolist() == list(range(5, 12))
    assert placed["vame_z0"].tolist() == list(range(7))


@pytest.mark.parametrize("time_window", [4, 5])
def test_the_center_is_time_window_floor_divided_by_two(time_window: int) -> None:
    df = _rows(range(10), 0)
    entry = build_sessions(df, _settings(time_window=time_window), "embed")
    latents = [np.zeros((10 - time_window + 1, _ZDIMS))]

    placed = place_latents(df, entry, latents, time_window=time_window)

    assert placed["frame"].iloc[0] == time_window // 2


def test_the_output_holds_the_identity_columns_and_float32_latents() -> None:
    placed = _place(_rows(range(10), 0))

    assert list(placed.columns) == [
        "frame",
        "group",
        "id",
        "sequence",
        "time",
        *latent_columns(_ZDIMS),
    ]
    assert (placed[latent_columns(_ZDIMS)].dtypes == np.float32).all()


def test_each_row_receives_its_own_ids_latent() -> None:
    df = _table(_rows(range(10), 0), _rows(range(10), 1))

    placed = _place(df)

    by_id = placed.groupby("id")["vame_z0"].apply(list).to_dict()
    assert by_id == {0: list(range(7)), 1: [100.0 + i for i in range(7)]}


def test_rows_without_a_latent_are_omitted() -> None:
    """Rows near a session's ends, and every row of an excluded session."""
    df = _table(_rows(range(10), 0), _rows(range(2), 1))

    placed = _place(df)

    assert placed["id"].tolist() == [0] * 7
    assert placed["frame"].tolist() == list(range(2, 9))


def test_a_center_frame_inside_an_interpolated_gap_is_dropped() -> None:
    df = _rows([*range(5), *range(7, 13)], 0)

    placed = _place(df)

    assert placed["frame"].tolist() == [2, 3, 4, 7, 8, 9, 10, 11]
    assert placed["vame_z0"].tolist() == [0, 1, 2, 5, 6, 7, 8, 9]


def test_rows_keep_the_order_of_the_table() -> None:
    df = _table(_rows(range(10), 0), _rows(range(10), 1)).sort_values(
        ["frame", "id"], ignore_index=True
    )

    placed = _place(df)

    assert placed["frame"].tolist() == sorted([*range(2, 9)] * 2)
    assert placed["id"].tolist() == [0, 1] * 7


def test_an_entry_without_a_placed_latent_names_its_exclusions() -> None:
    df = _rows(range(3), 7)

    with pytest.raises(ValueError, match=r"vame: no row .*frames 0 to 2 of id 7"):
        _ = _place(df)


def test_an_entry_without_a_placed_latent_names_its_gap_rows() -> None:
    """Frames 0 and 1 lack a kept reference point, and frames 2 to 4 are too short."""
    df = _rows(range(5), 7)
    _set_cells(df, [0, 1], ["poseP0"], 0.1)

    with pytest.raises(ValueError, match=r"of id 7: 3 frames, .*; 2 rows at frames"):
        _ = _place(df)


def test_latents_of_the_wrong_length_are_refused() -> None:
    df = _rows(range(10), 0)
    entry = build_sessions(df, _settings(), "embed")

    with pytest.raises(ValueError, match=r"shape \(6, 3\), .* 7 windows"):
        _ = place_latents(df, entry, [np.zeros((6, _ZDIMS))], time_window=_TIME_WINDOW)


def test_a_latent_array_per_session_is_required() -> None:
    df = _table(_rows(range(10), 0), _rows(range(10), 1))
    entry = build_sessions(df, _settings(), "embed")

    with pytest.raises(ValueError, match="2 sessions"):
        _ = place_latents(df, entry, [np.zeros((7, _ZDIMS))], time_window=_TIME_WINDOW)


def test_latents_of_different_widths_are_refused() -> None:
    df = _table(_rows(range(10), 0), _rows(range(10), 1))
    entry = build_sessions(df, _settings(), "embed")
    latents = [np.zeros((7, _ZDIMS)), np.zeros((7, _ZDIMS + 1))]

    with pytest.raises(
        ValueError,
        match=(
            r"frames 0 to 9 of id 1 have 4 dimensions, and those of frames 0 to 9 "
            r"of id 0 have 3"
        ),
    ):
        _ = place_latents(df, entry, latents, time_window=_TIME_WINDOW)


# --- Session files ---


def test_session_tracks_are_written_in_order_under_their_names(tmp_path: Path) -> None:
    df = _table(_rows([*range(10), *range(20, 30)], 0), _rows(range(10), 1))
    sessions = build_sessions(df, _settings(), "embed").sessions
    exchange = tmp_path / "exchange"

    written = write_session_tracks(sessions, exchange)

    assert list(written) == ["s000000", "s000001", "s000002"]
    assert sorted(exchange.iterdir()) == [exchange / f"{n}.npz" for n in written]
    for path, session in zip(written.values(), sessions, strict=True):
        with np.load(path) as track:
            assert sorted(track.files) == ["confidence", "frames", "position"]
            assert track["frames"].dtype == np.int64
            np.testing.assert_array_equal(track["frames"], session.track.frames)
            np.testing.assert_array_equal(track["position"], session.track.position)
            np.testing.assert_array_equal(track["confidence"], session.track.confidence)
