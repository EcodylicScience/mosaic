"""Track-quality measures, the events they flag, and the frames chosen from those events.

The criteria run on synthetic tables of ``frame``, ``id``, ``X`` and ``Y``, with
``poseX<k>``, ``poseY<k>`` and ``poseP<k>`` where a criterion reads keypoints or
scores.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mosaic.core.track_quality import (
    QualityRules,
    body_length,
    consecutive_runs,
    events,
    frame_flags,
    instance_body_lengths,
    presence,
    row_flags,
    select,
)

NAN = float("nan")


def test_head_to_tail_length_when_both_are_finite() -> None:
    xy = np.array([[[0.0, 0.0], [1.0, 5.0], [3.0, 4.0]]])
    assert instance_body_lengths(xy, 0, 2).tolist() == [5.0]


def test_tight_box_diagonal_when_head_or_tail_is_missing() -> None:
    xy = np.array([[[0.0, 0.0], [3.0, 4.0], [1.0, 1.0]]])
    assert instance_body_lengths(xy, None, None).tolist() == [5.0]
    assert instance_body_lengths(xy, 0, None).tolist() == [5.0]


def test_tight_box_diagonal_when_head_or_tail_is_not_finite() -> None:
    xy = np.array([[[0.0, 0.0], [np.nan, np.nan], [3.0, 4.0]]])
    assert instance_body_lengths(xy, 0, 1).tolist() == [5.0]


def test_an_out_of_range_index_falls_back_to_the_diagonal() -> None:
    xy = np.array([[[0.0, 0.0], [3.0, 4.0]]])
    assert instance_body_lengths(xy, 0, 7).tolist() == [5.0]


def test_nan_when_no_keypoint_is_finite() -> None:
    xy = np.full((1, 3, 2), np.nan)
    assert np.isnan(instance_body_lengths(xy, 0, 2)).all()


def test_one_length_per_instance() -> None:
    xy = np.array(
        [
            [[0.0, 0.0], [3.0, 4.0]],
            [[np.nan, np.nan], [np.nan, 1.0]],
            [[0.0, 0.0], [6.0, 8.0]],
        ]
    )
    lengths = instance_body_lengths(xy, 0, 1)
    assert lengths[0] == 5.0
    assert np.isnan(lengths[1])
    assert lengths[2] == 10.0


@pytest.mark.parametrize(
    ("frames", "keep", "runs"),
    [
        pytest.param([0, 1, 2, 3], [1, 1, 1, 1], [(0, 4)], id="one-run"),
        pytest.param([], [], [], id="empty"),
        pytest.param([7], [1], [(0, 1)], id="one-frame"),
        pytest.param([0, 1, 2, 5, 6, 7], [1] * 6, [(0, 3), (3, 6)], id="frame-gap"),
        pytest.param(
            [0, 1, 2, 3, 4], [1, 1, 0, 1, 1], [(0, 2), (3, 5)], id="unlabeled"
        ),
        pytest.param([0, 1, 2, 3], [1, 0, 1, 1], [(0, 1), (2, 4)], id="lone-frame"),
        pytest.param([0, 1, 2], [0, 0, 0], [], id="nothing-kept"),
        pytest.param([0, 1, 3, 4], [1, 1, 1, 1], [(0, 2), (2, 4)], id="gap-of-one"),
        pytest.param([4, 5, 6, 8], [0, 1, 1, 1], [(1, 3), (3, 4)], id="gap-after-skip"),
    ],
)
def test_a_run_ends_at_a_frame_gap_and_at_a_frame_that_is_not_kept(
    frames: list[int], keep: list[int], runs: list[tuple[int, int]]
) -> None:
    assert consecutive_runs(frames, [bool(k) for k in keep]) == runs


def _centers(frame: list[int], ids: list[int], x: list[float]) -> pd.DataFrame:
    """A table of centers on one line, ``Y`` zero wherever ``X`` is finite."""
    y = [0.0 if np.isfinite(value) else NAN for value in x]
    return pd.DataFrame({"frame": frame, "id": ids, "X": x, "Y": y})


def _poses(
    frame: list[int],
    ids: list[int],
    keypoints: list[list[tuple[float, float]]],
    scores: list[list[float]] | None = None,
) -> pd.DataFrame:
    """A table of keypoints, each row's center at the mean of its finite keypoints."""
    points = np.array(keypoints, dtype=float)
    columns: dict[str, object] = {"frame": frame, "id": ids}
    finite = np.isfinite(points).all(axis=-1)
    count = finite.sum(axis=1)
    total = np.where(finite[..., None], points, 0.0).sum(axis=1)
    center = np.where(count[:, None] > 0, total / np.maximum(count, 1)[:, None], NAN)
    columns["X"] = center[:, 0]
    columns["Y"] = center[:, 1]
    for k in range(points.shape[1]):
        columns[f"poseX{k}"] = points[:, k, 0]
        columns[f"poseY{k}"] = points[:, k, 1]
    if scores is not None:
        for k, score in enumerate(np.array(scores, dtype=float).T):
            columns[f"poseP{k}"] = score
    return pd.DataFrame(columns)


def test_a_row_is_present_when_its_x_and_y_are_finite() -> None:
    table = pd.DataFrame(
        {
            "frame": [0, 1, 2],
            "id": [1, 1, 1],
            "X": [1.0, NAN, 3.0],
            "Y": [1.0, 2.0, NAN],
        }
    )
    assert presence(table).tolist() == [True, False, False]


def test_the_body_length_is_the_median_head_to_tail_length_of_present_rows() -> None:
    table = _poses(
        [0, 1, 2, 3],
        [1, 1, 1, 1],
        [
            [(0.0, 0.0), (4.0, 0.0)],
            [(0.0, 0.0), (6.0, 0.0)],
            [(0.0, 0.0), (10.0, 0.0)],
            [(0.0, 0.0), (100.0, 0.0)],
        ],
    )
    table.loc[3, "X"] = NAN

    assert body_length(table, 0, 1) == 6.0


def test_the_body_length_falls_back_to_the_box_diagonal_without_a_body_axis() -> None:
    table = _poses([0], [1], [[(0.0, 0.0), (3.0, 4.0), (1.0, 1.0)]])
    assert body_length(table, None, None) == 5.0


def test_a_table_without_keypoint_columns_has_no_body_length() -> None:
    assert body_length(_centers([0, 1], [1, 1], [0.0, 1.0]), None, None) is None


def test_a_table_without_a_finite_keypoint_length_has_no_body_length() -> None:
    table = _poses([0], [1], [[(NAN, NAN), (NAN, NAN)]])
    table.loc[0, ["X", "Y"]] = [1.0, 1.0]
    assert body_length(table, 0, 1) is None


def test_a_single_keypoint_measures_no_body_length() -> None:
    table = _poses([0, 1], [1, 1], [[(0.0, 0.0)], [(5.0, 5.0)]])
    assert body_length(table, None, None) is None


def test_a_body_axis_past_the_tables_keypoints_is_refused() -> None:
    table = _poses([0], [1], [[(0.0, 0.0), (3.0, 4.0)]])
    with pytest.raises(ValueError, match="tail names keypoint 7"):
        body_length(table, 0, 7)
    with pytest.raises(ValueError, match="head names keypoint 2"):
        body_length(table, 2, None)
    with pytest.raises(ValueError, match="the table has 0 keypoints"):
        body_length(_centers([0], [1], [0.0]), 0, 1)


def test_a_jump_is_a_step_longer_than_its_body_lengths_between_consecutive_frames() -> (
    None
):
    table = _centers(
        [0, 1, 2, 4, 0, 1, 0, 1, 0, 1],
        [1, 1, 1, 1, 2, 2, 3, 3, 4, 4],
        [0.0, 1.0, 10.0, 100.0, 50.0, 50.0, 0.0, 8.0, NAN, 100.0],
    )

    flags = row_flags(table, QualityRules(jump_body_lengths=0.8), 10.0)

    # id 1 steps 9 > 8 into frame 2. Frame 4 follows a gap. id 3 steps 8, at the
    # limit. id 4 is absent at frame 0.
    assert flags["jump"].tolist() == [
        False,
        False,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    ]


def test_a_jump_compares_one_id_with_itself_and_not_with_another() -> None:
    table = _centers([0, 1], [1, 2], [0.0, 100.0])
    flags = row_flags(table, QualityRules(jump_body_lengths=0.8), 10.0)
    assert not flags["jump"].any()


def test_the_body_length_given_sets_the_scale_in_place_of_the_measured_one() -> None:
    table = _poses(
        [0, 1],
        [1, 1],
        [[(0.0, 0.0), (10.0, 0.0)], [(5.0, 0.0), (15.0, 0.0)]],
    )
    rules = QualityRules(jump_body_lengths=0.8)
    assert body_length(table, 0, 1) == 10.0

    measured = row_flags(table, rules, body_length(table, 0, 1))
    given = row_flags(table, rules, 5.0)

    assert measured["jump"].tolist() == [False, False]
    assert given["jump"].tolist() == [False, True]


def test_a_criterion_in_body_lengths_without_a_body_length_names_body_length_px() -> (
    None
):
    table = _centers([0, 1], [1, 1], [0.0, 1.0])
    with pytest.raises(ValueError, match="body_length_px"):
        row_flags(table, QualityRules(jump_body_lengths=0.8), None)
    with pytest.raises(ValueError, match="body_length_px"):
        frame_flags(
            table,
            2,
            QualityRules(duplicate_body_lengths=0.2),
            None,
            row_flags(table, QualityRules(), None),
        )


def test_a_low_score_is_a_mean_finite_confidence_below_the_threshold() -> None:
    table = _poses(
        [0, 1, 2, 3],
        [1, 1, 1, 1],
        [[(0.0, 0.0), (1.0, 0.0)]] * 4,
        scores=[[0.1, NAN], [0.4, 0.6], [0.9, 0.9], [0.6, 0.6]],
    )

    flags = row_flags(table, QualityRules(low_score_threshold=0.55), None)

    assert flags["low_score"].tolist() == [True, True, False, False]
    assert flags["score"].tolist() == pytest.approx([0.1, 0.5, 0.9, 0.6])


def test_a_low_score_is_below_the_entrys_own_quantile_of_its_means() -> None:
    table = _poses(
        [0, 1, 2, 3, 4],
        [1, 1, 1, 1, 1],
        [[(0.0, 0.0)]] * 5,
        scores=[[0.3], [0.1], [0.5], [0.2], [0.4]],
    )

    flags = row_flags(table, QualityRules(low_score_quantile=0.25), None)

    assert flags["low_score"].tolist() == [False, True, False, False, False]


def test_a_row_without_a_finite_confidence_has_no_low_score() -> None:
    table = _poses(
        [0, 1],
        [1, 1],
        [[(0.0, 0.0)]] * 2,
        scores=[[NAN], [0.9]],
    )
    flags = row_flags(table, QualityRules(low_score_threshold=0.5), None)
    assert flags["low_score"].tolist() == [False, False]


def test_low_score_is_skipped_for_a_table_without_confidence_columns() -> None:
    table = _poses([0], [1], [[(0.0, 0.0)]])
    flags = row_flags(table, QualityRules(low_score_quantile=0.05), None)
    assert "low_score" not in flags.columns


def test_a_low_score_threshold_and_quantile_together_are_refused() -> None:
    with pytest.raises(ValueError, match="low_score"):
        QualityRules(low_score_threshold=0.5, low_score_quantile=0.05)


def test_missing_keypoints_flags_a_present_row_missing_one_of_the_chosen() -> None:
    table = _poses(
        [0, 1, 2, 3],
        [1, 1, 1, 1],
        [
            [(NAN, NAN), (1.0, 1.0)],
            [(1.0, 1.0), (NAN, NAN)],
            [(1.0, 1.0), (1.0, 1.0)],
            [(NAN, NAN), (NAN, NAN)],
        ],
    )

    chosen = row_flags(table, QualityRules(missing_keypoints=(1,)), None)
    every = row_flags(table, QualityRules(missing_keypoints=()), None)

    assert chosen["missing_keypoints"].tolist() == [False, True, False, False]
    assert every["missing_keypoints"].tolist() == [True, True, False, False]


def test_missing_keypoints_refuses_a_keypoint_past_the_table() -> None:
    table = _poses([0], [1], [[(1.0, 1.0)]])
    with pytest.raises(ValueError, match="keypoint 3"):
        row_flags(table, QualityRules(missing_keypoints=(3,)), None)


def test_a_disabled_criterion_has_no_column() -> None:
    table = _poses([0], [1], [[(1.0, 1.0)]], scores=[[0.5]])
    flags = row_flags(table, QualityRules(), 10.0)
    assert {"jump", "low_score", "missing_keypoints"}.isdisjoint(flags.columns)
    assert flags["present"].tolist() == [True]


def _frames(
    table: pd.DataFrame, span: int, rules: QualityRules, length: float | None
) -> pd.DataFrame:
    return frame_flags(
        table, span, rules, length, row_flags(table, QualityRules(), length)
    )


def test_a_duplicate_is_two_present_rows_closer_than_its_body_lengths() -> None:
    table = _centers(
        [0, 0, 1, 1, 2, 2], [1, 2, 1, 2, 1, 2], [0.0, 1.0, 0.0, 5.0, 0.0, NAN]
    )

    flags = _frames(table, 3, QualityRules(duplicate_body_lengths=0.2), 10.0)

    assert flags["duplicate"].tolist() == [True, False, False]


def test_a_head_or_tail_apart_vetoes_a_duplicate_when_both_are_given() -> None:
    table = _poses(
        [0, 0],
        [1, 2],
        [[(0.0, 5.0), (0.0, -5.0)], [(1.0, -5.0), (1.0, 5.0)]],
    )
    axis = QualityRules(duplicate_body_lengths=0.2, head=0, tail=1)
    centers = QualityRules(duplicate_body_lengths=0.2)

    assert _frames(table, 1, axis, 10.0)["duplicate"].tolist() == [False]
    assert _frames(table, 1, centers, 10.0)["duplicate"].tolist() == [True]


def test_a_non_finite_head_or_tail_distance_does_not_veto_a_duplicate() -> None:
    table = _poses(
        [0, 0],
        [1, 2],
        [[(NAN, NAN), (0.0, 0.0)], [(1.0, 9.0), (1.0, 0.0)]],
    )
    rules = QualityRules(duplicate_body_lengths=0.2, head=0, tail=1)
    table.loc[:, ["X", "Y"]] = [[0.0, 0.0], [1.0, 0.0]]

    assert _frames(table, 1, rules, 10.0)["duplicate"].tolist() == [True]


def test_a_duplicate_veto_past_the_tables_keypoints_is_refused() -> None:
    table = _poses(
        [0, 0],
        [1, 2],
        [[(0.0, 0.0), (1.0, 0.0)], [(0.5, 0.0), (1.5, 0.0)]],
    )
    rules = QualityRules(duplicate_body_lengths=0.2, head=0, tail=5)
    with pytest.raises(ValueError, match="tail names keypoint 5"):
        _frames(table, 1, rules, 10.0)


def test_fewer_than_expected_counts_a_frame_without_rows_as_zero() -> None:
    table = _centers(
        [0, 0, 1, 3, 3, 4, 4],
        [1, 2, 1, 1, 2, 1, 2],
        [0.0, 50.0, 0.0, 0.0, 1.0, 0.0, NAN],
    )
    with_duplicates = QualityRules(duplicate_body_lengths=0.2, n_expected=2)
    without = QualityRules(n_expected=2)

    flags = _frames(table, 5, with_duplicates, 10.0)
    counted = _frames(table, 5, without, None)

    assert flags["n_detected"].tolist() == [2, 1, 0, 1, 1]
    assert flags["fewer_than_expected"].tolist() == [False, True, True, True, True]
    assert counted["n_detected"].tolist() == [2, 1, 0, 2, 1]
    assert counted["fewer_than_expected"].tolist() == [False, True, True, False, True]


def test_rows_joined_by_duplicate_pairs_are_detected_once() -> None:
    table = _centers([0, 0, 0, 0], [1, 2, 3, 4], [0.0, 1.0, 1.5, 50.0])
    flags = _frames(table, 1, QualityRules(duplicate_body_lengths=0.2), 10.0)
    assert flags["n_detected"].tolist() == [2]


def test_fewer_than_expected_is_off_without_n_expected() -> None:
    table = _centers([0], [1], [0.0])
    assert "fewer_than_expected" not in _frames(table, 1, QualityRules(), None).columns


def test_a_frame_outside_the_span_is_refused() -> None:
    table = _centers([0, 5], [1, 1], [0.0, 1.0])
    with pytest.raises(ValueError, match="span"):
        _frames(table, 5, QualityRules(n_expected=1), None)


def _row(frame: list[int], ids: list[str], jump: list[bool]) -> pd.DataFrame:
    n = len(frame)
    return pd.DataFrame(
        {
            "frame": frame,
            "id": ids,
            "x": [float(f) for f in frame],
            "y": [float(i) for i in range(n)],
            "score": [0.5] * n,
            "present": [True] * n,
            "jump": jump,
        }
    )


def _frame(duplicate: list[bool]) -> pd.DataFrame:
    n = len(duplicate)
    return pd.DataFrame(
        {
            "frame": np.arange(n),
            "n_present": [2] * n,
            "n_detected": [2] * n,
            "duplicate": duplicate,
        }
    )


def test_a_row_event_is_one_ids_run_of_consecutive_flagged_frames() -> None:
    row = _row(
        [3, 4, 5, 6, 8, 4, 5],
        ["1", "1", "1", "1", "1", "2", "2"],
        [True, True, True, False, True, True, False],
    )

    found = events(row, _frame([False] * 9), by_id=True, min_event_frames=1)

    assert found[["frame", "cause", "id", "event_id", "event_frames"]].to_dict(
        "records"
    ) == [
        {
            "frame": 3,
            "cause": "jump",
            "id": "1",
            "event_id": "jump:1:3",
            "event_frames": 3,
        },
        {
            "frame": 4,
            "cause": "jump",
            "id": "2",
            "event_id": "jump:2:4",
            "event_frames": 1,
        },
        {
            "frame": 8,
            "cause": "jump",
            "id": "1",
            "event_id": "jump:1:8",
            "event_frames": 1,
        },
    ]
    assert found["x"].tolist() == [3.0, 4.0, 8.0]
    assert found["y"].tolist() == [0.0, 5.0, 4.0]
    assert found["score"].tolist() == [0.5, 0.5, 0.5]


def test_a_row_event_formed_per_frame_runs_over_frames_whatever_the_id() -> None:
    row = _row([3, 4, 4, 6], ["1", "0", "1", "0"], [True, True, True, True])

    found = events(row, _frame([False] * 7), by_id=False, min_event_frames=1)

    assert found[["frame", "id", "event_id", "event_frames"]].to_dict("records") == [
        {"frame": 3, "id": "1", "event_id": "jump:1:3", "event_frames": 2},
        {"frame": 6, "id": "0", "event_id": "jump:0:6", "event_frames": 1},
    ]


def test_a_frame_event_is_a_run_of_consecutive_flagged_frames() -> None:
    row = _row([], [], [])

    found = events(
        row,
        _frame([False, False, True, True, False, True]),
        by_id=True,
        min_event_frames=1,
    )

    assert found[["frame", "cause", "id", "event_id", "event_frames"]].to_dict(
        "records"
    ) == [
        {
            "frame": 2,
            "cause": "duplicate",
            "id": "",
            "event_id": "duplicate:*:2",
            "event_frames": 2,
        },
        {
            "frame": 5,
            "cause": "duplicate",
            "id": "",
            "event_id": "duplicate:*:5",
            "event_frames": 1,
        },
    ]
    assert found[["x", "y", "score"]].isna().all().all()


def test_a_run_shorter_than_min_event_frames_is_no_event() -> None:
    row = _row([0, 1, 5], ["1", "1", "1"], [True, True, True])

    found = events(
        row,
        _frame([True, False, False, True, True, False]),
        by_id=True,
        min_event_frames=2,
    )

    assert found["event_id"].tolist() == ["jump:1:0", "duplicate:*:3"]


def test_events_are_ordered_by_onset_cause_and_id() -> None:
    row = _row([2, 2, 0], ["b", "a", "c"], [True, True, True])

    found = events(row, _frame([False, False, True]), by_id=True, min_event_frames=1)

    assert found["event_id"].tolist() == [
        "jump:c:0",
        "duplicate:*:2",
        "jump:a:2",
        "jump:b:2",
    ]


def test_events_of_the_flags_of_a_table() -> None:
    table = _centers([0, 1, 2, 2], [1, 1, 1, 2], [0.0, 0.0, 20.0, 20.5])
    rules = QualityRules(jump_body_lengths=0.8, duplicate_body_lengths=0.2)
    row = row_flags(table, rules, 10.0)

    found = events(
        row, frame_flags(table, 3, rules, 10.0, row), by_id=True, min_event_frames=1
    )

    assert found["event_id"].tolist() == ["duplicate:*:2", "jump:1:2"]
    assert found["x"].tolist()[1] == 20.0


def _candidates(frames: list[int], causes: list[str] | None = None) -> pd.DataFrame:
    cause = causes or ["jump"] * len(frames)
    ids = [str(n) for n in range(len(frames))]
    return pd.DataFrame(
        {
            "frame": frames,
            "cause": cause,
            "event_id": [
                f"{c}:{i}:{f}" for c, i, f in zip(cause, ids, frames, strict=True)
            ],
            "id": ids,
            "x": [0.0] * len(frames),
            "y": [0.0] * len(frames),
            "score": [NAN] * len(frames),
            "event_frames": [1] * len(frames),
        }
    )


def test_no_frame_is_chosen_within_the_separation_of_an_excluded_frame() -> None:
    chosen = select(
        _candidates([90, 140, 129, 71]),
        n_frames=10,
        min_separation=30,
        exclude=[100],
        seed=0,
    )
    assert chosen["frame"].tolist() == [140]


def test_no_two_chosen_frames_lie_within_the_separation() -> None:
    frames = list(range(0, 200, 7))
    for seed in range(20):
        chosen = select(
            _candidates(frames), n_frames=100, min_separation=15, exclude=[], seed=seed
        )["frame"].to_numpy()
        assert np.diff(chosen).min() >= 15


def test_frames_exactly_the_separation_apart_are_both_chosen() -> None:
    chosen = select(
        _candidates([0, 30]), n_frames=5, min_separation=30, exclude=[], seed=0
    )
    assert chosen["frame"].tolist() == [0, 30]


def test_a_candidate_equal_to_a_chosen_frame_is_never_taken() -> None:
    candidates = _candidates([5, 5, 7], ["jump", "low_score", "jump"])
    for seed in range(20):
        chosen = select(
            candidates, n_frames=5, min_separation=0, exclude=[7], seed=seed
        )
        assert chosen["frame"].tolist() == [5]


def test_at_most_n_frames_are_chosen() -> None:
    chosen = select(
        _candidates(list(range(0, 1000, 10))),
        n_frames=4,
        min_separation=0,
        exclude=[],
        seed=3,
    )
    assert len(chosen) == 4
    assert chosen["frame"].is_monotonic_increasing


def test_one_seed_gives_the_same_picks_on_every_call() -> None:
    candidates = _candidates(list(range(0, 1000, 10)))

    def picks(seed: int) -> list[str]:
        return select(
            candidates, n_frames=5, min_separation=40, exclude=[500], seed=seed
        )["event_id"].tolist()

    assert picks(11) == picks(11)
    assert picks(11) != picks(12)


def test_the_picks_do_not_depend_on_the_order_of_the_candidates() -> None:
    candidates = _candidates(list(range(0, 1000, 10)))
    shuffled = candidates.sample(frac=1.0, random_state=1)

    def picks(table: pd.DataFrame) -> list[str]:
        return select(table, n_frames=5, min_separation=40, exclude=[], seed=4)[
            "event_id"
        ].tolist()

    assert picks(candidates) == picks(shuffled)


def test_no_candidate_selects_nothing() -> None:
    chosen = select(_candidates([]), n_frames=5, min_separation=0, exclude=[], seed=0)
    assert chosen.empty
    assert list(chosen.columns) == list(_candidates([]).columns)
