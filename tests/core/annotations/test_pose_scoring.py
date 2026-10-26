"""Scoring pose predictions against annotated frames, and the summary pooled over them.

The cases are synthetic. Every truth instance is a four-keypoint body whose head
and tail lie 10 px apart, so its normalizer is 10 px unless a case says
otherwise, and every rate below is a count worked out by hand.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from mosaic.core.annotations.keypoint_map import KeypointMap, resolve_keypoint_map
from mosaic.core.annotations.model import (
    AnnotationFrame,
    AnnotationObject,
    Bbox,
    Keypoint,
    KeypointSchema,
    ObjectOrigin,
    Visibility,
)
from mosaic.core.annotations.pose_predictions import (
    FramePredictions,
    PredictedFrame,
    PredictedObject,
    PredictedPose,
)
from mosaic.core.annotations.scoring import (
    ErrorQuantiles,
    KeypointMapSummary,
    LabelRevision,
    PoseMetrics,
    PoseScoreParams,
    PoseScores,
    PoseScoreSummary,
    TruthFrame,
    match_instances,
    score_predictions,
    untouched_keypoints,
)
from mosaic.core.annotations.split import VideoFrame, frame_key
from mosaic.core.resample import percentile_interval, resampled_sums

SCHEMA = KeypointSchema(names=("head", "left", "right", "tail"), symmetries=((1, 2),))
IDENTITY = resolve_keypoint_map(SCHEMA, SCHEMA.names)
ORIGIN = "origin-a"
FRAMES_RUN = "kmeans-0123456789"
IMAGE = "media/frames/a.png"

PARAMS = PoseScoreParams(
    center=None,
    head=0,
    tail=3,
    match_radius=0.5,
    match_radius_px=None,
    duplicate_radius=0.2,
    oks_sigmas=(0.025, 0.025, 0.025, 0.025),
    pck_alphas=(0.05, 0.1, 0.2),
    min_instance_score=None,
    min_keypoint_score=None,
    n_boot=1000,
    confidence=0.95,
    random_state=0,
)

FRAME_COLUMNS = [
    "origin_uuid",
    "image",
    "frames_run_id",
    "group",
    "sequence",
    "camera",
    "frame_index",
    "frame_key",
    "status",
    "n_gt",
    "n_pred",
    "n_matched",
    "n_fp",
    "n_fp_duplicate",
    "all_found",
    "n_kpt_eligible",
    "n_kpt_found",
    "n_kpt_untouched",
    "n_kpt_unnormalized",
    "n_pck_0p05",
    "n_pck_0p1",
    "n_pck_0p2",
    "oks_sum",
    "n_oks",
    "n_swap_eligible",
    "n_swap",
    "min_oks",
]

INSTANCE_COLUMNS = [
    "origin_uuid",
    "image",
    "frame_key",
    "gt_index",
    "pred_index",
    "matched",
    "duplicate",
    "center_distance_px",
    "normalizer_px",
    "oks",
    "instance_score",
    "n_kpt_eligible",
    "n_kpt_found",
    "n_kpt_untouched",
    "error_px_0",
    "error_px_1",
    "error_px_2",
    "error_px_3",
    "source_ref",
]

Point = tuple[float, float] | None
Row = dict[Hashable, object]


def _body(dx: float = 0.0, dy: float = 0.0) -> tuple[Point, ...]:
    """Head, left, right and tail of a body whose head sits at ``(dx, dy)``."""
    return ((dx, dy), (dx + 5.0, dy - 2.0), (dx + 5.0, dy + 2.0), (dx + 10.0, dy))


def _tight_box(points: Sequence[Point]) -> Bbox:
    placed = [point for point in points if point is not None]
    if not placed:
        return Bbox(0.0, 0.0, 0.0, 0.0)
    xs = [x for x, _ in placed]
    ys = [y for _, y in placed]
    return Bbox(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))


def _truth_object(
    points: Sequence[Point],
    *,
    box: Bbox | None = None,
    visibilities: Sequence[Visibility] | None = None,
    source_ref: str | None = None,
    origin: ObjectOrigin | None = None,
) -> AnnotationObject:
    flags: Sequence[Visibility] = visibilities or (2,) * len(points)
    return AnnotationObject(
        keypoints=tuple(
            Keypoint.absent()
            if point is None
            else Keypoint(x=point[0], y=point[1], visibility=visibility)
            for point, visibility in zip(points, flags, strict=True)
        ),
        bbox=box or _tight_box(points),
        origin=origin,
        source_ref=source_ref,
    )


def _predicted(
    points: Sequence[Point],
    *,
    score: float | None = None,
    scores: Sequence[float | None] | None = None,
) -> PredictedObject:
    keypoint_scores: Sequence[float | None] = scores or (None,) * len(points)
    return PredictedObject(
        keypoints=tuple(
            Keypoint.absent()
            if point is None
            else Keypoint(x=point[0], y=point[1], visibility=2, score=point_score)
            for point, point_score in zip(points, keypoint_scores, strict=True)
        ),
        score=score,
    )


def _truth_frame(
    image: str, *objects: AnnotationObject, origin: str = ORIGIN
) -> TruthFrame:
    return TruthFrame(
        origin_uuid=origin,
        frame=AnnotationFrame(
            image_path=Path(image), width=640, height=480, objects=objects
        ),
    )


def _predicted_frame(
    image: str, *objects: PredictedObject, frame_index: int = 0
) -> PredictedFrame:
    return PredictedFrame(
        image=image, frame_index=frame_index, width=640, height=480, objects=objects
    )


def _run(
    *frames: PredictedFrame,
    names: tuple[str, ...] = SCHEMA.names,
    frames_run_id: str = FRAMES_RUN,
    sequence: str = "s",
    camera: str = "",
) -> FramePredictions:
    return FramePredictions(
        pose=PredictedPose(id=None, name="", schema=KeypointSchema(names=names)),
        frames_run_id=frames_run_id,
        group="g",
        sequence=sequence,
        camera=camera,
        frames=frames,
    )


def _score(
    truth: Sequence[TruthFrame],
    predictions: Sequence[FramePredictions],
    *,
    params: PoseScoreParams = PARAMS,
    keypoint_map: KeypointMap = IDENTITY,
    seen: frozenset[str] | None = None,
    untouched: Mapping[tuple[str, str, int], tuple[bool, ...]] | None = None,
) -> PoseScores:
    return score_predictions(
        truth,
        predictions,
        predictions_origin=ORIGIN,
        keypoint_map=keypoint_map,
        seen=seen,
        untouched=untouched or {},
        params=params,
    )


def _score_one(
    truth_objects: Sequence[AnnotationObject],
    predicted_objects: Sequence[PredictedObject],
    *,
    params: PoseScoreParams = PARAMS,
    keypoint_map: KeypointMap = IDENTITY,
    untouched: Mapping[tuple[str, str, int], tuple[bool, ...]] | None = None,
    names: tuple[str, ...] = SCHEMA.names,
) -> PoseScores:
    """Score one truth frame against one prediction frame of the same image."""
    return _score(
        [_truth_frame(IMAGE, *truth_objects)],
        [_run(_predicted_frame(IMAGE, *predicted_objects), names=names)],
        params=params,
        keypoint_map=keypoint_map,
        untouched=untouched,
    )


def _records(table: pd.DataFrame) -> list[Row]:
    """Return the rows of *table*, a null of a nullable column read as ``None``."""
    return [
        {name: None if value is pd.NA else value for name, value in row.items()}
        for row in table.to_dict("records")
    ]


def _is_nan(value: object) -> bool:
    """Return whether *value* is a float NaN, a null of a float column."""
    return isinstance(value, float) and math.isnan(value)


def _frame_row(scores: PoseScores, image: str = IMAGE) -> Row:
    (row,) = [row for row in _records(scores.frames) if row["image"] == image]
    return row


def _instance_rows(scores: PoseScores) -> list[Row]:
    return _records(scores.instances)


def _key(frame_index: int) -> str:
    return frame_key(
        VideoFrame(
            origin_uuid=ORIGIN,
            group="g",
            sequence="s",
            camera="",
            frame_index=frame_index,
            image_stem="a",
        )
    )


def _greedy(
    gt_xy: np.ndarray, pred_xy: np.ndarray, radius: np.ndarray
) -> list[tuple[int, int]]:
    """Nearest pair first, the rule that matching must beat."""
    distances = np.hypot(
        gt_xy[:, None, 0] - pred_xy[None, :, 0], gt_xy[:, None, 1] - pred_xy[None, :, 1]
    )
    pairs: list[tuple[int, int]] = []
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    for flat in np.argsort(distances, axis=None).tolist():
        gt, pred = divmod(int(flat), distances.shape[1])
        if gt in used_gt or pred in used_pred:
            continue
        if float(distances[gt, pred]) < float(radius[gt]):
            pairs.append((gt, pred))
            used_gt.add(gt)
            used_pred.add(pred)
    return pairs


# --- match_instances --------------------------------------------------------------


def test_matching_pairs_more_instances_than_greedy_nearest_first() -> None:
    gt_xy = np.array([[0.0, 0.0], [10.0, 0.0]])
    pred_xy = np.array([[4.0, 0.0], [-5.0, 0.0]])
    radius = np.array([7.0, 7.0])

    gt, pred = match_instances(gt_xy, pred_xy, radius)

    assert list(zip(gt.tolist(), pred.tolist(), strict=True)) == [(0, 1), (1, 0)]
    assert _greedy(gt_xy, pred_xy, radius) == [(0, 0)]


def test_matching_minimizes_total_distance_among_equal_counts() -> None:
    gt, pred = match_instances(
        np.array([[0.0, 0.0], [10.0, 0.0]]),
        np.array([[9.0, 0.0], [1.0, 0.0]]),
        np.array([20.0, 20.0]),
    )

    assert (gt.tolist(), pred.tolist()) == ([0, 1], [1, 0])


def test_matching_drops_a_pair_at_or_beyond_its_truth_radius() -> None:
    gt_xy = np.array([[0.0, 0.0]])
    pred_xy = np.array([[5.0, 0.0]])

    gt, pred = match_instances(gt_xy, pred_xy, np.array([5.0]))
    assert (gt.tolist(), pred.tolist()) == ([], [])

    gt, pred = match_instances(gt_xy, pred_xy, np.array([5.5]))
    assert (gt.tolist(), pred.tolist()) == ([0], [0])


def test_matching_reads_each_truth_instance_radius() -> None:
    gt, pred = match_instances(
        np.array([[0.0, 0.0], [20.0, 0.0]]),
        np.array([[5.0, 0.0]]),
        np.array([2.0, 20.0]),
    )

    assert (gt.tolist(), pred.tolist()) == ([1], [0])


def test_matching_never_pairs_a_center_that_is_not_finite() -> None:
    gt, pred = match_instances(
        np.array([[math.nan, math.nan], [0.0, 0.0]]),
        np.array([[0.0, 0.0], [math.nan, 0.0]]),
        np.array([5.0, 5.0]),
    )

    assert (gt.tolist(), pred.tolist()) == ([1], [0])


def test_matching_with_no_instance_on_either_side_pairs_nothing() -> None:
    gt, pred = match_instances(np.zeros((0, 2)), np.array([[0.0, 0.0]]), np.zeros((0,)))

    assert (gt.tolist(), pred.tolist()) == ([], [])


# --- radius -----------------------------------------------------------------------


@pytest.mark.parametrize(("match_radius", "matched"), [(0.5, 1), (0.3, 0)])
def test_the_radius_is_match_radius_body_lengths(
    match_radius: float, matched: int
) -> None:
    scores = _score_one(
        [_truth_object(_body())],
        [_predicted(_body(4.0))],
        params=replace(PARAMS, match_radius=match_radius),
    )

    row = _frame_row(scores)
    assert (row["n_matched"], row["n_fp"]) == (matched, 1 - matched)


@pytest.mark.parametrize(
    ("match_radius", "match_radius_px", "matched"), [(0.3, 4.5, 1), (0.5, 3.5, 0)]
)
def test_match_radius_px_replaces_the_body_length_radius(
    match_radius: float, match_radius_px: float, matched: int
) -> None:
    scores = _score_one(
        [_truth_object(_body())],
        [_predicted(_body(4.0))],
        params=replace(
            PARAMS, match_radius=match_radius, match_radius_px=match_radius_px
        ),
    )

    assert _frame_row(scores)["n_matched"] == matched


def test_the_center_keypoint_anchors_matching_when_placed() -> None:
    far_tail = (*_body()[:3], (40.0, 0.0))
    truth = [_truth_object(_body())]

    by_mean = _score_one(truth, [_predicted(far_tail)])
    by_left = _score_one(
        truth, [_predicted(far_tail)], params=replace(PARAMS, center=1)
    )
    left_unplaced = _score_one(
        truth,
        [_predicted((far_tail[0], None, far_tail[2], far_tail[3]))],
        params=replace(PARAMS, center=1),
    )

    assert _frame_row(by_mean)["n_matched"] == 0
    assert _frame_row(by_left)["n_matched"] == 1
    assert _frame_row(left_unplaced)["n_matched"] == 0


# --- false positives and duplicates -------------------------------------------------


def test_an_unmatched_prediction_near_a_matched_one_is_a_duplicate() -> None:
    predicted = [_predicted(_body()), _predicted(_body(1.0)), _predicted(_body(50.0))]

    scores = _score_one([_truth_object(_body())], predicted)

    row = _frame_row(scores)
    assert (row["n_pred"], row["n_matched"], row["n_fp"], row["n_fp_duplicate"]) == (
        3,
        1,
        2,
        1,
    )
    instances = _instance_rows(scores)
    assert [(r["gt_index"], r["pred_index"]) for r in instances] == [
        (0, 0),
        (None, 1),
        (None, 2),
    ]
    assert [r["matched"] for r in instances] == [True, False, False]
    assert [r["duplicate"] for r in instances] == [False, True, False]


def test_the_duplicate_radius_is_body_lengths_of_the_match() -> None:
    predicted = [_predicted(_body()), _predicted(_body(1.0))]

    scores = _score_one(
        [_truth_object(_body())],
        predicted,
        params=replace(PARAMS, duplicate_radius=0.05),
    )

    row = _frame_row(scores)
    assert (row["n_fp"], row["n_fp_duplicate"]) == (1, 0)


def test_predictions_below_min_instance_score_are_removed_before_matching() -> None:
    predicted = [_predicted(_body(), score=0.2), _predicted(_body(30.0), score=None)]

    scores = _score_one(
        [_truth_object(_body())],
        predicted,
        params=replace(PARAMS, min_instance_score=0.5),
    )

    row = _frame_row(scores)
    assert (row["n_pred"], row["n_matched"], row["n_fp"]) == (1, 0, 1)
    unmatched = _instance_rows(scores)[-1]
    assert unmatched["pred_index"] == 1
    assert _is_nan(unmatched["instance_score"])


# --- empty frames and all found -----------------------------------------------------


def test_a_usable_empty_frame_counts_its_false_positives_and_has_no_all_found() -> None:
    truth = [_truth_frame("a.png"), _truth_frame("b.png")]
    predictions = [
        _run(_predicted_frame("a.png", _predicted(_body())), _predicted_frame("b.png"))
    ]

    scores = _score(truth, predictions)

    with_prediction = _frame_row(scores, "a.png")
    assert with_prediction["status"] == "scored"
    assert (with_prediction["n_gt"], with_prediction["n_pred"]) == (0, 1)
    assert (with_prediction["n_matched"], with_prediction["n_fp"]) == (0, 1)
    assert with_prediction["all_found"] is None
    empty = _frame_row(scores, "b.png")
    assert (empty["status"], empty["n_fp"]) == ("scored", 0)
    assert empty["all_found"] is None


def test_all_found_needs_every_truth_instance_matched() -> None:
    truth = [_truth_object(_body()), _truth_object(_body(100.0))]

    every = _score_one(truth, [_predicted(_body()), _predicted(_body(100.0))])
    one = _score_one(truth, [_predicted(_body())])

    assert bool(_frame_row(every)["all_found"])
    assert not bool(_frame_row(one)["all_found"])


# --- pose metrics -------------------------------------------------------------------


def test_oks_and_pck_by_hand() -> None:
    truth = _truth_object(_body(), box=Bbox(0.0, -5.0, 10.0, 10.0))
    predicted = _predicted(((0.4, 0.0), (5.0, -2.0), (5.0, 2.8), None))
    params = replace(PARAMS, oks_sigmas=(0.025, 0.05, 0.075, 0.1))

    scores = _score_one([truth], [predicted], params=params)

    # Area 100 px^2; kappa = 2 * sigma; the tail is a miss and contributes 0.
    oks = (math.exp(-0.16 / (2 * 100 * 0.05**2)) + 1.0) / 4 + math.exp(
        -0.64 / (2 * 100 * 0.15**2)
    ) / 4
    row = _frame_row(scores)
    assert (row["n_kpt_eligible"], row["n_kpt_found"]) == (4, 3)
    # PCK thresholds at L = 10: 0.5, 1 and 2 px against errors 0.4, 0 and 0.8.
    assert (row["n_pck_0p05"], row["n_pck_0p1"], row["n_pck_0p2"]) == (2, 3, 3)
    assert row["n_oks"] == 1
    assert row["oks_sum"] == pytest.approx(oks)
    assert row["min_oks"] == pytest.approx(oks)
    (instance,) = _instance_rows(scores)
    assert instance["oks"] == pytest.approx(oks)
    assert instance["normalizer_px"] == 10.0
    assert instance["center_distance_px"] == pytest.approx(
        math.hypot(5.0 - 10.4 / 3, 0.8 / 3)
    )
    errors = [instance[f"error_px_{position}"] for position in range(4)]
    assert errors[:3] == pytest.approx([0.4, 0.0, 0.8])
    assert _is_nan(errors[3])


def test_a_missed_truth_instance_counts_for_detection_only() -> None:
    scores = _score_one([_truth_object(_body())], [])

    row = _frame_row(scores)
    assert (row["n_gt"], row["n_matched"], row["n_kpt_eligible"]) == (1, 0, 0)
    assert not bool(row["all_found"])
    assert (row["n_oks"], row["oks_sum"]) == (0, 0.0)
    assert _is_nan(row["min_oks"])
    (instance,) = _instance_rows(scores)
    assert not bool(instance["matched"])
    assert instance["pred_index"] is None
    assert instance["normalizer_px"] == 10.0
    assert _is_nan(instance["oks"])
    assert _is_nan(instance["center_distance_px"])


def test_a_keypoint_below_min_keypoint_score_is_a_miss() -> None:
    predicted = _predicted(_body(), scores=(0.1, 0.9, None, None))

    scores = _score_one(
        [_truth_object(_body())],
        [predicted],
        params=replace(PARAMS, min_keypoint_score=0.5),
    )

    row = _frame_row(scores)
    assert (row["n_kpt_eligible"], row["n_kpt_found"]) == (4, 3)
    (instance,) = _instance_rows(scores)
    assert _is_nan(instance["error_px_0"])
    assert instance["error_px_2"] == 0.0


def test_a_visibility_one_keypoint_is_not_eligible() -> None:
    truth = _truth_object(_body(), visibilities=(2, 1, 2, 2))

    scores = _score_one([truth], [_predicted(_body())])

    row = _frame_row(scores)
    assert (row["n_kpt_eligible"], row["n_kpt_found"]) == (3, 3)
    assert _is_nan(_instance_rows(scores)[0]["error_px_1"])


def test_an_instance_without_a_normalizer_is_counted_unnormalized() -> None:
    truth = _truth_object(((0.0, 0.0), None, None, None), box=Bbox(0.0, 0.0, 0.0, 0.0))
    predicted = _predicted(((1.0, 0.0), None, None, None))

    scores = _score_one(
        [truth], [predicted], params=replace(PARAMS, match_radius_px=5.0)
    )

    row = _frame_row(scores)
    assert (row["n_matched"], row["n_kpt_eligible"], row["n_kpt_found"]) == (1, 1, 1)
    assert row["n_kpt_unnormalized"] == 1
    assert (row["n_pck_0p05"], row["n_pck_0p1"], row["n_pck_0p2"]) == (0, 0, 0)
    assert _instance_rows(scores)[0]["normalizer_px"] == 0.0


def test_the_normalizer_falls_back_to_the_box_diagonal() -> None:
    one_point = _truth_object(
        ((0.0, 0.0), None, None, None), box=Bbox(0.0, 0.0, 6.0, 8.0)
    )

    head_to_tail = _score_one([_truth_object(_body())], [])
    tight_box = _score_one(
        [_truth_object(_body())], [], params=replace(PARAMS, head=None, tail=None)
    )
    drawn_box = _score_one([one_point], [])

    assert _instance_rows(head_to_tail)[0]["normalizer_px"] == 10.0
    assert _instance_rows(tight_box)[0]["normalizer_px"] == pytest.approx(
        math.sqrt(116.0)
    )
    assert _instance_rows(drawn_box)[0]["normalizer_px"] == 10.0


# --- swaps --------------------------------------------------------------------------


def test_swaps_over_the_symmetry_pair_and_the_head_and_tail_pair() -> None:
    mirrored = _predicted(((10.0, 0.0), (5.0, 2.0), (5.0, -2.0), (0.0, 0.0)))

    swapped = _score_one([_truth_object(_body())], [mirrored])
    correct = _score_one([_truth_object(_body())], [_predicted(_body())])

    assert (
        _frame_row(swapped)["n_swap_eligible"],
        _frame_row(swapped)["n_swap"],
    ) == (2, 2)
    assert (
        _frame_row(correct)["n_swap_eligible"],
        _frame_row(correct)["n_swap"],
    ) == (2, 0)


def test_the_head_and_tail_pair_counts_only_when_both_are_set() -> None:
    scores = _score_one(
        [_truth_object(_body())],
        [_predicted(_body())],
        params=replace(PARAMS, head=None),
    )

    assert _frame_row(scores)["n_swap_eligible"] == 1


def test_a_head_and_tail_pair_that_is_a_symmetry_pair_counts_once() -> None:
    schema = KeypointSchema(names=SCHEMA.names, symmetries=((0, 3), (1, 2)))

    scores = _score_one(
        [_truth_object(_body())],
        [_predicted(_body())],
        keypoint_map=resolve_keypoint_map(schema, schema.names),
    )

    assert _frame_row(scores)["n_swap_eligible"] == 2


def test_a_swap_needs_both_points_found() -> None:
    scores = _score_one(
        [_truth_object(_body())], [_predicted((*_body()[:2], None, _body()[3]))]
    )

    assert _frame_row(scores)["n_swap_eligible"] == 1


# --- untouched keypoints ------------------------------------------------------------


def test_untouched_keypoints_leave_pose_metrics_and_count_for_detection() -> None:
    untouched = {(ORIGIN, IMAGE, 0): (True, False, False, False)}

    scores = _score_one(
        [_truth_object(_body())], [_predicted(_body())], untouched=untouched
    )

    row = _frame_row(scores)
    assert (row["n_matched"], row["n_kpt_eligible"], row["n_kpt_found"]) == (1, 3, 3)
    assert row["n_kpt_untouched"] == 1
    (instance,) = _instance_rows(scores)
    assert _is_nan(instance["error_px_0"])
    assert instance["n_kpt_untouched"] == 1


def test_an_instance_untouched_everywhere_is_matched_and_has_no_oks() -> None:
    untouched = {(ORIGIN, IMAGE, 0): (True, True, True, True)}

    scores = _score_one(
        [_truth_object(_body())], [_predicted(_body(2.0))], untouched=untouched
    )

    row = _frame_row(scores)
    assert (row["n_matched"], row["n_kpt_eligible"], row["n_kpt_untouched"]) == (
        1,
        0,
        4,
    )
    assert (row["n_oks"], row["n_swap_eligible"]) == (0, 0)
    assert _is_nan(row["min_oks"])
    assert _is_nan(_instance_rows(scores)[0]["oks"])


def test_an_untouched_flag_on_a_keypoint_that_is_not_eligible_is_not_counted() -> None:
    truth = _truth_object(_body(), visibilities=(1, 2, 2, 2))
    untouched = {(ORIGIN, IMAGE, 0): (True, False, False, False)}

    scores = _score_one([truth], [_predicted(_body())], untouched=untouched)

    row = _frame_row(scores)
    assert (row["n_kpt_eligible"], row["n_kpt_untouched"]) == (3, 0)


# --- keypoint map coverage ----------------------------------------------------------


def test_only_target_keypoints_in_the_map_coverage_are_scored() -> None:
    names = ("right", "head", "left")
    keypoint_map = resolve_keypoint_map(SCHEMA, names)
    head, left, right, _ = _body()

    scores = _score_one(
        [_truth_object(_body())],
        [_predicted((right, head, left))],
        keypoint_map=keypoint_map,
        names=names,
    )

    row = _frame_row(scores)
    assert (row["n_kpt_eligible"], row["n_kpt_found"]) == (3, 3)
    assert row["min_oks"] == 1.0
    (instance,) = _instance_rows(scores)
    assert [instance[f"error_px_{position}"] for position in range(3)] == [0.0] * 3
    assert _is_nan(instance["error_px_3"])


# --- statuses -----------------------------------------------------------------------


def test_each_truth_frame_is_scored_excluded_or_not_covered() -> None:
    truth = [
        _truth_frame("a.png", _truth_object(_body())),
        _truth_frame("b.png", _truth_object(_body())),
        _truth_frame("c.png", _truth_object(_body())),
    ]
    predictions = [
        _run(
            _predicted_frame("a.png", _predicted(_body()), frame_index=10),
            _predicted_frame("b.png", _predicted(_body()), frame_index=20),
        )
    ]

    scores = _score(truth, predictions, seen=frozenset({_key(10)}))

    excluded = _frame_row(scores, "a.png")
    scored = _frame_row(scores, "b.png")
    not_covered = _frame_row(scores, "c.png")
    assert [excluded["status"], scored["status"], not_covered["status"]] == [
        "excluded_training",
        "scored",
        "not_covered",
    ]
    assert (excluded["frames_run_id"], excluded["frame_index"]) == (FRAMES_RUN, 10)
    assert excluded["frame_key"] == _key(10)
    assert (scored["frame_key"], scored["n_gt"], scored["n_matched"]) == (
        _key(20),
        1,
        1,
    )
    assert (excluded["n_gt"], excluded["n_pred"], excluded["n_matched"]) == (0, 0, 0)
    assert excluded["all_found"] is None
    assert [
        not_covered[column]
        for column in (
            "frames_run_id",
            "group",
            "sequence",
            "camera",
            "frame_key",
            "frame_index",
            "n_gt",
        )
    ] == ["", "", "", "", "", -1, 0]
    assert not_covered["all_found"] is None
    assert {row["image"] for row in _instance_rows(scores)} == {"b.png"}


def test_without_seen_frames_a_covered_frame_is_scored() -> None:
    truth = [_truth_frame("a.png", _truth_object(_body()))]
    predictions = [_run(_predicted_frame("a.png", frame_index=10))]

    scores = _score(truth, predictions, seen=None)

    assert _frame_row(scores, "a.png")["status"] == "scored"


def test_a_truth_frame_of_another_origin_is_not_covered() -> None:
    truth = [_truth_frame("a.png", _truth_object(_body()), origin="origin-b")]
    predictions = [_run(_predicted_frame("a.png", _predicted(_body())))]

    scores = _score(truth, predictions)

    assert _frame_row(scores, "a.png")["status"] == "not_covered"


# --- tables -------------------------------------------------------------------------


def test_the_frames_and_instances_columns_and_types() -> None:
    scores = _score_one([_truth_object(_body())], [_predicted(_body())])

    assert list(scores.frames.columns) == FRAME_COLUMNS
    assert list(scores.instances.columns) == INSTANCE_COLUMNS
    frame_types = {
        str(name): str(dtype) for name, dtype in scores.frames.dtypes.items()
    }
    assert frame_types["image"] == "string"
    assert frame_types["frame_index"] == "int64"
    assert frame_types["all_found"] == "boolean"
    assert frame_types["oks_sum"] == "float64"
    assert frame_types["min_oks"] == "float64"
    instance_types = {
        str(name): str(dtype) for name, dtype in scores.instances.dtypes.items()
    }
    assert (instance_types["gt_index"], instance_types["pred_index"]) == (
        "Int64",
        "Int64",
    )
    assert (instance_types["matched"], instance_types["duplicate"]) == ("bool", "bool")
    assert instance_types["error_px_0"] == "float64"
    assert instance_types["source_ref"] == "string"


def test_the_pck_columns_name_each_alpha_with_p_for_the_point() -> None:
    scores = _score_one(
        [_truth_object(_body())],
        [_predicted(_body())],
        params=replace(PARAMS, pck_alphas=(0.05, 1.0)),
    )

    pck = [
        str(column)
        for column in scores.frames.columns
        if str(column).startswith("n_pck_")
    ]
    assert pck == ["n_pck_0p05", "n_pck_1"]


def test_the_instance_row_records_the_truth_source_ref() -> None:
    truth = _truth_object(_body(), source_ref="predict-frames.0.1-abc#0")

    scores = _score_one([truth], [_predicted(_body(), score=0.9)])

    (instance,) = _instance_rows(scores)
    assert instance["source_ref"] == "predict-frames.0.1-abc#0"
    assert instance["instance_score"] == 0.9


def test_frames_sort_by_origin_and_image() -> None:
    truth = [
        _truth_frame("b.png", origin="o2"),
        _truth_frame("a.png", origin="o2"),
        _truth_frame("z.png", origin="o1"),
    ]

    scores = _score(truth, [_run(_predicted_frame("a.png"))])

    keys = [(row["origin_uuid"], row["image"]) for row in _records(scores.frames)]
    assert keys == [("o1", "z.png"), ("o2", "a.png"), ("o2", "b.png")]


def test_instances_sort_by_truth_then_prediction_with_nulls_last() -> None:
    truth = [_truth_object(_body(dx)) for dx in (0.0, 100.0, 200.0)]
    predicted = [_predicted(_body(dx)) for dx in (100.0, 0.0, 300.0, 400.0)]
    frames = [
        _truth_frame("b.png", *truth),
        _truth_frame("a.png", _truth_object(_body())),
    ]
    predictions = [
        _run(
            _predicted_frame("b.png", *predicted),
            _predicted_frame("a.png", _predicted(_body())),
        )
    ]

    scores = _score(frames, predictions)

    rows = [
        (r["image"], r["gt_index"], r["pred_index"]) for r in _instance_rows(scores)
    ]
    assert rows == [
        ("a.png", 0, 0),
        ("b.png", 0, 1),
        ("b.png", 1, 0),
        ("b.png", 2, None),
        ("b.png", None, 2),
        ("b.png", None, 3),
    ]


# --- summary ------------------------------------------------------------------------


def _two_sequences(
    *,
    predicted: tuple[bool, bool] = (True, True),
    params: PoseScoreParams = PARAMS,
) -> PoseMetrics:
    """Score one truth body in each of two sequences, each found exactly or not at all.

    Args:
        predicted: Whether ``s1`` and ``s2`` predict their body.
        params: The metric settings.
    """
    truth = [
        _truth_frame(image, _truth_object(_body())) for image in ("a.png", "b.png")
    ]
    predictions = [
        _run(
            _predicted_frame(image, *([_predicted(_body())] if found else [])),
            sequence=sequence,
        )
        for image, sequence, found in zip(
            ("a.png", "b.png"), ("s1", "s2"), predicted, strict=True
        )
    ]
    return _score(truth, predictions, params=params).metrics


def test_pooled_rates_are_the_rates_of_the_pooled_frames() -> None:
    """Two frames pool to the rates of one frame that contains their instances."""
    split = _score(
        [
            _truth_frame("a.png", _truth_object(_body())),
            _truth_frame(
                "b.png", *(_truth_object(_body(dx)) for dx in (0.0, 100.0, 200.0))
            ),
        ],
        [
            _run(
                _predicted_frame("a.png", _predicted(_body(0.8))),
                _predicted_frame(
                    "b.png", *(_predicted(_body(dx)) for dx in (0.0, 100.0, 400.0))
                ),
            )
        ],
    ).metrics
    merged = _score(
        [
            _truth_frame(
                "c.png",
                *(_truth_object(_body(dx)) for dx in (1000.0, 0.0, 100.0, 200.0)),
            )
        ],
        [
            _run(
                _predicted_frame(
                    "c.png",
                    *(_predicted(_body(dx)) for dx in (1000.8, 0.0, 100.0, 400.0)),
                )
            )
        ],
    ).metrics

    # Frame by frame, recall and precision are 1 and 2/3; pooled, 3 of 4.
    assert (split.recall, split.precision) == (0.75, 0.75)
    # The tight box is 10 x 4 px, and a.png misses each keypoint by 0.8 px.
    assert split.moks == pytest.approx((math.exp(-3.2) + 2.0) / 3.0)
    # a.png fails PCK@0.05 at its 4 keypoints, b.png passes at its 8.
    assert split.pck["0.05"] == pytest.approx(2 / 3)
    assert (split.recall, split.precision, split.swap_rate) == (
        merged.recall,
        merged.precision,
        merged.swap_rate,
    )
    assert split.moks == pytest.approx(merged.moks)
    assert split.pck == pytest.approx(merged.pck)
    assert (split.n_gt, split.n_pred, split.n_matched, split.n_fp) == (4, 4, 3, 1)


def test_a_rate_with_a_zero_denominator_is_null() -> None:
    negative = _score([_truth_frame("a.png")], [_run(_predicted_frame("a.png"))])
    uncovered = _score([_truth_frame("a.png", _truth_object(_body()))], [])

    metrics = negative.metrics
    assert [
        metrics.recall,
        metrics.precision,
        metrics.frac_frames_all_found,
        metrics.moks,
        metrics.swap_rate,
    ] == [None] * 5
    assert metrics.pck == {"0.05": None, "0.1": None, "0.2": None}
    assert metrics.fp_per_frame == 0.0
    assert metrics.error_px == ErrorQuantiles(median=None, p90=None)
    assert (uncovered.metrics.fp_per_frame, uncovered.metrics.recall) == (None, None)


def test_fp_per_frame_counts_the_negative_frames_in_its_denominator() -> None:
    truth = [
        _truth_frame("a.png", _truth_object(_body())),
        _truth_frame("b.png"),
        _truth_frame("c.png"),
    ]
    predictions = [
        _run(
            _predicted_frame("a.png", _predicted(_body()), _predicted(_body(100.0))),
            _predicted_frame("b.png", _predicted(_body())),
            _predicted_frame("c.png"),
        )
    ]

    metrics = _score(truth, predictions).metrics

    assert (metrics.n_scored, metrics.n_fp) == (3, 2)
    assert metrics.fp_per_frame == pytest.approx(2 / 3)


def test_frac_frames_all_found_is_over_the_scored_frames_with_truth() -> None:
    truth = [
        _truth_frame("a.png", _truth_object(_body())),
        _truth_frame("b.png", _truth_object(_body())),
        _truth_frame("c.png"),
        _truth_frame("d.png", _truth_object(_body())),
    ]
    predictions = [
        _run(
            _predicted_frame("a.png", _predicted(_body())),
            _predicted_frame("b.png"),
            _predicted_frame("c.png"),
        )
    ]

    metrics = _score(truth, predictions).metrics

    # a.png is all found and b.png is not; c.png has no truth, d.png is not covered.
    assert metrics.frac_frames_all_found == 0.5


def test_the_summary_counts_truth_frames_by_status() -> None:
    truth = [
        _truth_frame(image, _truth_object(_body()))
        for image in ("a.png", "b.png", "c.png")
    ]
    predictions = [
        _run(
            _predicted_frame("a.png", _predicted(_body()), frame_index=10),
            _predicted_frame("b.png", _predicted(_body()), frame_index=20),
        )
    ]

    metrics = _score(truth, predictions, seen=frozenset({_key(10)})).metrics

    assert (
        metrics.n_truth_frames,
        metrics.n_scored,
        metrics.n_excluded_training,
        metrics.n_not_covered,
    ) == (3, 1, 1, 1)
    assert metrics.coverage == pytest.approx(2 / 3)
    assert (metrics.n_gt, metrics.n_pred, metrics.n_matched) == (1, 1, 1)


def test_per_keypoint_rates_and_errors_by_hand() -> None:
    predicted = _predicted(((0.4, 0.0), (5.0, -2.0), (5.0, 2.8), None))

    metrics = _score_one([_truth_object(_body())], [predicted]).metrics

    # Errors 0.4, 0 and 0.8 px at L = 10; the tail is a miss.
    assert metrics.pck_per_keypoint == {
        "head": {"0.05": 1.0, "0.1": 1.0, "0.2": 1.0},
        "left": {"0.05": 1.0, "0.1": 1.0, "0.2": 1.0},
        "right": {"0.05": 0.0, "0.1": 1.0, "0.2": 1.0},
        "tail": {"0.05": 0.0, "0.1": 0.0, "0.2": 0.0},
    }
    assert metrics.miss_share_per_keypoint == {
        "head": 0.0,
        "left": 0.0,
        "right": 0.0,
        "tail": 1.0,
    }
    right = metrics.error_px_per_keypoint["right"]
    assert (right.median, right.p90) == pytest.approx((0.8, 0.8))
    assert metrics.error_px_per_keypoint["tail"] == ErrorQuantiles(
        median=None, p90=None
    )
    assert (metrics.error_px.median, metrics.error_px.p90) == pytest.approx((0.4, 0.72))
    assert (metrics.error_norm.median, metrics.error_norm.p90) == pytest.approx(
        (0.04, 0.072)
    )


def test_an_unnormalized_instance_is_left_out_of_pck_and_normalized_error() -> None:
    truth = _truth_object(((0.0, 0.0), None, None, None), box=Bbox(0.0, 0.0, 0.0, 0.0))
    predicted = _predicted(((1.0, 0.0), None, None, None))

    metrics = _score_one(
        [truth], [predicted], params=replace(PARAMS, match_radius_px=5.0)
    ).metrics

    assert (metrics.n_kpt_eligible, metrics.n_kpt_unnormalized) == (1, 1)
    assert metrics.pck == {"0.05": None, "0.1": None, "0.2": None}
    assert metrics.pck_per_keypoint["head"] == {"0.05": None, "0.1": None, "0.2": None}
    assert metrics.error_px == ErrorQuantiles(median=1.0, p90=1.0)
    assert metrics.error_norm == ErrorQuantiles(median=None, p90=None)


def test_the_summary_names_the_target_keypoints_and_the_map() -> None:
    names = ("right", "head", "left", "spare")
    head, left, right, _ = _body()

    mapped = _score_one(
        [_truth_object(_body())],
        [_predicted((right, head, left, None))],
        keypoint_map=resolve_keypoint_map(SCHEMA, names),
        names=names,
    ).metrics
    identity = _score_one([_truth_object(_body())], []).metrics

    assert mapped.keypoints == ["head", "left", "right", "tail"]
    assert mapped.keypoint_map == KeypointMapSummary(
        identity=False, coverage=["head", "left", "right"], unmapped_sources=["spare"]
    )
    assert mapped.miss_share_per_keypoint["tail"] is None
    assert identity.keypoint_map == KeypointMapSummary(
        identity=True, coverage=list(SCHEMA.names), unmapped_sources=[]
    )


def test_a_model_object_without_untouched_flags_has_an_unknown_source() -> None:
    truth = [
        _truth_frame(
            "a.png",
            _truth_object(_body(), origin="model"),
            _truth_object(_body(100.0), origin="model"),
            _truth_object(_body(200.0), origin="human"),
            _truth_object(_body(300.0)),
        ),
        _truth_frame("b.png", _truth_object(_body(), origin="model")),
    ]
    untouched = {(ORIGIN, "a.png", 0): (True, False, False, False)}

    metrics = _score(
        truth,
        [_run(_predicted_frame("a.png", _predicted(_body())))],
        untouched=untouched,
    ).metrics

    # Object 1 of a.png. b.png is not covered, so its object is counted nowhere.
    assert metrics.n_objects_source_unknown == 1
    assert (metrics.n_kpt_eligible, metrics.n_kpt_untouched) == (3, 1)


def test_the_score_summary_is_the_metrics_with_their_provenance() -> None:
    metrics = _two_sequences()
    provenance: dict[str, object] = {
        "prediction_runs": ["predict-frames.0.1-0123456789"],
        "model_kind": "yolo-pose",
        "model_id": "train-yolo-pose.0.1-abcdef0123",
        "model": ["train-yolo-pose.0.1-abcdef0123"],
        "labels": [
            {
                "origin_uuid": ORIGIN,
                "set_key": "set-a",
                "revision": 2,
                "digest": "0" * 64,
            }
        ],
        "training_frames_known": False,
        "training_frames_reason": "exclude_seen is false",
    }

    summary = PoseScoreSummary.model_validate({**metrics.model_dump(), **provenance})

    assert summary.labels == [
        LabelRevision(origin_uuid=ORIGIN, set_key="set-a", revision=2, digest="0" * 64)
    ]
    kept = summary.model_dump(include=set(PoseMetrics.model_fields))
    assert PoseMetrics.model_validate(kept) == metrics
    assert PoseScoreSummary.model_validate_json(summary.model_dump_json()) == summary
    with pytest.raises(ValidationError, match="unknown"):
        PoseScoreSummary.model_validate({**summary.model_dump(), "unknown": 1})


# --- intervals ----------------------------------------------------------------------


def test_the_interval_keys_are_the_bootstrapped_rates() -> None:
    metrics = _two_sequences()

    assert metrics.intervals is not None
    assert set(metrics.intervals) == {
        "recall",
        "precision",
        "moks",
        "frac_frames_all_found",
        "pck:0.05",
        "pck:0.1",
        "pck:0.2",
    }
    assert metrics.intervals["recall"] == {"estimate": 1.0, "lower": 1.0, "upper": 1.0}
    assert (metrics.n_sequences, metrics.interval_reason) == (2, "")


def test_the_pck_keys_are_each_alpha_as_its_repr() -> None:
    metrics = _two_sequences(params=replace(PARAMS, pck_alphas=(0.05, 1.0)))

    assert list(metrics.pck) == ["0.05", "1.0"]
    assert list(metrics.pck_per_keypoint["head"]) == ["0.05", "1.0"]
    assert metrics.intervals is not None
    assert {key for key in metrics.intervals if key.startswith("pck:")} == {
        "pck:0.05",
        "pck:1.0",
    }


def test_a_rate_with_a_null_estimate_has_no_interval() -> None:
    metrics = _two_sequences(predicted=(False, False))

    assert (metrics.recall, metrics.precision, metrics.moks) == (0.0, None, None)
    assert metrics.intervals is not None
    assert set(metrics.intervals) == {"recall", "frac_frames_all_found"}


def test_fewer_than_two_scored_sequences_have_no_intervals() -> None:
    truth = [
        _truth_frame(image, _truth_object(_body()))
        for image in ("a.png", "b.png", "c.png")
    ]
    predictions = [
        _run(_predicted_frame("a.png", _predicted(_body())), camera="left"),
        _run(_predicted_frame("b.png"), camera="right"),
    ]

    one = _score(truth, predictions).metrics
    none = _score(truth, []).metrics

    # Two cameras of one sequence are one unit, and c.png, not covered, is in none.
    assert (one.n_sequences, one.intervals) == (1, None)
    assert "1 scored sequence" in one.interval_reason
    assert (none.n_sequences, none.intervals) == (0, None)
    assert "0 scored sequences" in none.interval_reason
    assert (one.n_boot, one.confidence, one.random_state) == (1000, 0.95, 0)


def test_the_same_random_state_pairs_two_runs_replicate_by_replicate() -> None:
    """Both runs read their intervals off one draw of the same three sequences."""
    params = replace(PARAMS, n_boot=200, random_state=7)
    n_gt = (1, 2, 3)
    # Over the identity matrix, each replicate's sums are its draw count per sequence.
    weights = resampled_sums(np.eye(3), n_boot=200, random_state=7)

    for found in ((1, 1, 0), (0, 2, 1)):
        truth: list[TruthFrame] = []
        predictions: list[FramePredictions] = []
        for index, (count, hits) in enumerate(zip(n_gt, found, strict=True)):
            image = f"{index}.png"
            bodies = [_body(100.0 * position) for position in range(count)]
            truth.append(_truth_frame(image, *(_truth_object(body) for body in bodies)))
            predictions.append(
                _run(
                    _predicted_frame(
                        image, *(_predicted(body) for body in bodies[:hits])
                    ),
                    sequence=f"s{index}",
                )
            )

        metrics = _score(truth, predictions, params=params).metrics

        replicates = (weights @ np.array(found, dtype=np.float64)) / (
            weights @ np.array(n_gt, dtype=np.float64)
        )
        expected = percentile_interval(sum(found) / 6, replicates, 0.95)
        assert metrics.intervals is not None
        assert metrics.intervals["recall"] == pytest.approx(expected)


def test_a_replicate_with_a_zero_denominator_is_left_out_of_its_interval() -> None:
    """A replicate that draws ``s1`` alone keeps no prediction and has no precision."""
    params = replace(PARAMS, n_boot=200, random_state=7)
    truth = [_truth_frame(f"{index}.png", _truth_object(_body())) for index in range(3)]
    # s0 keeps a match and a false positive, s1 no prediction and s2 a match.
    predictions = [
        _run(
            _predicted_frame("0.png", _predicted(_body()), _predicted(_body(300.0))),
            sequence="s0",
        ),
        _run(_predicted_frame("1.png"), sequence="s1"),
        _run(_predicted_frame("2.png", _predicted(_body())), sequence="s2"),
    ]

    metrics = _score(truth, predictions, params=params).metrics

    # Over the identity matrix, each replicate's sums are its draw count per sequence.
    weights = resampled_sums(np.eye(3), n_boot=200, random_state=7)
    matched = weights @ np.array([1.0, 0.0, 1.0])
    kept = weights @ np.array([2.0, 0.0, 1.0])
    defined = kept > 0
    assert not defined.all()
    expected = percentile_interval(2 / 3, matched[defined] / kept[defined], 0.95)
    assert metrics.intervals is not None
    assert metrics.intervals["precision"] == pytest.approx(expected)


def test_a_rate_without_a_replicate_of_a_positive_denominator_has_no_interval() -> None:
    """One replicate draws ``s2`` twice, so only the rates over truth are defined."""
    seed = next(
        seed
        for seed in range(100)
        if resampled_sums(np.eye(2), n_boot=1, random_state=seed)[0, 0] == 0
    )

    metrics = _two_sequences(
        predicted=(True, False), params=replace(PARAMS, n_boot=1, random_state=seed)
    )

    assert (metrics.precision, metrics.moks, metrics.pck["0.05"]) == (1.0, 1.0, 1.0)
    assert metrics.intervals is not None
    assert set(metrics.intervals) == {"recall", "frac_frames_all_found"}


# --- refusals -----------------------------------------------------------------------


def test_one_oks_sigma_per_target_keypoint_is_required() -> None:
    with pytest.raises(ValueError, match="oks_sigmas"):
        _score_one(
            [_truth_object(_body())],
            [],
            params=replace(PARAMS, oks_sigmas=(0.025,)),
        )


def test_an_oks_sigma_that_is_not_positive_is_refused() -> None:
    with pytest.raises(ValueError, match="sigma"):
        _score_one(
            [_truth_object(_body())],
            [],
            params=replace(PARAMS, oks_sigmas=(0.025, 0.0, 0.025, 0.025)),
        )


def test_a_keypoint_position_outside_the_target_pose_is_refused() -> None:
    with pytest.raises(ValueError, match="center position 4"):
        _score_one([_truth_object(_body())], [], params=replace(PARAMS, center=4))


def test_two_pck_alphas_of_one_column_name_are_refused() -> None:
    with pytest.raises(ValueError, match="repeat a column name"):
        _score_one(
            [_truth_object(_body())],
            [],
            params=replace(PARAMS, pck_alphas=(0.1, 0.10001)),
        )


@pytest.mark.parametrize(
    ("params", "message"),
    [
        (replace(PARAMS, n_boot=0), "n_boot"),
        (replace(PARAMS, confidence=1.0), "confidence"),
    ],
)
def test_bootstrap_settings_outside_their_range_are_refused(
    params: PoseScoreParams, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _score_one([_truth_object(_body())], [], params=params)


def test_a_truth_object_of_another_keypoint_count_is_refused() -> None:
    with pytest.raises(ValueError, match="has 3 keypoints, and the target pose has 4"):
        _score_one([_truth_object(_body()[:3])], [])


def test_a_truth_object_without_a_box_is_refused() -> None:
    truth = replace(_truth_object(_body()), bbox=None)

    with pytest.raises(ValueError, match="has no box"):
        _score_one([truth], [])


def test_an_untouched_entry_of_another_flag_count_is_refused() -> None:
    untouched = {(ORIGIN, IMAGE, 0): (True, False)}

    with pytest.raises(ValueError, match="2 flags for 4 target keypoints"):
        _score_one([_truth_object(_body())], [], untouched=untouched)


def test_an_image_key_in_two_truth_frames_is_refused() -> None:
    truth = [_truth_frame("a.png"), _truth_frame("a.png")]

    with pytest.raises(ValueError, match=r"a\.png"):
        _score(truth, [])


def test_an_image_in_two_prediction_runs_is_refused() -> None:
    predictions = [_run(_predicted_frame("a.png")), _run(_predicted_frame("a.png"))]

    with pytest.raises(ValueError, match=r"a\.png"):
        _score([_truth_frame("a.png")], predictions)


# --- untouched_keypoints ------------------------------------------------------------


def _points(*points: Point) -> tuple[Keypoint, ...]:
    return tuple(
        Keypoint.absent()
        if point is None
        else Keypoint(x=point[0], y=point[1], visibility=2)
        for point in points
    )


def test_positional_untouched_compares_each_label_keypoint_with_its_source() -> None:
    label = _points((1.0, 1.0), (2.0, 2.0), None, (4.0, 4.0))
    source = _points((1.005, 1.0), (2.5, 2.0), (3.0, 3.0), None)

    assert untouched_keypoints(label, source) == (True, False, False, False)
    assert untouched_keypoints(label, source, tol_px=0.6) == (True, True, False, False)


def test_positional_untouched_needs_one_source_keypoint_per_label_keypoint() -> None:
    with pytest.raises(ValueError, match="positional"):
        untouched_keypoints(_points((1.0, 1.0), (2.0, 2.0)), _points((1.0, 1.0)))


def test_map_free_untouched_matches_any_placed_source_keypoint() -> None:
    label = _points((1.0, 1.0), (2.0, 2.0), (3.0, 3.0), None)
    source = (
        Keypoint(x=2.0, y=2.005, visibility=2),
        Keypoint(x=1.0, y=1.0, visibility=2),
        Keypoint(x=3.0, y=3.0, visibility=0),
    )

    assert untouched_keypoints(label, source, positional=False) == (
        True,
        True,
        False,
        False,
    )
