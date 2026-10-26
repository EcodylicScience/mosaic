"""Scores a pose model's predictions against annotated frames.

A truth frame is a usable frame of a set narrowed to one pose. Its image key is the
manifest uuid of the dataset that owns the image and the image's dataset-relative
path. A prediction frame with the same image key covers it. Predictions are mapped
onto the truth's pose first, and only target keypoints inside the map's coverage are
scored. A target keypoint outside the coverage is neither a hit nor a miss.

Per frame, truth and predicted instances are matched by their centers
(:func:`match_instances`), each truth instance within its own radius. Each frame's
counts are written to one row of the frames table, and each truth instance and each
unmatched prediction to one row of the instances table. A rate pooled over frames is
therefore a ratio of summed counts. Its interval resamples whole sequences, because
the frames of one sequence are not independent.

Keypoint counts, PCK, OKS, errors and swaps are taken on matched pairs, at the
eligible truth keypoints: visibility 2, inside the map's coverage, and not
untouched. An untouched keypoint lies where a model placed it
(:func:`untouched_keypoints`). It still counts for detection, because a truth
center uses every placed keypoint.

pandas is imported at module scope and scipy inside the function that uses it. The
module reads no dataset.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Final, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from mosaic.core.annotations.keypoint_map import KeypointMap
from mosaic.core.annotations.model import (
    AnnotationFrame,
    AnnotationObject,
    Bbox,
    Keypoint,
)
from mosaic.core.annotations.pose_predictions import (
    FramePredictions,
    PredictedFrame,
    finite_or_none,
    keypoint_accepted,
    score_passes,
)
from mosaic.core.annotations.split import VideoFrame, frame_key
from mosaic.core.resample import (
    Interval,
    check_bootstrap_settings,
    percentile_interval,
    resampled_sums,
)
from mosaic.core.strict_model import StrictModel
from mosaic.core.track_quality import instance_body_lengths
from mosaic.core.wording import counted

__all__ = [
    "DEFAULT_PCK_ALPHAS",
    "ErrorQuantiles",
    "FrameStatus",
    "KeypointMapSummary",
    "LabelRevision",
    "PoseMetrics",
    "PoseScoreParams",
    "PoseScoreSummary",
    "PoseScores",
    "TruthFrame",
    "UntouchedKey",
    "empty_frame_table",
    "empty_instance_table",
    "frame_table_columns",
    "match_instances",
    "score_predictions",
    "untouched_keypoints",
]

DEFAULT_PCK_ALPHAS: Final = (0.05, 0.1, 0.2)
"""The PCK thresholds of a score that names none."""

FrameStatus = Literal["scored", "excluded_training", "not_covered"]
"""How a truth frame enters a score.

``not_covered``: no prediction frame has its image key. ``excluded_training``: the
frame key of its prediction frame is among the frames the model trained or
validated on. ``scored``: otherwise. Counts are 0 on a frame that is not scored.
"""

UntouchedKey = tuple[str, str, int]
"""``(origin_uuid, image, object position)`` of a truth object."""

_AREA_EPSILON: Final = float(np.spacing(1.0))
"""Added to a truth box's area, as COCO adds it, so a box without area gives an OKS
term of 0 instead of a division by zero."""

_FloatArray = npt.NDArray[np.float64]
_BoolArray = npt.NDArray[np.bool_]
_Cell = str | int | float | bool | None
_Record = dict[str, _Cell]
_PlacedRule = Callable[[Keypoint], bool]


@dataclass(frozen=True, slots=True)
class TruthFrame:
    """Records one annotated frame and the dataset that owns its image.

    Attributes:
        origin_uuid: The manifest uuid of the dataset whose frames index lists the
            image.
        frame: A usable frame of a set narrowed to the target pose. Its
            ``image_path`` is relative to that dataset's root, and every object has
            a box, drawn or derived. A frame without objects is a negative.
    """

    origin_uuid: str
    frame: AnnotationFrame

    @property
    def image(self) -> str:
        """The image's dataset-relative POSIX path, the second half of its key."""
        return self.frame.image_path.as_posix()


@dataclass(frozen=True, slots=True)
class PoseScoreParams:
    """Declares the metric settings of a score, resolved against the target pose.

    Attributes:
        center: The target position whose keypoint is an instance's center when
            placed, or ``None`` for the mean of the placed keypoints.
        head: The target position of the head, or ``None``.
        tail: The target position of the tail, or ``None``.
        match_radius: The matching radius, in normalizers of each truth instance.
        match_radius_px: A fixed matching radius in pixels. When set, it replaces
            *match_radius*.
        duplicate_radius: The distance from a matched prediction's center, in
            normalizers of that match's truth instance, below which an unmatched
            prediction is a duplicate.
        oks_sigmas: One OKS sigma per target keypoint, in target order.
        pck_alphas: The PCK thresholds, in normalizers.
        min_instance_score: The lowest instance score kept, or ``None`` to keep
            every prediction. A prediction without a score is kept.
        min_keypoint_score: The lowest keypoint score that counts as placed, or
            ``None``. A keypoint without a score counts as placed.
        n_boot: The number of bootstrap replicates of the pooled rates.
        confidence: The confidence of the bootstrap intervals.
        random_state: The seed of the bootstrap.
    """

    center: int | None
    head: int | None
    tail: int | None
    match_radius: float
    match_radius_px: float | None
    duplicate_radius: float
    oks_sigmas: tuple[float, ...]
    pck_alphas: tuple[float, ...]
    min_instance_score: float | None
    min_keypoint_score: float | None
    n_boot: int
    confidence: float
    random_state: int


class KeypointMapSummary(StrictModel):
    """Records how a score maps the model's keypoints onto the target pose.

    Attributes:
        identity: Whether the model's keypoints are the target's, each at its
            own position.
        coverage: The target keypoints that a model keypoint fills, in target
            order. Only these are scored.
        unmapped_sources: The model's keypoints that fill no target keypoint, in
            the model's order.
    """

    identity: bool
    coverage: list[str]
    unmapped_sources: list[str]


class ErrorQuantiles(StrictModel):
    """Records the median and the 90th percentile of keypoint errors.

    Attributes:
        median: The median, or ``None`` without a found keypoint.
        p90: The 90th percentile, or ``None`` without a found keypoint.
    """

    median: float | None
    p90: float | None


class LabelRevision(StrictModel):
    """Names a revision of an annotated set that a score is taken against.

    Attributes:
        origin_uuid: The manifest uuid of the dataset that owns the set.
        set_key: The set's key.
        revision: The revision's number.
        digest: The digest of the revision's content.
    """

    origin_uuid: str
    set_key: str
    revision: int
    digest: str


class PoseMetrics(StrictModel):
    """Records a score's metrics, pooled over its scored frames.

    A pooled rate is a ratio of the scored frames' summed counts, ``None`` when
    its denominator is 0, so it equals the rate of one frame that contains every
    scored frame's instances. Keypoint counts, PCK and errors are over matched
    pairs at eligible keypoints. A PCK value is keyed by its alpha as
    ``repr(alpha)``, such as ``"0.05"``.

    Attributes:
        keypoints: The target pose's keypoint names, by position.
        keypoint_map: How the model's keypoints map onto the target pose.
        n_truth_frames: The truth frames.
        n_scored: The truth frames scored.
        n_excluded_training: The truth frames excluded as seen in training.
        n_not_covered: The truth frames whose image key no prediction frame has.
        coverage: ``(n_scored + n_excluded_training) / n_truth_frames``, 0
            without a truth frame.
        n_gt: The truth instances of the scored frames.
        n_pred: The kept predictions of the scored frames.
        n_matched: The matched pairs.
        n_fp: The unmatched predictions.
        n_fp_duplicate: The unmatched predictions near a matched prediction.
        recall: ``n_matched / n_gt``.
        precision: ``n_matched / n_pred``.
        fp_per_frame: ``n_fp / n_scored``, frames without truth included.
        frac_frames_all_found: The share of the scored frames with truth whose
            every truth instance is matched.
        moks: The mean OKS of the matched pairs that have one.
        swap_rate: The swapped keypoint pairs over the pairs checked for a swap.
        pck: By alpha, the share of the eligible keypoints of instances with a
            positive normalizer that are found within *alpha* normalizers.
        pck_per_keypoint: ``pck`` by target keypoint name.
        miss_share_per_keypoint: By target keypoint name, the share of its
            eligible keypoints that are missed.
        error_px_per_keypoint: By target keypoint name, the pixel error of its
            found keypoints.
        error_px: The pixel error of every found keypoint.
        error_norm: The error of the found keypoints in normalizers, over
            instances with a positive normalizer.
        n_kpt_eligible: The eligible keypoints.
        n_kpt_untouched: The keypoints left out of the pose metrics as untouched.
        n_kpt_unnormalized: The eligible keypoints of instances whose normalizer
            is 0.
        n_objects_source_unknown: The model-placed truth objects of the scored
            frames without untouched flags, scored as touched everywhere.
        intervals: The percentile intervals of ``recall``, ``precision``,
            ``moks``, ``frac_frames_all_found`` and ``pck:<alpha>``, from a
            bootstrap over the scored sequences. A rate whose estimate is
            ``None``, or whose every replicate has a denominator of 0, has no
            key. ``None`` with fewer than 2 scored sequences.
        interval_reason: Why ``intervals`` is ``None``, ``""`` otherwise.
        n_sequences: The scored sequences ``(origin_uuid, group, sequence)``,
            the units that the bootstrap draws.
        n_boot: The bootstrap's replicates.
        confidence: The confidence of the intervals.
        random_state: The bootstrap's seed. The same seed over the same
            sequences draws the same replicates, so two scores pair replicate by
            replicate.
    """

    keypoints: list[str]
    keypoint_map: KeypointMapSummary
    n_truth_frames: int
    n_scored: int
    n_excluded_training: int
    n_not_covered: int
    coverage: float
    n_gt: int
    n_pred: int
    n_matched: int
    n_fp: int
    n_fp_duplicate: int
    recall: float | None
    precision: float | None
    fp_per_frame: float | None
    frac_frames_all_found: float | None
    moks: float | None
    swap_rate: float | None
    pck: dict[str, float | None]
    pck_per_keypoint: dict[str, dict[str, float | None]]
    miss_share_per_keypoint: dict[str, float | None]
    error_px_per_keypoint: dict[str, ErrorQuantiles]
    error_px: ErrorQuantiles
    error_norm: ErrorQuantiles
    n_kpt_eligible: int
    n_kpt_untouched: int
    n_kpt_unnormalized: int
    n_objects_source_unknown: int
    intervals: dict[str, Interval] | None
    interval_reason: str
    n_sequences: int
    n_boot: int
    confidence: float
    random_state: int


class PoseScoreSummary(PoseMetrics):
    """Declares a score run's ``summary.json``: its metrics and their provenance.

    A truth object accepted from one of the scored runs is compared with its
    source prediction through the score's keypoint map, so a keypoint that the
    annotator moved onto its partner's predicted position counts as edited. An
    object accepted from any other run is compared without a map, because the
    map applied at accept is not recorded. That comparison reads a label
    keypoint placed exactly on another keypoint of its source as untouched,
    which only a swap produces.

    Attributes:
        prediction_runs: The scored predict-frames runs, sorted.
        model_kind: The training kind of their model.
        model_id: The id of their model.
        model: Their model references.
        labels: The revisions scored against.
        training_frames_known: Whether the frames that the model trained or
            validated on are known and excluded.
        training_frames_reason: Why they are not, ``""`` when they are.
    """

    prediction_runs: list[str]
    model_kind: str
    model_id: str
    model: list[str]
    labels: list[LabelRevision]
    training_frames_known: bool
    training_frames_reason: str


@dataclass(frozen=True, slots=True)
class PoseScores:
    """Records a score's counts, per frame and per instance, and its metrics.

    Attributes:
        frames: One row per truth frame, sorted by ``(origin_uuid, image)``.
        instances: One row per truth instance and per unmatched prediction of the
            scored frames, sorted by ``(origin_uuid, image, gt_index, pred_index)``
            with nulls last.
        metrics: The metrics pooled over the scored frames.
    """

    frames: pd.DataFrame
    instances: pd.DataFrame
    metrics: PoseMetrics


# --- Matching ----------------------------------------------------------------------


def match_instances(
    gt_xy: np.ndarray, pred_xy: np.ndarray, radius: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the truth and prediction positions of one frame's matched pairs.

    The assignment first maximizes the number of pairs whose center distance is
    below the truth instance's radius, then minimizes their total distance. A pair
    at or beyond the radius, or with a center that is not finite, costs more than
    every set of in-radius pairs together, and is dropped when assigned. Greedy
    nearest-first matching can take a close pair that leaves a third instance
    without a partner.

    Args:
        gt_xy: Truth centers, shape ``(n, 2)``.
        pred_xy: Predicted centers, shape ``(m, 2)``.
        radius: One radius per truth instance, shape ``(n,)``, each finite and 0
            or more.

    Returns:
        The truth positions and the prediction positions of the matched pairs, as
        two integer arrays of one length, ascending by truth position.

    Raises:
        ValueError: *radius* differs in length from *gt_xy*, or a radius is
            negative or not finite.
    """
    from scipy.optimize import linear_sum_assignment

    gt = np.asarray(gt_xy, dtype=np.float64).reshape(-1, 2)
    pred = np.asarray(pred_xy, dtype=np.float64).reshape(-1, 2)
    radii = np.asarray(radius, dtype=np.float64).reshape(-1)
    if radii.shape[0] != gt.shape[0]:
        msg = f"{radii.shape[0]} radii were given for {gt.shape[0]} truth instances"
        raise ValueError(msg)
    if not bool(np.all(np.isfinite(radii) & (radii >= 0.0))):
        msg = f"every matching radius must be finite and 0 or more: {radii.tolist()}"
        raise ValueError(msg)
    if gt.shape[0] == 0 or pred.shape[0] == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    distance = _center_distances(gt, pred)
    within = np.isfinite(distance) & (distance < radii[:, None])
    over = float(radii.max()) * (min(distance.shape) + 1)
    rows, columns = linear_sum_assignment(np.where(within, distance, over))
    keep = within[rows, columns]
    return rows[keep].astype(np.int64), columns[keep].astype(np.int64)


def _center_distances(gt: _FloatArray, pred: _FloatArray) -> _FloatArray:
    """Return the ``(n, m)`` distances between truth and predicted centers."""
    return np.hypot(
        gt[:, None, 0] - pred[None, :, 0], gt[:, None, 1] - pred[None, :, 1]
    )


# --- Untouched keypoints -------------------------------------------------------------


def untouched_keypoints(
    label: Sequence[Keypoint],
    source: Sequence[Keypoint],
    *,
    tol_px: float = 0.01,
    positional: bool = True,
) -> tuple[bool, ...]:
    """Return one flag per label keypoint, true where it lies on a source keypoint.

    A label keypoint is untouched when it is placed and a placed source keypoint
    lies within *tol_px* of it in x and in y. Whether a keypoint is untouched is
    derived by this comparison and is not stored.

    Args:
        label: An annotated object's keypoints, in its pose's order.
        source: The predicted object that the annotated object was accepted from.
            With *positional*, its keypoints are mapped onto the label's pose
            order, and each label keypoint is compared with the source keypoint
            at its position. Without it, they are in the model's order, and each
            label keypoint is compared with every source keypoint, so no keypoint
            map is needed. A label keypoint placed on another keypoint's source
            position then reads as untouched, which only a swap produces.
        tol_px: The largest difference in x and in y, in pixels.
        positional: Whether *source* is aligned to *label* position by position.

    Raises:
        ValueError: With *positional*, *source* differs in length from *label*.
    """
    if positional:
        if len(source) != len(label):
            msg = (
                f"a positional comparison needs one source keypoint per label "
                f"keypoint, and {len(source)} were given for {len(label)}"
            )
            raise ValueError(msg)
        return tuple(
            point.is_placed and other.is_placed and _near(point, other, tol_px)
            for point, other in zip(label, source, strict=True)
        )
    placed = [other for other in source if other.is_placed]
    return tuple(
        point.is_placed and any(_near(point, other, tol_px) for other in placed)
        for point in label
    )


def _near(point: Keypoint, other: Keypoint, tol_px: float) -> bool:
    return abs(point.x - other.x) <= tol_px and abs(point.y - other.y) <= tol_px


# --- Scoring -----------------------------------------------------------------------


def score_predictions(
    truth: Sequence[TruthFrame],
    predictions: Sequence[FramePredictions],
    *,
    predictions_origin: str,
    keypoint_map: KeypointMap,
    seen: frozenset[str] | None,
    untouched: Mapping[UntouchedKey, tuple[bool, ...]],
    params: PoseScoreParams,
) -> PoseScores:
    """Score *predictions* against *truth*, frame by frame and instance by instance.

    A truth center is the ``center`` keypoint when placed, else the mean of every
    placed keypoint, untouched ones included. A predicted center is taken likewise
    over the mapped keypoints that pass ``min_keypoint_score``. A truth instance's
    normalizer ``L`` is its head-to-tail distance when both are placed, else the
    diagonal of the tight box around its placed keypoints. When that is 0 or
    undefined, ``L`` is the diagonal of the object's box. The instance's matching
    radius is ``match_radius_px`` when set, else ``match_radius * L``.

    On each matched pair, an eligible keypoint is found when its mapped prediction
    is placed and passes ``min_keypoint_score``, and missed otherwise. Keypoint
    counts are over matched pairs, so a missed truth instance counts for detection
    only. An instance with ``L = 0`` adds its eligible keypoints to
    ``n_kpt_unnormalized`` and none to a PCK count. OKS follows COCO, with the
    truth box's area and ``kappa = 2 * sigma``, a miss contributing 0. A pair
    without an eligible keypoint has no OKS. A swap pair is each mirror pair of the
    pose, and the head and tail when both are set, with both keypoints found. It
    is swapped when the crossed distances sum below the straight ones.

    The metrics pool the scored frames' counts. Their intervals come from
    :func:`~mosaic.core.resample.resampled_sums`, seeded with ``random_state``,
    over one row of summed counts per scored sequence ``(origin_uuid, group,
    sequence)``, the sequences in sorted order. A replicate whose denominator is
    0 is left out of its rate's interval. A model-placed truth object of a scored
    frame without an *untouched* entry counts in ``n_objects_source_unknown``.

    Args:
        truth: Usable frames of sets narrowed to the target pose. Each image key
            appears in one frame.
        predictions: The prediction frames of every run scored together, with
            each image in one of them.
        predictions_origin: The manifest uuid of the dataset that contains the
            prediction runs, the origin of every prediction frame's image key.
        keypoint_map: Maps the model's keypoints onto the target pose, which is
            the pose of *truth*.
        seen: The frame keys excluded as seen in training, or ``None`` to exclude
            none.
        untouched: One flag per target keypoint of a truth object, from
            :func:`untouched_keypoints`. An object without an entry is touched
            everywhere.
        params: The metric settings.

    Returns:
        The frames and instances tables and the metrics.

    Raises:
        ValueError: An image key is in two truth frames, or an image is in two
            prediction frames. *params* names a keypoint position outside the
            target pose, gives other than one positive OKS sigma per target
            keypoint, gives two PCK alphas of one column name, gives ``n_boot``
            below 1, or gives a ``confidence`` outside (0, 1). A truth object
            has a keypoint count other than the pose's, or has no box. An
            untouched entry has other than one flag per target keypoint.
    """
    layout = _Layout.of(keypoint_map, params)
    located = _locate(predictions, predictions_origin)
    frames: list[_Record] = []
    instances: list[_Record] = []
    scored_frames: list[tuple[TruthFrame, _FrameScore]] = []
    for item in _sorted_truth(truth):
        found = (
            located.get(item.image) if item.origin_uuid == predictions_origin else None
        )
        if found is None:
            frames.append(_frame_record(item, None, "not_covered", None, layout))
        elif seen is not None and found.key in seen:
            frames.append(_frame_record(item, found, "excluded_training", None, layout))
        else:
            scored = _score_frame(item, found.frame, untouched, layout)
            frames.append(_frame_record(item, found, "scored", scored, layout))
            instances.extend(_instance_records(item, found.key, scored, layout))
            scored_frames.append((item, scored))
    frames_table = _table(frames, layout.frame_columns())
    return PoseScores(
        frames=frames_table,
        instances=_table(instances, layout.instance_columns()),
        metrics=_metrics(frames_table, scored_frames, untouched, layout),
    )


@dataclass(frozen=True, slots=True)
class _Layout:
    """Records the target pose and the params, checked against each other once."""

    keypoint_map: KeypointMap
    params: PoseScoreParams
    covered: _BoolArray
    sigmas: _FloatArray
    alphas: _FloatArray
    swap_pairs: tuple[tuple[int, int], ...]
    pck_columns: tuple[str, ...]

    @classmethod
    def of(cls, keypoint_map: KeypointMap, params: PoseScoreParams) -> _Layout:
        target = keypoint_map.target
        count = target.num_keypoints
        check_bootstrap_settings(params.n_boot, params.confidence)
        for field, position in (
            ("center", params.center),
            ("head", params.head),
            ("tail", params.tail),
        ):
            if position is not None and not 0 <= position < count:
                msg = (
                    f"{field} position {position} is outside the target pose's "
                    f"keypoints {list(target.names)}"
                )
                raise ValueError(msg)
        if len(params.oks_sigmas) != count:
            msg = (
                f"oks_sigmas gives {len(params.oks_sigmas)} sigmas for the "
                f"{count} target keypoints {list(target.names)}; give one each"
            )
            raise ValueError(msg)
        if not all(math.isfinite(sigma) and sigma > 0.0 for sigma in params.oks_sigmas):
            msg = (
                "every OKS sigma must be finite and positive: "
                f"{list(params.oks_sigmas)}"
            )
            raise ValueError(msg)
        columns = tuple(_pck_column(alpha) for alpha in params.pck_alphas)
        if len(set(columns)) != len(columns):
            msg = (
                f"pck_alphas {list(params.pck_alphas)} repeat a column name: "
                f"{list(columns)}"
            )
            raise ValueError(msg)
        pairs = list(target.symmetries)
        head, tail = params.head, params.tail
        if head is not None and tail is not None and head != tail:
            axis = (min(head, tail), max(head, tail))
            if axis not in pairs:
                pairs.append(axis)
        return cls(
            keypoint_map=keypoint_map,
            params=params,
            covered=np.array(
                [source is not None for source in keypoint_map.sources], dtype=np.bool_
            ),
            sigmas=np.array(params.oks_sigmas, dtype=np.float64),
            alphas=np.array(params.pck_alphas, dtype=np.float64),
            swap_pairs=tuple(pairs),
            pck_columns=columns,
        )

    @property
    def count(self) -> int:
        return self.keypoint_map.target.num_keypoints

    def frame_columns(self) -> dict[str, str]:
        """Return each frames-table column and its pandas type, in column order."""
        return frame_table_columns(self.params.pck_alphas)

    def instance_columns(self) -> dict[str, str]:
        """Return each instances-table column and its pandas type, in column order."""
        return _instance_table_columns(self.count)


def _instance_table_columns(count: int) -> dict[str, str]:
    """Return each instances-table column and its pandas type, in column order.

    Args:
        count: The number of keypoints, one error column each.
    """
    return {
        **dict.fromkeys(("origin_uuid", "image", "frame_key"), "string"),
        **dict.fromkeys(("gt_index", "pred_index"), "Int64"),
        **dict.fromkeys(("matched", "duplicate"), "bool"),
        **dict.fromkeys(
            ("center_distance_px", "normalizer_px", "oks", "instance_score"),
            "float64",
        ),
        **dict.fromkeys(("n_kpt_eligible", "n_kpt_found", "n_kpt_untouched"), "int64"),
        **dict.fromkeys(_error_columns(count), "float64"),
        "source_ref": "string",
    }


def frame_table_columns(pck_alphas: Sequence[float]) -> dict[str, str]:
    """Return each frames-table column and its pandas type, in column order.

    Args:
        pck_alphas: The PCK thresholds, one ``n_pck_<a>`` column each.
    """
    return {
        **dict.fromkeys(
            (
                "origin_uuid",
                "image",
                "frames_run_id",
                "group",
                "sequence",
                "camera",
            ),
            "string",
        ),
        "frame_index": "int64",
        "frame_key": "string",
        "status": "string",
        **dict.fromkeys(
            ("n_gt", "n_pred", "n_matched", "n_fp", "n_fp_duplicate"), "int64"
        ),
        "all_found": "boolean",
        **dict.fromkeys(
            (
                "n_kpt_eligible",
                "n_kpt_found",
                "n_kpt_untouched",
                "n_kpt_unnormalized",
            ),
            "int64",
        ),
        **dict.fromkeys(tuple(_pck_column(alpha) for alpha in pck_alphas), "int64"),
        "oks_sum": "float64",
        **dict.fromkeys(("n_oks", "n_swap_eligible", "n_swap"), "int64"),
        "min_oks": "float64",
    }


def _pck_column(alpha: float) -> str:
    """Return ``n_pck_<a>``: *alpha* to 4 decimals, trimmed, its point as ``p``."""
    text = f"{alpha:.4f}".rstrip("0").rstrip(".")
    return "n_pck_" + text.replace(".", "p")


def _error_columns(count: int) -> tuple[str, ...]:
    return tuple(f"error_px_{position}" for position in range(count))


@dataclass(frozen=True, slots=True)
class _Located:
    """Records a prediction frame, the run that predicted it, and its frame key."""

    run: FramePredictions
    frame: PredictedFrame
    key: str


def _locate(
    predictions: Sequence[FramePredictions], origin: str
) -> dict[str, _Located]:
    """Return every prediction frame by its image, refusing an image predicted twice."""
    located: dict[str, _Located] = {}
    for run in predictions:
        for frame in run.frames:
            earlier = located.get(frame.image)
            if earlier is not None:
                msg = (
                    f"the image {frame.image} is predicted on the frames of both "
                    f"{earlier.run.frames_run_id} and {run.frames_run_id}; each image "
                    "is scored from one prediction frame"
                )
                raise ValueError(msg)
            key = frame_key(
                VideoFrame(
                    origin_uuid=origin,
                    group=run.group,
                    sequence=run.sequence,
                    camera=run.camera,
                    frame_index=frame.frame_index,
                    image_stem=PurePosixPath(frame.image).stem,
                )
            )
            located[frame.image] = _Located(run=run, frame=frame, key=key)
    return located


def _sorted_truth(truth: Sequence[TruthFrame]) -> list[TruthFrame]:
    """Return *truth* sorted by image key, refusing a key held by two frames."""
    ordered = sorted(truth, key=lambda item: (item.origin_uuid, item.image))
    for earlier, later in zip(ordered, ordered[1:]):
        if (earlier.origin_uuid, earlier.image) == (later.origin_uuid, later.image):
            msg = (
                f"the image {later.image} of dataset {later.origin_uuid} is in two "
                "truth frames; an image key is scored once"
            )
            raise ValueError(msg)
    return ordered


@dataclass(frozen=True, slots=True)
class _Truths:
    """Records a truth frame's instances as arrays over the target pose."""

    xy: _FloatArray
    centers: _FloatArray
    visible: _BoolArray
    untouched: _BoolArray
    normalizers: _FloatArray
    areas: _FloatArray


@dataclass(frozen=True, slots=True)
class _Predictions:
    """Records a prediction frame's kept instances, mapped onto the target pose."""

    positions: tuple[int, ...]
    scores: tuple[float | None, ...]
    xy: _FloatArray
    centers: _FloatArray


@dataclass(frozen=True, slots=True)
class _Pair:
    """Records one matched pair's keypoint scores.

    ``pck_hits`` has one row per PCK alpha and one column per target keypoint,
    all false when ``normalizer`` is 0.
    """

    gt: int
    pred: int
    center_distance: float
    normalizer: float
    eligible: _BoolArray
    found: _BoolArray
    untouched: _BoolArray
    errors: _FloatArray
    oks: float | None
    pck_hits: _BoolArray
    unnormalized: int
    swap_eligible: int
    swaps: int


@dataclass(frozen=True, slots=True)
class _FrameScore:
    """Records a scored frame's instances, pairs and duplicate flags."""

    truths: _Truths
    predictions: _Predictions
    pairs: tuple[_Pair, ...]
    duplicate: tuple[bool, ...]


def _score_frame(
    item: TruthFrame,
    frame: PredictedFrame,
    untouched: Mapping[UntouchedKey, tuple[bool, ...]],
    layout: _Layout,
) -> _FrameScore:
    truths = _truths(item, untouched, layout)
    predictions = _predictions(frame, layout)
    params = layout.params
    radius = (
        np.full(truths.normalizers.shape, params.match_radius_px, dtype=np.float64)
        if params.match_radius_px is not None
        else params.match_radius * truths.normalizers
    )
    gt, pred = match_instances(truths.centers, predictions.centers, radius)
    pairs = tuple(
        _pair(truths, predictions, int(i), int(j), layout)
        for i, j in zip(gt.tolist(), pred.tolist(), strict=True)
    )
    return _FrameScore(
        truths=truths,
        predictions=predictions,
        pairs=pairs,
        duplicate=_duplicates(truths, predictions, pairs, params.duplicate_radius),
    )


def _truths(
    item: TruthFrame,
    untouched: Mapping[UntouchedKey, tuple[bool, ...]],
    layout: _Layout,
) -> _Truths:
    count = layout.count
    objects = item.frame.objects
    boxes: list[Bbox] = []
    flags: list[tuple[bool, ...]] = []
    for index, obj in enumerate(objects):
        boxes.append(_checked_box(item, index, obj, count))
        flag = untouched.get((item.origin_uuid, item.image, index), (False,) * count)
        if len(flag) != count:
            msg = (
                f"the untouched entry of object {index} of {item.image} has "
                f"{len(flag)} flags for {count} target keypoints"
            )
            raise ValueError(msg)
        flags.append(flag)
    xy = _xy([obj.keypoints for obj in objects], count, lambda point: point.is_placed)
    visible = np.array(
        [[point.visibility == 2 for point in obj.keypoints] for obj in objects],
        dtype=np.bool_,
    ).reshape(-1, count) & np.isfinite(xy).all(axis=2)
    params = layout.params
    lengths = instance_body_lengths(xy, params.head, params.tail)
    normalizers = np.array(
        [
            _normalizer(float(length), box)
            for length, box in zip(lengths.tolist(), boxes, strict=True)
        ],
        dtype=np.float64,
    )
    return _Truths(
        xy=xy,
        centers=_centers(xy, params.center),
        visible=visible,
        untouched=np.array(flags, dtype=np.bool_).reshape(-1, count),
        normalizers=normalizers,
        areas=np.array([box.width * box.height for box in boxes], dtype=np.float64),
    )


def _checked_box(
    item: TruthFrame, index: int, obj: AnnotationObject, count: int
) -> Bbox:
    if len(obj.keypoints) != count:
        msg = (
            f"object {index} of {item.image} has {len(obj.keypoints)} keypoints, "
            f"and the target pose has {count}"
        )
        raise ValueError(msg)
    if obj.bbox is None:
        msg = (
            f"object {index} of {item.image} has no box; a narrowed set gives "
            "every object a drawn or derived box"
        )
        raise ValueError(msg)
    return obj.bbox


def _normalizer(length: float, box: Bbox) -> float:
    """Return *length* when positive, else the box diagonal, else 0."""
    if math.isfinite(length) and length > 0.0:
        return length
    diagonal = math.hypot(box.width, box.height)
    return diagonal if math.isfinite(diagonal) else 0.0


def _predictions(frame: PredictedFrame, layout: _Layout) -> _Predictions:
    params = layout.params
    kept = [
        (position, obj)
        for position, obj in enumerate(frame.objects)
        if score_passes(obj.score, params.min_instance_score)
    ]
    mapped = [layout.keypoint_map.apply(obj.keypoints) for _, obj in kept]
    xy = _xy(
        mapped,
        layout.count,
        lambda point: keypoint_accepted(point, params.min_keypoint_score),
    )
    return _Predictions(
        positions=tuple(position for position, _ in kept),
        scores=tuple(finite_or_none(obj.score) for _, obj in kept),
        xy=xy,
        centers=_centers(xy, params.center),
    )


def _xy(
    instances: Sequence[Sequence[Keypoint]],
    count: int,
    placed: _PlacedRule,
) -> _FloatArray:
    """Return ``(n, count, 2)`` coordinates, NaN where *placed* refuses a point."""
    return np.array(
        [
            [
                (point.x, point.y) if placed(point) else (math.nan, math.nan)
                for point in keypoints
            ]
            for keypoints in instances
        ],
        dtype=np.float64,
    ).reshape(-1, count, 2)


def _centers(xy: _FloatArray, center: int | None) -> _FloatArray:
    """Return each instance's center keypoint when placed, else its placed mean."""
    centers = np.full((xy.shape[0], 2), math.nan, dtype=np.float64)
    placed = np.isfinite(xy).all(axis=2)
    for index in range(xy.shape[0]):
        if center is not None and bool(placed[index, center]):
            centers[index] = xy[index, center]
        elif bool(placed[index].any()):
            centers[index] = xy[index, placed[index]].mean(axis=0)
    return centers


def _pair(
    truths: _Truths, predictions: _Predictions, gt: int, pred: int, layout: _Layout
) -> _Pair:
    candidates = truths.visible[gt] & layout.covered
    untouched = candidates & truths.untouched[gt]
    eligible = candidates & ~truths.untouched[gt]
    truth_xy = truths.xy[gt]
    predicted_xy = predictions.xy[pred]
    found = eligible & np.isfinite(predicted_xy).all(axis=1)
    distances = np.hypot(*(predicted_xy - truth_xy).T)
    errors = np.where(found, distances, math.nan)
    normalizer = float(truths.normalizers[gt])
    n_eligible = int(eligible.sum())
    if normalizer > 0.0:
        pck_hits = found[None, :] & (
            errors[None, :] <= layout.alphas[:, None] * normalizer
        )
        unnormalized = 0
    else:
        pck_hits = np.zeros((layout.alphas.size, layout.count), dtype=np.bool_)
        unnormalized = n_eligible
    oks: float | None = None
    if n_eligible:
        scale = 2.0 * (float(truths.areas[gt]) + _AREA_EPSILON)
        terms = np.exp(
            -np.square(np.where(found, errors, 0.0))
            / (scale * np.square(2.0 * layout.sigmas))
        )
        oks = float(np.where(found, terms, 0.0).sum()) / n_eligible
    swap_eligible = 0
    swaps = 0
    for a, b in layout.swap_pairs:
        if not (bool(found[a]) and bool(found[b])):
            continue
        swap_eligible += 1
        crossed = math.dist(predicted_xy[a], truth_xy[b]) + math.dist(
            predicted_xy[b], truth_xy[a]
        )
        if crossed < float(distances[a]) + float(distances[b]):
            swaps += 1
    return _Pair(
        gt=gt,
        pred=pred,
        center_distance=math.dist(truths.centers[gt], predictions.centers[pred]),
        normalizer=normalizer,
        eligible=eligible,
        found=found,
        untouched=untouched,
        errors=errors,
        oks=oks,
        pck_hits=pck_hits,
        unnormalized=unnormalized,
        swap_eligible=swap_eligible,
        swaps=swaps,
    )


def _duplicates(
    truths: _Truths,
    predictions: _Predictions,
    pairs: Sequence[_Pair],
    duplicate_radius: float,
) -> tuple[bool, ...]:
    """Flag each kept prediction that is unmatched and near a matched one."""
    matched = {pair.pred for pair in pairs}
    flags: list[bool] = []
    for index in range(len(predictions.positions)):
        center = predictions.centers[index]
        flags.append(
            index not in matched
            and any(
                math.dist(center, predictions.centers[pair.pred])
                < duplicate_radius * float(truths.normalizers[pair.gt])
                for pair in pairs
            )
        )
    return tuple(flags)


def _frame_record(
    item: TruthFrame,
    found: _Located | None,
    status: FrameStatus,
    scored: _FrameScore | None,
    layout: _Layout,
) -> _Record:
    record: _Record = {
        "origin_uuid": item.origin_uuid,
        "image": item.image,
        "frames_run_id": "" if found is None else found.run.frames_run_id,
        "group": "" if found is None else found.run.group,
        "sequence": "" if found is None else found.run.sequence,
        "camera": "" if found is None else found.run.camera,
        "frame_index": -1 if found is None else found.frame.frame_index,
        "frame_key": "" if found is None else found.key,
        "status": status,
    }
    pairs = () if scored is None else scored.pairs
    n_gt = 0 if scored is None else scored.truths.xy.shape[0]
    n_pred = 0 if scored is None else len(scored.predictions.positions)
    oks = [pair.oks for pair in pairs if pair.oks is not None]
    record.update(
        {
            "n_gt": n_gt,
            "n_pred": n_pred,
            "n_matched": len(pairs),
            "n_fp": n_pred - len(pairs),
            "n_fp_duplicate": 0 if scored is None else sum(scored.duplicate),
            "all_found": len(pairs) == n_gt if n_gt else None,
            "n_kpt_eligible": sum(int(pair.eligible.sum()) for pair in pairs),
            "n_kpt_found": sum(int(pair.found.sum()) for pair in pairs),
            "n_kpt_untouched": sum(int(pair.untouched.sum()) for pair in pairs),
            "n_kpt_unnormalized": sum(pair.unnormalized for pair in pairs),
        }
    )
    for position, column in enumerate(layout.pck_columns):
        record[column] = sum(int(pair.pck_hits[position].sum()) for pair in pairs)
    record.update(
        {
            "oks_sum": math.fsum(oks),
            "n_oks": len(oks),
            "n_swap_eligible": sum(pair.swap_eligible for pair in pairs),
            "n_swap": sum(pair.swaps for pair in pairs),
            "min_oks": min(oks, default=None),
        }
    )
    return record


def _instance_records(
    item: TruthFrame, key: str, scored: _FrameScore, layout: _Layout
) -> list[_Record]:
    by_truth = {pair.gt: pair for pair in scored.pairs}
    predictions = scored.predictions
    columns = _error_columns(layout.count)
    records: list[_Record] = []
    for index, obj in enumerate(item.frame.objects):
        pair = by_truth.get(index)
        errors = (
            [None] * layout.count
            if pair is None
            else [finite_or_none(error) for error in pair.errors.tolist()]
        )
        records.append(
            {
                "origin_uuid": item.origin_uuid,
                "image": item.image,
                "frame_key": key,
                "gt_index": index,
                "pred_index": None
                if pair is None
                else predictions.positions[pair.pred],
                "matched": pair is not None,
                "duplicate": False,
                "center_distance_px": None if pair is None else pair.center_distance,
                "normalizer_px": float(scored.truths.normalizers[index]),
                "oks": None if pair is None else pair.oks,
                "instance_score": None
                if pair is None
                else predictions.scores[pair.pred],
                "n_kpt_eligible": 0 if pair is None else int(pair.eligible.sum()),
                "n_kpt_found": 0 if pair is None else int(pair.found.sum()),
                "n_kpt_untouched": 0 if pair is None else int(pair.untouched.sum()),
                **dict(zip(columns, errors, strict=True)),
                "source_ref": obj.source_ref,
            }
        )
    matched = {pair.pred for pair in scored.pairs}
    for index, position in enumerate(predictions.positions):
        if index in matched:
            continue
        records.append(
            {
                "origin_uuid": item.origin_uuid,
                "image": item.image,
                "frame_key": key,
                "gt_index": None,
                "pred_index": position,
                "matched": False,
                "duplicate": scored.duplicate[index],
                "center_distance_px": None,
                "normalizer_px": None,
                "oks": None,
                "instance_score": predictions.scores[index],
                "n_kpt_eligible": 0,
                "n_kpt_found": 0,
                "n_kpt_untouched": 0,
                **dict.fromkeys(columns),
                "source_ref": None,
            }
        )
    return records


def _table(records: Sequence[_Record], columns: Mapping[str, str]) -> pd.DataFrame:
    """Return *records* as a table of *columns*, each of its declared type."""
    return pd.DataFrame(list(records), columns=list(columns)).astype(dict(columns))


def empty_frame_table(pck_alphas: Sequence[float]) -> pd.DataFrame:
    """Return a frames table with no rows, of the columns *pck_alphas* give."""
    return _table([], frame_table_columns(pck_alphas))


def empty_instance_table(count: int) -> pd.DataFrame:
    """Return an instances table with no rows, for *count* keypoints."""
    return _table([], _instance_table_columns(count))


# --- Summary -----------------------------------------------------------------------

_UNIT: Final = ("origin_uuid", "group", "sequence")
"""The frames-table columns that name a scored sequence, the bootstrap's unit."""


@dataclass(frozen=True, slots=True)
class _Rate:
    """Names a pooled rate by the count columns of its numerator and denominator."""

    key: str
    numerator: str
    denominator: str
    bootstrapped: bool


def _alpha_key(alpha: float) -> str:
    """Return the summary key of a PCK alpha, its ``repr`` as a float."""
    return repr(float(alpha))


def _rates(layout: _Layout) -> tuple[_Rate, ...]:
    """Return every pooled rate of the summary, the PCK rates in alpha order."""
    pck = (
        _Rate(f"pck:{_alpha_key(alpha)}", column, "n_kpt_normalized", True)
        for alpha, column in zip(
            layout.params.pck_alphas, layout.pck_columns, strict=True
        )
    )
    return (
        _Rate("recall", "n_matched", "n_gt", True),
        _Rate("precision", "n_matched", "n_pred", True),
        _Rate("fp_per_frame", "n_fp", "n_frames", False),
        _Rate("frac_frames_all_found", "n_frames_all_found", "n_frames_with_gt", True),
        _Rate("moks", "oks_sum", "n_oks", True),
        _Rate("swap_rate", "n_swap", "n_swap_eligible", False),
        *pck,
    )


def _ratio(numerator: float, denominator: float) -> float | None:
    """Return *numerator* over *denominator*, or ``None`` when that is 0."""
    return numerator / denominator if denominator > 0 else None


def _frame_counts(scored: pd.DataFrame, layout: _Layout) -> dict[str, _FloatArray]:
    """Return each additive count of the scored frames, one value per frame."""
    counts = {
        name: scored[name].to_numpy(dtype=np.float64)
        for name in (
            "n_gt",
            "n_pred",
            "n_matched",
            "n_fp",
            "n_fp_duplicate",
            "n_kpt_eligible",
            "n_kpt_untouched",
            "n_kpt_unnormalized",
            *layout.pck_columns,
            "oks_sum",
            "n_oks",
            "n_swap_eligible",
            "n_swap",
        )
    }
    counts["n_frames"] = np.ones(len(scored), dtype=np.float64)
    counts["n_frames_all_found"] = scored["all_found"].to_numpy(
        dtype=np.float64, na_value=0.0
    )
    counts["n_frames_with_gt"] = (counts["n_gt"] > 0).astype(np.float64)
    counts["n_kpt_normalized"] = counts["n_kpt_eligible"] - counts["n_kpt_unnormalized"]
    return counts


def _intervals(
    by_sequence: Mapping[str, _FloatArray],
    n_sequences: int,
    estimates: Mapping[str, float | None],
    layout: _Layout,
) -> tuple[dict[str, Interval] | None, str]:
    """Return the bootstrapped rates' intervals, and the reason when there are none.

    Args:
        by_sequence: Each count summed per scored sequence, the sequences sorted.
        n_sequences: The scored sequences.
        estimates: Each rate's pooled value, by key.
        layout: The target pose and the params.
    """
    if n_sequences < 2:
        sequences = counted(n_sequences, "scored sequence")
        return None, f"{sequences}; a sequence bootstrap needs 2 or more"
    rates = [rate for rate in _rates(layout) if rate.bootstrapped]
    names = sorted(
        {name for rate in rates for name in (rate.numerator, rate.denominator)}
    )
    params = layout.params
    sums = resampled_sums(
        np.column_stack([by_sequence[name] for name in names]),
        n_boot=params.n_boot,
        random_state=params.random_state,
    )
    replicate = {name: sums[:, index] for index, name in enumerate(names)}
    intervals: dict[str, Interval] = {}
    for rate in rates:
        estimate = estimates[rate.key]
        denominator = replicate[rate.denominator]
        defined = denominator > 0
        if estimate is None or not bool(defined.any()):
            continue
        ratios = replicate[rate.numerator][defined] / denominator[defined]
        intervals[rate.key] = percentile_interval(estimate, ratios, params.confidence)
    return intervals, ""


def _quantiles(errors: _FloatArray) -> ErrorQuantiles:
    """Return the median and the 90th percentile of the finite *errors*."""
    finite = errors[np.isfinite(errors)]
    if finite.size == 0:
        return ErrorQuantiles(median=None, p90=None)
    return ErrorQuantiles(
        median=float(np.quantile(finite, 0.5)), p90=float(np.quantile(finite, 0.9))
    )


@dataclass(frozen=True, slots=True)
class _KeypointMetrics:
    """Records the matched pairs' metrics by target keypoint, and their errors."""

    pck: dict[str, dict[str, float | None]]
    miss_share: dict[str, float | None]
    error_px: dict[str, ErrorQuantiles]
    error_px_pooled: ErrorQuantiles
    error_norm: ErrorQuantiles

    @classmethod
    def of(cls, pairs: Sequence[_Pair], layout: _Layout) -> _KeypointMetrics:
        shape = (len(pairs), layout.count)
        eligible = np.array([p.eligible for p in pairs], dtype=np.bool_).reshape(shape)
        found = np.array([p.found for p in pairs], dtype=np.bool_).reshape(shape)
        errors = np.array([p.errors for p in pairs], dtype=np.float64).reshape(shape)
        hits = np.array([p.pck_hits for p in pairs], dtype=np.bool_).reshape(
            (len(pairs), layout.alphas.size, layout.count)
        )
        normalizers = np.array([p.normalizer for p in pairs], dtype=np.float64)
        normalized = normalizers > 0.0
        n_eligible = eligible.sum(axis=0)
        n_missed = (eligible & ~found).sum(axis=0)
        n_normalized = eligible[normalized].sum(axis=0)
        n_hits = hits.sum(axis=0)
        keys = [_alpha_key(alpha) for alpha in layout.params.pck_alphas]
        names = layout.keypoint_map.target.names
        return cls(
            pck={
                name: {
                    key: _ratio(
                        float(n_hits[row, position]), float(n_normalized[position])
                    )
                    for row, key in enumerate(keys)
                }
                for position, name in enumerate(names)
            },
            miss_share={
                name: _ratio(float(n_missed[position]), float(n_eligible[position]))
                for position, name in enumerate(names)
            },
            error_px={
                name: _quantiles(errors[:, position])
                for position, name in enumerate(names)
            },
            error_px_pooled=_quantiles(errors),
            error_norm=_quantiles(errors[normalized] / normalizers[normalized, None]),
        )


def _source_unknown(
    scored_frames: Sequence[tuple[TruthFrame, _FrameScore]],
    untouched: Mapping[UntouchedKey, tuple[bool, ...]],
) -> int:
    """Count the model-placed truth objects of scored frames without untouched flags."""
    return sum(
        1
        for item, _ in scored_frames
        for index, obj in enumerate(item.frame.objects)
        if obj.origin == "model"
        and (item.origin_uuid, item.image, index) not in untouched
    )


def _metrics(
    frames: pd.DataFrame,
    scored_frames: Sequence[tuple[TruthFrame, _FrameScore]],
    untouched: Mapping[UntouchedKey, tuple[bool, ...]],
    layout: _Layout,
) -> PoseMetrics:
    """Pool the counts of the frames table's scored rows and the scored pairs."""
    status = frames["status"]
    n_truth = len(frames)
    n_by_status = {
        name: int((status == name).sum())
        for name in ("scored", "excluded_training", "not_covered")
    }
    n_covered = n_by_status["scored"] + n_by_status["excluded_training"]
    scored = frames.loc[status == "scored"]
    counts = _frame_counts(scored, layout)
    totals = {name: float(values.sum()) for name, values in counts.items()}
    estimates = {
        rate.key: _ratio(totals[rate.numerator], totals[rate.denominator])
        for rate in _rates(layout)
    }
    codes = scored.groupby(list(_UNIT), sort=True).ngroup().to_numpy(dtype=np.int64)
    n_sequences = int(codes.max()) + 1 if codes.size else 0
    by_sequence = {
        name: np.bincount(codes, weights=values, minlength=n_sequences)
        for name, values in counts.items()
    }
    intervals, reason = _intervals(by_sequence, n_sequences, estimates, layout)
    per_keypoint = _KeypointMetrics.of(
        [pair for _, score in scored_frames for pair in score.pairs], layout
    )
    keypoint_map = layout.keypoint_map
    params = layout.params
    return PoseMetrics(
        keypoints=list(keypoint_map.target.names),
        keypoint_map=KeypointMapSummary(
            identity=keypoint_map.is_identity,
            coverage=list(keypoint_map.coverage),
            unmapped_sources=list(keypoint_map.unmapped_sources),
        ),
        n_truth_frames=n_truth,
        n_scored=n_by_status["scored"],
        n_excluded_training=n_by_status["excluded_training"],
        n_not_covered=n_by_status["not_covered"],
        coverage=n_covered / n_truth if n_truth else 0.0,
        n_gt=int(totals["n_gt"]),
        n_pred=int(totals["n_pred"]),
        n_matched=int(totals["n_matched"]),
        n_fp=int(totals["n_fp"]),
        n_fp_duplicate=int(totals["n_fp_duplicate"]),
        recall=estimates["recall"],
        precision=estimates["precision"],
        fp_per_frame=estimates["fp_per_frame"],
        frac_frames_all_found=estimates["frac_frames_all_found"],
        moks=estimates["moks"],
        swap_rate=estimates["swap_rate"],
        pck={
            _alpha_key(alpha): estimates[f"pck:{_alpha_key(alpha)}"]
            for alpha in params.pck_alphas
        },
        pck_per_keypoint=per_keypoint.pck,
        miss_share_per_keypoint=per_keypoint.miss_share,
        error_px_per_keypoint=per_keypoint.error_px,
        error_px=per_keypoint.error_px_pooled,
        error_norm=per_keypoint.error_norm,
        n_kpt_eligible=int(totals["n_kpt_eligible"]),
        n_kpt_untouched=int(totals["n_kpt_untouched"]),
        n_kpt_unnormalized=int(totals["n_kpt_unnormalized"]),
        n_objects_source_unknown=_source_unknown(scored_frames, untouched),
        intervals=intervals,
        interval_reason=reason,
        n_sequences=n_sequences,
        n_boot=params.n_boot,
        confidence=params.confidence,
        random_state=params.random_state,
    )
