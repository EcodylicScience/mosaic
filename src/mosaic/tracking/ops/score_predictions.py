"""Defines the score-predictions op, which scores prediction runs against annotations.

A score compares the predictions of one model's predict-frames runs with the
annotated frames of keypoint-set revisions. Its run id contains the revisions'
content, the keypoint map as resolved, the frames that the model trained or
validated on, and a digest of each predictions file that the score reads. The
run writes a frames table, an instances table, a summary and its identity
payload to its run root, and records one row in the score-predictions index.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, ClassVar, Final

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    field_validator,
)

from mosaic.core.annotations.bbox import BboxPolicy
from mosaic.core.annotations.keypoint_map import (
    KeypointMap,
    KeypointMapParams,
    resolve_keypoint_map,
)
from mosaic.core.annotations.model import AnnotationSet, KeypointSchema
from mosaic.core.annotations.narrow import narrow_pose_sets, narrowed_pose
from mosaic.core.annotations.pose_annotations import PoseDefinition
from mosaic.core.annotations.pose_predictions import (
    FramePredictions,
    PredictedFrame,
    parse_prediction_source_ref,
    read_pose_predictions,
)
from mosaic.core.annotations.scoring import (
    DEFAULT_PCK_ALPHAS,
    LabelRevision,
    PoseMetrics,
    PoseScoreParams,
    PoseScoreSummary,
    TruthFrame,
    UntouchedKey,
    score_predictions,
    untouched_keypoints,
)
from mosaic.core.json_value import JsonValue
from mosaic.core.keypoint_refs import resolve_keypoint_refs
from mosaic.core.params import Declared, Params
from mosaic.core.pipeline._utils import atomic_write, hash_params
from mosaic.core.pipeline.file_digest import file_digest
from mosaic.core.pipeline.identity_scheme import write_identity_scheme
from mosaic.core.pipeline.index_csv import index_records
from mosaic.core.pipeline.op_identity import OP_IDENTITY_SCHEME, op_run_id
from mosaic.core.pipeline.ops import IdentityDeferred, Op, OpIdentity, register_op
from mosaic.core.pipeline.writers import write_parquet_atomic
from mosaic.core.strict_model import terse
from mosaic.tracking.frame_predictions.errors import (
    PredictionRunNotFound,
    ScorePredictionsConflict,
    ScorePredictionsInvalid,
)
from mosaic.tracking.frame_predictions.index import (
    RUN_PARAMS_NAME,
    SCORE_RUN_OUTPUTS,
    ScoreRunRow,
    prediction_index_path,
    prediction_kind_root,
    score_predictions_index,
    score_run_is_complete,
    write_run_params,
)
from mosaic.tracking.frame_predictions.read import (
    list_prediction_runs,
    predictions_file,
)
from mosaic.tracking.keypoint_sets import (
    KeypointSetRef,
    ResolvedSet,
    consumed_sets_cell,
    in_identity_order,
    read_resolved_set,
    resolve_keypoint_sets,
    set_label,
)
from mosaic.tracking.training_provenance import SeenFrames, seen_frames_of_model

from ._common import claim_run_root, empty_claimed_run_root, renew_run_root_claim

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset
    from mosaic.core.pipeline._utils import ResolvedScope
    from mosaic.core.pipeline.job import JobContext

__all__ = [
    "ScorePredictionsOp",
    "ScorePredictionsParams",
    "pose_score_params",
    "score_predictions_payload",
    "score_predictions_run_id",
]

_KIND: Final = "score-predictions"

_ALPHA_DECIMALS: Final = 4
"""The most decimals of a PCK alpha, all of which its ``n_pck_<a>`` column spells."""

_BODY_LENGTHS: Final = "body lengths"
"""The unit of a radius scaled by each annotated instance's normalizer."""

_PREDICTION_RUNS_DESCRIPTION = (
    "The predict-frames run ids to score, all of one model, scored as one pool. "
    "Give one run per extraction run, so a set built from several extraction "
    "runs is scored as a whole."
)

_LABELS_DESCRIPTION = "The annotation sets that the predictions are scored against."

_POSE_DESCRIPTION = (
    "Which pose of the sets is scored, by its id or its name. Left unset, every "
    "set must hold exactly one pose."
)

_KEYPOINT_MAP_DESCRIPTION = (
    "How the model's keypoints map onto the scored pose. Unset, by name."
)

_EXCLUDE_SEEN_DESCRIPTION = (
    "Leave out the frames that the model trained or validated on."
)

_MATCH_RADIUS_DESCRIPTION = (
    "How far from an annotated instance's center a prediction's center matches "
    "it, as a fraction of the instance's normalizer."
)

_MATCH_RADIUS_PX_DESCRIPTION = "A fixed matching radius, used in place of match_radius."

_DUPLICATE_RADIUS_DESCRIPTION = (
    "How near a matched prediction's center an unmatched prediction counts as a "
    "duplicate, as a fraction of the matched instance's normalizer."
)

_CENTER_DESCRIPTION = (
    "The keypoint, by index or name, whose position is an instance's center "
    "when placed. Unset, the mean of the placed keypoints."
)

_HEAD_DESCRIPTION = (
    "The keypoint at the head of the instance's axis, by index or name. Unset, "
    "the set's bbox_policy head_index."
)

_TAIL_DESCRIPTION = (
    "The keypoint at the tail of the instance's axis, by index or name. Unset, "
    "the set's bbox_policy tail_index."
)

_OKS_SIGMA_DESCRIPTION = "The OKS sigma of a keypoint that oks_sigmas does not name."

_OKS_SIGMAS_DESCRIPTION = (
    "The OKS sigma of a keypoint of the scored pose, by its name. A name that "
    "is not a keypoint of the pose is refused."
)

_PCK_ALPHAS_DESCRIPTION = (
    "The PCK thresholds, as fractions of each annotated instance's normalizer. "
    "Each is a multiple of 0.0001, and the thresholds are stored sorted."
)

_MIN_INSTANCE_SCORE_DESCRIPTION = (
    "The lowest instance score kept. Predictions below it are dropped before "
    "matching. Unset keeps every prediction."
)

_MIN_KEYPOINT_SCORE_DESCRIPTION = (
    "The lowest keypoint score that counts as placed. Unset, every predicted "
    "keypoint counts as placed."
)

_UNTOUCHED_TOL_DESCRIPTION = (
    "The largest offset on each axis at which an annotated keypoint still lies "
    "where a model placed it. Such a keypoint is untouched and is left out of "
    "the keypoint metrics."
)

_N_BOOT_DESCRIPTION = "The number of bootstrap replicates of the pooled rates."

_CONFIDENCE_DESCRIPTION = "The confidence of the bootstrap intervals."

_RANDOM_STATE_DESCRIPTION = "The seed of the bootstrap."

_KeypointReference = int | str | None
"""A keypoint named by its index or its name, or ``None`` for none."""


class ScorePredictionsParams(Params):
    """Declares how the predictions of one model are scored against annotations.

    ``prediction_runs`` and ``pck_alphas`` are stored sorted, and two orders of
    either give one run id. ``labels`` and ``keypoint_map`` enter the run id as
    the revisions' content and as the resolved map.
    """

    prediction_runs: Annotated[
        tuple[str, ...], Field(min_length=1), Declared(_PREDICTION_RUNS_DESCRIPTION)
    ]
    labels: Annotated[
        tuple[KeypointSetRef, ...], Field(min_length=1), Declared(_LABELS_DESCRIPTION)
    ]
    pose: Annotated[int | str | None, Declared(_POSE_DESCRIPTION)] = None
    keypoint_map: Annotated[KeypointMapParams, Declared(_KEYPOINT_MAP_DESCRIPTION)] = (
        Field(default_factory=KeypointMapParams)
    )
    exclude_seen: Annotated[bool, Declared(_EXCLUDE_SEEN_DESCRIPTION)] = True
    match_radius: Annotated[
        float, Field(gt=0.0), Declared(_MATCH_RADIUS_DESCRIPTION, unit=_BODY_LENGTHS)
    ] = 0.5
    match_radius_px: Annotated[
        float | None,
        Field(gt=0.0),
        Declared(_MATCH_RADIUS_PX_DESCRIPTION, unit="px"),
    ] = None
    duplicate_radius: Annotated[
        float,
        Field(gt=0.0),
        Declared(_DUPLICATE_RADIUS_DESCRIPTION, unit=_BODY_LENGTHS),
    ] = 0.2
    center: Annotated[_KeypointReference, Declared(_CENTER_DESCRIPTION)] = None
    head: Annotated[_KeypointReference, Declared(_HEAD_DESCRIPTION)] = None
    tail: Annotated[_KeypointReference, Declared(_TAIL_DESCRIPTION)] = None
    oks_sigma: Annotated[float, Field(gt=0.0), Declared(_OKS_SIGMA_DESCRIPTION)] = 0.025
    oks_sigmas: Annotated[
        dict[str, Annotated[float, Field(gt=0.0)]], Declared(_OKS_SIGMAS_DESCRIPTION)
    ] = Field(default_factory=dict)
    pck_alphas: Annotated[
        tuple[Annotated[float, Field(gt=0.0, le=1.0)], ...],
        Field(min_length=1),
        Declared(_PCK_ALPHAS_DESCRIPTION),
    ] = DEFAULT_PCK_ALPHAS
    min_instance_score: Annotated[
        float | None, Declared(_MIN_INSTANCE_SCORE_DESCRIPTION)
    ] = None
    min_keypoint_score: Annotated[
        float | None, Declared(_MIN_KEYPOINT_SCORE_DESCRIPTION)
    ] = None
    untouched_tol_px: Annotated[
        float, Field(ge=0.0), Declared(_UNTOUCHED_TOL_DESCRIPTION, unit="px")
    ] = 0.01
    n_boot: Annotated[int, Field(ge=1), Declared(_N_BOOT_DESCRIPTION)] = 1000
    confidence: Annotated[
        float, Field(gt=0.0, lt=1.0), Declared(_CONFIDENCE_DESCRIPTION)
    ] = 0.95
    random_state: Annotated[int, Declared(_RANDOM_STATE_DESCRIPTION)] = 0

    @field_validator("prediction_runs")
    @classmethod
    def _sorted_runs(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Sort the runs, and refuse a run named twice."""
        repeated = sorted({run for run in value if value.count(run) > 1})
        if repeated:
            msg = (
                f"prediction_runs names {repeated} more than once. Name each run once."
            )
            raise ValueError(msg)
        return tuple(sorted(value))

    @field_validator("pck_alphas")
    @classmethod
    def _sorted_alphas(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        """Refuse an alpha with more than 4 decimals or named twice, and sort them."""
        inexact = [alpha for alpha in value if round(alpha, _ALPHA_DECIMALS) != alpha]
        if inexact:
            msg = (
                f"pck_alphas {inexact} are not multiples of 0.0001. Give each "
                "alpha to at most 4 decimals."
            )
            raise ValueError(msg)
        repeated = sorted({alpha for alpha in value if value.count(alpha) > 1})
        if repeated:
            msg = f"pck_alphas names {repeated} more than once. Name each alpha once."
            raise ValueError(msg)
        return tuple(sorted(value))


def pose_score_params(
    params: ScorePredictionsParams, schema: KeypointSchema, bbox_policy: BboxPolicy
) -> PoseScoreParams:
    """Resolve the metric settings of *params* against the scored pose.

    ``center``, ``head`` and ``tail`` become positions in *schema*. An unset
    ``head`` or ``tail`` takes the index that *bbox_policy* declares. Each
    keypoint of *schema* takes its sigma from ``oks_sigmas``, else
    ``oks_sigma``.

    Args:
        params: The score's parameters.
        schema: The pose that the labels are narrowed to.
        bbox_policy: The box policy of the labels, which declares the default
            head and tail.

    Returns:
        The settings that the scorer reads.

    Raises:
        ValueError: An ``oks_sigmas`` key names no keypoint of *schema*, or
            ``center``, ``head`` or ``tail`` names a keypoint that *schema*
            does not have.
    """
    unknown = sorted(set(params.oks_sigmas) - set(schema.names))
    if unknown:
        msg = (
            f"oks_sigmas names {unknown}, which the scored pose does not have; "
            f"its keypoints are {list(schema.names)}"
        )
        raise ValueError(msg)
    head = _position(params.head, schema, who="head")
    tail = _position(params.tail, schema, who="tail")
    return PoseScoreParams(
        center=_position(params.center, schema, who="center"),
        head=bbox_policy.head_index if head is None else head,
        tail=bbox_policy.tail_index if tail is None else tail,
        match_radius=params.match_radius,
        match_radius_px=params.match_radius_px,
        duplicate_radius=params.duplicate_radius,
        oks_sigmas=tuple(
            params.oks_sigmas.get(name, params.oks_sigma) for name in schema.names
        ),
        pck_alphas=params.pck_alphas,
        min_instance_score=params.min_instance_score,
        min_keypoint_score=params.min_keypoint_score,
        n_boot=params.n_boot,
        confidence=params.confidence,
        random_state=params.random_state,
    )


def _position(
    reference: _KeypointReference, schema: KeypointSchema, *, who: str
) -> int | None:
    """Return the position in *schema* that *reference* names, or ``None``."""
    if reference is None:
        return None
    (position,) = resolve_keypoint_refs(
        [reference], schema.names, schema.num_keypoints, who=who
    )
    return position


def score_predictions_payload(
    params: ScorePredictionsParams,
    *,
    labels: Sequence[tuple[str, str, str]],
    keypoint_map: KeypointMap,
    seen_digest: str | None,
    sources: Collection[str],
    read: Collection[tuple[str, str, str, str, str]],
) -> dict[str, object]:
    """Return the identity payload of a score-predictions run.

    The run id digests it, and the run writes it as ``run_params.json``.
    ``labels`` and ``keypoint_map`` leave the parameter dump and enter as the
    revisions' content and the resolved map. A revision number selects and is
    not hashed. Two map spellings that resolve to the same pairs give one
    payload. The map enters only when it is not the identity, the seen digest
    only when it is set, and the sources only when there are any.

    Args:
        params: The score's parameters.
        labels: ``(origin_uuid, set_key, digest)`` of each revision scored
            against.
        keypoint_map: The map from the model's keypoints onto the scored pose.
        seen_digest: The digest of the frames that the model trained or
            validated on, or ``None`` when ``exclude_seen`` is false.
        sources: The source run ids of model-origin label objects that have a
            finished predict-frames row.
        read: ``(run_id, group, sequence, camera, sha256)`` of each
            predictions file that the score reads.
    """
    dumped = params.identity_dump()
    _ = dumped.pop("labels", None)
    _ = dumped.pop("keypoint_map", None)
    payload: dict[str, object] = {
        "params": dumped,
        "labels": sorted([*item] for item in labels),
    }
    if not keypoint_map.is_identity:
        payload["keypoint_map"] = keypoint_map.identity_payload()
    if seen_digest is not None:
        payload["seen"] = seen_digest
    if sources:
        payload["sources"] = sorted(sources)
    payload["read"] = sorted([*item] for item in read)
    return payload


def score_predictions_run_id(
    version: str,
    params: ScorePredictionsParams,
    *,
    labels: Sequence[tuple[str, str, str]],
    keypoint_map: KeypointMap,
    seen_digest: str | None,
    sources: Collection[str],
    read: Collection[tuple[str, str, str, str, str]],
) -> str:
    """Mint a score-predictions run identifier over :func:`score_predictions_payload`.

    The golden corpus calls it with fixed arguments and no filesystem.

    Args:
        version: The op's declared version.
        params: The score's parameters.
        labels: ``(origin_uuid, set_key, digest)`` of each revision scored
            against.
        keypoint_map: The map from the model's keypoints onto the scored pose.
        seen_digest: The digest of the frames that the model trained or
            validated on, or ``None`` when ``exclude_seen`` is false.
        sources: The source run ids of model-origin label objects that have a
            finished predict-frames row.
        read: ``(run_id, group, sequence, camera, sha256)`` of each
            predictions file that the score reads.

    Returns:
        ``score-predictions.<version>-<digest>``.
    """
    payload = score_predictions_payload(
        params,
        labels=labels,
        keypoint_map=keypoint_map,
        seen_digest=seen_digest,
        sources=sources,
        read=read,
    )
    return op_run_id(_KIND, version, payload)


# --- The op ----------------------------------------------------------------------


_PREDICT_KIND: Final = "predict-frames"

_SCORE_IDLE_SECONDS: Final = 600.0
"""The inactivity bound of the run root's claim.

The scoring gives no activity signal, so the claim is renewed once, between the
scoring and the first write.
"""

_UNSEEN_REASON: Final = "exclude_seen is false"
"""Why the frames that the model saw are unknown to a score that keeps them."""

_FRAMES_TABLE, _INSTANCES_TABLE, _SUMMARY = SCORE_RUN_OUTPUTS

_ReadItem = tuple[str, str, str, str, str]
"""``(run_id, group, sequence, camera, sha256)`` of one predictions file read."""

_MODEL_REFERENCES: Final = TypeAdapter(list[str])
"""Reads a predict-frames row's ``model`` cell, a JSON list of run ids."""


class _RecordedPrediction(BaseModel):
    """Reads the settings that a predict-frames run recorded in its identity payload."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore")

    params: dict[str, JsonValue]


@dataclass(frozen=True, slots=True)
class _PredictionRun:
    """A scored predict-frames run, with its finished rows.

    Attributes:
        run_id: The run.
        model: The model references that its rows record.
        model_kind: The training kind of the references.
        model_id: The identity of the model.
        rows: The finished rows, one per entry camera, in entry camera order.
    """

    run_id: str
    model: tuple[str, ...]
    model_kind: str
    model_id: str
    rows: tuple[dict[str, str], ...]


@dataclass(frozen=True, slots=True)
class _PredictionFile:
    """One entry camera's predictions, with the digest of the file read.

    Attributes:
        run_id: The predict-frames run.
        group: The entry's group.
        sequence: The entry's sequence.
        camera: The entry's camera.
        digest: The file's digest.
        predictions: The predictions in the file.
    """

    run_id: str
    group: str
    sequence: str
    camera: str
    digest: str
    predictions: FramePredictions

    @property
    def read_item(self) -> _ReadItem:
        """The item that names the file in the run id's ``read`` term."""
        return (self.run_id, self.group, self.sequence, self.camera, self.digest)

    @property
    def images(self) -> frozenset[str]:
        """The images that the file predicts."""
        return frozenset(frame.image for frame in self.predictions.frames)


@dataclass(frozen=True, slots=True)
class _Untouched:
    """The untouched flags of the truth objects that a model placed.

    Attributes:
        flags: One flag per target keypoint of each object whose source was read.
        sources: The source runs named by the objects that have a finished row.
        opened: The predictions files that the comparison read.
    """

    flags: dict[UntouchedKey, tuple[bool, ...]]
    sources: frozenset[str]
    opened: frozenset[_ReadItem]


@dataclass(frozen=True, slots=True)
class _ScoreInputs:
    """Everything that a score reads, and the terms of its run id.

    Attributes:
        model: The first scored run, whose model every scored run shares.
        sets: The revisions scored against, in identity order.
        pose: The pose that the revisions narrow to.
        truth: The usable frames of the pose in the revisions.
        origin: The manifest uuid of the dataset, the origin of every
            prediction frame's image key.
        predictions: The predictions of every file that holds an annotated
            image.
        keypoint_map: Maps the model's keypoints onto the pose.
        metric_settings: The metric settings, resolved against the pose.
        seen: The frames that the model trained or validated on, or ``None``
            when ``exclude_seen`` is false.
        untouched: The untouched flags and the sources that they read.
        read: Every predictions file that the score reads.
    """

    model: _PredictionRun
    sets: tuple[ResolvedSet, ...]
    pose: PoseDefinition
    truth: tuple[TruthFrame, ...]
    origin: str
    predictions: tuple[FramePredictions, ...]
    keypoint_map: KeypointMap
    metric_settings: PoseScoreParams
    seen: SeenFrames | None
    untouched: _Untouched
    read: frozenset[_ReadItem]

    @property
    def labels(self) -> tuple[tuple[str, str, str], ...]:
        """``(origin_uuid, set_key, digest)`` of each revision."""
        return tuple(
            (item.origin_uuid, item.set_key, item.digest) for item in self.sets
        )

    @property
    def seen_digest(self) -> str | None:
        """The digest of the seen frames, ``None`` when they are not asked for."""
        if self.seen is None:
            return None
        return hash_params(
            {"keys": sorted(self.seen.keys), "complete": self.seen.complete}
        )

    @property
    def excluded(self) -> frozenset[str] | None:
        """The frame keys that the score excludes, ``None`` without a complete walk."""
        if self.seen is None or not self.seen.complete:
            return None
        return self.seen.keys

    @property
    def training_frames_reason(self) -> str:
        """Why the seen frames are not excluded, ``""`` when they are."""
        return _UNSEEN_REASON if self.seen is None else self.seen.reason


def _score_inputs(
    ds: Dataset, params: ScorePredictionsParams, *, require_data: bool
) -> _ScoreInputs:
    """Read what the score reads, and refuse what cannot be scored.

    Args:
        ds: The dataset that contains the prediction runs and the revisions.
        params: The score's parameters.
        require_data: Whether a prediction run without a finished row defers
            the identity, as a planner asks, instead of being refused.

    Raises:
        IdentityDeferred: A prediction run has no finished row and
            *require_data* is true, or a revision is not indexed or not on
            disk.
        PredictionRunNotFound: A prediction run has no finished row and
            *require_data* is false.
        ScorePredictionsInvalid: The runs are of different models or settings
            or name their keypoints differently, the revisions do not narrow to
            one pose, or the keypoint map does not resolve.
        ScorePredictionsConflict: Two revisions contain one image, two runs
            predict one image, or no annotated image is predicted.
        FileNotFoundError: A finished row of a scored run, or the identity
            payload of a pooled run, names a file that is not on disk.
        ValueError: A file of a scored run is not one that this mosaic reads,
            or a metric setting names a keypoint that the pose does not have.
        KeyError: A label reference matches the sets of several datasets.
    """
    finished = _finished_rows(ds)
    runs = _prediction_runs(params.prediction_runs, finished, require_data=require_data)
    _refuse_mixed_models(runs)
    _refuse_mixed_settings(ds, runs)
    sets = tuple(
        in_identity_order(resolve_keypoint_sets(ds, params.labels, kind=_KIND))
    )
    states = {set_label(item): read_resolved_set(item) for item in sets}
    try:
        narrowed = narrow_pose_sets(states, pose=params.pose)
        pose = narrowed_pose(states, pose=params.pose)
    except ValueError as exc:
        raise ScorePredictionsInvalid(str(exc)) from exc
    truth = _truth(sets, narrowed)
    files = {
        run.run_id: tuple(_read_file(ds, run.run_id, row) for row in run.rows)
        for run in runs
    }
    _refuse_shared_images(files)
    keypoint_map = _keypoint_map(params.keypoint_map, pose.schema, files)
    origin = ds.uuid or ""
    covering = _covering(truth, files, origin=origin, sets=sets)
    reference = states[set_label(sets[0])]
    metric_settings = pose_score_params(params, pose.schema, reference.bbox_policy)
    seen = (
        seen_frames_of_model(ds, runs[0].model_kind, runs[0].model)
        if params.exclude_seen
        else None
    )
    untouched = _untouched(
        ds,
        truth,
        origin=origin,
        finished=finished,
        scored=files,
        keypoint_map=keypoint_map,
        tol_px=params.untouched_tol_px,
    )
    return _ScoreInputs(
        model=runs[0],
        sets=sets,
        pose=pose,
        truth=truth,
        origin=origin,
        predictions=tuple(file.predictions for file in covering),
        keypoint_map=keypoint_map,
        metric_settings=metric_settings,
        seen=seen,
        untouched=untouched,
        read=frozenset(file.read_item for file in covering) | untouched.opened,
    )


def _finished_rows(ds: Dataset) -> dict[str, tuple[dict[str, str], ...]]:
    """Return the finished predict-frames rows of each run, in entry camera order."""
    rows: dict[str, list[dict[str, str]]] = {}
    for record in index_records(list_prediction_runs(ds)):
        if record.get("finished_at", ""):
            rows.setdefault(record.get("run_id", ""), []).append(record)
    return {
        run_id: tuple(
            sorted(
                found,
                key=lambda row: (
                    row.get("group", ""),
                    row.get("sequence", ""),
                    row.get("camera", ""),
                ),
            )
        )
        for run_id, found in rows.items()
    }


def _prediction_runs(
    run_ids: Sequence[str],
    finished: Mapping[str, tuple[dict[str, str], ...]],
    *,
    require_data: bool,
) -> tuple[_PredictionRun, ...]:
    """Return each run of *run_ids* with its finished rows.

    Raises:
        IdentityDeferred: A run has no finished row and *require_data* is true.
        PredictionRunNotFound: A run has no finished row and *require_data* is
            false.
    """
    missing = [run_id for run_id in run_ids if run_id not in finished]
    if missing:
        named = ", ".join(missing)
        if require_data:
            because = (
                f"no finished predict-frames row names {named} yet, and this "
                "run's identity covers the predictions that it reads"
            )
            raise IdentityDeferred(_KIND, because)
        message = (
            f"No finished predict-frames row names {named}. `mosaic inventory "
            "--kind frame-predictions` lists the prediction runs."
        )
        raise PredictionRunNotFound(message)
    runs: list[_PredictionRun] = []
    for run_id in run_ids:
        rows = finished[run_id]
        first = rows[0]
        runs.append(
            _PredictionRun(
                run_id=run_id,
                model=tuple(_MODEL_REFERENCES.validate_json(first.get("model", ""))),
                model_kind=first.get("model_kind", ""),
                model_id=first.get("model_id", ""),
                rows=rows,
            )
        )
    return tuple(runs)


def _refuse_mixed_models(runs: Sequence[_PredictionRun]) -> None:
    """Refuse runs of more than one model.

    Raises:
        ScorePredictionsInvalid: Two runs differ in their model's id or kind.
            The message names each run with its model's id and kind.
    """
    if len({(run.model_kind, run.model_id) for run in runs}) < 2:
        return
    named = "; ".join(
        f"{run.run_id} predicted with the {run.model_kind} model {run.model_id}"
        for run in runs
    )
    message = f"{_KIND} scores the runs of one model together. {named}."
    raise ScorePredictionsInvalid(message)


def _refuse_mixed_settings(ds: Dataset, runs: Sequence[_PredictionRun]) -> None:
    """Refuse runs that predicted at different settings.

    Each run's settings are the params of its identity payload. The extraction
    run is left out, because a pool takes one run per extraction run.

    Raises:
        ScorePredictionsInvalid: Two runs differ in a setting. The message
            names each run and the settings that differ, with their values.
        FileNotFoundError: A run's identity payload is not on disk.
        ValueError: A run's identity payload is not one that predict-frames
            writes.
    """
    if len(runs) < 2:
        return
    settings = {run.run_id: _predict_settings(ds, run.run_id) for run in runs}
    names = sorted({name for values in settings.values() for name in values})
    differing = [
        name
        for name in names
        if len({_json_text(values.get(name)) for values in settings.values()}) > 1
    ]
    if not differing:
        return
    named = "; ".join(
        f"{run_id} predicted at "
        + ", ".join(f"{name}={_json_text(values.get(name))}" for name in differing)
        for run_id, values in settings.items()
    )
    message = (
        f"The prediction runs predicted at different settings: {named}. A pool "
        "scores one operating point, so score runs of other settings apart."
    )
    raise ScorePredictionsInvalid(message)


def _predict_settings(ds: Dataset, run_id: str) -> dict[str, JsonValue]:
    """Return the settings that the predict-frames run *run_id* recorded.

    Raises:
        FileNotFoundError: The run's identity payload is not on disk.
        ValueError: The payload is not one that predict-frames writes.
    """
    path = prediction_kind_root(ds, _PREDICT_KIND) / run_id / RUN_PARAMS_NAME
    try:
        recorded = _RecordedPrediction.model_validate_json(path.read_bytes())
    except ValidationError as exc:
        message = (
            f"{path} is not the identity payload of a predict-frames run: {terse(exc)}"
        )
        raise ValueError(message) from exc
    return {
        name: value
        for name, value in recorded.params.items()
        if name != "frames_run_id"
    }


def _json_text(value: JsonValue) -> str:
    """Return *value* as compact JSON, so equal values compare equal as text."""
    return json.dumps(value, sort_keys=True)


def _truth(
    sets: Sequence[ResolvedSet], narrowed: Mapping[str, AnnotationSet]
) -> tuple[TruthFrame, ...]:
    """Return every narrowed frame of *sets* as a truth frame of its revision's origin.

    Raises:
        ScorePredictionsConflict: Two revisions contain one image key. The
            message names the image and both set keys.
    """
    holders: dict[tuple[str, str], str] = {}
    truth: list[TruthFrame] = []
    for item in sets:
        for frame in narrowed[set_label(item)].frames:
            found = TruthFrame(origin_uuid=item.origin_uuid, frame=frame)
            holder = holders.setdefault((found.origin_uuid, found.image), item.set_key)
            if holder != item.set_key:
                message = (
                    f"The image {found.image} of dataset {found.origin_uuid!r} is a "
                    f"usable frame of the scored pose in both {holder!r} and "
                    f"{item.set_key!r}. A set pair that shares an annotated image "
                    "is scored in separate runs."
                )
                raise ScorePredictionsConflict(message)
            truth.append(found)
    return tuple(truth)


def _read_file(ds: Dataset, run_id: str, row: Mapping[str, str]) -> _PredictionFile:
    """Read the predictions file that a finished row names, and digest it.

    Raises:
        FileNotFoundError: The file is not on disk.
        ValueError: The file is not a predictions file that this mosaic reads.
    """
    path = predictions_file(ds, row)
    return _PredictionFile(
        run_id=run_id,
        group=row.get("group", ""),
        sequence=row.get("sequence", ""),
        camera=row.get("camera", ""),
        digest=file_digest(path),
        predictions=read_pose_predictions(path),
    )


def _refuse_shared_images(files: Mapping[str, Sequence[_PredictionFile]]) -> None:
    """Refuse an image that two scored runs both predict.

    Raises:
        ScorePredictionsConflict: Two runs predict one image. The message
            names the image and both runs.
    """
    owners: dict[str, str] = {}
    for run_id, run_files in files.items():
        for file in run_files:
            for frame in file.predictions.frames:
                owner = owners.setdefault(frame.image, run_id)
                if owner != run_id:
                    message = (
                        f"The image {frame.image} is predicted by both {owner} and "
                        f"{run_id}. A pool scores each image once, so give one "
                        "prediction run per extraction run."
                    )
                    raise ScorePredictionsConflict(message)


def _keypoint_map(
    given: KeypointMapParams,
    target: KeypointSchema,
    files: Mapping[str, Sequence[_PredictionFile]],
) -> KeypointMap:
    """Resolve the map from the keypoints that the files name onto *target*.

    Raises:
        ScorePredictionsInvalid: The files name the model's keypoints in more
            than one way, or the map does not resolve. The second message is
            the keypoint map's own.
    """
    named: dict[tuple[str, ...], set[str]] = {}
    for run_files in files.values():
        for file in run_files:
            names = file.predictions.pose.schema.names
            named.setdefault(names, set()).add(file.run_id)
    if len(named) > 1:
        ways = "; ".join(
            f"{', '.join(sorted(runs))} name them {list(names)}"
            for names, runs in sorted(named.items())
        )
        message = (
            f"The prediction runs name their model's keypoints in more than one "
            f"way: {ways}. A pool is scored through one keypoint map, so score "
            "runs that name them otherwise apart."
        )
        raise ScorePredictionsInvalid(message)
    (source_names,) = named
    try:
        return resolve_keypoint_map(
            target, source_names, overrides=given.keypoints, by_name=given.by_name
        )
    except ValueError as exc:
        raise ScorePredictionsInvalid(str(exc)) from exc


def _covering(
    truth: Sequence[TruthFrame],
    files: Mapping[str, Sequence[_PredictionFile]],
    *,
    origin: str,
    sets: Sequence[ResolvedSet],
) -> tuple[_PredictionFile, ...]:
    """Return the files that predict an annotated image.

    Raises:
        ScorePredictionsConflict: No file predicts an annotated image. The
            message names the origin of the predictions and of the labels.
    """
    annotated = {item.image for item in truth if item.origin_uuid == origin}
    covering = tuple(
        file
        for run_files in files.values()
        for file in run_files
        if not annotated.isdisjoint(file.images)
    )
    if covering:
        return covering
    message = (
        f"None of the annotated images is predicted by {', '.join(files)}. The "
        f"predictions name images of dataset {origin!r}, and the labels images of "
        f"{sorted({item.origin_uuid for item in sets})}. Predictions and labels must "
        "name images of one dataset, and a set from a linked library is predicted "
        "in that library."
    )
    raise ScorePredictionsConflict(message)


def _untouched(
    ds: Dataset,
    truth: Sequence[TruthFrame],
    *,
    origin: str,
    finished: Mapping[str, tuple[dict[str, str], ...]],
    scored: Mapping[str, Sequence[_PredictionFile]],
    keypoint_map: KeypointMap,
    tol_px: float,
) -> _Untouched:
    """Compare each truth object that a model placed with its source prediction.

    An object of a scored run is compared position by position through
    *keypoint_map*, the map that its run is scored through. An object of any
    other run is compared without a map. An object whose source run, image or
    instance cannot be read gets no flags, and this raises nothing for it.
    """
    flags: dict[UntouchedKey, tuple[bool, ...]] = {}
    sources: set[str] = set()
    opened: set[_ReadItem] = set()
    by_run: dict[str, dict[str, tuple[_PredictionFile, PredictedFrame]]] = {}
    for item in truth:
        for index, obj in enumerate(item.frame.objects):
            parsed = (
                parse_prediction_source_ref(obj.source_ref)
                if obj.origin == "model" and obj.source_ref is not None
                else None
            )
            if parsed is None or parsed[0] not in finished:
                continue
            source_run, instance = parsed
            sources.add(source_run)
            if item.origin_uuid != origin:
                continue
            frames = by_run.get(source_run)
            if frames is None:
                run_files = scored.get(source_run)
                if run_files is None:
                    run_files = _readable_files(ds, source_run, finished[source_run])
                frames = {
                    frame.image: (file, frame)
                    for file in run_files
                    for frame in file.predictions.frames
                }
                by_run[source_run] = frames
            found = frames.get(item.image)
            if found is None:
                continue
            file, frame = found
            opened.add(file.read_item)
            if instance >= len(frame.objects):
                continue
            source = frame.objects[instance].keypoints
            flags[(item.origin_uuid, item.image, index)] = (
                untouched_keypoints(
                    obj.keypoints, keypoint_map.apply(source), tol_px=tol_px
                )
                if source_run in scored
                else untouched_keypoints(
                    obj.keypoints, source, tol_px=tol_px, positional=False
                )
            )
    return _Untouched(flags=flags, sources=frozenset(sources), opened=frozenset(opened))


def _readable_files(
    ds: Dataset, run_id: str, rows: Sequence[Mapping[str, str]]
) -> list[_PredictionFile]:
    """Read every file of a source run that can be read, and skip the others."""
    files: list[_PredictionFile] = []
    for row in rows:
        try:
            files.append(_read_file(ds, run_id, row))
        except (OSError, ValueError):
            continue
    return files


def _already_scored(ds: Dataset, run_id: str) -> bool:
    """Return whether *run_id* has a finished row and every output on disk."""
    path = prediction_index_path(ds, _KIND)
    if not path.is_file():
        return False
    rows = index_records(score_predictions_index(path).read())
    return any(
        row.get("run_id", "") == run_id and score_run_is_complete(ds, row)
        for row in rows
    )


def _summary(
    params: ScorePredictionsParams, inputs: _ScoreInputs, metrics: PoseMetrics
) -> PoseScoreSummary:
    """Add the provenance of the score to its metrics."""
    model = inputs.model
    provenance: dict[str, object] = {
        "prediction_runs": list(params.prediction_runs),
        "model_kind": model.model_kind,
        "model_id": model.model_id,
        "model": list(model.model),
        "labels": [
            LabelRevision(
                origin_uuid=item.origin_uuid,
                set_key=item.set_key,
                revision=item.revision,
                digest=item.digest,
            )
            for item in inputs.sets
        ],
        "training_frames_known": inputs.excluded is not None,
        "training_frames_reason": inputs.training_frames_reason,
    }
    return PoseScoreSummary.model_validate({**metrics.model_dump(), **provenance})


@register_op
class ScorePredictionsOp(Op[ScorePredictionsParams]):
    """Score the predictions of one model's predict-frames runs against annotations."""

    kind = _KIND
    category = "convert"
    domain = "tracking"
    version = "0.1"
    writes_tracks = False
    scope_takes = "none"
    scope_dependent = False
    Params = ScorePredictionsParams

    def plan_identity(
        self,
        ds: Dataset,
        params: ScorePredictionsParams,
        scope: ResolvedScope,
        *,
        require_data: bool = True,
    ) -> OpIdentity:
        """Name the score from the content of everything that it reads.

        The revisions enter by content, the keypoint map as resolved, the
        frames that the model saw by digest, and each predictions file read by
        its digest. A prediction run without a finished row defers the identity
        when *require_data* is true and is refused otherwise.
        """
        _ = scope
        inputs = _score_inputs(ds, params, require_data=require_data)
        return OpIdentity(
            run_id=score_predictions_run_id(
                self.version,
                params,
                labels=inputs.labels,
                keypoint_map=inputs.keypoint_map,
                seen_digest=inputs.seen_digest,
                sources=inputs.untouched.sources,
                read=inputs.read,
            )
        )

    def run(
        self,
        ds: Dataset,
        params: ScorePredictionsParams,
        scope: ResolvedScope,
        overwrite: bool,
        ctx: JobContext,
    ) -> str:
        """Score the prediction runs against the revisions, and record the score.

        Without *overwrite*, a score whose finished row and outputs exist is
        reused. Every refusal comes before an output is written.
        """
        run_id = self.plan_identity(ds, params, scope, require_data=False).run_id
        ctx.set_run_id(run_id)
        if not overwrite and _already_scored(ds, run_id):
            ctx.cache_hit()
            return run_id

        run_root = prediction_kind_root(ds, _KIND) / run_id
        run_root.mkdir(parents=True, exist_ok=True)
        marker = claim_run_root(
            ds, ctx.execution_id, run_root, _KIND, _SCORE_IDLE_SECONDS
        )
        empty_claimed_run_root(run_root)
        inputs = _score_inputs(ds, params, require_data=False)
        scores = score_predictions(
            inputs.truth,
            inputs.predictions,
            predictions_origin=inputs.origin,
            keypoint_map=inputs.keypoint_map,
            seen=inputs.excluded,
            untouched=inputs.untouched.flags,
            params=inputs.metric_settings,
        )
        _ = renew_run_root_claim(run_root, marker, _KIND, _SCORE_IDLE_SECONDS)
        payload = score_predictions_payload(
            params,
            labels=inputs.labels,
            keypoint_map=inputs.keypoint_map,
            seen_digest=inputs.seen_digest,
            sources=inputs.untouched.sources,
            read=inputs.read,
        )
        summary = _summary(params, inputs, scores.metrics)
        _ = write_parquet_atomic(scores.frames, run_root / _FRAMES_TABLE)
        _ = write_parquet_atomic(scores.instances, run_root / _INSTANCES_TABLE)
        text = json.dumps(
            summary.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        )
        atomic_write(run_root / _SUMMARY, lambda target: target.write_text(text))
        write_run_params(run_root, payload)
        write_identity_scheme(run_root, OP_IDENTITY_SCHEME)

        metrics = scores.metrics
        index = score_predictions_index(prediction_index_path(ds, _KIND))
        index.append(
            [
                ScoreRunRow(
                    run_id=run_id,
                    abs_path=Path(ds.relative_to_root(run_root)),
                    prediction_runs=json.dumps(list(params.prediction_runs)),
                    model=json.dumps(list(inputs.model.model)),
                    model_kind=inputs.model.model_kind,
                    model_id=inputs.model.model_id,
                    consumed_sets=consumed_sets_cell(inputs.sets),
                    pose_id=str(inputs.pose.id),
                    n_scored=metrics.n_scored,
                    n_excluded_training=metrics.n_excluded_training,
                    n_not_covered=metrics.n_not_covered,
                    training_frames_known=summary.training_frames_known,
                    params_hash=hash_params(payload),
                )
            ]
        )
        index.mark_finished(run_id)
        print(
            f"[{_KIND}] completed run_id={run_id}: {metrics.n_scored} of "
            f"{metrics.n_truth_frames} annotated frames scored -> {run_root}"
        )
        return run_id
