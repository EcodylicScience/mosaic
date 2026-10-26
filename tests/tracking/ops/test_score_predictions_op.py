"""Checks the params, the errors, the run identity and the run of score-predictions.

The params store the runs and the alphas sorted, and two orders of either name
one run. The run identity contains the revisions' content and the map as
resolved, and leaves out an empty term. A run reads its identity's terms from
disk, refuses what cannot be scored before it writes, and writes two tables, a
summary, its identity payload and one index row.
"""

from __future__ import annotations

import datetime
import json
from collections.abc import Callable, Collection, Mapping, Sequence
from pathlib import Path
from typing import Final

import pandas as pd
import pytest
from pydantic import TypeAdapter, ValidationError

import mosaic.tracking.ops.score_predictions as score_predictions
from mosaic.core.annotations.bbox import BboxPolicy
from mosaic.core.annotations.keypoint_map import KeypointMap, resolve_keypoint_map
from mosaic.core.annotations.model import Keypoint, KeypointSchema
from mosaic.core.annotations.pose_annotations import (
    PoseDefinition,
    PoseFrame,
    PoseObject,
)
from mosaic.core.annotations.pose_predictions import (
    FramePredictions,
    PredictedFrame,
    PredictedObject,
    PredictedPose,
)
from mosaic.core.annotations.projection import write_keypoint_set_revision
from mosaic.core.annotations.scoring import PoseScoreParams, PoseScoreSummary
from mosaic.core.annotations.split import VideoFrame, frame_key
from mosaic.core.dataset import Dataset
from mosaic.core.helpers import entry_camera_path
from mosaic.core.pipeline._utils import hash_params
from mosaic.core.pipeline.error_envelope import error_envelope
from mosaic.core.pipeline.error_kind import ClassifiedError, ErrorKind
from mosaic.core.pipeline.file_digest import file_digest
from mosaic.core.pipeline.graph import Recipe, plan_pipeline
from mosaic.core.pipeline.identity_scheme import MARKER_NAME, read_identity_scheme
from mosaic.core.pipeline.index_csv import index_records
from mosaic.core.pipeline.inventory.model import PredictionScoreRef
from mosaic.core.pipeline.markers import (
    INFLIGHT_MARKER_NAME,
    new_inflight,
    parse_instant,
    read_inflight,
    write_inflight,
)
from mosaic.core.pipeline.models import SPLIT_RECORD_FILENAME, model_run_root
from mosaic.core.pipeline.op_identity import (
    OP_IDENTITY_SCHEME,
    op_run_id,
    parse_op_run_id,
)
from mosaic.core.pipeline.ops import OPS, IdentityDeferred, op_resource_class, run_op
from mosaic.runlog import RunLogSnapshot, read_run, run_log_dir
from mosaic.tracking import register_ops
from mosaic.tracking.frame_predictions.errors import (
    PredictionRunNotFound,
    ScorePredictionsConflict,
    ScorePredictionsInvalid,
)
from mosaic.tracking.frame_predictions.index import (
    prediction_index_path,
    prediction_kind_root,
    score_predictions_index,
    write_run_params,
)
from mosaic.tracking.frame_predictions.params import PredictFramesParams
from mosaic.tracking.ops._common import (
    RunRootHeld,
    empty_claimed_run_root,
    resolve_training_data,
)
from mosaic.tracking.ops.predict_frames import predict_frames_payload
from mosaic.tracking.ops.prepare import PrepareTrainingDataParams, check_preparation
from mosaic.tracking.ops.score_predictions import (
    ScorePredictionsParams,
    pose_score_params,
    score_predictions_run_id,
)
from mosaic.tracking.training_provenance import SeenFrames
from tests.helpers import (
    MOUSE,
    extract_entry_camera,
    make_dataset,
    pose_frame,
    pose_object,
    pose_set,
    record_entry_predictions,
    register_trained_model,
)

register_ops()

VERSION: Final = "0.1"
PREDICTION_RUN: Final = "predict-frames.0.1-abcdef0123"
OTHER_PREDICTION_RUN: Final = "predict-frames.0.1-0123456789"

SCHEMA: Final = KeypointSchema(names=("nose", "neck", "tail_base"))
IDENTITY_MAP: Final = resolve_keypoint_map(SCHEMA, SCHEMA.names)
INDEX_MAP: Final = resolve_keypoint_map(
    SCHEMA,
    ("kp0", "kp1", "kp2"),
    overrides={"nose": 0, "neck": 1, "tail_base": 2},
    by_name=False,
)

LABELS: Final = (("origin-a", "set-a", "digest-a"),)
READ: Final = ((PREDICTION_RUN, "group-a", "sequence-a", "", "sha-a"),)

ReadItem = tuple[str, str, str, str, str]


def _required() -> dict[str, object]:
    return {"prediction_runs": [PREDICTION_RUN], "labels": [{"set_key": "set-a"}]}


def _params(**overrides: object) -> ScorePredictionsParams:
    return ScorePredictionsParams.model_validate({**_required(), **overrides})


def _run_id(
    params: ScorePredictionsParams | None = None,
    *,
    labels: Sequence[tuple[str, str, str]] = LABELS,
    keypoint_map: KeypointMap = IDENTITY_MAP,
    seen_digest: str | None = None,
    sources: Collection[str] = (),
    read: Collection[ReadItem] = READ,
) -> str:
    return score_predictions_run_id(
        VERSION,
        params or _params(),
        labels=labels,
        keypoint_map=keypoint_map,
        seen_digest=seen_digest,
        sources=sources,
        read=read,
    )


@pytest.fixture
def payloads(monkeypatch: pytest.MonkeyPatch) -> list[Mapping[str, object]]:
    """Record each payload that the run id is taken over."""
    recorded: list[Mapping[str, object]] = []

    def record(kind: str, version: str, payload: Mapping[str, object]) -> str:
        recorded.append(payload)
        return f"{kind}.{version}-0000000000"

    monkeypatch.setattr(score_predictions, "op_run_id", record)
    return recorded


# --- Errors -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("error", "kind", "base"),
    [
        (PredictionRunNotFound, "not_found", LookupError),
        (ScorePredictionsInvalid, "invalid", ValueError),
        (ScorePredictionsConflict, "conflict", ValueError),
    ],
)
def test_each_error_declares_its_kind(
    error: type[ClassifiedError], kind: ErrorKind, base: type[Exception]
) -> None:
    raised = error("the message")

    assert error.error_kind == kind
    assert isinstance(raised, base)
    envelope = error_envelope(raised)
    assert (envelope.kind, envelope.message) == (kind, "the message")


# --- Params -----------------------------------------------------------------------


def test_every_optional_field_has_its_default() -> None:
    params = _params()

    assert params.pose is None
    assert params.keypoint_map.keypoints == {}
    assert params.keypoint_map.by_name is True
    assert params.exclude_seen is True
    assert params.match_radius == 0.5
    assert params.match_radius_px is None
    assert params.duplicate_radius == 0.2
    assert (params.center, params.head, params.tail) == (None, None, None)
    assert params.oks_sigma == 0.025
    assert params.oks_sigmas == {}
    assert params.pck_alphas == (0.05, 0.1, 0.2)
    assert params.min_instance_score is None
    assert params.min_keypoint_score is None
    assert params.untouched_tol_px == 0.01
    assert params.n_boot == 1000
    assert params.confidence == 0.95
    assert params.random_state == 0


def test_every_field_enters_the_identity() -> None:
    assert set(_params().identity_dump()) == set(ScorePredictionsParams.model_fields)


@pytest.mark.parametrize("field", ["prediction_runs", "labels"])
def test_the_prediction_runs_and_the_labels_are_required(field: str) -> None:
    given = _required()
    del given[field]

    with pytest.raises(ValidationError) as caught:
        _ = ScorePredictionsParams.model_validate(given)

    assert [(error["loc"], error["type"]) for error in caught.value.errors()] == [
        ((field,), "missing")
    ]


@pytest.mark.parametrize("field", ["prediction_runs", "labels"])
def test_empty_prediction_runs_or_labels_are_refused(field: str) -> None:
    with pytest.raises(ValidationError) as caught:
        _ = _params(**{field: ()})

    assert [(error["loc"], error["type"]) for error in caught.value.errors()] == [
        ((field,), "too_short")
    ]


def test_the_prediction_runs_are_stored_sorted() -> None:
    params = _params(prediction_runs=[PREDICTION_RUN, OTHER_PREDICTION_RUN])

    assert params.prediction_runs == (OTHER_PREDICTION_RUN, PREDICTION_RUN)
    assert _run_id(params) == _run_id(
        _params(prediction_runs=[OTHER_PREDICTION_RUN, PREDICTION_RUN])
    )


def test_a_repeated_prediction_run_is_refused() -> None:
    with pytest.raises(ValidationError, match=PREDICTION_RUN):
        _ = _params(
            prediction_runs=[PREDICTION_RUN, OTHER_PREDICTION_RUN, PREDICTION_RUN]
        )


@pytest.mark.parametrize(
    "alphas",
    [(0.1, 0.05, 0.1), (0.0,), (-0.05,), (1.5,), (0.00015,), (0.12345,), ()],
    ids=["repeat", "zero", "negative", "above-one", "finer", "five-decimals", "none"],
)
def test_a_pck_alpha_set_that_names_no_exact_column_is_refused(
    alphas: tuple[float, ...],
) -> None:
    with pytest.raises(ValidationError, match="pck_alphas"):
        _ = _params(pck_alphas=alphas)


@pytest.mark.parametrize(
    "alphas", [(1.0,), (0.0001,), (0.3,), (0.05, 0.1, 0.2)], ids=str
)
def test_a_pck_alpha_in_steps_of_one_ten_thousandth_is_kept(
    alphas: tuple[float, ...],
) -> None:
    assert _params(pck_alphas=alphas).pck_alphas == alphas


def test_unsorted_pck_alphas_are_stored_sorted_with_the_sorted_identity() -> None:
    unsorted = _params(pck_alphas=(0.2, 0.05, 0.1))
    ordered = _params(pck_alphas=(0.05, 0.1, 0.2))

    assert unsorted.pck_alphas == (0.05, 0.1, 0.2)
    assert unsorted.identity_dump() == ordered.identity_dump()
    assert _run_id(unsorted) == _run_id(ordered)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("match_radius", 0.0),
        ("match_radius_px", 0.0),
        ("match_radius_px", -1.0),
        ("duplicate_radius", 0.0),
        ("oks_sigma", 0.0),
        ("oks_sigmas", {"nose": 0.0}),
        ("untouched_tol_px", -0.01),
        ("n_boot", 0),
        ("confidence", 0.0),
        ("confidence", 1.0),
    ],
)
def test_a_value_outside_its_range_is_refused(field: str, value: object) -> None:
    with pytest.raises(ValidationError, match=field):
        _ = _params(**{field: value})


# --- Metric settings --------------------------------------------------------------


def test_the_metric_settings_are_resolved_against_the_target_pose() -> None:
    params = _params(
        center="neck",
        head=0,
        tail="tail_base",
        match_radius=0.75,
        match_radius_px=12.0,
        duplicate_radius=0.3,
        oks_sigma=0.03,
        oks_sigmas={"neck": 0.05},
        pck_alphas=(0.1, 0.25),
        min_instance_score=0.4,
        min_keypoint_score=0.2,
        untouched_tol_px=0.5,
        n_boot=50,
        confidence=0.9,
        random_state=7,
    )

    resolved = pose_score_params(params, SCHEMA, BboxPolicy())

    assert resolved == PoseScoreParams(
        center=1,
        head=0,
        tail=2,
        match_radius=0.75,
        match_radius_px=12.0,
        duplicate_radius=0.3,
        oks_sigmas=(0.03, 0.05, 0.03),
        pck_alphas=(0.1, 0.25),
        min_instance_score=0.4,
        min_keypoint_score=0.2,
        n_boot=50,
        confidence=0.9,
        random_state=7,
    )


def test_an_oks_sigmas_key_naming_no_target_keypoint_is_refused() -> None:
    params = _params(oks_sigmas={"neck": 0.05, "ear": 0.04})

    with pytest.raises(ValueError, match="oks_sigmas") as raised:
        _ = pose_score_params(params, SCHEMA, BboxPolicy())

    assert "'ear'" in str(raised.value)
    assert "['nose', 'neck', 'tail_base']" in str(raised.value)


@pytest.mark.parametrize(
    ("given", "policy", "expected"),
    [
        ({}, BboxPolicy(head_index=0, tail_index=2), (0, 2)),
        ({"head": "neck"}, BboxPolicy(head_index=0, tail_index=2), (1, 2)),
        ({"tail": 1}, BboxPolicy(head_index=0, tail_index=2), (0, 1)),
        ({}, BboxPolicy(), (None, None)),
    ],
    ids=["policy", "own-head", "own-tail", "neither"],
)
def test_head_and_tail_fall_back_to_the_sets_bbox_policy(
    given: dict[str, object],
    policy: BboxPolicy,
    expected: tuple[int | None, int | None],
) -> None:
    resolved = pose_score_params(_params(**given), SCHEMA, policy)

    assert (resolved.head, resolved.tail) == expected


@pytest.mark.parametrize("field", ["center", "head", "tail"])
@pytest.mark.parametrize("reference", ["ear", 3], ids=["name", "index"])
def test_a_reference_to_no_target_keypoint_is_refused(
    field: str, reference: str | int
) -> None:
    with pytest.raises(ValueError, match=field):
        _ = pose_score_params(_params(**{field: reference}), SCHEMA, BboxPolicy())


# --- Run identity -----------------------------------------------------------------


def test_the_run_id_names_the_kind_and_the_version() -> None:
    parsed = parse_op_run_id(_run_id())

    assert parsed is not None
    assert (parsed.kind, parsed.version) == ("score-predictions", VERSION)


def test_an_empty_term_is_left_out_of_the_payload(
    payloads: list[Mapping[str, object]],
) -> None:
    """The identity map, no seen digest and no sources each leave their key out."""
    _ = _run_id()

    (payload,) = payloads
    assert set(payload) == {"params", "labels", "read"}


def test_the_labels_and_the_map_leave_the_params(
    payloads: list[Mapping[str, object]],
) -> None:
    _ = _run_id()

    (payload,) = payloads
    dumped = payload["params"]
    assert isinstance(dumped, dict)
    assert "labels" not in dumped
    assert "keypoint_map" not in dumped
    assert "prediction_runs" in dumped


def test_a_mapped_payload_records_the_map_as_resolved(
    payloads: list[Mapping[str, object]],
) -> None:
    _ = _run_id(keypoint_map=INDEX_MAP)

    (payload,) = payloads
    assert payload["keypoint_map"] == INDEX_MAP.identity_payload()


def test_a_seen_digest_and_sources_enter_the_payload(
    payloads: list[Mapping[str, object]],
) -> None:
    sources = ["predict-frames.0.1-ffffffffff", OTHER_PREDICTION_RUN]

    _ = _run_id(seen_digest="seen-digest", sources=sources)

    (payload,) = payloads
    assert payload["seen"] == "seen-digest"
    assert payload["sources"] == sorted(sources)


MOVED: Final[dict[str, Callable[[], str]]] = {
    "map": lambda: _run_id(keypoint_map=INDEX_MAP),
    "seen": lambda: _run_id(seen_digest="seen-digest"),
    "sources": lambda: _run_id(sources=[OTHER_PREDICTION_RUN]),
    "label-digest": lambda: _run_id(labels=(("origin-a", "set-a", "digest-b"),)),
    "read-digest": lambda: _run_id(
        read=((PREDICTION_RUN, "group-a", "sequence-a", "", "sha-b"),)
    ),
    "read-entry": lambda: _run_id(
        read=(*READ, (PREDICTION_RUN, "group-a", "sequence-b", "", "sha-c"))
    ),
}
"""A run id taken with one term changed from the defaults of :func:`_run_id`."""


@pytest.mark.parametrize("moved", MOVED.values(), ids=MOVED.keys())
def test_each_term_moves_the_run_id(moved: Callable[[], str]) -> None:
    assert moved() != _run_id()


def test_a_revision_number_and_a_map_spelling_leave_the_run_id() -> None:
    """A revision number selects, and the map enters as resolved."""
    spelled = _params(
        labels=[{"set_key": "set-a", "revision": 3}],
        keypoint_map={"keypoints": {"nose": "nose"}, "by_name": True},
    )

    assert _run_id(spelled) == _run_id()


def test_the_order_of_labels_sources_and_reads_leaves_the_run_id() -> None:
    labels = [("origin-a", "set-a", "digest-a"), ("origin-b", "set-b", "digest-b")]
    sources = [PREDICTION_RUN, OTHER_PREDICTION_RUN]
    read = [*READ, (PREDICTION_RUN, "group-a", "sequence-b", "", "sha-c")]

    forward = _run_id(labels=labels, sources=sources, read=read)

    assert forward == _run_id(
        labels=labels[::-1], sources=sources[::-1], read=read[::-1]
    )


# --- The op -----------------------------------------------------------------------


KIND: Final = "score-predictions"
GROUP: Final = "g"
FRAMES_RUN: Final = "uniform-0123456789"
OTHER_FRAMES_RUN: Final = "uniform-9876543210"
SOURCE_RUN: Final = "predict-frames.0.1-5555555555"
MODEL: Final = "train-pose.0.1-0123456789"
OTHER_MODEL: Final = "train-pose.0.1-9999999999"
NOSE: Final = (10.0, 20.0)
TAIL: Final = (30.0, 20.0)
ELSEWHERE: Final = (50.0, 40.0)
NAMED: Final = ("nose", "tail")
INDEXED: Final = ("kp0", "kp1")

FLY: Final = PoseDefinition(id=4, name="fly", schema=KeypointSchema(names=NAMED))
"""A second pose with the mouse's keypoint names, which a set of mice cannot share."""

Point = tuple[float, float]

_JSON_OBJECT: Final = TypeAdapter(dict[str, object])

_LONG_AGO: Final = "2020-01-01T00:00:00+00:00"
"""A claim expiry far enough in the past that no clock skew reaches it."""


def _image(sequence: str, frames_run: str = FRAMES_RUN) -> str:
    """The dataset-relative image that an extraction run wrote for *sequence*."""
    root = Path("media/frames/uniform") / frames_run
    return (
        entry_camera_path(root, GROUP, sequence, "") / "frame_000004.png"
    ).as_posix()


def _frame(
    sequence: str,
    *objects: PoseObject,
    frames_run: str = FRAMES_RUN,
    frame_index: int = 4,
) -> PoseFrame:
    return pose_frame(
        _image(sequence, frames_run),
        *objects,
        sequence=sequence,
        frame_index=frame_index,
    )


def _accepted(*points: Point, source: str) -> PoseObject:
    """An object that the annotator accepted from the prediction *source*."""
    return pose_object(*points, origin="model", source_ref=source)


def _save(
    ds: Dataset,
    set_key: str,
    *frames: PoseFrame,
    poses: tuple[PoseDefinition, ...] = (MOUSE,),
) -> Path:
    """Save *frames* as the next revision of *set_key*, and return its file."""
    saved = write_keypoint_set_revision(
        ds,
        set_key=set_key,
        annotations=pose_set(frames, poses=poses, image_root=ds.base_dir),
        origin={},
    )
    return saved.path


def _predicted(*points: Point) -> PredictedObject:
    return PredictedObject(
        keypoints=tuple(Keypoint(x=x, y=y, visibility=2, score=0.8) for x, y in points),
        score=0.9,
    )


def _record(
    ds: Dataset,
    *objects: PredictedObject,
    run_id: str = PREDICTION_RUN,
    sequence: str = "s1",
    frames_run: str = FRAMES_RUN,
    names: tuple[str, ...] = NAMED,
    model: str = MODEL,
    finished: bool = True,
) -> Path:
    """Record one entry camera of a prediction run: one frame holding *objects*."""
    predictions = FramePredictions(
        pose=PredictedPose(id=MOUSE.id, name=MOUSE.name, schema=KeypointSchema(names)),
        frames_run_id=frames_run,
        group=GROUP,
        sequence=sequence,
        camera="",
        frames=(
            PredictedFrame(
                image=_image(sequence, frames_run),
                frame_index=4,
                width=64,
                height=48,
                objects=objects,
            ),
        ),
    )
    return record_entry_predictions(
        ds, predictions, run_id=run_id, model=model, finished=finished
    )


def _settle(
    ds: Dataset, run_id: str, *, frames_run: str = FRAMES_RUN, **values: object
) -> None:
    """Write the run_params.json that a predict-frames run of MODEL writes."""
    params = PredictFramesParams.model_validate(
        {"model": [MODEL], "frames_run_id": frames_run, **values}
    )
    payload = predict_frames_payload(params, model_id=MODEL, model_kind="train-pose")
    write_run_params(prediction_kind_root(ds, "predict-frames") / run_id, payload)


def _given(
    runs: Sequence[str] = (PREDICTION_RUN,),
    labels: Sequence[str] = ("set-a",),
    **overrides: object,
) -> dict[str, object]:
    return {
        "prediction_runs": list(runs),
        "labels": [{"set_key": key} for key in labels],
        **overrides,
    }


def _score(
    ds: Dataset,
    runs: Sequence[str] = (PREDICTION_RUN,),
    labels: Sequence[str] = ("set-a",),
    *,
    execution_id: str | None = None,
    overwrite: bool = False,
    **overrides: object,
) -> str:
    return run_op(
        ds,
        KIND,
        _given(runs, labels, **overrides),
        execution_id=execution_id,
        overwrite=overwrite,
    )


def _plan(
    ds: Dataset, params: ScorePredictionsParams, *, require_data: bool = True
) -> str:
    op = OPS[KIND]()
    scope = ds.resolve_scope(None)
    return op.plan_identity(ds, params, scope, require_data=require_data).run_id


def _run_root(ds: Dataset, run_id: str) -> Path:
    return prediction_kind_root(ds, KIND) / run_id


def _frames(ds: Dataset, run_id: str) -> pd.DataFrame:
    return pd.read_parquet(_run_root(ds, run_id) / "frames.parquet")


def _summary(ds: Dataset, run_id: str) -> PoseScoreSummary:
    text = (_run_root(ds, run_id) / "summary.json").read_bytes()
    return PoseScoreSummary.model_validate_json(text)


def _row(ds: Dataset, run_id: str) -> dict[str, str]:
    index = score_predictions_index(prediction_index_path(ds, KIND))
    (row,) = [
        record for record in index_records(index.read()) if record["run_id"] == run_id
    ]
    return row


def _snapshot(ds: Dataset, execution_id: str) -> RunLogSnapshot:
    found = read_run(run_log_dir(ds.base_dir), execution_id)
    assert found is not None
    return found


def _tree(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _walk_returning(
    monkeypatch: pytest.MonkeyPatch, seen: SeenFrames
) -> list[tuple[str, tuple[str, ...]]]:
    """Make the seen walk return *seen*, and record each walk's model kind and runs."""
    calls: list[tuple[str, tuple[str, ...]]] = []

    def walk(ds: Dataset, kind: str, run_ids: Sequence[str]) -> SeenFrames:
        _ = ds
        calls.append((kind, tuple(run_ids)))
        return seen

    monkeypatch.setattr(score_predictions, "seen_frames_of_model", walk)
    return calls


@pytest.fixture
def ds(tmp_path: Path) -> Dataset:
    """One annotated frame of ``s1``, predicted where it is annotated."""
    dataset = make_dataset(tmp_path / "ds")
    _ = _save(dataset, "set-a", _frame("s1", pose_object(NOSE, TAIL)))
    _ = _record(dataset, _predicted(NOSE, TAIL))
    return dataset


def test_the_op_declares_a_scope_free_conversion() -> None:
    op = OPS[KIND]

    assert op.category == "convert"
    assert op.domain == "tracking"
    assert op.version == VERSION
    assert op.writes_tracks is False
    assert op.scope_takes == "none"
    assert op.scope_dependent is False
    assert op.model_reference is None
    assert op_resource_class(KIND) == "cpu"


# --- Identity ---------------------------------------------------------------------


def test_the_run_id_names_revisions_map_seen_frames_sources_and_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the predictions file that holds an annotated image is read."""
    dataset = make_dataset(tmp_path / "ds")
    accepted = _accepted(NOSE, TAIL, source=f"{PREDICTION_RUN}#0")
    revision = _save(dataset, "set-a", _frame("s1", accepted))
    covering = _record(dataset, _predicted(NOSE, TAIL), names=INDEXED)
    _ = _record(dataset, _predicted(NOSE, TAIL), sequence="s2", names=INDEXED)
    seen = SeenFrames(keys=frozenset({"elsewhere"}), complete=True, reason="")
    walks = _walk_returning(monkeypatch, seen)
    overrides = {"nose": "kp0", "tail": "kp1"}
    params = _params(keypoint_map={"keypoints": overrides})

    planned = _plan(dataset, params)

    assert walks == [("train-pose", (MODEL,))]
    assert planned == score_predictions_run_id(
        VERSION,
        params,
        labels=[(dataset.uuid or "", "set-a", file_digest(revision))],
        keypoint_map=resolve_keypoint_map(MOUSE.schema, INDEXED, overrides=overrides),
        seen_digest=hash_params({"keys": ["elsewhere"], "complete": True}),
        sources=[PREDICTION_RUN],
        read=[(PREDICTION_RUN, GROUP, "s1", "", file_digest(covering))],
    )


def test_a_plan_defers_until_a_prediction_run_is_finished(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path / "ds")
    _ = _save(dataset, "set-a", _frame("s1", pose_object(NOSE, TAIL)))

    with pytest.raises(IdentityDeferred):
        _ = _plan(dataset, _params())


@pytest.mark.parametrize("recorded", [True, False], ids=["unfinished", "no-index"])
def test_a_run_refuses_a_prediction_run_without_a_finished_row(
    tmp_path: Path, recorded: bool
) -> None:
    dataset = make_dataset(tmp_path / "ds")
    _ = _save(dataset, "set-a", _frame("s1", pose_object(NOSE, TAIL)))
    if recorded:
        _ = _record(dataset, _predicted(NOSE, TAIL), finished=False)

    with pytest.raises(PredictionRunNotFound, match=PREDICTION_RUN):
        _ = _plan(dataset, _params(), require_data=False)


def test_a_revision_that_is_not_indexed_defers_the_run_too(ds: Dataset) -> None:
    with pytest.raises(IdentityDeferred, match="set-z"):
        _ = _plan(ds, _params(labels=[{"set_key": "set-z"}]), require_data=False)


def test_a_finished_row_whose_file_is_gone_is_refused(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path / "ds")
    _ = _save(dataset, "set-a", _frame("s1", pose_object(NOSE, TAIL)))
    written = _record(dataset, _predicted(NOSE, TAIL))
    written.unlink()

    with pytest.raises(FileNotFoundError, match=PREDICTION_RUN) as caught:
        _ = _score(dataset)

    assert str(written) in str(caught.value)


def test_a_planned_step_names_its_score_run(ds: Dataset) -> None:
    recipe = Recipe.model_validate(
        {"steps": [{"id": "score", "type": "op", "kind": KIND, "params": _given()}]}
    )

    plan = plan_pipeline(ds, recipe)

    assert plan.step("score").artifact == PredictionScoreRef(
        run_id=_plan(ds, _params())
    )


# --- Outputs ----------------------------------------------------------------------


def test_a_run_writes_its_tables_summary_identity_payload_and_row(
    ds: Dataset,
) -> None:
    run_id = _score(ds)
    root = _run_root(ds, run_id)
    revision = ds.uuid or ""

    frames = _frames(ds, run_id)
    assert frames["status"].tolist() == ["scored"]
    assert frames["n_matched"].tolist() == [1]
    instances = pd.read_parquet(root / "instances.parquet")
    assert instances["matched"].tolist() == [True]
    summary = _summary(ds, run_id)
    assert summary.prediction_runs == [PREDICTION_RUN]
    assert (summary.model, summary.model_kind, summary.model_id) == (
        [MODEL],
        "train-pose",
        MODEL,
    )
    (label,) = summary.labels
    assert (label.origin_uuid, label.set_key, label.revision) == (revision, "set-a", 1)
    assert summary.keypoints == list(NAMED)
    assert (summary.n_scored, summary.recall) == (1, 1.0)
    payload = _JSON_OBJECT.validate_json((root / "run_params.json").read_bytes())
    assert op_run_id(KIND, VERSION, payload) == run_id
    assert read_identity_scheme(root) == OP_IDENTITY_SCHEME
    assert (root / MARKER_NAME).is_file()

    row = _row(ds, run_id)
    assert row["abs_path"] == ds.relative_to_root(root)
    assert json.loads(row["prediction_runs"]) == [PREDICTION_RUN]
    assert json.loads(row["model"]) == [MODEL]
    assert (row["model_kind"], row["model_id"]) == ("train-pose", MODEL)
    assert json.loads(row["consumed_sets"]) == [
        {
            "origin_uuid": revision,
            "set_key": "set-a",
            "revision": 1,
            "digest": label.digest,
        }
    ]
    assert row["pose_id"] == str(MOUSE.id)
    assert (row["n_scored"], row["n_excluded_training"], row["n_not_covered"]) == (
        "1",
        "0",
        "0",
    )
    assert row["params_hash"] == hash_params(payload)
    assert row["finished_at"]


def test_a_finished_run_with_its_three_outputs_is_reused(ds: Dataset) -> None:
    run_id = _score(ds, execution_id="exec-first")
    written = _tree(_run_root(ds, run_id))

    again = _score(ds, execution_id="exec-again")

    assert again == run_id
    assert _snapshot(ds, "exec-first")["cache_hit"] is False
    assert _snapshot(ds, "exec-again")["cache_hit"] is True
    assert _tree(_run_root(ds, run_id)) == written


def test_a_run_missing_an_output_is_scored_again(ds: Dataset) -> None:
    run_id = _score(ds, execution_id="exec-first")
    (_run_root(ds, run_id) / "summary.json").unlink()

    _ = _score(ds, execution_id="exec-again")

    assert _snapshot(ds, "exec-again")["cache_hit"] is False
    assert _summary(ds, run_id).n_scored == 1


def test_without_exclude_seen_no_walk_runs(
    ds: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    walks = _walk_returning(
        monkeypatch, SeenFrames(keys=frozenset(), complete=True, reason="")
    )

    run_id = _score(ds, exclude_seen=False)

    assert walks == []
    summary = _summary(ds, run_id)
    assert summary.training_frames_known is False
    assert summary.training_frames_reason == "exclude_seen is false"
    assert _row(ds, run_id)["training_frames_known"] == "False"


@pytest.mark.parametrize("complete", [True, False], ids=["complete", "incomplete"])
def test_only_a_complete_walk_excludes_the_frames_the_model_saw(
    ds: Dataset, monkeypatch: pytest.MonkeyPatch, complete: bool
) -> None:
    key = frame_key(
        VideoFrame(
            origin_uuid=ds.uuid or "",
            group=GROUP,
            sequence="s1",
            camera="",
            frame_index=4,
        )
    )
    reason = "" if complete else "the run recorded no data path"
    seen = SeenFrames(keys=frozenset({key}), complete=complete, reason=reason)
    _ = _walk_returning(monkeypatch, seen)

    run_id = _score(ds)

    status = "excluded_training" if complete else "scored"
    assert _frames(ds, run_id)["status"].tolist() == [status]
    summary = _summary(ds, run_id)
    assert summary.training_frames_known is complete
    assert summary.training_frames_reason == reason


SPLIT_UUID: Final = "00000000-0000-4000-8000-000000000001"
"""The ``uuid`` of the dataset whose preparation is scored, which enters every unit key.

Under it, the preparation places frames in every split.
"""


def test_the_frames_a_preparation_placed_to_train_or_validate_are_excluded(
    tmp_path: Path,
) -> None:
    """The scorer's frame keys are the keys that the preparation's split record names.

    A model trained on a real preparation of the scored set predicts every
    annotated image, and the walk of its training is not replaced.
    """
    dataset = make_dataset(tmp_path / "ds", uuid=SPLIT_UUID)
    images = extract_entry_camera(
        dataset,
        run_id=FRAMES_RUN,
        group=GROUP,
        sequence="s1",
        camera="",
        sizes={index: (64, 48) for index in range(100, 116)},
    )
    annotated = pose_object(NOSE, ELSEWHERE)
    _ = _save(
        dataset,
        "set-a",
        *(
            pose_frame(image, annotated, sequence="s1", frame_index=index)
            for index, image in images.items()
        ),
    )
    prepared = run_op(
        dataset,
        "prepare-training-data",
        {
            "sets": [{"set_key": "set-a"}],
            "holdout": [],
            "pose": MOUSE.name,
            "target": "yolo-pose",
            "split_by": "frame",
            "split": (0.1, 0.1, 0.8),
        },
    )
    weights = model_run_root(dataset, "train-pose", MODEL) / "best.pt"
    weights.parent.mkdir(parents=True)
    _ = weights.write_bytes(b"weights")
    register_trained_model(
        dataset,
        "train-pose",
        MODEL,
        weights,
        data_path=resolve_training_data(dataset, prepared),
    )
    predictions = FramePredictions(
        pose=PredictedPose(id=MOUSE.id, name=MOUSE.name, schema=MOUSE.schema),
        frames_run_id=FRAMES_RUN,
        group=GROUP,
        sequence="s1",
        camera="",
        frames=tuple(
            PredictedFrame(
                image=image.as_posix(),
                frame_index=index,
                width=64,
                height=48,
                objects=(_predicted(NOSE, ELSEWHERE),),
            )
            for index, image in images.items()
        ),
    )
    _ = record_entry_predictions(
        dataset, predictions, run_id=PREDICTION_RUN, model=MODEL
    )
    record = model_run_root(dataset, "prepare-training-data", prepared)
    placed = index_records(
        pd.read_csv(
            record / SPLIT_RECORD_FILENAME, dtype="string", keep_default_na=False
        )
    )

    run_id = _score(dataset, exclude_seen=True)

    assert {row["split"] for row in placed} == {"train", "valid", "test"}
    frames = _frames(dataset, run_id)
    assert dict(zip(frames["frame_key"], frames["status"], strict=True)) == {
        row["frame_key"]: "scored" if row["split"] == "test" else "excluded_training"
        for row in placed
    }
    assert _summary(dataset, run_id).training_frames_known is True


def test_a_frame_predicted_after_a_score_names_a_new_score(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path / "ds")
    annotated = pose_object(NOSE, TAIL)
    _ = _save(dataset, "set-a", _frame("s1", annotated), _frame("s2", annotated))
    _ = _record(dataset, _predicted(NOSE, TAIL))
    first = _score(dataset)
    assert _summary(dataset, first).n_not_covered == 1

    _ = _record(dataset, _predicted(NOSE, TAIL), sequence="s2")
    second = _score(dataset)

    assert second != first
    assert _summary(dataset, second).n_not_covered == 0


def test_a_frame_predicted_during_a_score_stays_out_of_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The run id, the tables and the identity payload come from one read."""
    dataset = make_dataset(tmp_path / "ds")
    annotated = pose_object(NOSE, TAIL)
    _ = _save(dataset, "set-a", _frame("s1", annotated), _frame("s2", annotated))
    _ = _record(dataset, _predicted(NOSE, TAIL))

    def empty_then_predict(run_root: Path) -> None:
        empty_claimed_run_root(run_root)
        _ = _record(dataset, _predicted(NOSE, TAIL), sequence="s2")

    monkeypatch.setattr(score_predictions, "empty_claimed_run_root", empty_then_predict)

    run_id = _score(dataset)

    root = _run_root(dataset, run_id)
    payload = _JSON_OBJECT.validate_json((root / "run_params.json").read_bytes())
    assert op_run_id(KIND, VERSION, payload) == run_id
    assert _row(dataset, run_id)["params_hash"] == hash_params(payload)
    assert _summary(dataset, run_id).n_not_covered == 1
    assert _frames(dataset, run_id)["status"].tolist() == ["scored", "not_covered"]


def test_a_score_whose_root_a_peer_took_over_writes_nothing(
    ds: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A score that outlives its claim leaves the root to the execution that took it."""
    peer = new_inflight(
        execution_id="exec-peer",
        host="elsewhere",
        pid=1,
        phase=None,
        idle_seconds=600.0,
    )

    def empty_then_lose(run_root: Path) -> None:
        empty_claimed_run_root(run_root)
        write_inflight(run_root, peer)

    monkeypatch.setattr(score_predictions, "empty_claimed_run_root", empty_then_lose)

    with pytest.raises(RunRootHeld, match="exec-peer"):
        _ = _score(ds)

    root = _run_root(ds, _plan(ds, _params()))
    assert [path.name for path in root.iterdir()] == [INFLIGHT_MARKER_NAME]
    assert not prediction_index_path(ds, KIND).is_file()


def test_a_score_renews_its_claim_before_it_writes(
    ds: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A score that outlives its claim, with no peer in the root, writes its outputs."""

    def empty_then_go_stale(run_root: Path) -> None:
        empty_claimed_run_root(run_root)
        held = read_inflight(run_root)
        assert held is not None
        write_inflight(run_root, held.model_copy(update={"expires_at": _LONG_AGO}))

    monkeypatch.setattr(
        score_predictions, "empty_claimed_run_root", empty_then_go_stale
    )
    started = datetime.datetime.now(datetime.timezone.utc)

    run_id = _score(ds)

    held = read_inflight(_run_root(ds, run_id))
    assert held is not None
    expires = parse_instant(held.expires_at)
    assert expires is not None
    assert expires > started
    assert _summary(ds, run_id).n_scored == 1


def test_two_runs_of_one_model_are_scored_as_one_pool(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path / "ds")
    annotated = pose_object(NOSE, TAIL)
    _ = _save(
        dataset,
        "set-a",
        _frame("s1", annotated),
        _frame("s2", annotated, frames_run=OTHER_FRAMES_RUN),
    )
    _ = _record(dataset, _predicted(NOSE, TAIL))
    _ = _record(
        dataset,
        _predicted(NOSE, TAIL),
        run_id=OTHER_PREDICTION_RUN,
        sequence="s2",
        frames_run=OTHER_FRAMES_RUN,
    )
    _settle(dataset, PREDICTION_RUN)
    _settle(dataset, OTHER_PREDICTION_RUN, frames_run=OTHER_FRAMES_RUN)

    run_id = _score(dataset, [PREDICTION_RUN, OTHER_PREDICTION_RUN])

    summary = _summary(dataset, run_id)
    assert summary.prediction_runs == [OTHER_PREDICTION_RUN, PREDICTION_RUN]
    assert (summary.n_scored, summary.n_matched) == (2, 2)
    assert json.loads(_row(dataset, run_id)["prediction_runs"]) == [
        OTHER_PREDICTION_RUN,
        PREDICTION_RUN,
    ]


@pytest.fixture
def pooled(tmp_path: Path) -> Dataset:
    """Annotated frames of ``s1`` and ``s2``, and two runs that predict one each."""
    dataset = make_dataset(tmp_path / "ds")
    annotated = pose_object(NOSE, TAIL)
    _ = _save(dataset, "set-a", _frame("s1", annotated), _frame("s2", annotated))
    _ = _record(dataset, _predicted(NOSE, TAIL))
    return dataset


def test_runs_of_two_models_are_refused(pooled: Dataset) -> None:
    _ = _record(
        pooled,
        _predicted(NOSE, TAIL),
        run_id=OTHER_PREDICTION_RUN,
        sequence="s2",
        model=OTHER_MODEL,
    )

    with pytest.raises(ScorePredictionsInvalid) as caught:
        _ = _score(pooled, [PREDICTION_RUN, OTHER_PREDICTION_RUN])

    message = str(caught.value)
    for named in (PREDICTION_RUN, OTHER_PREDICTION_RUN, MODEL, OTHER_MODEL):
        assert named in message
    assert "train-pose" in message


def test_runs_at_two_operating_points_are_refused(pooled: Dataset) -> None:
    _ = _record(
        pooled, _predicted(NOSE, TAIL), run_id=OTHER_PREDICTION_RUN, sequence="s2"
    )
    _settle(pooled, PREDICTION_RUN, conf_threshold=0.25)
    _settle(pooled, OTHER_PREDICTION_RUN, conf_threshold=0.5)

    with pytest.raises(ScorePredictionsInvalid) as caught:
        _ = _score(pooled, [PREDICTION_RUN, OTHER_PREDICTION_RUN])

    message = str(caught.value)
    for named in (PREDICTION_RUN, OTHER_PREDICTION_RUN, "conf_threshold=0.25"):
        assert named in message
    assert "conf_threshold=0.5" in message
    assert "frames_run_id" not in message
    assert "one operating point" in message


def test_an_image_that_two_runs_predict_is_refused(pooled: Dataset) -> None:
    _ = _record(pooled, _predicted(NOSE, TAIL), run_id=OTHER_PREDICTION_RUN)
    _settle(pooled, PREDICTION_RUN)
    _settle(pooled, OTHER_PREDICTION_RUN)

    with pytest.raises(ScorePredictionsConflict) as caught:
        _ = _score(pooled, [PREDICTION_RUN, OTHER_PREDICTION_RUN])

    message = str(caught.value)
    for named in (_image("s1"), PREDICTION_RUN, OTHER_PREDICTION_RUN):
        assert named in message


def test_runs_that_name_their_keypoints_apart_are_refused(pooled: Dataset) -> None:
    """One map scores a pool, and a map is resolved from the names it maps."""
    _ = _record(
        pooled,
        _predicted(NOSE, TAIL),
        run_id=OTHER_PREDICTION_RUN,
        sequence="s2",
        names=INDEXED,
    )
    _settle(pooled, PREDICTION_RUN)
    _settle(pooled, OTHER_PREDICTION_RUN)

    with pytest.raises(ScorePredictionsInvalid) as caught:
        _ = _score(pooled, [PREDICTION_RUN, OTHER_PREDICTION_RUN])

    message = str(caught.value)
    for named in (PREDICTION_RUN, OTHER_PREDICTION_RUN, str(list(INDEXED))):
        assert named in message
    assert str(list(NAMED)) in message


# --- Refusals of the labels -------------------------------------------------------


def test_revisions_of_two_poses_are_refused_as_prepare_refuses_them(
    ds: Dataset,
) -> None:
    _ = _save(
        ds, "set-b", _frame("s2", pose_object(NOSE, TAIL, pose_id=FLY.id)), poses=(FLY,)
    )
    labels = ("set-a", "set-b")
    prepare = PrepareTrainingDataParams.model_validate(
        {"sets": [{"set_key": key} for key in labels]}
    )
    with pytest.raises(ValueError) as prepared:
        check_preparation(ds, prepare)

    with pytest.raises(ScorePredictionsInvalid) as scored:
        _ = _score(ds, labels=labels)

    assert str(scored.value) == str(prepared.value)


def test_a_keypoint_map_that_maps_nothing_is_refused_with_its_message(
    tmp_path: Path,
) -> None:
    dataset = make_dataset(tmp_path / "ds")
    _ = _save(dataset, "set-a", _frame("s1", pose_object(NOSE, TAIL)))
    _ = _record(dataset, _predicted(NOSE, TAIL), names=INDEXED)
    with pytest.raises(ValueError) as resolved:
        _ = resolve_keypoint_map(MOUSE.schema, INDEXED)

    with pytest.raises(ScorePredictionsInvalid) as scored:
        _ = _score(dataset)

    assert str(scored.value) == str(resolved.value)


def test_labels_that_no_prediction_frame_covers_are_refused(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path / "ds")
    _ = _save(dataset, "set-a", _frame("s2", pose_object(NOSE, TAIL)))
    _ = _record(dataset, _predicted(NOSE, TAIL))

    with pytest.raises(ScorePredictionsConflict) as caught:
        _ = _score(dataset)

    message = str(caught.value)
    assert message.count(dataset.uuid or "") == 2
    assert "one dataset" in message
    assert "linked library" in message


def test_two_sets_that_share_an_annotated_image_are_refused(ds: Dataset) -> None:
    _ = _save(ds, "set-b", _frame("s1", pose_object(TAIL, NOSE)))

    with pytest.raises(ScorePredictionsConflict) as caught:
        _ = _score(ds, labels=("set-a", "set-b"))

    message = str(caught.value)
    for named in (_image("s1"), "'set-a'", "'set-b'", "separate runs"):
        assert named in message


# --- Untouched keypoints ----------------------------------------------------------


@pytest.mark.parametrize(
    "keypoint_map",
    [
        {"keypoints": {"nose": 0, "tail": 1}, "by_name": False},
        {"keypoints": {"nose": "kp0", "tail": "kp1"}},
    ],
    ids=["index", "override"],
)
def test_untouched_points_of_another_runs_object_are_found_without_a_map(
    tmp_path: Path, keypoint_map: dict[str, object]
) -> None:
    """The source run orders its keypoints otherwise, and no map says how."""
    dataset = make_dataset(tmp_path / "ds")
    accepted = _accepted(NOSE, TAIL, source=f"{SOURCE_RUN}#0")
    _ = _save(dataset, "set-a", _frame("s1", accepted))
    _ = _record(dataset, _predicted(NOSE, TAIL), names=INDEXED)
    _ = _record(
        dataset,
        _predicted(TAIL, ELSEWHERE, NOSE),
        run_id=SOURCE_RUN,
        names=("kp0", "kp1", "kp2"),
        model=OTHER_MODEL,
    )

    run_id = _score(dataset, keypoint_map=keypoint_map)

    frames = _frames(dataset, run_id)
    assert (frames["n_kpt_untouched"].tolist(), frames["n_kpt_eligible"].tolist()) == (
        [2],
        [0],
    )
    assert _summary(dataset, run_id).n_objects_source_unknown == 0


@pytest.mark.parametrize("source", ["unreadable", "unrecorded"])
def test_an_object_whose_source_cannot_be_read_is_touched_everywhere(
    tmp_path: Path, source: str
) -> None:
    dataset = make_dataset(tmp_path / "ds")
    accepted = _accepted(NOSE, TAIL, source=f"{SOURCE_RUN}#0")
    _ = _save(dataset, "set-a", _frame("s1", accepted))
    _ = _record(dataset, _predicted(NOSE, TAIL))
    if source == "unreadable":
        written = _record(dataset, _predicted(NOSE, TAIL), run_id=SOURCE_RUN)
        _ = written.write_text("not a predictions file")

    run_id = _score(dataset)

    frames = _frames(dataset, run_id)
    assert (frames["n_kpt_untouched"].tolist(), frames["n_kpt_eligible"].tolist()) == (
        [0],
        [2],
    )
    assert _summary(dataset, run_id).n_objects_source_unknown == 1


@pytest.mark.parametrize(
    ("label", "untouched"),
    [((NOSE, TAIL), 2), ((TAIL, NOSE), 0)],
    ids=["kept", "swapped"],
)
def test_a_point_moved_onto_its_partner_is_edited_for_a_scored_run(
    tmp_path: Path, label: tuple[Point, Point], untouched: int
) -> None:
    dataset = make_dataset(tmp_path / "ds")
    accepted = _accepted(*label, source=f"{PREDICTION_RUN}#0")
    _ = _save(dataset, "set-a", _frame("s1", accepted))
    _ = _record(dataset, _predicted(NOSE, TAIL))

    run_id = _score(dataset)

    assert _frames(dataset, run_id)["n_kpt_untouched"].tolist() == [untouched]


def test_an_image_annotated_without_a_frame_index_takes_its_prediction_frames_key(
    tmp_path: Path,
) -> None:
    dataset = make_dataset(tmp_path / "ds")
    unindexed = _frame("s1", pose_object(NOSE, TAIL), frame_index=-1)
    _ = _save(dataset, "set-a", unindexed)
    _ = _record(dataset, _predicted(NOSE, TAIL))

    run_id = _score(dataset)

    expected = frame_key(
        VideoFrame(
            origin_uuid=dataset.uuid or "",
            group=GROUP,
            sequence="s1",
            camera="",
            frame_index=4,
        )
    )
    assert _frames(dataset, run_id)["frame_key"].tolist() == [expected]
