"""A predict-frames run predicts one trained model on the frames of an extraction run.

The run claims every entry it predicts, hands every held entry's images to the
frame adapter in one call, writes one predictions file and index row per entry
camera, and releases every claim whatever happened. The probe and its checks run
before any claim. These tests run without a tool. They replace the probe and the
adapter, or for a SLEAP or Lightning Pose model the environment behind its
adapter.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import pandas as pd
import pytest
from pydantic import TypeAdapter

import mosaic.tracking.common.ultralytics_env as tool_env
import mosaic.tracking.pose_training.ultralytics_infer as ultralytics_infer
from mosaic.core.annotations.model import Keypoint, KeypointSchema
from mosaic.core.annotations.pose_predictions import (
    FramePredictions,
    PredictedObject,
    PredictedPose,
)
from mosaic.core.dataset import Dataset
from mosaic.core.entry import Entry, make_entry_key
from mosaic.core.pipeline.graph import Recipe, plan_pipeline
from mosaic.core.pipeline.identity_scheme import MARKER_NAME, read_identity_scheme
from mosaic.core.pipeline.index_csv import index_records
from mosaic.core.pipeline.inventory.model import FramePredictionsRef
from mosaic.core.pipeline.job import Cancelled, CancelToken, JobContext, job_context
from mosaic.core.pipeline.markers import (
    INFLIGHT_MARKER_NAME,
    new_inflight,
    read_inflight,
    write_inflight,
)
from mosaic.core.pipeline.models import model_index_path, model_run_root
from mosaic.core.pipeline.op_identity import OP_IDENTITY_SCHEME
from mosaic.core.pipeline.ops import OPS, ModelReference, op_resource_class, run_op
from mosaic.core.pipeline.progress import NullProgressCallback
from mosaic.core.pipeline.run import AllEntriesFailed
from mosaic.core.pipeline.subprocess_util import ProcessCancelled
from mosaic.core.scope import Scope
from mosaic.runlog import RunLogSnapshot, read_run, run_log_dir
from mosaic.tracking import register_ops
from mosaic.tracking.common.ultralytics_env import UnsupportedModelError
from mosaic.tracking.external.runner.ultralytics_protocol import ProbeResponse
from mosaic.tracking.frame_predictions.adapter import (
    AdapterResult,
    FrameImage,
    ToolProbe,
)
from mosaic.tracking.frame_predictions.errors import (
    PredictFramesConflict,
    PredictFramesInvalid,
    PredictFramesNotFound,
    PredictFramesUnavailable,
)
from mosaic.tracking.frame_predictions.index import prediction_kind_root
from mosaic.tracking.frame_predictions.params import PredictFramesParams
from mosaic.tracking.frame_predictions.read import (
    list_prediction_runs,
    read_entry_predictions,
)
from mosaic.tracking.keypoint_sets import keypoint_schema_cell
from mosaic.tracking.model_refs import ModelReferenceRefusedError, ResolvedModel
from mosaic.tracking.ops.predict_frames import pose_after_return
from mosaic.tracking.sleap.frames import SleapInstanceReadback
from mosaic.tracking.sleap.run import SleapNotFoundError
from tests.helpers import (
    FakeLitposeFrames,
    FakeSleapFrames,
    LitposeSnippetRun,
    SleapTrackCall,
    entry_error_lines,
    fake_tool_probe,
    install_fake_litpose_frames,
    install_fake_point_probe,
    install_fake_pose_probe,
    install_fake_sleap_frames,
    make_dataset,
    record_frames_row,
    register_sleap_model,
    register_trained_model,
    sleap_instance,
    ultralytics_probe_response,
    write_frames_run,
    write_litpose_model,
    write_sleap_model,
)

register_ops()

KIND: Final = "predict-frames"
FRAMES_RUN: Final = "uniform-0123456789"
POSE_RUN: Final = "train-pose.0.2-abcdef0123"
POINTS_RUN: Final = "train-points.0.2-abcdef0123"
SLEAP_RUN: Final = "train-sleap.0.3-abcdef0123"
CENTROID_RUN: Final = "train-sleap.0.3-0123456789"
INSTANCE_RUN: Final = "train-sleap.0.3-fedcba9876"
LITPOSE_RUN: Final = "train-litpose.0.1-abcdef0123"
PREPARED_RUN: Final = "prepare-training-data.0.3-0123456789"
OTHER_PREPARED_RUN: Final = "prepare-training-data.0.3-9876543210"
SIZES: Final = {20: (64, 48), 10: (32, 24)}
"""Two frames of each extracted entry camera, listed out of frame order."""

S1: Final[Entry] = ("", "s1")
S2: Final[Entry] = ("", "s2")
TWO_CAMERAS: Final[Entry] = ("g", "s3")

EARED: Final = PredictedPose(
    id=5,
    name="eared",
    schema=KeypointSchema(names=("nose", "tail"), skeleton=((0, 1),)),
)
"""A recorded pose of two keypoints, the count that the fake pose probe reports."""

THREE_KEYPOINTS: Final = PredictedPose(
    id=6, name="wide", schema=KeypointSchema(names=("nose", "neck", "tail"))
)

LONG_AGO: Final = "2020-01-01T00:00:00+00:00"
PEER: Final = "01JPEER000000000000000000"

_JSON_OBJECT: Final = TypeAdapter(dict[str, object])


# --- the stand-in adapter ----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AdapterCall:
    """One call of the stand-in adapter, with what it was handed."""

    images: tuple[FrameImage, ...]
    model: ResolvedModel
    params: PredictFramesParams
    probe: ToolProbe
    work_dir: Path
    on_output: Callable[[str], None]


@dataclass
class FakeAdapter:
    """Stand in for a frame adapter, and record every call.

    Each image gets one object of ``n_keypoints`` keypoints, except the images
    named in ``nothing_in``, which get none. ``during`` runs inside the call,
    while the run holds its claims, and an error it raises reaches the run as
    the tool's failure.

    Attributes:
        n_keypoints: The keypoints of each object and of the tool's pose.
        nothing_in: The file names of the images in which nothing is found.
        short_in: The directory names of the images whose object has one
            keypoint fewer than the pose.
        during: Called with the call's record before the adapter returns.
        calls: Every call, in order.
    """

    n_keypoints: int = 2
    nothing_in: frozenset[str] = frozenset()
    short_in: frozenset[str] = frozenset()
    during: Callable[[AdapterCall], None] | None = None
    calls: list[AdapterCall] = field(default_factory=list[AdapterCall])

    def __call__(
        self,
        images: Sequence[FrameImage],
        model: ResolvedModel,
        params: PredictFramesParams,
        probe: ToolProbe,
        work_dir: Path,
        *,
        on_output: Callable[[str], None],
        cancel_check: Callable[[], bool],
    ) -> AdapterResult:
        _ = cancel_check
        call = AdapterCall(tuple(images), model, params, probe, work_dir, on_output)
        self.calls.append(call)
        work_dir.mkdir(parents=True, exist_ok=True)
        _ = (work_dir / "exchange.json").write_text("{}")
        if self.during is not None:
            self.during(call)
        return AdapterResult(
            objects=tuple(self._found(image) for image in images),
            tool_pose=KeypointSchema(
                names=tuple(f"kp{k}" for k in range(self.n_keypoints))
            ),
        )

    def _found(self, image: FrameImage) -> tuple[PredictedObject, ...]:
        if image.path.name in self.nothing_in:
            return ()
        count = self.n_keypoints - (1 if image.path.parent.name in self.short_in else 0)
        keypoints = tuple(
            Keypoint(x=float(image.position + k), y=1.0, visibility=2, score=0.5)
            for k in range(count)
        )
        return (PredictedObject(keypoints=keypoints, score=0.9),)


@pytest.fixture
def fake_pose(monkeypatch: pytest.MonkeyPatch) -> FakeAdapter:
    """Stand in for the probe and the frame adapter of a yolo-pose model."""
    _ = install_fake_pose_probe(monkeypatch)
    fake = FakeAdapter()
    monkeypatch.setattr(ultralytics_infer, "predict_pose_frames", fake)
    return fake


@pytest.fixture
def fake_points(monkeypatch: pytest.MonkeyPatch) -> FakeAdapter:
    """Stand in for the probe and the frame adapter of a POLO model."""
    _ = install_fake_point_probe(monkeypatch)
    fake = FakeAdapter(n_keypoints=1)
    monkeypatch.setattr(ultralytics_infer, "predict_point_frames", fake)
    return fake


def _probe_reporting(monkeypatch: pytest.MonkeyPatch, response: ProbeResponse) -> None:
    """Make every probe of a tool environment report *response*."""

    def probe(model_path: str, **_kwargs: object) -> ProbeResponse:
        _ = model_path
        return response

    monkeypatch.setattr(tool_env, "probe_environment", probe)


# --- the dataset -------------------------------------------------------------------


def _frames_directory(ds: Dataset, entry: Entry, camera: str = "") -> Path:
    group, sequence = entry
    directory = (
        ds.get_root("frames") / "uniform" / FRAMES_RUN / make_entry_key(group, sequence)
    )
    return directory / camera if camera else directory


def _extract(
    ds: Dataset,
    entry: Entry,
    *,
    camera: str = "",
    composition: str = "c1",
    directory: Path | None = None,
) -> None:
    """Write and record one entry camera of the extraction run."""
    target = directory or _frames_directory(ds, entry, camera)
    write_frames_run(target, SIZES)
    stored = str(target) if directory is not None else ds.relative_to_root(target)
    _record(ds, entry, stored, camera=camera, composition=composition)


def _record(
    ds: Dataset, entry: Entry, stored: str, *, camera: str = "", composition: str
) -> None:
    group, sequence = entry
    record_frames_row(
        ds,
        stored,
        run_id=FRAMES_RUN,
        group=group,
        sequence=sequence,
        camera=camera,
        media_composition=composition,
    )


def _recut(ds: Dataset, entry: Entry, composition: str) -> None:
    """Record the entry's frames again, cut from media of another composition."""
    _record(
        ds,
        entry,
        ds.relative_to_root(_frames_directory(ds, entry)),
        composition=composition,
    )


def _record_preparation(
    ds: Dataset, pose: PredictedPose, prepared_run: str = PREPARED_RUN
) -> Path:
    """Record a preparation whose row names *pose*, and return its data file."""
    data_yaml = model_run_root(ds, "prepare-training-data", prepared_run) / "data.yaml"
    data_yaml.parent.mkdir(parents=True)
    _ = data_yaml.write_text("names: [animal]\n")
    row = {
        "run_id": prepared_run,
        "pose_id": "" if pose.id is None else str(pose.id),
        "pose_name": pose.name,
        "keypoint_schema": keypoint_schema_cell(pose.schema),
        "consumed_sets": "[]",
    }
    index_path = model_index_path(ds, "prepare-training-data")
    earlier = (
        index_records(pd.read_csv(index_path, dtype=str, keep_default_na=False))
        if index_path.exists()
        else []
    )
    pd.DataFrame([*earlier, row]).to_csv(index_path, index=False)
    return data_yaml


def _register(
    ds: Dataset, kind: str, run_id: str, pose: PredictedPose | None = None
) -> Path:
    """Register a trained model, trained on a preparation of *pose* when given.

    Returns the model's run root.
    """
    run_root = model_run_root(ds, kind, run_id)
    weights = run_root / "best.pt"
    weights.parent.mkdir(parents=True)
    _ = weights.write_bytes(b"weights")
    data_path = _record_preparation(ds, pose) if pose is not None else None
    register_trained_model(ds, kind, run_id, weights, data_path=data_path)
    return run_root


def _register_sleap(
    ds: Dataset,
    run_id: str,
    *,
    head: str = "single_instance",
    pose: PredictedPose | None = None,
    prepared_run: str = PREPARED_RUN,
) -> Path:
    """Register a train-sleap model of *head*, trained on *pose* when given.

    Returns the model directory.
    """
    data_path = None if pose is None else _record_preparation(ds, pose, prepared_run)
    return register_sleap_model(
        ds, run_id, run_id.encode(), head=head, data_path=data_path
    )


@pytest.fixture
def ds(tmp_path: Path) -> Dataset:
    """Two extracted entries, ``s1`` and ``s2``, and a yolo-pose model."""
    dataset = make_dataset(tmp_path / "ds")
    _extract(dataset, S1)
    _extract(dataset, S2)
    _ = _register(dataset, "train-pose", POSE_RUN)
    return dataset


# --- running -------------------------------------------------------------------------


def _params(model: Sequence[str], **overrides: object) -> dict[str, object]:
    return {"model": list(model), "frames_run_id": FRAMES_RUN, **overrides}


def _run(
    ds: Dataset,
    *,
    model: Sequence[str] = (POSE_RUN,),
    entries: Sequence[Entry] | None = None,
    overwrite: bool = False,
    execution_id: str | None = None,
    cancel_token: CancelToken | None = None,
    progress: NullProgressCallback | None = None,
    **overrides: object,
) -> str:
    return run_op(
        ds,
        KIND,
        _params(model, **overrides),
        scope=None if entries is None else Scope(entries=list(entries)),
        overwrite=overwrite,
        execution_id=execution_id,
        cancel_token=cancel_token,
        progress_callback=progress,
    )


def _planned_run_id(ds: Dataset, model: Sequence[str] = (POSE_RUN,)) -> str:
    op = OPS[KIND]
    params = op.Params.model_validate(_params(model))
    return op().plan_identity(ds, params, ds.resolve_scope(None)).run_id


def _run_root(ds: Dataset, run_id: str) -> Path:
    return prediction_kind_root(ds, KIND) / run_id


def _claims(ds: Dataset) -> list[Path]:
    """Every claim marker under the frame-predictions root."""
    return sorted(ds.get_root("frame_predictions").rglob(INFLIGHT_MARKER_NAME))


def _predictions(
    ds: Dataset, run_id: str, entry: Entry, camera: str = ""
) -> FramePredictions:
    group, sequence = entry
    found = read_entry_predictions(ds, run_id, group, sequence, camera)
    assert found is not None
    return found


def _rows(ds: Dataset, run_id: str) -> list[dict[str, str]]:
    records = index_records(list_prediction_runs(ds))
    selected = [record for record in records if record["run_id"] == run_id]
    return sorted(selected, key=lambda record: (record["sequence"], record["camera"]))


def _snapshot(ds: Dataset, execution_id: str) -> RunLogSnapshot:
    found = read_run(run_log_dir(ds.base_dir), execution_id)
    assert found is not None
    return found


def _tree(root: Path) -> dict[str, bytes]:
    """Every file under *root*, by relative path, with its bytes."""
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


# --- declaration ---------------------------------------------------------------------


def test_the_op_declares_an_inference_run_over_any_scope() -> None:
    op = OPS[KIND]

    assert op.category == "infer"
    assert op.domain == "tracking"
    assert op.version == "0.1"
    assert op.writes_tracks is False
    assert op.scope_takes == "any"
    assert op.scope_dependent is False
    assert op.model_reference == ModelReference(
        field="model",
        kinds=("train-pose", "train-points", "train-sleap", "train-litpose"),
        many=True,
    )
    assert op_resource_class(KIND) == "gpu"


def test_a_planned_step_names_its_predictions_by_run_and_extraction(
    ds: Dataset,
) -> None:
    recipe = Recipe.model_validate(
        {
            "steps": [
                {
                    "id": "predict",
                    "type": "op",
                    "kind": KIND,
                    "params": _params([POSE_RUN]),
                }
            ]
        }
    )

    plan = plan_pipeline(ds, recipe, scope=Scope(entries=[S1, S2]))

    assert plan.step("predict").artifact == FramePredictionsRef(
        run_id=_planned_run_id(ds), frames_run_id=FRAMES_RUN
    )


# --- refusals before any claim -----------------------------------------------------


def test_a_bare_weights_path_is_refused_by_the_run_itself(
    ds: Dataset, fake_pose: FakeAdapter
) -> None:
    with pytest.raises(ModelReferenceRefusedError) as caught:
        _ = _run(ds, model=["models/train-pose/run/best.pt"])

    assert caught.value.reason == "not_a_run_id"
    assert fake_pose.calls == []


def test_references_of_mixed_kinds_are_refused_by_the_run_itself(
    ds: Dataset, fake_pose: FakeAdapter
) -> None:
    with pytest.raises(PredictFramesInvalid) as caught:
        _ = _run(ds, model=[POSE_RUN, POINTS_RUN])

    assert caught.value.error_kind == "invalid"
    message = str(caught.value)
    assert f"{POSE_RUN!r} is a train-pose run" in message
    assert f"{POINTS_RUN!r} is a train-points run" in message
    assert fake_pose.calls == []


def test_a_pose_model_whose_recorded_pose_has_another_count_is_a_conflict(
    tmp_path: Path, fake_pose: FakeAdapter
) -> None:
    dataset = make_dataset(tmp_path / "ds")
    _extract(dataset, S1)
    _ = _register(dataset, "train-pose", POSE_RUN, THREE_KEYPOINTS)

    with pytest.raises(PredictFramesConflict) as caught:
        _ = _run(dataset, execution_id="exec-count")

    assert caught.value.error_kind == "conflict"
    message = str(caught.value)
    assert POSE_RUN in message
    assert "predicts 2 keypoints" in message
    assert "3 keypoints" in message
    assert _claims(dataset) == []
    assert entry_error_lines(dataset, "exec-count") == []
    assert fake_pose.calls == []


def test_a_points_model_whose_recorded_pose_has_two_keypoints_is_a_conflict(
    tmp_path: Path, fake_points: FakeAdapter
) -> None:
    dataset = make_dataset(tmp_path / "ds")
    _extract(dataset, S1)
    _ = _register(dataset, "train-points", POINTS_RUN, EARED)

    with pytest.raises(PredictFramesConflict) as caught:
        _ = _run(dataset, model=[POINTS_RUN], execution_id="exec-points")

    assert caught.value.error_kind == "conflict"
    message = str(caught.value)
    assert POINTS_RUN in message
    assert "predicts 1 keypoint," in message
    assert "2 keypoints" in message
    assert _claims(dataset) == []
    assert entry_error_lines(dataset, "exec-points") == []
    assert fake_points.calls == []


def test_weights_that_the_model_check_refuses_are_refused_before_any_claim(
    ds: Dataset, fake_pose: FakeAdapter, monkeypatch: pytest.MonkeyPatch
) -> None:
    _probe_reporting(
        monkeypatch,
        ultralytics_probe_response("locate", has_locate=True, n_keypoints=1),
    )

    with pytest.raises(UnsupportedModelError):
        _ = _run(ds)

    assert _claims(ds) == []
    assert fake_pose.calls == []


def test_an_absent_extraction_run_is_not_found(
    ds: Dataset, fake_pose: FakeAdapter
) -> None:
    with pytest.raises(PredictFramesNotFound) as caught:
        _ = _run(ds, frames_run_id="uniform-ffffffffff")

    assert caught.value.error_kind == "not_found"
    assert "uniform-ffffffffff" in str(caught.value)
    assert fake_pose.calls == []


def test_a_scope_without_frames_in_the_run_is_a_conflict(
    ds: Dataset, fake_pose: FakeAdapter
) -> None:
    with pytest.raises(PredictFramesConflict) as caught:
        _ = _run(ds, entries=[("", "elsewhere")])

    assert caught.value.error_kind == "conflict"
    message = str(caught.value)
    assert FRAMES_RUN in message
    assert "elsewhere" in message
    assert _claims(ds) == []
    assert fake_pose.calls == []


# --- the pose that is written --------------------------------------------------------


def test_a_recorded_pose_is_written_with_its_names_and_id(
    tmp_path: Path, fake_pose: FakeAdapter
) -> None:
    dataset = make_dataset(tmp_path / "ds")
    _extract(dataset, S1)
    _ = _register(dataset, "train-pose", POSE_RUN, EARED)

    run_id = _run(dataset)

    assert _predictions(dataset, run_id, S1).pose == EARED
    assert len(fake_pose.calls) == 1


def test_without_a_recorded_pose_the_tools_pose_is_written(
    ds: Dataset, fake_pose: FakeAdapter
) -> None:
    run_id = _run(ds)

    assert _predictions(ds, run_id, S1).pose == PredictedPose(
        id=None, name="", schema=KeypointSchema(names=("kp0", "kp1"))
    )


def test_a_points_model_writes_its_one_keypoint(
    tmp_path: Path, fake_points: FakeAdapter
) -> None:
    dataset = make_dataset(tmp_path / "ds")
    _extract(dataset, S1)
    _ = _register(dataset, "train-points", POINTS_RUN)

    run_id = _run(dataset, model=[POINTS_RUN])

    written = _predictions(dataset, run_id, S1)
    assert written.pose.schema.names == ("kp0",)
    assert [len(frame.objects) for frame in written.frames] == [1, 1]
    assert fake_points.calls[0].probe == fake_tool_probe("locate")


def test_the_after_return_check_fails_every_held_entry_on_other_names(
    ds: Dataset,
) -> None:
    with job_context(ds, kind=KIND, target=KIND, execution_id="exec-names") as ctx:
        pose = pose_after_return(
            ctx,
            ["s1", "s2"],
            EARED,
            KeypointSchema(names=("head", "tail")),
            model=[POSE_RUN],
        )

    assert pose is None
    assert ctx.failed_keys == ["s1", "s2"]
    lines = entry_error_lines(ds, "exec-names")
    assert len(lines) == 2
    assert all("PredictFramesConflict" in line for line in lines)
    assert all("head" in line and "nose" in line for line in lines)


def test_the_after_return_check_adopts_the_tools_pose_without_a_recorded_one(
    ds: Dataset,
) -> None:
    tool_pose = KeypointSchema(names=("head", "tail"), skeleton=((1, 0),))
    with job_context(ds, kind=KIND, target=KIND) as ctx:
        adopted = pose_after_return(ctx, ["s1"], None, tool_pose, model=[POSE_RUN])
        kept = pose_after_return(
            ctx, ["s1"], EARED, KeypointSchema(names=("nose", "tail")), model=[POSE_RUN]
        )

    assert adopted == PredictedPose(id=None, name="", schema=tool_pose)
    assert kept == EARED
    assert ctx.failed_keys == []


# --- the entries a run predicts ------------------------------------------------------


def test_scoped_entries_without_frames_are_skipped_and_counted(
    ds: Dataset, fake_pose: FakeAdapter, capsys: pytest.CaptureFixture[str]
) -> None:
    run_id = _run(ds, entries=[S1, ("", "other1"), ("", "other2")])

    assert "2 scoped entries" in capsys.readouterr().out
    assert [row["sequence"] for row in _rows(ds, run_id)] == ["s1"]


def test_an_entry_whose_frames_lie_outside_the_dataset_is_that_entrys_error(
    ds: Dataset, fake_pose: FakeAdapter, tmp_path: Path
) -> None:
    _extract(ds, ("", "s3"), directory=tmp_path / "elsewhere" / "s3")

    run_id = _run(ds, execution_id="exec-outside")

    lines = entry_error_lines(ds, "exec-outside")
    assert len(lines) == 1
    assert _JSON_OBJECT.validate_json(lines[0])["key"] == "s3"
    assert "ValueError" in lines[0]
    snapshot = _snapshot(ds, "exec-outside")
    assert snapshot["entries_failed"] == 1
    assert snapshot["entries_written"] == 2
    assert [row["sequence"] for row in _rows(ds, run_id)] == ["s1", "s2"]


def test_an_entry_whose_extraction_directory_is_gone_is_that_entrys_error(
    ds: Dataset, fake_pose: FakeAdapter
) -> None:
    shutil.rmtree(_frames_directory(ds, S2))

    run_id = _run(ds, execution_id="exec-gone")

    lines = entry_error_lines(ds, "exec-gone")
    assert len(lines) == 1
    assert _JSON_OBJECT.validate_json(lines[0])["key"] == "s2"
    assert "run_info.json" in lines[0]
    snapshot = _snapshot(ds, "exec-gone")
    assert snapshot["entries_failed"] == 1
    assert snapshot["entries_written"] == 1
    assert [row["sequence"] for row in _rows(ds, run_id)] == ["s1"]


def test_an_entry_whose_manifest_record_lacks_a_field_is_that_entrys_error(
    ds: Dataset, fake_pose: FakeAdapter
) -> None:
    manifest = _frames_directory(ds, S2) / "run_info.json"
    _ = manifest.write_text(json.dumps({"files": [{"frame_index": 10}]}))

    run_id = _run(ds, execution_id="exec-damaged")

    lines = entry_error_lines(ds, "exec-damaged")
    assert len(lines) == 1
    assert _JSON_OBJECT.validate_json(lines[0])["key"] == "s2"
    assert _snapshot(ds, "exec-damaged")["entries_written"] == 1
    assert [row["sequence"] for row in _rows(ds, run_id)] == ["s1"]


def test_a_run_whose_every_entry_lies_outside_the_dataset_fails_with_no_claim(
    tmp_path: Path, fake_pose: FakeAdapter
) -> None:
    dataset = make_dataset(tmp_path / "ds")
    _extract(dataset, S1, directory=tmp_path / "elsewhere" / "s1")
    _ = _register(dataset, "train-pose", POSE_RUN)

    with pytest.raises(AllEntriesFailed):
        _ = _run(dataset, execution_id="exec-all-outside")

    assert len(entry_error_lines(dataset, "exec-all-outside")) == 1
    assert _claims(dataset) == []
    assert fake_pose.calls == []


def test_the_adapter_is_called_once_for_every_held_entry(
    ds: Dataset, fake_pose: FakeAdapter
) -> None:
    _ = _run(ds)

    assert len(fake_pose.calls) == 1
    images = fake_pose.calls[0].images
    expected = [
        (_frames_directory(ds, entry) / f"frame_{index:06d}.png", SIZES[index])
        for entry in (S1, S2)
        for index in sorted(SIZES)
    ]
    assert [image.position for image in images] == list(range(len(expected)))
    assert [
        (image.path.resolve(), (image.width, image.height)) for image in images
    ] == [(path.resolve(), size) for path, size in expected]


def test_each_entry_camera_gets_its_file_and_row(
    ds: Dataset, fake_pose: FakeAdapter
) -> None:
    _extract(ds, TWO_CAMERAS, camera="cam1", composition="c3")
    _extract(ds, TWO_CAMERAS, camera="cam2", composition="c4")

    run_id = _run(ds, entries=[TWO_CAMERAS])

    rows = _rows(ds, run_id)
    assert [(row["camera"], row["consumed_frames_composition"]) for row in rows] == [
        ("cam1", "c3"),
        ("cam2", "c4"),
    ]
    for camera in ("cam1", "cam2"):
        written = _predictions(ds, run_id, TWO_CAMERAS, camera)
        assert written.camera == camera
        assert written.frames_run_id == FRAMES_RUN
        assert [frame.frame_index for frame in written.frames] == sorted(SIZES)
        assert all(
            frame.image.startswith(f"media/frames/uniform/{FRAMES_RUN}/g__s3/{camera}/")
            for frame in written.frames
        )


def test_frames_with_nothing_found_are_written_empty(
    ds: Dataset, fake_pose: FakeAdapter
) -> None:
    fake_pose.nothing_in = frozenset({"frame_000020.png"})

    run_id = _run(ds, entries=[S1])

    frames = _predictions(ds, run_id, S1).frames
    assert [(frame.frame_index, len(frame.objects)) for frame in frames] == [
        (10, 1),
        (20, 0),
    ]
    assert _rows(ds, run_id)[0]["n_objects"] == "1"


# --- claims ---------------------------------------------------------------------------


def test_each_entry_is_claimed_while_the_tool_runs(
    ds: Dataset, fake_pose: FakeAdapter
) -> None:
    held: list[str] = []

    def record_claims(_call: AdapterCall) -> None:
        for marker in _claims(ds):
            claim = read_inflight(marker.parent)
            assert claim is not None
            held.append(f"{marker.parent.name}:{claim.execution_id}")

    fake_pose.during = record_claims

    _ = _run(ds, execution_id="exec-claims")

    assert held == ["s1:exec-claims", "s2:exec-claims"]
    assert _claims(ds) == []


def test_every_held_claim_is_refreshed_by_the_tools_output(
    ds: Dataset, fake_pose: FakeAdapter
) -> None:
    refreshed: list[bool] = []

    def go_stale_then_print(call: AdapterCall) -> None:
        markers = _claims(ds)
        for marker in markers:
            claim = read_inflight(marker.parent)
            assert claim is not None
            write_inflight(
                marker.parent, claim.model_copy(update={"expires_at": LONG_AGO})
            )
        call.on_output("line")
        for marker in markers:
            claim = read_inflight(marker.parent)
            assert claim is not None
            refreshed.append(claim.expires_at != LONG_AGO)

    fake_pose.during = go_stale_then_print

    _ = _run(ds)

    assert refreshed == [True, True]


def test_every_claim_is_released_after_a_tool_failure(
    ds: Dataset, fake_pose: FakeAdapter
) -> None:
    def fail(_call: AdapterCall) -> None:
        raise RuntimeError("the runner exited with status 1")

    fake_pose.during = fail

    with pytest.raises(AllEntriesFailed):
        _ = _run(ds, execution_id="exec-failed")

    assert _claims(ds) == []
    lines = entry_error_lines(ds, "exec-failed")
    assert len(lines) == 2
    assert all("the runner exited with status 1" in line for line in lines)
    assert _rows(ds, _planned_run_id(ds)) == []


def test_every_claim_is_released_after_a_cancel(
    ds: Dataset, fake_pose: FakeAdapter
) -> None:
    token = CancelToken()

    def cancel(_call: AdapterCall) -> None:
        token.cancel()
        raise ProcessCancelled(["runner"])

    fake_pose.during = cancel

    with pytest.raises(Cancelled):
        _ = _run(ds, execution_id="exec-cancel", cancel_token=token)

    assert _claims(ds) == []
    assert entry_error_lines(ds, "exec-cancel") == []
    assert sorted(
        path.name for path in _run_root(ds, _planned_run_id(ds)).iterdir()
    ) == [MARKER_NAME, "run_params.json"]


@pytest.mark.parametrize("fails", [False, True])
def test_no_work_directory_remains_and_the_model_is_unchanged(
    ds: Dataset, fake_pose: FakeAdapter, fails: bool
) -> None:
    model_root = model_run_root(ds, "train-pose", POSE_RUN)
    before = _tree(model_root)
    if fails:

        def fail(_call: AdapterCall) -> None:
            raise RuntimeError("the runner exited with status 1")

        fake_pose.during = fail
        with pytest.raises(AllEntriesFailed):
            _ = _run(ds)
    else:
        _ = _run(ds)

    work_dir = fake_pose.calls[0].work_dir
    run_root = _run_root(ds, _planned_run_id(ds))
    assert work_dir.parent == run_root
    assert not work_dir.exists()
    left = sorted(path.name for path in run_root.iterdir())
    written = [] if fails else ["s1", "s2"]
    assert left == [MARKER_NAME, "run_params.json", *written]
    assert _tree(model_root) == before


def test_an_entry_held_elsewhere_is_not_attempted(
    ds: Dataset, fake_pose: FakeAdapter
) -> None:
    held = _run_root(ds, _planned_run_id(ds)) / "s2"
    held.mkdir(parents=True)
    write_inflight(
        held,
        new_inflight(
            execution_id=PEER, host="peer", pid=1, phase=None, idle_seconds=3600
        ),
    )

    run_id = _run(ds, execution_id="exec-contended")

    assert [image.path.parent.name for image in fake_pose.calls[0].images] == [
        "s1",
        "s1",
    ]
    assert [row["sequence"] for row in _rows(ds, run_id)] == ["s1"]
    assert entry_error_lines(ds, "exec-contended") == []
    claim = read_inflight(held)
    assert claim is not None
    assert claim.execution_id == PEER


# --- reuse and the cache hit ---------------------------------------------------------


def test_a_rerun_reuses_every_written_entry_and_is_a_cache_hit(
    ds: Dataset, fake_pose: FakeAdapter
) -> None:
    first = _run(ds, execution_id="exec-first")
    second = _run(ds, execution_id="exec-second")

    assert second == first
    assert len(fake_pose.calls) == 1
    assert _snapshot(ds, "exec-first")["cache_hit"] is False
    assert _snapshot(ds, "exec-second")["cache_hit"] is True
    assert _snapshot(ds, "exec-second")["entries_written"] == 2


def test_a_rerun_over_a_wider_scope_than_the_extraction_is_a_cache_hit(
    ds: Dataset, fake_pose: FakeAdapter
) -> None:
    _ = _run(ds, entries=[S1, S2])
    _ = _run(ds, entries=[S1, S2, ("", "other")], execution_id="exec-wider")

    assert len(fake_pose.calls) == 1
    assert _snapshot(ds, "exec-wider")["cache_hit"] is True


def test_an_entry_held_elsewhere_blocks_the_cache_hit(
    ds: Dataset, fake_pose: FakeAdapter
) -> None:
    run_id = _run(ds)
    write_inflight(
        _run_root(ds, run_id) / "s2",
        new_inflight(
            execution_id=PEER, host="peer", pid=1, phase=None, idle_seconds=3600
        ),
    )

    _ = _run(ds, execution_id="exec-held")

    assert len(fake_pose.calls) == 1
    assert _snapshot(ds, "exec-held")["cache_hit"] is False


def test_an_entry_missing_one_cameras_file_is_predicted_again(
    ds: Dataset, fake_pose: FakeAdapter
) -> None:
    _extract(ds, TWO_CAMERAS, camera="cam1")
    _extract(ds, TWO_CAMERAS, camera="cam2")
    run_id = _run(ds, entries=[TWO_CAMERAS])
    (_run_root(ds, run_id) / "g__s3" / "cam2" / "predictions.json").unlink()

    _ = _run(ds, entries=[TWO_CAMERAS], execution_id="exec-missing")

    assert len(fake_pose.calls) == 2
    assert {image.path.parent.name for image in fake_pose.calls[1].images} == {
        "cam1",
        "cam2",
    }
    assert _snapshot(ds, "exec-missing")["cache_hit"] is False
    assert _predictions(ds, run_id, TWO_CAMERAS, "cam2").camera == "cam2"


def test_overwrite_predicts_a_drifted_entry_again_under_the_same_run_id(
    ds: Dataset, fake_pose: FakeAdapter
) -> None:
    first = _run(ds, entries=[S1])
    _recut(ds, S1, "c2")

    reused = _run(ds, entries=[S1])
    assert len(fake_pose.calls) == 1
    assert _rows(ds, first)[0]["consumed_frames_composition"] == "c1"

    again = _run(ds, entries=[S1], overwrite=True)

    assert again == reused == first
    assert len(fake_pose.calls) == 2
    rows = _rows(ds, first)
    assert len(rows) == 1
    assert rows[0]["consumed_frames_composition"] == "c2"
    assert rows[0]["finished_at"]


# --- losses and coverage -------------------------------------------------------------


def test_a_failed_write_loses_that_entry_alone(
    ds: Dataset, fake_pose: FakeAdapter
) -> None:
    fake_pose.short_in = frozenset({"s1"})

    run_id = _run(ds, execution_id="exec-partial")

    lines = entry_error_lines(ds, "exec-partial")
    assert len(lines) == 1
    assert _JSON_OBJECT.validate_json(lines[0])["key"] == "s1"
    snapshot = _snapshot(ds, "exec-partial")
    assert snapshot["entries_written"] == 1
    assert snapshot["entries_failed"] == 1
    assert [row["sequence"] for row in _rows(ds, run_id)] == ["s2"]
    assert _claims(ds) == []


def test_a_run_that_loses_every_entry_raises(
    ds: Dataset, fake_pose: FakeAdapter
) -> None:
    fake_pose.short_in = frozenset({"s1", "s2"})

    with pytest.raises(AllEntriesFailed) as caught:
        _ = _run(ds, execution_id="exec-lost")

    assert "s1" in str(caught.value)
    assert "s2" in str(caught.value)
    assert _snapshot(ds, "exec-lost")["entries_written"] == 0


def test_the_tool_probe_carries_the_ultralytics_version(
    ds: Dataset, fake_pose: FakeAdapter
) -> None:
    _ = _run(ds)

    assert fake_pose.calls[0].probe == fake_tool_probe("pose")
    assert fake_pose.calls[0].probe.tool_version == "8.4.63"


def test_the_tool_version_is_recorded_and_outside_the_identity(
    ds: Dataset, fake_pose: FakeAdapter, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = _run(ds)
    assert {row["tool_version"] for row in _rows(ds, first)} == {"8.4.63"}

    _probe_reporting(
        monkeypatch, ultralytics_probe_response("pose", n_keypoints=2, version="8.4.99")
    )
    second = _run(ds, overwrite=True)

    assert second == first
    assert {row["tool_version"] for row in _rows(ds, first)} == {"8.4.99"}


def test_each_row_records_what_the_run_read_and_ran(
    ds: Dataset, fake_pose: FakeAdapter
) -> None:
    run_id = _run(ds)

    rows = _rows(ds, run_id)
    assert len(rows) == 2
    for row, entry in zip(rows, (S1, S2), strict=True):
        assert (row["group"], row["sequence"], row["camera"]) == (*entry, "")
        assert row["frames_method"] == "uniform"
        assert row["frames_run_id"] == FRAMES_RUN
        assert row["consumed_frames_composition"] == "c1"
        assert json.loads(row["model"]) == [POSE_RUN]
        assert row["model_kind"] == "train-pose"
        assert row["model_id"] == POSE_RUN
        assert row["n_frames"] == "2"
        assert row["n_objects"] == "2"
        assert run_id.endswith(f"-{row['params_hash']}")
        assert (
            row["abs_path"]
            == f"frame_predictions/{KIND}/{run_id}/{entry[1]}/predictions.json"
        )
        assert row["finished_at"]


def test_run_params_holds_the_identity_payload(
    ds: Dataset, fake_pose: FakeAdapter
) -> None:
    run_id = _run(ds)

    run_root = _run_root(ds, run_id)
    written = json.loads((run_root / "run_params.json").read_text())
    params = PredictFramesParams.model_validate(_params([POSE_RUN]))
    assert written == {
        "params": params.identity_dump(),
        "model_id": POSE_RUN,
        "model_kind": "train-pose",
    }
    assert read_identity_scheme(run_root) == OP_IDENTITY_SCHEME


class _EntryEnds(NullProgressCallback):
    """Record each entry's end, beside the heartbeats recorded by the test."""

    def __init__(self, events: list[tuple[str, int, str]]) -> None:
        self.events = events

    def on_entry_end(self, index: int, total: int, key: str) -> None:
        self.events.append(("end", index, key))


def test_each_written_entry_reports_its_end_and_a_heartbeat(
    ds: Dataset, fake_pose: FakeAdapter, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[tuple[str, int, str]] = []
    original = JobContext.heartbeat

    def heartbeat(self: JobContext, done: int | None = None) -> None:
        if done is not None:
            events.append(("heartbeat", done, ""))
        original(self, done)

    monkeypatch.setattr(JobContext, "heartbeat", heartbeat)

    _ = _run(ds, progress=_EntryEnds(events))

    assert events == [
        ("end", 1, "s1"),
        ("heartbeat", 1, ""),
        ("end", 2, "s2"),
        ("heartbeat", 2, ""),
    ]


# --- SLEAP models --------------------------------------------------------------------


@pytest.fixture
def fake_sleap(monkeypatch: pytest.MonkeyPatch) -> FakeSleapFrames:
    """Stand in for the SLEAP environment behind the SLEAP frame adapter."""
    return install_fake_sleap_frames(monkeypatch)


def _payload(ds: Dataset, run_id: str, entry: Entry) -> dict[str, object]:
    """Return the predictions file of *entry* as the JSON object written."""
    path = _run_root(ds, run_id) / entry[1] / "predictions.json"
    return _JSON_OBJECT.validate_json(path.read_bytes())


def test_a_sleap_model_without_a_recorded_pose_writes_its_node_names(
    ds: Dataset, fake_sleap: FakeSleapFrames
) -> None:
    _ = _register_sleap(ds, SLEAP_RUN)

    run_id = _run(ds, model=[SLEAP_RUN])

    for entry in (S1, S2):
        written = _predictions(ds, run_id, entry)
        assert written.pose == PredictedPose(
            id=None,
            name="",
            schema=KeypointSchema(names=("head", "tail"), skeleton=((0, 1),)),
        )
        assert [len(frame.objects) for frame in written.frames] == [1, 1]
    assert fake_sleap.written[0].annotations.schema == KeypointSchema(names=("kp0",))


def test_the_sleap_input_names_each_image_by_its_key_under_the_dataset_root(
    ds: Dataset, fake_sleap: FakeSleapFrames
) -> None:
    _ = _register_sleap(ds, SLEAP_RUN)

    run_id = _run(ds, model=[SLEAP_RUN])

    written = fake_sleap.written[0].annotations
    assert written.image_root == ds.base_dir
    assert [frame.image_path.as_posix() for frame in written.frames] == [
        frame.image
        for entry in (S1, S2)
        for frame in _predictions(ds, run_id, entry).frames
    ]


def test_a_sleap_readback_edge_is_written_lower_keypoint_first(
    ds: Dataset, fake_sleap: FakeSleapFrames
) -> None:
    _ = _register_sleap(ds, SLEAP_RUN)
    fake_sleap.nodes = ("nose", "neck", "tail", "tip")
    fake_sleap.edges = ((3, 1),)

    run_id = _run(ds, model=[SLEAP_RUN], entries=[S1])

    pose = _payload(ds, run_id, S1)["pose"]
    assert isinstance(pose, dict)
    assert pose["skeleton"] == [[1, 3]]


def test_a_sleap_model_with_a_recorded_pose_writes_that_pose(
    ds: Dataset, fake_sleap: FakeSleapFrames
) -> None:
    _ = _register_sleap(ds, SLEAP_RUN, pose=EARED)
    fake_sleap.nodes = ("nose", "tail")

    run_id = _run(ds, model=[SLEAP_RUN])

    assert _predictions(ds, run_id, S1).pose == EARED
    assert fake_sleap.written[0].annotations.schema == EARED.schema


def test_a_sleap_model_whose_recorded_pose_names_other_nodes_is_a_conflict(
    ds: Dataset, fake_sleap: FakeSleapFrames
) -> None:
    _ = _register_sleap(ds, SLEAP_RUN, pose=EARED)

    with pytest.raises(AllEntriesFailed):
        _ = _run(ds, model=[SLEAP_RUN], execution_id="exec-sleap-names")

    lines = entry_error_lines(ds, "exec-sleap-names")
    assert len(lines) == 2
    assert all("PredictFramesConflict" in line for line in lines)
    assert all("head" in line and "nose" in line for line in lines)
    assert _claims(ds) == []


def test_two_entries_that_share_frame_names_get_their_own_predictions(
    ds: Dataset, fake_sleap: FakeSleapFrames
) -> None:
    def by_entry(image: Path) -> list[SleapInstanceReadback]:
        x = 1.0 if image.parent.name == "s1" else 2.0
        return [sleap_instance([(x, 0.0, 0.5), (x, 1.0, 0.5)])]

    _ = _register_sleap(ds, SLEAP_RUN)
    fake_sleap.instances = by_entry

    run_id = _run(ds, model=[SLEAP_RUN])

    for entry, x in ((S1, 1.0), (S2, 2.0)):
        frames = _predictions(ds, run_id, entry).frames
        assert [frame.image.rsplit("/", 1)[1] for frame in frames] == [
            "frame_000010.png",
            "frame_000020.png",
        ]
        assert {obj.keypoints[0].x for frame in frames for obj in frame.objects} == {x}


def test_an_image_where_sleap_finds_nothing_is_written_without_objects(
    ds: Dataset, fake_sleap: FakeSleapFrames
) -> None:
    def nothing_in_one(image: Path) -> list[SleapInstanceReadback]:
        if image.parent.name == "s1" and image.name == "frame_000020.png":
            return []
        return [sleap_instance([(1.0, 2.0, 0.5), (3.0, 4.0, 0.5)])]

    _ = _register_sleap(ds, SLEAP_RUN)
    fake_sleap.instances = nothing_in_one

    run_id = _run(ds, model=[SLEAP_RUN])

    found = [
        [(frame.frame_index, len(frame.objects)) for frame in frames]
        for frames in (_predictions(ds, run_id, entry).frames for entry in (S1, S2))
    ]
    assert found == [[(10, 1), (20, 0)], [(10, 1), (20, 1)]]
    assert [row["n_objects"] for row in _rows(ds, run_id)] == ["1", "2"]


def test_a_sleap_path_that_matches_no_request_image_fails_every_held_entry(
    ds: Dataset, fake_sleap: FakeSleapFrames
) -> None:
    _ = _register_sleap(ds, SLEAP_RUN)
    fake_sleap.reported = lambda image: (
        str(image.with_name("moved.png")) if image.parent.name == "s2" else str(image)
    )

    with pytest.raises(AllEntriesFailed):
        _ = _run(ds, model=[SLEAP_RUN], execution_id="exec-sleap-moved")

    lines = entry_error_lines(ds, "exec-sleap-moved")
    keys = [_JSON_OBJECT.validate_json(line)["key"] for line in lines]
    assert set(keys) == {"s1", "s2"}
    assert len(keys) == 2
    assert all("moved.png" in line for line in lines)
    assert _claims(ds) == []


def test_a_sleap_pair_runs_its_centroid_model_first(
    ds: Dataset, fake_sleap: FakeSleapFrames
) -> None:
    centroid = _register_sleap(ds, CENTROID_RUN, head="centroid")
    instance = _register_sleap(ds, INSTANCE_RUN, head="centered_instance")

    run_id = _run(ds, model=[CENTROID_RUN, INSTANCE_RUN])

    assert fake_sleap.tracked[0].settings["model_paths"] == [centroid, instance]
    assert {row["model_kind"] for row in _rows(ds, run_id)} == {"train-sleap"}


@pytest.mark.parametrize(
    ("heads", "named"),
    [
        (
            ("centered_instance", "centroid"),
            ("a centered_instance model", "a centroid model"),
        ),
        (
            ("centroid", "single_instance"),
            ("a centroid model", "a single_instance model"),
        ),
        (("centroid", ""), ("a centroid model", "a model whose config names no head")),
    ],
    ids=["reversed", "not-top-down", "unknown-head"],
)
def test_a_sleap_pair_out_of_order_is_refused_before_any_claim(
    ds: Dataset,
    fake_sleap: FakeSleapFrames,
    heads: tuple[str, str],
    named: tuple[str, str],
) -> None:
    first, second = heads
    _ = _register_sleap(ds, CENTROID_RUN, head=first)
    _ = _register_sleap(ds, INSTANCE_RUN, head=second)

    with pytest.raises(PredictFramesInvalid) as caught:
        _ = _run(ds, model=[CENTROID_RUN, INSTANCE_RUN])

    assert caught.value.error_kind == "invalid"
    assert f"{CENTROID_RUN!r} is {named[0]} and {INSTANCE_RUN!r} is {named[1]}." in str(
        caught.value
    )
    assert _claims(ds) == []
    assert fake_sleap.probed == []
    assert fake_sleap.tracked == []


@pytest.mark.parametrize(
    "head", ["centroid", "centered_instance", "multi_class_topdown"]
)
def test_a_lone_top_down_head_is_refused_before_any_claim(
    ds: Dataset, fake_sleap: FakeSleapFrames, head: str
) -> None:
    _ = _register_sleap(ds, SLEAP_RUN, head=head)

    with pytest.raises(PredictFramesInvalid) as caught:
        _ = _run(ds, model=[SLEAP_RUN])

    assert caught.value.error_kind == "invalid"
    message = str(caught.value)
    assert f"{SLEAP_RUN!r} is a {head} model" in message
    assert "takes a top-down model as a pair" in message
    assert _claims(ds) == []
    assert fake_sleap.tracked == []


@pytest.mark.parametrize(
    "model",
    [(SLEAP_RUN,), (CENTROID_RUN, SLEAP_RUN)],
    ids=["alone", "in-a-pair"],
)
def test_an_embedding_model_is_refused_before_any_claim(
    ds: Dataset, fake_sleap: FakeSleapFrames, model: tuple[str, ...]
) -> None:
    """An embedding model embeds detections, and predicts no pose to write.

    Its directory holds the merged config that sleap-nn saves, which also names
    every head that was not trained.
    """
    _ = _register_sleap(ds, CENTROID_RUN, head="centroid")
    directory = _register_sleap(ds, SLEAP_RUN, head="embedding")
    _ = write_sleap_model(directory, SLEAP_RUN.encode(), head="embedding", merged=True)

    with pytest.raises(PredictFramesInvalid) as caught:
        _ = _run(ds, model=model)

    assert caught.value.error_kind == "invalid"
    message = str(caught.value)
    assert f"{SLEAP_RUN!r} is an embedding (re-identification) model" in message
    assert "cannot predict poses" in message
    assert _claims(ds) == []
    assert fake_sleap.probed == []
    assert fake_sleap.tracked == []


def test_a_sleap_pair_writes_the_pose_that_one_training_records(
    ds: Dataset, fake_sleap: FakeSleapFrames
) -> None:
    _ = _register_sleap(ds, CENTROID_RUN, head="centroid")
    _ = _register_sleap(ds, INSTANCE_RUN, head="centered_instance", pose=EARED)
    fake_sleap.nodes = EARED.schema.names

    run_id = _run(ds, model=[CENTROID_RUN, INSTANCE_RUN])

    assert fake_sleap.written[0].annotations.schema == EARED.schema
    assert [_predictions(ds, run_id, entry).pose for entry in (S1, S2)] == [EARED] * 2


def test_a_sleap_pair_whose_trainings_record_different_poses_is_invalid(
    ds: Dataset, fake_sleap: FakeSleapFrames
) -> None:
    _ = _register_sleap(ds, CENTROID_RUN, head="centroid", pose=EARED)
    _ = _register_sleap(
        ds,
        INSTANCE_RUN,
        head="centered_instance",
        pose=THREE_KEYPOINTS,
        prepared_run=OTHER_PREPARED_RUN,
    )

    with pytest.raises(PredictFramesInvalid) as caught:
        _ = _run(ds, model=[CENTROID_RUN, INSTANCE_RUN])

    assert caught.value.error_kind == "invalid"
    message = str(caught.value)
    assert CENTROID_RUN in message
    assert INSTANCE_RUN in message
    assert str(list(EARED.schema.names)) in message
    assert str(list(THREE_KEYPOINTS.schema.names)) in message
    assert _claims(ds) == []
    assert fake_sleap.tracked == []


@pytest.mark.parametrize(
    ("other", "named"),
    [
        (KeypointSchema(names=("nose", "tail")), ("edges [[0, 1]]", "edges []")),
        (
            KeypointSchema(
                names=("nose", "tail"), skeleton=((0, 1),), symmetries=((0, 1),)
            ),
            ("mirror pairs []", "mirror pairs [[0, 1]]"),
        ),
    ],
    ids=["edges", "mirror-pairs"],
)
def test_a_sleap_pair_whose_poses_share_their_names_is_refused_by_the_difference(
    ds: Dataset,
    fake_sleap: FakeSleapFrames,
    other: KeypointSchema,
    named: tuple[str, str],
) -> None:
    _ = _register_sleap(ds, CENTROID_RUN, head="centroid", pose=EARED)
    _ = _register_sleap(
        ds,
        INSTANCE_RUN,
        head="centered_instance",
        pose=PredictedPose(id=7, name="linked", schema=other),
        prepared_run=OTHER_PREPARED_RUN,
    )

    with pytest.raises(PredictFramesInvalid) as caught:
        _ = _run(ds, model=[CENTROID_RUN, INSTANCE_RUN])

    names = list(EARED.schema.names)
    assert (
        f"{CENTROID_RUN!r} was trained on the keypoints {names} with the {named[0]} "
        f"and {INSTANCE_RUN!r} on the keypoints {names} with the {named[1]}."
    ) in str(caught.value)
    assert fake_sleap.tracked == []


def test_the_sleap_nn_version_is_the_tool_version(
    ds: Dataset, fake_sleap: FakeSleapFrames
) -> None:
    _ = _register_sleap(ds, SLEAP_RUN)
    fake_sleap.version = "0.3.9"

    run_id = _run(ds, model=[SLEAP_RUN])

    assert fake_sleap.probed == [None]
    assert {row["tool_version"] for row in _rows(ds, run_id)} == {"0.3.9"}


def test_an_environment_without_sleap_nn_is_refused_before_any_claim(
    ds: Dataset, fake_sleap: FakeSleapFrames
) -> None:
    _ = _register_sleap(ds, SLEAP_RUN)
    fake_sleap.has_sleap_nn = False

    with pytest.raises(SleapNotFoundError):
        _ = _run(ds, model=[SLEAP_RUN])

    assert _claims(ds) == []
    assert fake_sleap.tracked == []


def test_every_held_claim_is_restamped_by_a_line_of_sleap_output(
    ds: Dataset, fake_sleap: FakeSleapFrames
) -> None:
    _ = _register_sleap(ds, SLEAP_RUN)
    refreshed: list[bool] = []

    def go_stale_then_print(call: SleapTrackCall) -> None:
        markers = _claims(ds)
        assert len(markers) == 2
        for marker in markers:
            claim = read_inflight(marker.parent)
            assert claim is not None
            write_inflight(
                marker.parent, claim.model_copy(update={"expires_at": LONG_AGO})
            )
        assert call.on_activity is not None
        call.on_activity("an error line")
        for marker in markers:
            claim = read_inflight(marker.parent)
            assert claim is not None
            refreshed.append(claim.expires_at != LONG_AGO)

    fake_sleap.during = go_stale_then_print

    _ = _run(ds, model=[SLEAP_RUN])

    assert refreshed == [True, True]
    assert [call.on_output for call in fake_sleap.tracked] == [None]


# --- Lightning Pose models -----------------------------------------------------------


@pytest.fixture
def fake_litpose(monkeypatch: pytest.MonkeyPatch) -> FakeLitposeFrames:
    """Stand in for the Lightning Pose interpreter, with no inherited device list."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    return install_fake_litpose_frames(monkeypatch)


def _register_litpose(
    ds: Dataset, *, model_type: str = "heatmap", pose: PredictedPose | None = None
) -> Path:
    """Register a train-litpose model of *model_type*, trained on *pose* when given.

    Returns the model directory.
    """
    directory = write_litpose_model(
        model_run_root(ds, "train-litpose", LITPOSE_RUN) / "model",
        model_type=model_type,
    )
    data_path = None if pose is None else _record_preparation(ds, pose)
    register_trained_model(
        ds,
        "train-litpose",
        LITPOSE_RUN,
        next(directory.rglob("*.ckpt")),
        directory=directory,
        data_path=data_path,
    )
    return directory


@pytest.mark.parametrize(
    "model_type", ["heatmap_mhcrnn", "heatmap_multiview_transformer"]
)
def test_a_litpose_model_that_single_frames_cannot_run_is_refused_before_any_claim(
    ds: Dataset, fake_litpose: FakeLitposeFrames, model_type: str
) -> None:
    _ = _register_litpose(ds, model_type=model_type)

    with pytest.raises(PredictFramesInvalid) as caught:
        _ = _run(ds, model=[LITPOSE_RUN], execution_id="exec-litpose-type")

    assert caught.value.error_kind == "invalid"
    message = str(caught.value)
    assert f"{LITPOSE_RUN!r} is a {model_type} model" in message
    assert "single-frame prediction does not support" in message
    assert _claims(ds) == []
    assert entry_error_lines(ds, "exec-litpose-type") == []
    assert fake_litpose.runs == []


def test_an_old_lightning_pose_is_unavailable_before_any_claim(
    ds: Dataset, fake_litpose: FakeLitposeFrames
) -> None:
    _ = _register_litpose(ds)
    fake_litpose.version = "2.2.0"

    with pytest.raises(PredictFramesUnavailable) as caught:
        _ = _run(ds, model=[LITPOSE_RUN], execution_id="exec-litpose-old")

    assert caught.value.error_kind == "unavailable"
    message = str(caught.value)
    assert "2.2.0" in message
    assert "2.3.1" in message
    assert _claims(ds) == []
    assert entry_error_lines(ds, "exec-litpose-old") == []
    assert len(fake_litpose.probes) == 1
    assert fake_litpose.predictions == []


def test_the_lightning_pose_version_is_the_tool_version(
    ds: Dataset, fake_litpose: FakeLitposeFrames
) -> None:
    _ = _register_litpose(ds)
    fake_litpose.version = "2.3.4"

    run_id = _run(ds, model=[LITPOSE_RUN])

    assert [run.argv == () for run in fake_litpose.runs] == [True, False]
    assert {row["tool_version"] for row in _rows(ds, run_id)} == {"2.3.4"}
    assert {row["model_kind"] for row in _rows(ds, run_id)} == {"train-litpose"}


@pytest.mark.parametrize(("device", "visible"), [("auto", None), ("1", "1")])
def test_the_device_reaches_the_lightning_pose_snippet(
    ds: Dataset, fake_litpose: FakeLitposeFrames, device: str, visible: str | None
) -> None:
    _ = _register_litpose(ds)

    _ = _run(ds, model=[LITPOSE_RUN], device=device)

    (run,) = fake_litpose.predictions
    assert run.env is not None
    assert run.env.get("CUDA_VISIBLE_DEVICES") == visible


def test_cpu_is_refused_for_a_litpose_model_at_plan_time(
    ds: Dataset, fake_litpose: FakeLitposeFrames
) -> None:
    _ = _register_litpose(ds)
    op = OPS[KIND]
    params = op.Params.model_validate(_params([LITPOSE_RUN], device="cpu"))

    with pytest.raises(PredictFramesInvalid) as caught:
        _ = op().plan_identity(ds, params, ds.resolve_scope(None))

    assert caught.value.error_kind == "invalid"
    assert "'cpu'" in str(caught.value)
    assert fake_litpose.runs == []


def test_a_litpose_model_without_a_recorded_pose_writes_its_keypoint_names(
    ds: Dataset, fake_litpose: FakeLitposeFrames
) -> None:
    _ = _register_litpose(ds)
    fake_litpose.keypoint_names = ("snout", "tail")

    run_id = _run(ds, model=[LITPOSE_RUN])

    for entry in (S1, S2):
        assert _predictions(ds, run_id, entry).pose == PredictedPose(
            id=None, name="", schema=KeypointSchema(names=("snout", "tail"))
        )


def test_a_litpose_model_with_a_recorded_pose_writes_that_pose(
    ds: Dataset, fake_litpose: FakeLitposeFrames
) -> None:
    _ = _register_litpose(ds, pose=EARED)
    fake_litpose.keypoint_names = EARED.schema.names

    run_id = _run(ds, model=[LITPOSE_RUN])

    assert [_predictions(ds, run_id, entry).pose for entry in (S1, S2)] == [EARED] * 2


def test_a_litpose_model_whose_recorded_pose_names_other_keypoints_is_a_conflict(
    ds: Dataset, fake_litpose: FakeLitposeFrames
) -> None:
    _ = _register_litpose(ds, pose=EARED)
    fake_litpose.keypoint_names = ("head", "tail")

    with pytest.raises(AllEntriesFailed):
        _ = _run(ds, model=[LITPOSE_RUN], execution_id="exec-litpose-names")

    lines = entry_error_lines(ds, "exec-litpose-names")
    assert len(lines) == 2
    assert all("PredictFramesConflict" in line for line in lines)
    assert all("head" in line and "nose" in line for line in lines)
    assert _claims(ds) == []


def test_two_entries_that_share_frame_names_get_their_own_litpose_predictions(
    ds: Dataset, fake_litpose: FakeLitposeFrames
) -> None:
    def by_entry(image: Path) -> list[tuple[float, float, float]]:
        x = 1.0 if image.parent.name == "s1" else 2.0
        return [(x, 0.0, 0.5), (x, 1.0, 0.5)]

    _ = _register_litpose(ds)
    fake_litpose.points = by_entry

    run_id = _run(ds, model=[LITPOSE_RUN])

    for entry, x in ((S1, 1.0), (S2, 2.0)):
        frames = _predictions(ds, run_id, entry).frames
        assert [frame.image.rsplit("/", 1)[1] for frame in frames] == [
            "frame_000010.png",
            "frame_000020.png",
        ]
        assert {obj.keypoints[0].x for frame in frames for obj in frame.objects} == {x}


def test_each_litpose_image_is_written_with_one_object_without_score_track_or_box(
    ds: Dataset, fake_litpose: FakeLitposeFrames
) -> None:
    _ = _register_litpose(ds)

    run_id = _run(ds, model=[LITPOSE_RUN], entries=[S1])

    frames = _predictions(ds, run_id, S1).frames
    assert [len(frame.objects) for frame in frames] == [1, 1]
    for frame in frames:
        (obj,) = frame.objects
        assert (obj.score, obj.track_id, obj.bbox) == (None, None, None)
        assert [keypoint.score for keypoint in obj.keypoints] == [0.5, 0.5]
    assert _rows(ds, run_id)[0]["n_objects"] == "2"


def test_every_held_claim_is_restamped_by_a_line_of_lightning_pose_output(
    ds: Dataset, fake_litpose: FakeLitposeFrames
) -> None:
    _ = _register_litpose(ds)
    refreshed: list[bool] = []

    def go_stale_then_print(run: LitposeSnippetRun) -> None:
        markers = _claims(ds)
        assert len(markers) == 2
        for marker in markers:
            claim = read_inflight(marker.parent)
            assert claim is not None
            write_inflight(
                marker.parent, claim.model_copy(update={"expires_at": LONG_AGO})
            )
        assert run.on_activity is not None
        run.on_activity("an error line")
        for marker in markers:
            claim = read_inflight(marker.parent)
            assert claim is not None
            refreshed.append(claim.expires_at != LONG_AGO)

    fake_litpose.during = go_stale_then_print

    _ = _run(ds, model=[LITPOSE_RUN])

    assert refreshed == [True, True]
    assert [run.on_output for run in fake_litpose.predictions] == [None]


def test_a_litpose_run_leaves_its_model_unchanged_and_no_work_directory(
    ds: Dataset, fake_litpose: FakeLitposeFrames
) -> None:
    _ = _register_litpose(ds)
    model_root = model_run_root(ds, "train-litpose", LITPOSE_RUN)
    before = _tree(model_root)

    run_id = _run(ds, model=[LITPOSE_RUN])

    assert _tree(model_root) == before
    (run,) = fake_litpose.predictions
    work_dir = Path(run.argv[1]).parent
    assert work_dir.parent == _run_root(ds, run_id)
    assert not work_dir.exists()
