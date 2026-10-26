"""What mosaic sends against what the runner accepts, checked side by side.

Every other test on this boundary fakes one half of it. The marker suite
replaces both seams and asserts on the ``TrackRequest`` mosaic *built*; the
runner has no suite of its own, because CI never builds the Ultralytics
environment. So a field mosaic populates with one meaning and the runner reads
with another, a renamed argument, a required value never sent -- each of those
passes the whole suite green and fails on a user's machine, at the one place
mosaic cannot see. This file is what stands between the two.

It costs nothing to run. Every ``ultralytics`` import in the runner program is
deferred into a function body, so the module imports here on the strength of
numpy, pandas, pydantic and mosaic-media alone -- all of which mosaic's own
environment holds -- and no AGPL-licensed code is loaded by any of this.
"""

from __future__ import annotations

import ast
import dataclasses
import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Final, Literal, Self, TypeAlias, override

import numpy as np
import pandas as pd
import pytest
from mosaic_media import MediaFacts
from pydantic import BaseModel

import mosaic.tracking.common.ultralytics_env as ultralytics_env
import mosaic.tracking.ops.infer as ops_infer
import mosaic.tracking.pose_training.ultralytics_infer as ultralytics_infer
import mosaic.tracking.ultralytics_track.dataset_runs as dataset_runs
from mosaic.tracking.ultralytics_track.params import UltralyticsParams
import mosaic.tracking.ultralytics_track.run as ultralytics_run
from mosaic.core.dataset import Dataset
from mosaic.core.media.read_target import verified_read_facts
from mosaic.core.pipeline.ops import run_op
from mosaic.core.pipeline.tracks_index import read_tracks_index
from mosaic.core.scope import Scope
from mosaic.core.track_library.ultralytics_tracks import raw_columns
from mosaic.tracking.common.tool_input import ToolFile
from mosaic.tracking.external.runner.ultralytics_protocol import (
    POINT_COLUMNS,
    POINT_DTYPES,
    ImageSource,
    InferPointsRequest,
    InferPoseRequest,
    InferTask,
    PredictImagesRequest,
    PredictImagesResponse,
    ProbeResponse,
    TrackRequest,
    UltralyticsInteropError,
    pose_rows_from_result,
    request_columns,
    rows_from_result,
)
from mosaic.tracking.ultralytics_track.tracker_defaults import TRACKER_NAMES

from tests.helpers import (
    FakeDetections,
    FakeResult,
    image_mark,
    install_fake_point_probe,
    install_fake_pose_probe,
    install_ultralytics_config,
    make_dataset,
    stub_media_dataset,
    write_marked_images,
    write_painted_entry,
)

# Selected by CI's `tracking` job with `-m tracker` rather than by a filename
# list in the workflow, so a new file here is covered the day it lands.
pytestmark = pytest.mark.tracker

_N_KEYPOINTS = 2

InferKind: TypeAlias = Literal["infer-pose", "infer-points"]
"""The two inference ops whose runner is this program."""


# --- the runner program, imported rather than spawned ----------------------


# --- what each side says about the request ---------------------------------


def _module_source(module: ModuleType) -> ast.Module:
    return ast.parse(Path(str(module.__file__)).read_text())


def _keywords_of_call(module: ModuleType, name: str) -> set[str]:
    """Every keyword *module* passes when it constructs *name*.

    Read out of the syntax rather than restated here, so the assertion moves
    with the code instead of pinning a list somebody has to remember to edit.
    """
    sent: set[str] = set()
    for node in ast.walk(_module_source(module)):
        if not isinstance(node, ast.Call):
            continue
        called = node.func
        if isinstance(called, ast.Name) and called.id == name:
            sent.update(
                keyword.arg for keyword in node.keywords if keyword.arg is not None
            )
    return sent


def _attributes_read_from(module: ModuleType, *annotations: str) -> set[str]:
    """Every attribute *module* reads off a parameter annotated one of *annotations*.

    Scoped to the functions that take one, rather than to every local called
    ``request``: the runner's probe path takes a ``ProbeRequest`` under that same
    name, and a ``tracker`` read there is not a field of the track request.
    """
    read: set[str] = set()
    for node in ast.walk(_module_source(module)):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        names = {
            argument.arg
            for argument in node.args.args
            if isinstance(argument.annotation, ast.Name)
            and argument.annotation.id in annotations
        }
        if not names:
            continue
        for inner in ast.walk(node):
            if (
                isinstance(inner, ast.Attribute)
                and isinstance(inner.value, ast.Name)
                and inner.value.id in names
            ):
                read.add(inner.attr)
    return read


def _attributes_read_off(modules: Sequence[ModuleType], local: str) -> set[str]:
    """Every attribute *modules* read off a local named *local*."""
    read: set[str] = set()
    for module in modules:
        for node in ast.walk(_module_source(module)):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == local
            ):
                read.add(node.attr)
    return read


def test_mosaic_sends_exactly_the_track_fields_the_runner_reads(
    runner_module: ModuleType,
) -> None:
    """Neither side carries a field the other has no use for.

    A field mosaic stops sending is a validation error on the first real run; a
    field the runner stops reading is worse, because it is silent -- a knob that
    reads as configured and does nothing.
    """
    sent = _keywords_of_call(dataset_runs, "TrackRequest")
    read = _attributes_read_from(runner_module, "TrackRequest")
    declared = set(TrackRequest.model_fields)

    # The extraction has to have found something before its verdict means
    # anything: an empty pair of sets agrees with itself.
    assert len(declared) > 15
    assert sent == declared
    assert read == declared


def test_the_runner_reports_exactly_the_probe_fields_mosaic_reads(
    runner_module: ModuleType,
) -> None:
    """Every reported field is read, and every read field is reported.

    Both branches of the probe are checked, because the one that answers for an
    environment with no Ultralytics fills the same model from different values --
    a field only the other branch sets would raise there and nowhere else.
    """
    reported = _keywords_of_call(runner_module, "ProbeResponse")
    read = _attributes_read_off(
        [dataset_runs, ultralytics_run, ultralytics_env, ultralytics_infer, ops_infer],
        "probe",
    )
    declared = set(ProbeResponse.model_fields)

    assert len(declared) > 5
    assert reported == declared
    assert read == declared


@pytest.mark.parametrize(
    ("name", "model"),
    [
        ("InferPoseRequest", InferPoseRequest),
        ("InferPointsRequest", InferPointsRequest),
    ],
)
def test_mosaic_sends_exactly_the_inference_fields_the_runner_reads(
    runner_module: ModuleType, name: str, model: type[BaseModel]
) -> None:
    """Neither side of an inference request carries a field the other ignores.

    The same check the track request gets, and it matters more here, because two
    requests share a base: a field read only off the base, or sent only on one
    subclass, is exactly the asymmetry a single-request contract cannot express.

    Reads are collected off the base annotations as well as the subclass, since
    the runner's video path takes an ``InferRequestBase``, its detection
    arguments are read off a ``PredictRequestBase``, and only the two entry
    points take the narrower type.
    """
    sent = _keywords_of_call(ops_infer, name)
    read = _attributes_read_from(
        runner_module, name, "InferRequestBase", "PredictRequestBase"
    )
    declared = set(model.model_fields)

    # The extraction has to have found something before its verdict means
    # anything: an empty pair of sets agrees with itself.
    assert len(declared) > 15
    assert sent == declared
    assert read == declared


# --- the column contract ---------------------------------------------------


def test_the_point_types_name_every_point_column_in_order() -> None:
    """The keys of the point types equal the point columns, in order.

    The runner casts its table to these types, and CI does not run the runner's
    point inference. A column added to one declaration and not the other would fail
    on a user's machine and nowhere else.
    """
    assert tuple(POINT_DTYPES) == POINT_COLUMNS


@pytest.mark.parametrize("n_keypoints", [1, _N_KEYPOINTS, 17])
def test_the_columns_mosaic_names_match_the_block_the_runner_writes(
    n_keypoints: int,
) -> None:
    """One name per value, in the order the extraction fills them.

    Mosaic owns the names and sends them; the runner owns the layout. ``n = 1``
    is the arm most likely to drift: a detect model makes the probe report one
    keypoint and leaves ``result.keypoints`` ``None``, so the box centre and the
    detection confidence are written as the single keypoint.
    """
    columns = raw_columns(n_keypoints)
    assert len(columns) == 8 + 3 * n_keypoints

    box = (10.0, 20.0, 30.0, 60.0)
    keypoints = (
        None
        if n_keypoints == 1
        else FakeDetections(
            np.array([[[1.0 + k, 2.0 + k, 0.5] for k in range(n_keypoints)]], float)
        )
    )
    result = FakeResult(
        boxes=FakeDetections(np.array([[*box, 7.0, 0.25, 3.0]], dtype=float)),
        keypoints=keypoints,
    )
    block = rows_from_result(result, 5, n_keypoints=n_keypoints)
    assert block is not None
    row = pd.DataFrame(block, columns=list(columns)).iloc[0]

    assert row["frame"] == 5.0
    assert row["track_id"] == 7.0
    assert (row["x1"], row["y1"], row["x2"], row["y2"]) == box
    assert row["conf"] == 0.25
    assert row["cls"] == 3.0
    if n_keypoints == 1:
        assert (row["kpx0"], row["kpy0"]) == (20.0, 40.0)  # the box centre
        assert row["kpp0"] == 0.25  # its detection confidence
    else:
        assert (row["kpx0"], row["kpy0"], row["kpp0"]) == (1.0, 2.0, 0.5)


@dataclasses.dataclass
class _Found:
    """One frame's result from weights that find poses and points in it.

    ``boxes`` contains the box of each pose, when a test gives them.
    """

    keypoints: FakeDetections
    locations: FakeDetections
    names: dict[int, str]
    boxes: FakeDetections | None = None

    def plot(self) -> np.ndarray:
        return np.zeros((1, 1, 3), np.uint8)


def _posed(keypoints: np.ndarray, boxes: np.ndarray | None = None) -> _Found:
    """Return a frame's result with *keypoints* and, when given, their *boxes*."""
    return _Found(
        keypoints=FakeDetections(keypoints),
        locations=FakeDetections(np.empty((0, 4))),
        names={0: "bee"},
        boxes=None if boxes is None else FakeDetections(boxes),
    )


_POSES: Final = np.array(
    [
        [[1.0, 2.0, 0.9], [5.0, 8.0, 0.8]],
        [[11.0, 12.0, 0.7], [15.0, 18.0, 0.6]],
    ]
)
"""Two poses of two keypoints, each keypoint with its visibility score."""

_INFER_POSE_BLOCK: Final = np.array(
    [
        [7.0, 0.0, 1.0, 2.0, 0.9, 5.0, 8.0, 0.8],
        [7.0, 1.0, 11.0, 12.0, 0.7, 15.0, 18.0, 0.6],
    ]
)
"""The rows that ``infer-pose`` writes for :data:`_POSES` in frame 7."""


def test_pose_rows_by_default_are_the_block_infer_pose_writes() -> None:
    """Frame, per-frame ordinal, then each keypoint's x, y and score, byte for byte."""
    block = pose_rows_from_result(_posed(_POSES), 7, n_keypoints=2)

    assert block is not None
    assert block.dtype == np.float64
    assert block.shape == _INFER_POSE_BLOCK.shape
    assert block.tobytes() == _INFER_POSE_BLOCK.tobytes()


def test_pose_rows_without_a_box_are_the_default_block() -> None:
    """``with_box=False`` and a score for absent visibility leave the block as it is.

    The poses carry their visibility, so *missing_score* writes nothing.
    """
    result = _posed(_POSES)

    for block in (
        pose_rows_from_result(result, 7, n_keypoints=2, with_box=False),
        pose_rows_from_result(result, 7, n_keypoints=2, missing_score=math.nan),
    ):
        assert block is not None
        assert block.tobytes() == _INFER_POSE_BLOCK.tobytes()


@pytest.mark.parametrize(
    "boxes",
    [
        np.array(
            [[0.5, 1.5, 6.5, 9.5, 0.95, 0.0], [10.5, 11.5, 16.5, 19.5, 0.85, 0.0]]
        ),
        np.array(
            [
                [0.5, 1.5, 6.5, 9.5, 4.0, 0.95, 0.0],
                [10.5, 11.5, 16.5, 19.5, 2.0, 0.85, 0.0],
            ]
        ),
    ],
    ids=["predicted", "tracked"],
)
def test_pose_rows_with_a_box_append_its_confidence_and_corners(
    boxes: np.ndarray,
) -> None:
    """Each pose row is followed by its detection's confidence and box corners.

    A predicted box row is ``(x1, y1, x2, y2, conf, cls)``, and a tracked one has
    its track id before ``conf``. Both give the confidence, not the track id.
    """
    block = pose_rows_from_result(
        _posed(_POSES, boxes), 7, n_keypoints=2, with_box=True
    )

    assert block is not None
    table = pd.DataFrame(block, columns=request_columns("pose", 2))
    assert table["det_conf"].tolist() == [0.95, 0.85]
    corners = table[["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]].to_numpy()
    assert corners.tolist() == boxes[:, 0:4].tolist()
    assert np.array_equal(block[:, : _INFER_POSE_BLOCK.shape[1]], _INFER_POSE_BLOCK)


@pytest.mark.parametrize(
    ("boxes", "refusal"),
    [
        (None, "boxes for 2 poses"),
        (np.array([[0.5, 1.5, 6.5, 9.5, 0.95, 0.0]]), "boxes for 2 poses"),
        (
            np.array([[0.5, 1.5, 6.5, 9.5, 0.95], [10.5, 11.5, 16.5, 19.5, 0.85]]),
            "should carry 6 columns",
        ),
    ],
    ids=["no-boxes", "one-box-for-two-poses", "boxes-without-a-class"],
)
def test_pose_rows_with_a_box_refuse_boxes_that_are_not_the_poses(
    boxes: np.ndarray | None, refusal: str
) -> None:
    """A pose row gains a box only when the boxes are the poses' detections.

    A box row narrower than ``(x1, y1, x2, y2, conf, cls)`` has no confidence at
    the place the block reads it from.
    """
    with pytest.raises(UltralyticsInteropError, match=refusal):
        _ = pose_rows_from_result(
            _posed(_POSES, boxes), 7, n_keypoints=2, with_box=True
        )


def test_a_pose_without_visibility_scores_its_keypoints_with_the_missing_score() -> (
    None
):
    """Keypoints of two channels get *missing_score*, which is 1.0 by default."""
    result = _posed(_POSES[:, :, 0:2])

    scored = pose_rows_from_result(result, 7, n_keypoints=2)
    unscored = pose_rows_from_result(result, 7, n_keypoints=2, missing_score=math.nan)

    assert scored is not None
    assert unscored is not None
    scores = [4, 7]
    positions = [2, 3, 5, 6]
    assert scored[:, scores].tolist() == [[1.0, 1.0], [1.0, 1.0]]
    assert np.isnan(unscored[:, scores]).all()
    assert np.array_equal(scored[:, positions], _INFER_POSE_BLOCK[:, positions])
    assert np.array_equal(unscored[:, positions], _INFER_POSE_BLOCK[:, positions])


# --- the whole exchange, over a supervisor that answers as the runner does ---


@pytest.fixture
def ds(tmp_path: Path) -> Dataset:
    return stub_media_dataset(tmp_path, ["vid1"])


@pytest.fixture
def model(tmp_path: Path) -> Path:
    path = tmp_path / "yolo" / "best.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_bytes(b"weights")
    return path


class RunnerStandIn:
    """A supervised process that answers exactly as the runner program would.

    It parses the argv mosaic built with the runner's *own* parser, validates the
    request file with the runner's *own* models, and writes its answers through
    the runner's own writer. Nothing here restates what either side does, so a
    renamed flag, a moved field or a changed response shape fails here rather
    than on a machine with the environment built.
    """

    def __init__(self, module: ModuleType, n_keypoints: int) -> None:
        self.module: ModuleType = module
        self.n_keypoints: int = n_keypoints
        self.commands: list[str] = []
        self.payloads: list[str] = []
        """Each request as it was written, kept so a test can read it back.

        Whether the runner *accepts* one is settled below, by validating it with
        the runner's own model; re-reading a payload here with mosaic's model is
        for inspecting field values, which the two agree on by then.
        """
        self.reconstructed_facts: list[MediaFacts] = []

    def __call__(
        self,
        argv: Sequence[str],
        *,
        env: dict[str, str] | None = None,
        cancel_check: Callable[[], bool] | None = None,
        timeout: float | None = None,
        idle_timeout: float | None = None,
        poll_interval: float = 0.5,
        on_output: Callable[[str], None] | None = None,
    ) -> tuple[str, str, int]:
        tokens = [str(token) for token in argv]
        assert Path(tokens[1]).is_file(), "mosaic spawns a program that is there"
        parsed = self.module._parser().parse_args(tokens[2:])
        command = str(parsed.command)
        self.commands.append(command)
        payload = Path(str(parsed.request)).read_text()
        self.payloads.append(payload)
        answer = self._probe(payload) if command == "probe" else self._track(payload)
        Path(str(parsed.out)).write_text(answer)
        return ("", "", 0)

    def _probe(self, payload: str) -> str:
        # Validated with the runner's own model: this is where a field mosaic
        # renamed, dropped or retyped stops the test.
        _ = self.module.ProbeRequest.model_validate_json(payload)
        return self.module.ProbeResponse(
            has_ultralytics=True,
            has_lap=True,
            has_locate=False,
            ultralytics_version="8.4.63",
            tracker_names=list(TRACKER_NAMES),
            model_task="pose",
            n_keypoints=self.n_keypoints,
            model_load_error="",
            installed_tracker_table={"track_buffer": 30},
        ).model_dump_json()

    def _track(self, payload: str) -> str:
        request = self.module.TrackRequest.model_validate_json(payload)
        # The facts mosaic flattened, rebuilt the way the runner rebuilds them
        # before handing them to its reader.
        self.reconstructed_facts.extend(
            self.module._media_facts(source.media_facts) for source in request.sources
        )
        columns = list(request.columns)
        rows = [
            [float(frame), 1.0, 10.0, 20.0, 15.0, 25.0, 0.9, 0.0]
            + [0.0] * (len(columns) - 8)
            for frame in range(4)
        ]
        table = pd.DataFrame(np.array(rows, dtype=float), columns=columns)
        table = table.astype({"frame": "int64", "track_id": "int64", "cls": "int64"})
        self.module._publish_parquet(table, request.output_parquet)
        return self.module.TrackResponse(n_frames=4, n_ids=1).model_dump_json()


@pytest.fixture
def stand_in(
    runner_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> Iterator[RunnerStandIn]:
    fake = RunnerStandIn(runner_module, _N_KEYPOINTS)
    monkeypatch.setattr(ultralytics_env, "run_supervised", fake)
    yield fake


def test_a_run_speaks_a_protocol_the_runner_program_accepts(
    ds: Dataset, model: Path, stand_in: RunnerStandIn
) -> None:
    """One run, end to end, with only the process boundary replaced.

    Production builds the argv, the request and the tracker configuration; the
    runner's own parser and models are what accept them; and what mosaic reads
    back is what the runner's own models serialize. The tracks row at the end is
    the proof that the columns mosaic sent are the ones its converter reads.
    """
    run_id = dataset_runs.run_ultralytics(
        ds,
        UltralyticsParams(model_path=str(model)),
        ultralytics_bin="/x/bin/yolo",
    )

    assert stand_in.commands == ["probe", "track"]
    request = TrackRequest.model_validate_json(stand_in.payloads[1])
    assert list(request.columns) == list(raw_columns(_N_KEYPOINTS))
    assert Path(request.output_parquet).is_file()
    # The `project` argument Ultralytics computes its run directory from, which
    # is the run root and not the entry directory beside it.
    assert Path(request.project_dir) == dataset_runs.ultralytics_run_root(ds, run_id)

    tracks = read_tracks_index(ds)
    assert len(tracks) == 1
    assert int(tracks.iloc[0]["n_rows"]) == 4
    assert str(tracks.iloc[0]["producer_run_id"]) == run_id


def test_gated_media_facts_survive_the_crossing(
    ds: Dataset,
    model: Path,
    stand_in: RunnerStandIn,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    write_cfr_mp4: Callable[..., None],
) -> None:
    """Measured, gated, flattened, serialized, and equal on the far side.

    The facts are what keeps a raw stream reading with its true frame count
    instead of the garbage one its header declares, so a field lost in the
    crossing is a misindexed track under a valid identifier. Compared against
    the same gate mosaic ran, on the same file.
    """
    video = tmp_path / "clip.mp4"
    write_cfr_mp4(video, frames=6, size=(64, 48))
    measured = verified_read_facts(video, None, "analysis")[0]

    def resolved(*_args: object, **_kwargs: object) -> tuple[ToolFile, ...]:
        return (ToolFile(video, measured),)

    monkeypatch.setattr(dataset_runs, "entry_runner_sources", resolved)
    _ = dataset_runs.run_ultralytics(
        ds,
        UltralyticsParams(model_path=str(model)),
        ultralytics_bin="/x/bin/yolo",
    )

    assert stand_in.reconstructed_facts == [measured]
    # Flattened rather than pickled: what crossed is JSON a reader can inspect.
    request = TrackRequest.model_validate_json(stand_in.payloads[1])
    (source,) = request.sources
    assert source.media_facts == dataclasses.asdict(measured)


# --- a frame window, read by the runner's own loop ---------------------------


class _FindsOneOfEach:
    """Weights that find one two-keypoint pose and one point in every frame."""

    def predict(self, source: list[np.ndarray], **_kwargs: object) -> list[_Found]:
        pose = np.array([[[1.0, 2.0, 0.9], [5.0, 8.0, 0.8]]])
        point = np.array([[3.0, 4.0, 0.9, 0.0]])
        return [
            _Found(FakeDetections(pose), FakeDetections(point), {0: "bee"})
            for _ in source
        ]


class _RecordsItsArguments(_FindsOneOfEach):
    """Weights that find one of each and keep the keywords of every call."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    @override
    def predict(self, source: list[np.ndarray], **kwargs: object) -> list[_Found]:
        self.calls.append(kwargs)
        return super().predict(source, **kwargs)


_TRAINED_RADII: dict[int, float] = {0: 100.0, 1: 40.0}

_PREDICTED_WITH: Final[Mapping[str, object]] = {
    "device": "0",
    "conf": 0.4,
    "imgsz": 320,
    "max_det": 50,
    "agnostic_nms": True,
    "augment": False,
    "verbose": False,
}
"""The detection arguments of every ``predict`` call below, precision aside."""


@dataclasses.dataclass
class _LocateModel:
    radii: dict[int, float]


class _Checkpointed(_RecordsItsArguments):
    """Recording weights whose checkpoint was trained at ``dor`` 0.3 with radii.

    Shaped as a POLO checkpoint loads: ``train_args`` under ``ckpt``, and the
    radii the trainer copies from the dataset on the model it saved.
    """

    def __init__(self) -> None:
        super().__init__()
        self.ckpt: dict[str, object] = {"train_args": {"dor": 0.3, "epochs": 200}}
        self.model = _LocateModel(radii=_TRAINED_RADII)


def _gray(_frame: int) -> np.ndarray:
    return np.full((48, 64, 3), 100, np.uint8)


class _FramesInMemory:
    """A reader of blank frames under the given numbers, which records its closing."""

    def __init__(self, numbers: Sequence[int]) -> None:
        self._numbers: list[int] = list(numbers)
        self._count: int = len(self._numbers)
        self.closed: bool = False

    def __len__(self) -> int:
        return self._count

    def read_batch(self, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        taken = self._numbers[:batch_size]
        self._numbers = self._numbers[batch_size:]
        return np.array(taken, np.int64), np.zeros((len(taken), 4, 4, 3), np.uint8)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        self.closed = True


def _two_keypoint_rows(result: _Found, frame_index: int) -> np.ndarray | None:
    return pose_rows_from_result(result, frame_index, n_keypoints=2)


def test_the_frame_loop_numbers_rows_by_what_any_sized_reader_returns(
    runner_module: ModuleType, tmp_path: Path
) -> None:
    """The loop reads a reader that is not an entry's video, and closes it.

    Rows and drawn frames take the numbers that the reader returns, and
    ``max_frames`` ends the reading inside a batch.
    """
    reader = _FramesInMemory([3, 8, 9, 12, 20])
    weights = _RecordsItsArguments()
    annotated = tmp_path / "annotated"

    outcome = runner_module._infer_frames(
        reader,
        weights,
        {"conf": 0.5},
        _two_keypoint_rows,
        batch_size=2,
        prefetch=False,
        max_frames=3,
        annotated_dir=annotated,
    )

    assert outcome.n_frames == 3
    assert [float(block[0, 0]) for block in outcome.blocks] == [3.0, 8.0, 9.0]
    assert outcome.names == {0: "bee"}
    assert weights.calls == [{"conf": 0.5}, {"conf": 0.5}]
    assert reader.closed
    assert sorted(path.name for path in annotated.iterdir()) == [
        "frame_00000003.jpg",
        "frame_00000008.jpg",
        "frame_00000009.jpg",
    ]


@dataclasses.dataclass
class _LocateWeights:
    """Loaded weights that report the ``locate`` task, as a POLO checkpoint does."""

    task: str = "locate"


def test_weights_of_a_task_other_than_the_declared_one_are_refused(
    runner_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The weights at the given path load, and are checked against the given task."""
    package = install_ultralytics_config(monkeypatch)
    loaded: list[str] = []

    def load(path: str) -> _LocateWeights:
        loaded.append(path)
        return _LocateWeights()

    monkeypatch.setattr(package, "YOLO", load, raising=False)

    assert runner_module._loaded_for("points.pt", "locate") == _LocateWeights()
    with pytest.raises(runner_module.RunnerError, match="declares task='pose'"):
        _ = runner_module._loaded_for("points.pt", "pose")
    assert loaded == ["points.pt", "points.pt"]


def _infer_in_process(
    runner_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    kind: InferKind,
    params: dict[str, object],
    weights: _FindsOneOfEach,
    *,
    quantize: bool = True,
) -> Dataset:
    """Run *kind* with *params* over a twenty-frame clip at 25 fps.

    The runner program runs in this process, through its own parser, request
    models, reader and frame loop. Only the probe, *weights* and the release's
    shipped configuration, whose half-precision keyword *quantize* picks, are
    stood in for.
    """
    ds = make_dataset(tmp_path / "ds")
    _ = write_painted_entry(ds, "sess", [(20, 25.0)], _gray)
    model = tmp_path / "weights" / "best.pt"
    model.parent.mkdir()
    _ = model.write_bytes(b"weights")
    if kind == "infer-pose":
        _ = install_fake_pose_probe(monkeypatch)
    else:
        _ = install_fake_point_probe(monkeypatch)
    _ = install_ultralytics_config(monkeypatch, quantize=quantize)

    def loaded_for(*_args: object) -> _FindsOneOfEach:
        return weights

    monkeypatch.setattr(runner_module, "_loaded_for", loaded_for)

    def run_runner(
        _env: object,
        _failure: object,
        subcommand: str,
        request_path: Path,
        response_path: Path,
        **_kwargs: object,
    ) -> tuple[str, str]:
        argv = [subcommand, "--request", str(request_path), "--out", str(response_path)]
        assert runner_module.main(argv) == 0
        return "", ""

    monkeypatch.setattr(ultralytics_infer, "run_runner", run_runner)

    _ = run_op(
        ds,
        kind,
        {"model": str(model), **params},
        scope=Scope(entries=[("", "sess")]),
    )
    return ds


@pytest.mark.media
@pytest.mark.parametrize("kind", ["infer-pose", "infer-points"])
def test_a_windowed_run_publishes_the_frames_that_the_runner_read(
    runner_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    kind: InferKind,
    requires_ffmpeg: None,
) -> None:
    """``start_frame`` 5 and ``frame_step`` 2 publish frames 5, 7, 9 and so on."""
    ds = _infer_in_process(
        runner_module,
        monkeypatch,
        tmp_path,
        kind,
        {"start_frame": 5, "frame_step": 2},
        _FindsOneOfEach(),
    )

    (path,) = read_tracks_index(ds)["abs_path"].tolist()
    table = pd.read_parquet(ds.resolve_path(str(path)))
    read = list(range(5, 20, 2))
    assert table["frame"].tolist() == read
    assert table["time"].tolist() == pytest.approx([frame / 25.0 for frame in read])
    # The one detection of each frame is number 0 of that frame, under the name
    # that the schema gives it.
    assert table["id"].tolist() == [0] * len(read)
    assert "detection_id" not in table.columns


@pytest.mark.media
@pytest.mark.parametrize(
    ("params", "weights", "expected"),
    [
        ({"dor": 0.55}, _RecordsItsArguments, {"dor": 0.55}),
        ({"dor": 0.55}, _Checkpointed, {"dor": 0.55, "radii": _TRAINED_RADII}),
        ({}, _Checkpointed, {"dor": 0.3, "radii": _TRAINED_RADII}),
        ({}, _RecordsItsArguments, {}),
    ],
    ids=["stated", "stated-over-checkpoint", "checkpoint", "neither"],
)
def test_point_suppression_takes_the_stated_dor_or_else_the_checkpoints(
    runner_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    requires_ffmpeg: None,
    params: dict[str, object],
    weights: Callable[[], _RecordsItsArguments],
    expected: dict[str, object],
) -> None:
    """Every ``predict`` call gets the stated ``dor``, else what the checkpoint recorded.

    0.55 is neither the fork's default nor the checkpoint's, so a stated value
    lost before the call cannot pass by falling back to either. A checkpoint
    that records neither ``dor`` nor radii leaves both keywords out, and the
    fork its own defaults.
    """
    recorder = weights()
    _ = _infer_in_process(
        runner_module, monkeypatch, tmp_path, "infer-points", params, recorder
    )

    assert recorder.calls
    for call in recorder.calls:
        assert {key: call[key] for key in ("dor", "radii") if key in call} == expected


@pytest.mark.media
@pytest.mark.parametrize("kind", ["infer-pose", "infer-points"])
@pytest.mark.parametrize(
    ("quantize", "half_precision"),
    [(True, {"quantize": "fp16"}), (False, {"half": True})],
    ids=["quantize", "half"],
)
def test_the_predictor_receives_every_detection_argument(
    runner_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    kind: InferKind,
    quantize: bool,
    half_precision: dict[str, object],
    requires_ffmpeg: None,
) -> None:
    """No detection argument is left to the installed release's default.

    A lock refresh could retune a default under an identifier already minted.
    Precision travels under the keyword the installed release takes, and
    ``iou`` reaches the pose predictor alone, because POLO's locate NMS reads
    none. The weights record no checkpoint, so ``infer-points`` sends no
    ``dor`` or radii.
    """
    params: dict[str, object] = {
        "conf_threshold": 0.4,
        "imgsz": 320,
        "max_det": 50,
        "agnostic_nms": True,
        "precision": "fp16",
    }
    expected: dict[str, object] = {**_PREDICTED_WITH, **half_precision}
    if kind == "infer-pose":
        params["iou"] = 0.5
        expected["iou"] = 0.5
    recorder = _RecordsItsArguments()

    _ = _infer_in_process(
        runner_module, monkeypatch, tmp_path, kind, params, recorder, quantize=quantize
    )

    assert recorder.calls
    assert all(call == expected for call in recorder.calls), recorder.calls


@pytest.mark.media
def test_the_point_predictor_receives_its_suppression_with_the_detection_arguments(
    runner_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    requires_ffmpeg: None,
) -> None:
    """Point inference sends the stated ``dor``, the trained radii and no ``iou``."""
    params: dict[str, object] = {
        "conf_threshold": 0.4,
        "imgsz": 320,
        "max_det": 50,
        "agnostic_nms": True,
        "precision": "fp16",
        "dor": 0.55,
    }
    expected: dict[str, object] = {
        **_PREDICTED_WITH,
        "quantize": "fp16",
        "dor": 0.55,
        "radii": _TRAINED_RADII,
    }
    recorder = _Checkpointed()

    _ = _infer_in_process(
        runner_module, monkeypatch, tmp_path, "infer-points", params, recorder
    )

    assert recorder.calls
    assert all(call == expected for call in recorder.calls), recorder.calls


# --- a list of images, read by the runner's loop -----------------------------


_IMAGE_SHAPES: Final = ((48, 64), (32, 40), (48, 64), (32, 40), (32, 40))
"""The ``(height, width)`` of the image at each position: two shapes, interleaved."""

_IMAGE_BOX: Final = (0.5, 1.5, 6.5, 9.5)
"""The corners of every pose's box."""

_IMAGE_BOX_CONF: Final = 0.95


class _ReadsEachImage(_RecordsItsArguments):
    """Recording weights that find one boxed pose and one point in each image.

    Both keypoints and the point lie at an x equal to the image's blue, which
    :func:`image_mark` sets from the image's position. The pose lacks a
    visibility channel. The shapes of each call's images are kept.
    """

    def __init__(self) -> None:
        super().__init__()
        self.shapes: list[list[tuple[int, ...]]] = []

    @override
    def predict(self, source: list[np.ndarray], **kwargs: object) -> list[_Found]:
        self.calls.append(kwargs)
        self.shapes.append([image.shape for image in source])
        return [self._found(float(image[0, 0, 0])) for image in source]

    @staticmethod
    def _found(x: float) -> _Found:
        return _Found(
            keypoints=FakeDetections(np.array([[[x, 2.0], [x, 8.0]]])),
            locations=FakeDetections(np.array([[x, 4.0, 0.9, 0.0]])),
            names={0: "bee"},
            boxes=FakeDetections(np.array([[*_IMAGE_BOX, _IMAGE_BOX_CONF, 0.0]])),
        )


class _CheckpointedImages(_ReadsEachImage, _Checkpointed):
    """Image-reading weights whose checkpoint was trained at ``dor`` 0.3 with radii."""


def _images_request(
    images: Sequence[ImageSource],
    rows_path: Path,
    *,
    task: InferTask,
    iou: float | None,
    dor: float | None = None,
    columns: Sequence[str] | None = None,
) -> PredictImagesRequest:
    """Return a request of *task* over *images*, under the detection arguments above.

    The columns are those that mosaic sends for *task* unless *columns* is given.
    """
    sent = request_columns(task, _N_KEYPOINTS)
    return PredictImagesRequest(
        model_path="best.pt",
        task=task,
        images=list(images),
        columns=sent if columns is None else list(columns),
        conf=0.4,
        iou=iou,
        imgsz=320,
        max_det=50,
        agnostic_nms=True,
        device="0",
        precision="fp16",
        batch_size=2,
        n_keypoints=_N_KEYPOINTS if task == "pose" else 1,
        dor=dor,
        rows_path=str(rows_path),
    )


def _answer_predict_images(
    runner_module: ModuleType, tmp_path: Path, request: PredictImagesRequest
) -> PredictImagesResponse:
    """Run *request* through the runner's parser, request model and loop."""
    request_path = tmp_path / "request.json"
    response_path = tmp_path / "response.json"
    _ = request_path.write_text(request.model_dump_json())
    argv = [
        "predict-images",
        "--request",
        str(request_path),
        "--out",
        str(response_path),
    ]
    assert runner_module.main(argv) == 0
    return PredictImagesResponse.model_validate_json(response_path.read_text())


def _predict_images(
    runner_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    request: PredictImagesRequest,
    weights: _ReadsEachImage,
    *,
    quantize: bool = True,
) -> PredictImagesResponse:
    """Run *request* over *weights*, standing in for nothing else of Ultralytics.

    The release's shipped configuration, whose half-precision keyword *quantize*
    picks, is the one other stand-in.
    """
    _ = install_ultralytics_config(monkeypatch, quantize=quantize)

    def loaded_for(*_args: object) -> _ReadsEachImage:
        return weights

    monkeypatch.setattr(runner_module, "_loaded_for", loaded_for)
    return _answer_predict_images(runner_module, tmp_path, request)


def test_the_runner_reads_every_predict_images_field(
    runner_module: ModuleType,
) -> None:
    """The runner reads each field of the request and fills each of the response.

    The detection arguments are read off the ``PredictRequestBase`` that the
    image path shares with the video path.
    """
    read = _attributes_read_from(
        runner_module, "PredictImagesRequest", "PredictRequestBase"
    )
    reported = _keywords_of_call(runner_module, "PredictImagesResponse")
    declared = set(PredictImagesRequest.model_fields)

    assert len(declared) > 10
    assert read == declared
    assert reported == set(PredictImagesResponse.model_fields)


def test_mosaic_sends_every_predict_images_field() -> None:
    """The frame adapter names each field of the request that it builds."""
    sent = _keywords_of_call(ultralytics_infer, "PredictImagesRequest")

    assert sent == set(PredictImagesRequest.model_fields)


def test_predict_images_predicts_each_batch_on_images_of_one_shape(
    runner_module: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Each call gets up to a batch of images of one shape, the smaller shape first."""
    images = write_marked_images(tmp_path, _IMAGE_SHAPES)
    rows_path = tmp_path / "rows.parquet"
    weights = _ReadsEachImage()

    response = _predict_images(
        runner_module,
        monkeypatch,
        tmp_path,
        _images_request(images, rows_path, task="pose", iou=0.5),
        weights,
    )

    small, large = (32, 40, 3), (48, 64, 3)
    assert weights.shapes == [[small, small], [small], [large, large]]
    assert response == PredictImagesResponse(
        n_images=len(_IMAGE_SHAPES), n_rows=len(_IMAGE_SHAPES), rows_path=str(rows_path)
    )


@pytest.mark.parametrize(
    ("quantize", "half_precision"),
    [(True, {"quantize": "fp16"}), (False, {"half": True})],
    ids=["quantize", "half"],
)
@pytest.mark.parametrize(
    ("task", "iou", "dor", "weights", "suppression"),
    [
        ("pose", 0.5, None, _ReadsEachImage, {"iou": 0.5}),
        (
            "locate",
            None,
            0.55,
            _CheckpointedImages,
            {"dor": 0.55, "radii": _TRAINED_RADII},
        ),
    ],
    ids=["pose", "locate"],
)
def test_predict_images_passes_every_detection_argument(
    runner_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    quantize: bool,
    half_precision: dict[str, object],
    task: InferTask,
    iou: float | None,
    dor: float | None,
    weights: Callable[[], _ReadsEachImage],
    suppression: dict[str, object],
) -> None:
    """The arguments of each call equal the detection arguments that inference passes.

    Only the pose predictor gets ``iou``. The point predictor gets the request's
    ``dor`` and the radii that its checkpoint recorded.
    """
    recorder = weights()
    request = _images_request(
        write_marked_images(tmp_path, _IMAGE_SHAPES),
        tmp_path / "rows.parquet",
        task=task,
        iou=iou,
        dor=dor,
    )

    _ = _predict_images(
        runner_module, monkeypatch, tmp_path, request, recorder, quantize=quantize
    )

    expected = {**_PREDICTED_WITH, **half_precision, **suppression}
    assert recorder.calls
    assert all(call == expected for call in recorder.calls), recorder.calls


@pytest.mark.parametrize(
    ("dor", "weights", "expected"),
    [
        (0.55, _ReadsEachImage, {"dor": 0.55}),
        (0.55, _CheckpointedImages, {"dor": 0.55, "radii": _TRAINED_RADII}),
        (None, _CheckpointedImages, {"dor": 0.3, "radii": _TRAINED_RADII}),
        (None, _ReadsEachImage, {}),
    ],
    ids=["stated", "stated-over-checkpoint", "checkpoint", "neither"],
)
def test_predict_images_suppresses_points_at_the_request_dor_and_the_trained_radii(
    runner_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    dor: float | None,
    weights: Callable[[], _ReadsEachImage],
    expected: dict[str, object],
) -> None:
    """A point model gets the stated ``dor``, else the checkpoint's, and its radii.

    A checkpoint moved off the machine that trained it otherwise suppresses at
    radius 1. The request omits radii.
    """
    recorder = weights()
    request = _images_request(
        write_marked_images(tmp_path, _IMAGE_SHAPES),
        tmp_path / "rows.parquet",
        task="locate",
        iou=None,
        dor=dor,
    )

    _ = _predict_images(runner_module, monkeypatch, tmp_path, request, recorder)

    assert recorder.calls
    for call in recorder.calls:
        assert {key: call[key] for key in ("dor", "radii") if key in call} == expected


def _assert_refused_before_the_weights_load(
    runner_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    request: PredictImagesRequest,
    refusal: str,
) -> None:
    """Assert that *request* is refused with *refusal*, with no weights loaded.

    No row is written either.
    """
    loaded: list[str] = []

    def loaded_for(model_path: str, _task: str) -> _ReadsEachImage:
        loaded.append(model_path)
        return _ReadsEachImage()

    monkeypatch.setattr(runner_module, "_loaded_for", loaded_for)

    with pytest.raises(runner_module.RunnerError, match=refusal):
        _ = _answer_predict_images(runner_module, tmp_path, request)
    assert loaded == []
    assert not Path(request.rows_path).exists()


@pytest.mark.parametrize(
    ("task", "iou", "refusal"),
    [
        ("locate", 0.5, "locate request sets iou"),
        ("pose", None, "pose request leaves iou unset"),
    ],
    ids=["locate-with-iou", "pose-without-iou"],
)
def test_predict_images_refuses_an_iou_that_its_task_does_not_read(
    runner_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    task: InferTask,
    iou: float | None,
    refusal: str,
) -> None:
    """POLO's point suppression ignores ``iou``, and a pose NMS reads it."""
    request = _images_request(
        write_marked_images(tmp_path, _IMAGE_SHAPES),
        tmp_path / "rows.parquet",
        task=task,
        iou=iou,
    )

    _assert_refused_before_the_weights_load(
        runner_module, monkeypatch, tmp_path, request, refusal
    )


def _two_at_one_position(directory: Path) -> list[ImageSource]:
    """Return the marked images, the second moved to the first's position."""
    first, second, *rest = write_marked_images(directory, _IMAGE_SHAPES)
    return [first, ImageSource(position=first.position, path=second.path), *rest]


def _one_not_a_png(directory: Path) -> list[ImageSource]:
    """Return the marked images and, last, a file that is not a PNG."""
    path = directory / "not_a_png.png"
    _ = path.write_bytes(b"GIF89a")
    images = write_marked_images(directory, _IMAGE_SHAPES)
    return [*images, ImageSource(position=len(images), path=str(path))]


def _one_absent(directory: Path) -> list[ImageSource]:
    """Return the marked images and, last, a path with no file at it."""
    images = write_marked_images(directory, _IMAGE_SHAPES)
    absent = str(directory / "absent.png")
    return [*images, ImageSource(position=len(images), path=absent)]


@pytest.mark.parametrize(
    ("images", "refusal"),
    [
        (_two_at_one_position, r"positions \[0\]"),
        (_one_not_a_png, "not_a_png.png is not a PNG"),
        (_one_absent, "absent.png could not be opened"),
    ],
    ids=["two-at-one-position", "not-a-png", "absent"],
)
def test_predict_images_refuses_its_images_before_the_weights_load(
    runner_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    images: Callable[[Path], list[ImageSource]],
    refusal: str,
) -> None:
    """The refusals of the images read only their PNG headers.

    A cold checkpoint load is the longest step of a run, and a request that the
    headers refuse needs none.
    """
    request = _images_request(
        images(tmp_path), tmp_path / "rows.parquet", task="pose", iou=0.5
    )

    _assert_refused_before_the_weights_load(
        runner_module, monkeypatch, tmp_path, request, refusal
    )


def test_a_pose_request_against_locate_weights_is_refused(
    runner_module: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``predict-images`` checks the loaded weights against the request's task."""
    package = install_ultralytics_config(monkeypatch)

    def load(_path: str) -> _LocateWeights:
        return _LocateWeights()

    monkeypatch.setattr(package, "YOLO", load, raising=False)
    rows_path = tmp_path / "rows.parquet"
    request = _images_request(
        write_marked_images(tmp_path, _IMAGE_SHAPES), rows_path, task="pose", iou=0.5
    )

    with pytest.raises(runner_module.RunnerError, match="declares task='pose'"):
        _ = _answer_predict_images(runner_module, tmp_path, request)
    assert not rows_path.exists()


def test_predict_images_writes_pose_rows_by_position_with_their_boxes(
    runner_module: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Each pose row is numbered by its image's position, under ``request.columns``.

    The pose lacks a visibility channel, and each keypoint score is therefore
    NaN. The row continues with its detection's confidence and box.
    """
    rows_path = tmp_path / "rows.parquet"
    request = _images_request(
        write_marked_images(tmp_path, _IMAGE_SHAPES), rows_path, task="pose", iou=0.5
    )

    _ = _predict_images(
        runner_module, monkeypatch, tmp_path, request, _ReadsEachImage()
    )

    table = pd.read_parquet(rows_path)
    n = len(_IMAGE_SHAPES)
    assert list(table.columns) == request.columns
    assert (str(table["frame"].dtype), str(table["id"].dtype)) == ("int64", "int64")
    assert sorted(table["frame"].tolist()) == list(range(n))
    named = [float(image_mark(int(frame))[0]) for frame in table["frame"]]
    assert table["poseX0"].tolist() == named
    assert table["poseX1"].tolist() == named
    assert table["id"].tolist() == [0] * n
    assert table[["poseP0", "poseP1"]].isna().all().all()
    assert table["det_conf"].tolist() == [_IMAGE_BOX_CONF] * n
    corners = table[["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]].to_numpy()
    assert corners.tolist() == [list(_IMAGE_BOX)] * n


def test_predict_images_writes_point_rows_by_position_in_the_requested_order(
    runner_module: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Point rows take the order of ``request.columns`` and the point types.

    The columns are sent reversed. A table written in the runner's order then
    fails.
    """
    rows_path = tmp_path / "rows.parquet"
    request = _images_request(
        write_marked_images(tmp_path, _IMAGE_SHAPES),
        rows_path,
        task="locate",
        iou=None,
        columns=POINT_COLUMNS[::-1],
    )

    _ = _predict_images(
        runner_module, monkeypatch, tmp_path, request, _ReadsEachImage()
    )

    table = pd.read_parquet(rows_path)
    n = len(_IMAGE_SHAPES)
    assert list(table.columns) == list(POINT_COLUMNS[::-1])
    numeric = {
        column: kind for column, kind in POINT_DTYPES.items() if kind != "object"
    }
    assert {column: str(table[column].dtype) for column in numeric} == numeric
    assert sorted(table["frame"].tolist()) == list(range(n))
    named = [float(image_mark(int(frame))[0]) for frame in table["frame"]]
    assert table["x"].tolist() == named
    assert table["class_name"].tolist() == ["bee"] * n


@pytest.mark.parametrize(("task", "iou"), [("pose", 0.5), ("locate", None)])
def test_predict_images_on_no_images_writes_an_empty_table_of_every_column(
    runner_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    task: InferTask,
    iou: float | None,
) -> None:
    """The rows file of an empty run is written with its full column set."""
    rows_path = tmp_path / "rows.parquet"
    request = _images_request([], rows_path, task=task, iou=iou)

    response = _predict_images(
        runner_module, monkeypatch, tmp_path, request, _ReadsEachImage()
    )

    assert (response.n_images, response.n_rows) == (0, 0)
    table = pd.read_parquet(rows_path)
    assert list(table.columns) == request.columns
    assert table.empty
