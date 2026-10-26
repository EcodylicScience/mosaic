"""The SLEAP frame adapter, which predicts extracted images with ``sleap-nn track``.

The adapter writes the request's images as a frames-only ``.slp``, runs
``sleap-nn track`` on it without tracking, and reads the predictions back. A fake
stands in for the writer, the track call and the readback. The readback tests
stand in for the SLEAP interpreter alone.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Final

import pytest

import mosaic.tracking.sleap.frames as sleap_frames
from mosaic.core.annotations.model import Keypoint, KeypointSchema
from mosaic.tracking.frame_predictions.adapter import (
    AdapterResult,
    FrameAdapter,
    FrameImage,
    ToolProbe,
)
from mosaic.tracking.frame_predictions.params import PredictFramesParams
from mosaic.tracking.model_refs import ModelArtifact, ResolvedModel
from mosaic.tracking.sleap.frames import (
    SleapFrameAdapter,
    SleapInstanceReadback,
    read_sleap_predictions,
)
from mosaic.tracking.sleap.run import SleapError, sleap_track_device_args
from tests.helpers import FakeSleapFrames, install_fake_sleap_frames, sleap_instance

SLEAP_RUN: Final = "train-sleap.0.3-abcdef0123"
PROBE: Final = ToolProbe(tool_version="0.3.1", n_keypoints=None)
RECORDED: Final = KeypointSchema(names=("nose", "tail"), skeleton=((0, 1),))


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeSleapFrames:
    """Stand in for the SLEAP environment."""
    return install_fake_sleap_frames(monkeypatch)


def _model(tmp_path: Path, names: Sequence[str] = ("model",)) -> ResolvedModel:
    """Return a resolved model of one SLEAP directory per name, in order."""
    artifacts = tuple(
        ModelArtifact(
            root=tmp_path / name, files=(), exec_path=tmp_path / name, digest="d"
        )
        for name in names
    )
    return ResolvedModel(artifacts=artifacts, run_id="", digest="d", members=())


def _images(tmp_path: Path, *directories: str) -> list[FrameImage]:
    """Return ``frame_000100.png`` and ``frame_000200.png`` of each directory.

    Each image's key is its path relative to *tmp_path*, the dataset root.
    """
    images: list[FrameImage] = []
    for directory in directories:
        for index in (100, 200):
            key = f"frames/{directory}/frame_{index:06d}.png"
            image = FrameImage(
                position=len(images),
                key=key,
                path=tmp_path / key,
                width=64,
                height=48,
            )
            images.append(image)
    return images


def _params(**overrides: object) -> PredictFramesParams:
    return PredictFramesParams.model_validate(
        {"model": [SLEAP_RUN], "frames_run_id": "uniform-0123456789", **overrides}
    )


def _predict(
    tmp_path: Path,
    images: Sequence[FrameImage],
    *,
    schema: KeypointSchema | None = None,
    model: ResolvedModel | None = None,
    lines: list[str] | None = None,
    **overrides: object,
) -> AdapterResult:
    adapter: FrameAdapter = SleapFrameAdapter(schema=schema, images_root=tmp_path)
    received: list[str] = [] if lines is None else lines
    return adapter(
        images,
        model or _model(tmp_path),
        _params(**overrides),
        PROBE,
        tmp_path / "work",
        on_output=received.append,
        cancel_check=lambda: False,
    )


# --- the frames-only input -----------------------------------------------------------


@pytest.mark.parametrize(
    ("recorded", "written"),
    [(None, KeypointSchema(names=("kp0",))), (RECORDED, RECORDED)],
    ids=["placeholder", "recorded"],
)
def test_the_input_names_the_recorded_pose_or_the_placeholder(
    tmp_path: Path,
    fake: FakeSleapFrames,
    recorded: KeypointSchema | None,
    written: KeypointSchema,
) -> None:
    _ = _predict(tmp_path, _images(tmp_path, "s1"), schema=recorded)

    assert fake.written[0].annotations.schema == written


def test_the_input_lists_every_image_where_it_is_and_no_object(
    tmp_path: Path, fake: FakeSleapFrames
) -> None:
    images = _images(tmp_path, "s1", "s2")

    _ = _predict(tmp_path, images)

    written = fake.written[0]
    assert written.path == tmp_path / "work" / "frames.slp"
    assert written.images == [image.path.resolve() for image in images]
    assert [(f.width, f.height) for f in written.annotations.frames] == [(64, 48)] * 4
    assert all(not frame.objects for frame in written.annotations.frames)


def test_the_input_names_each_image_by_its_key_under_the_dataset_root(
    tmp_path: Path, fake: FakeSleapFrames
) -> None:
    images = _images(tmp_path, "s1", "s2")

    _ = _predict(tmp_path, images)

    written = fake.written[0].annotations
    assert written.image_root == tmp_path
    assert [frame.image_path.as_posix() for frame in written.frames] == [
        image.key for image in images
    ]


# --- sleap-nn track ------------------------------------------------------------------


def test_sleap_nn_predicts_the_input_with_the_models_and_no_tracking(
    tmp_path: Path, fake: FakeSleapFrames
) -> None:
    model = _model(tmp_path, ("centroid", "instance"))

    _ = _predict(
        tmp_path,
        _images(tmp_path, "s1"),
        model=model,
        peak_threshold=0.3,
        max_instances=2,
        batch_size=8,
    )

    call = fake.tracked[0]
    assert call.input_slp == tmp_path / "work" / "frames.slp"
    assert call.output_slp.parent == tmp_path / "work"
    assert fake.read == [call.output_slp]
    assert call.settings["model_paths"] == [
        tmp_path / "centroid",
        tmp_path / "instance",
    ]
    assert call.settings["tracking"] is False
    assert call.settings["peak_threshold"] == 0.3
    assert call.settings["max_instances"] == 2
    assert call.settings["batch_size"] == 8


@pytest.mark.parametrize(
    ("device", "sent"),
    [("auto", []), ("cpu", ["-d", "cpu"]), ("0", ["-d", "cuda:0"])],
)
def test_the_device_reaches_sleap_nn_and_auto_sends_none(
    tmp_path: Path, fake: FakeSleapFrames, device: str, sent: list[str]
) -> None:
    _ = _predict(tmp_path, _images(tmp_path, "s1"), device=device)

    given = fake.tracked[0].settings["device"]
    assert isinstance(given, str)
    assert sleap_track_device_args(given) == sent


def test_sleap_nn_output_reaches_on_output_through_on_activity_alone(
    tmp_path: Path, fake: FakeSleapFrames
) -> None:
    lines: list[str] = []

    _ = _predict(tmp_path, _images(tmp_path, "s1"), lines=lines)

    call = fake.tracked[0]
    assert call.on_output is None
    assert call.on_activity is not None
    assert lines == ["predicted 2 frames"]


# --- the readback ----------------------------------------------------------------------


def test_objects_are_matched_to_their_images_by_path(
    tmp_path: Path, fake: FakeSleapFrames
) -> None:
    def by_directory(image: Path) -> list[SleapInstanceReadback]:
        x = 1.0 if image.parent.name == "s1" else 2.0
        y = float(image.stem.removeprefix("frame_"))
        return [sleap_instance([(x, y, 0.5), (x, y, 0.5)])]

    fake.instances = by_directory
    images = _images(tmp_path, "s1", "s2")

    result = _predict(tmp_path, images)

    found = [(obj.keypoints[0].x, obj.keypoints[0].y) for (obj,) in result.objects]
    assert found == [(1.0, 100.0), (1.0, 200.0), (2.0, 100.0), (2.0, 200.0)]


def test_each_object_has_its_scores_and_its_track(
    tmp_path: Path, fake: FakeSleapFrames
) -> None:
    fake.instances = lambda _image: [
        sleap_instance([(1.0, 2.0, 0.7), (3.0, 4.0, 0.6)], score=0.8, track="track_0"),
        sleap_instance([(5.0, 6.0, 0.5), (7.0, 8.0, 0.4)], score=0.3),
    ]

    result = _predict(tmp_path, _images(tmp_path, "s1"))

    first, second = result.objects[0]
    assert first.keypoints == (
        Keypoint(x=1.0, y=2.0, visibility=2, score=0.7),
        Keypoint(x=3.0, y=4.0, visibility=2, score=0.6),
    )
    assert (first.score, first.track_id, first.bbox) == (0.8, "track_0", None)
    assert (second.score, second.track_id) == (0.3, None)


def test_a_nan_point_is_an_unplaced_keypoint(
    tmp_path: Path, fake: FakeSleapFrames
) -> None:
    nan = math.nan
    fake.instances = lambda _image: [
        sleap_instance([(nan, nan, nan), (3.0, 4.0, nan)], score=nan)
    ]

    result = _predict(tmp_path, _images(tmp_path, "s1"))

    (obj,) = result.objects[0]
    unplaced, placed = obj.keypoints
    assert unplaced.visibility == 0
    assert not unplaced.is_placed
    assert unplaced.score is None
    assert placed == Keypoint(x=3.0, y=4.0, visibility=2, score=None)
    assert obj.score is None


def test_an_image_without_a_predicted_frame_has_no_objects(
    tmp_path: Path, fake: FakeSleapFrames
) -> None:
    images = _images(tmp_path, "s1")
    fake.empty_frames = False
    fake.instances = lambda image: (
        [] if image.name == "frame_000200.png" else [sleap_instance([(1, 2, 3)] * 2)]
    )

    result = _predict(tmp_path, images)

    (answer,) = fake.answers
    assert [frame.image for frame in answer.frames] == [str(images[0].path.resolve())]
    assert len(answer.images) == 2
    assert [len(objects) for objects in result.objects] == [1, 0]


def test_a_frame_without_instances_has_no_objects(
    tmp_path: Path, fake: FakeSleapFrames
) -> None:
    """Read a frame reported with no instances as an image with no objects.

    sleap-nn 0.2.0 and 0.3.1 report every frame of a frames-only input, with no
    instances where they find none.
    """
    images = _images(tmp_path, "s1")
    fake.instances = lambda image: (
        [] if image.name == "frame_000200.png" else [sleap_instance([(1, 2, 3)] * 2)]
    )

    result = _predict(tmp_path, images)

    (answer,) = fake.answers
    # The fake reports the second image as a frame with no instances.
    reported = {frame.image: len(frame.instances) for frame in answer.frames}
    assert reported == {
        str(image.path.resolve()): n for image, n in zip(images, (1, 0), strict=True)
    }
    assert [len(objects) for objects in result.objects] == [1, 0]


def test_the_tool_pose_is_the_readbacks_nodes_edges_and_mirror_pairs(
    tmp_path: Path, fake: FakeSleapFrames
) -> None:
    fake.nodes = ("nose", "left", "tail", "right")
    fake.edges = ((3, 1), (0, 2))
    fake.symmetries = ((3, 1),)

    result = _predict(tmp_path, _images(tmp_path, "s1"))

    assert result.tool_pose == KeypointSchema(
        names=("nose", "left", "tail", "right"),
        skeleton=((3, 1), (0, 2)),
        symmetries=((1, 3),),
    )


def _first_image_always(images: Sequence[FrameImage]) -> Callable[[Path], str | None]:
    first = str(images[0].path.resolve())
    return lambda _image: first


def _unlist_the_first(images: Sequence[FrameImage]) -> Callable[[Path], str | None]:
    first = images[0].path.resolve()
    return lambda image: None if image == first else str(image)


def _another_image(images: Sequence[FrameImage]) -> Callable[[Path], str | None]:
    first = images[0].path.resolve()
    return lambda image: (
        str(image.with_name("other.png")) if image == first else str(image)
    )


@pytest.mark.parametrize(
    ("reporting", "message"),
    [
        (_first_image_always, "twice"),
        (_unlist_the_first, "do not list 1 of 2 images of the request"),
        (_another_image, "other.png, which is not an image of the request"),
    ],
    ids=["repeated", "unlisted", "another"],
)
def test_predictions_that_do_not_name_each_image_once_are_refused(
    tmp_path: Path,
    fake: FakeSleapFrames,
    reporting: Callable[[Sequence[FrameImage]], Callable[[Path], str | None]],
    message: str,
) -> None:
    images = _images(tmp_path, "s1")
    fake.reported = reporting(images)

    with pytest.raises(ValueError, match=message):
        _ = _predict(tmp_path, images)


# --- the readback snippet ------------------------------------------------------------


def _answering(
    monkeypatch: pytest.MonkeyPatch,
    document: str,
    *,
    returncode: int = 0,
) -> list[list[str]]:
    """Stand in for the SLEAP interpreter, writing *document* at the answer path."""
    commands: list[list[str]] = []

    def invocation(_env: object, *, executable: str) -> list[str]:
        return [f"/fake/bin/{executable}"]

    def run(cmd: list[str], **_kwargs: object) -> tuple[str, str, int]:
        commands.append(cmd)
        if returncode == 0:
            _ = Path(cmd[-1]).write_text(document)
        return ("", "the snippet failed", returncode)

    monkeypatch.setattr(sleap_frames, "tool_invocation", invocation)
    monkeypatch.setattr(sleap_frames, "run_supervised", run)
    return commands


def test_the_readback_takes_its_paths_in_argv_and_reads_nan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document = {
        "nodes": ["head", "tail"],
        "edges": [[0, 1]],
        "symmetries": [],
        "images": ["/frames/s1/frame_000100.png", "/frames/s1/frame_000200.png"],
        "frames": [
            {
                "image": "/frames/s1/frame_000100.png",
                "instances": [
                    {
                        "score": 0.9,
                        "track": None,
                        "points": [[1.0, 2.0, 0.5], [math.nan] * 3],
                    }
                ],
            }
        ],
    }
    commands = _answering(monkeypatch, json.dumps(document))
    predicted = tmp_path / "predicted.slp"
    answer = tmp_path / "predicted.json"

    readback = read_sleap_predictions(predicted, answer)

    (cmd,) = commands
    assert cmd[:2] == ["/fake/bin/python", "-c"]
    assert cmd[3:] == [str(predicted), str(answer)]
    assert str(tmp_path) not in cmd[2]
    (point,) = readback.frames[0].instances[0].points[1:]
    assert all(value is not None and math.isnan(value) for value in point)
    assert readback.nodes == ["head", "tail"]
    assert readback.images == [
        "/frames/s1/frame_000100.png",
        "/frames/s1/frame_000200.png",
    ]


def test_a_readback_that_exits_non_zero_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ = _answering(monkeypatch, "", returncode=1)

    with pytest.raises(SleapError, match="the snippet failed"):
        _ = read_sleap_predictions(tmp_path / "predicted.slp", tmp_path / "out.json")
