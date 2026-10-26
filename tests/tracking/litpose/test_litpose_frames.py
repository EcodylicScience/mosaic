"""The Lightning Pose frame adapter, which predicts extracted images one at a time.

The adapter runs a snippet in the Lightning Pose environment that calls
``Model.predict_frame`` on each image of a request, and joins its results to the
images by position. A version probe runs before it. A fake stands in for the
interpreter that runs both snippets. The tests of the snippets themselves run
them in this interpreter, against a stand-in ``lightning_pose`` package.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Final

import cv2
import numpy as np
import pytest

from mosaic.core.annotations.model import Keypoint, KeypointSchema
from mosaic.tracking.frame_predictions.adapter import (
    AdapterResult,
    FrameAdapter,
    FrameImage,
    ToolProbe,
)
from mosaic.tracking.frame_predictions.errors import PredictFramesUnavailable
from mosaic.tracking.frame_predictions.params import PredictFramesParams
from mosaic.tracking.litpose.frames import (
    LITPOSE_PREDICT_FRAME_MIN,
    LitposeFrameResult,
    predict_litpose_frames,
    probe_litpose_version,
    require_predict_frame,
)
from mosaic.tracking.litpose.run import LITPOSE_ENV, LitposeError
from mosaic.tracking.model_refs import ModelArtifact, ResolvedModel
from tests.helpers import (
    FakeLitposeFrames,
    install_fake_litpose_frames,
    install_fake_tool_python,
)

LITPOSE_RUN: Final = "train-litpose.0.1-abcdef0123"
PROBE: Final = ToolProbe(tool_version="2.3.1", n_keypoints=None)

type Rewrite = Callable[[list[LitposeFrameResult]], list[LitposeFrameResult]]


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeLitposeFrames:
    """Stand in for the Lightning Pose interpreter, with no inherited device list."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    return install_fake_litpose_frames(monkeypatch)


def _model(tmp_path: Path) -> ResolvedModel:
    """Return a resolved model of the Lightning Pose directory ``model``."""
    root = tmp_path / "model"
    artifact = ModelArtifact(root=root, files=(), exec_path=root, digest="d")
    return ResolvedModel(artifacts=(artifact,), run_id="", digest="d", members=())


def _images(tmp_path: Path, *directories: str) -> list[FrameImage]:
    """Return ``frame_000100.png`` and ``frame_000200.png`` of each directory."""
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
        {"model": [LITPOSE_RUN], "frames_run_id": "uniform-0123456789", **overrides}
    )


def _predict(
    tmp_path: Path,
    images: Sequence[FrameImage],
    *,
    lines: list[str] | None = None,
    **overrides: object,
) -> AdapterResult:
    adapter: FrameAdapter = predict_litpose_frames
    received: list[str] = [] if lines is None else lines
    work_dir = tmp_path / "work"
    work_dir.mkdir(exist_ok=True)
    return adapter(
        images,
        _model(tmp_path),
        _params(**overrides),
        PROBE,
        work_dir,
        on_output=received.append,
        cancel_check=lambda: False,
    )


# --- the version ---------------------------------------------------------------------


def test_the_minimum_is_the_release_whose_predict_frame_is_verified() -> None:
    assert LITPOSE_PREDICT_FRAME_MIN == "2.3.1"


@pytest.mark.parametrize("version", ["2.3.1", "2.3.2", "2.4.0", "3.0", "2.3.1.post1"])
def test_a_release_from_the_minimum_on_is_accepted(version: str) -> None:
    require_predict_frame(version)


@pytest.mark.parametrize("version", ["2.3.0", "2.3", "2.2.9", "1.7.0", "unknown"])
def test_an_older_release_is_unavailable_naming_both_versions(version: str) -> None:
    with pytest.raises(PredictFramesUnavailable) as caught:
        require_predict_frame(version)

    assert caught.value.error_kind == "unavailable"
    message = str(caught.value)
    assert version in message
    assert "2.3.1" in message


def test_the_probe_prints_the_installed_version_from_argv_alone(
    fake: FakeLitposeFrames,
) -> None:
    fake.version = "2.3.4"

    found = probe_litpose_version()

    assert found == "2.3.4"
    (probe,) = fake.probes
    assert probe.argv == ()
    assert "lightning_pose" in probe.source
    assert fake.predictions == []


def test_the_probe_reads_the_last_line_that_it_prints(
    monkeypatch: pytest.MonkeyPatch, fake: FakeLitposeFrames
) -> None:
    def noisy(cmd: Sequence[str], **_kwargs: object) -> tuple[str, str, int]:
        _ = cmd
        return ("activating the environment\n2.3.1\n\n", "a warning", 0)

    monkeypatch.setattr("mosaic.tracking.litpose.frames.run_supervised", noisy)

    assert probe_litpose_version() == "2.3.1"
    assert fake.runs == []


@pytest.mark.parametrize(
    ("answer", "raised"),
    [
        (("", "no module named lightning_pose", 1), LitposeError),
        (("", "", 0), ValueError),
    ],
    ids=["exits-non-zero", "prints-nothing"],
)
def test_a_probe_without_a_version_raises(
    monkeypatch: pytest.MonkeyPatch,
    fake: FakeLitposeFrames,
    answer: tuple[str, str, int],
    raised: type[Exception],
) -> None:
    _ = fake

    def failing(cmd: Sequence[str], **_kwargs: object) -> tuple[str, str, int]:
        _ = cmd
        return answer

    monkeypatch.setattr("mosaic.tracking.litpose.frames.run_supervised", failing)

    with pytest.raises(raised):
        _ = probe_litpose_version()


# --- the prediction snippet ----------------------------------------------------------


def test_the_request_lists_each_image_by_position_in_request_order(
    tmp_path: Path, fake: FakeLitposeFrames
) -> None:
    images = _images(tmp_path, "s1", "s2")

    _ = _predict(tmp_path, images)

    (run,) = fake.predictions
    assert run.request == [(image.position, str(image.path)) for image in images]


def test_the_snippet_takes_its_paths_and_the_minimum_in_argv(
    tmp_path: Path, fake: FakeLitposeFrames
) -> None:
    _ = _predict(tmp_path, _images(tmp_path, "s1"))

    (run,) = fake.predictions
    model_dir, request, answer, minimum, too_old = run.argv
    assert model_dir == str(tmp_path / "model")
    assert Path(request).parent == tmp_path / "work"
    assert Path(answer).parent == tmp_path / "work"
    assert minimum == LITPOSE_PREDICT_FRAME_MIN
    assert too_old.isdigit()
    assert int(too_old) not in (0, 1)
    assert str(tmp_path) not in run.source


def test_the_snippet_predicts_single_frames_at_fp32_and_never_a_label_csv(
    tmp_path: Path, fake: FakeLitposeFrames
) -> None:
    _ = probe_litpose_version()
    _ = _predict(tmp_path, _images(tmp_path, "s1"))

    (run,) = fake.predictions
    assert ".predict_frame(" in run.source
    assert 'precision="fp32"' in run.source
    assert all("predict_on_label_csv" not in each.source for each in fake.runs)
    assert all("predict_on_video_file" not in each.source for each in fake.runs)


def test_both_snippets_are_python_programs(
    tmp_path: Path, fake: FakeLitposeFrames
) -> None:
    _ = probe_litpose_version()
    _ = _predict(tmp_path, _images(tmp_path, "s1"))

    assert len(fake.runs) == 2
    for run in fake.runs:
        _ = compile(run.source, "<snippet>", "exec")


@pytest.mark.parametrize(
    ("device", "inherited", "visible"),
    [
        ("auto", None, None),
        ("0", None, "0"),
        ("1", None, "1"),
        ("1", "4,6", "6"),
        ("auto", "4,6", "4,6"),
    ],
    ids=["auto", "index-0", "index-1", "within-inherited", "auto-keeps"],
)
def test_the_device_reaches_the_snippet_as_its_visible_cuda_devices(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake: FakeLitposeFrames,
    device: str,
    inherited: str | None,
    visible: str | None,
) -> None:
    if inherited is not None:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", inherited)

    _ = _predict(tmp_path, _images(tmp_path, "s1"), device=device)

    (run,) = fake.predictions
    assert run.env is not None
    assert run.env.get("CUDA_VISIBLE_DEVICES") == visible


def test_the_snippet_output_reaches_on_output_through_on_activity_alone(
    tmp_path: Path, fake: FakeLitposeFrames
) -> None:
    lines: list[str] = []

    _ = _predict(tmp_path, _images(tmp_path, "s1"), lines=lines)

    (run,) = fake.predictions
    assert run.on_output is None
    assert run.on_activity is not None
    assert lines == ["predicted 2 images"]


def test_the_guard_exit_is_the_version_refusal(
    tmp_path: Path, fake: FakeLitposeFrames
) -> None:
    fake.version = "2.2.0"
    fake.guard_exits = True

    with pytest.raises(PredictFramesUnavailable) as caught:
        _ = _predict(tmp_path, _images(tmp_path, "s1"))

    message = str(caught.value)
    assert "2.2.0" in message
    assert LITPOSE_PREDICT_FRAME_MIN in message


def test_a_snippet_that_exits_non_zero_raises(
    tmp_path: Path, fake: FakeLitposeFrames
) -> None:
    fake.returncode = 1

    with pytest.raises(LitposeError, match="the snippet failed"):
        _ = _predict(tmp_path, _images(tmp_path, "s1"))


# --- the answer ----------------------------------------------------------------------


def test_results_are_joined_to_their_images_by_position(
    tmp_path: Path, fake: FakeLitposeFrames
) -> None:
    def by_directory(image: Path) -> list[tuple[float, float, float]]:
        x = 1.0 if image.parent.name == "s1" else 2.0
        y = float(image.stem.removeprefix("frame_"))
        return [(x, y, 0.5), (x, y, 0.5)]

    fake.points = by_directory

    result = _predict(tmp_path, _images(tmp_path, "s1", "s2"))

    found = [(obj.keypoints[0].x, obj.keypoints[0].y) for (obj,) in result.objects]
    assert found == [(1.0, 100.0), (1.0, 200.0), (2.0, 100.0), (2.0, 200.0)]


def test_each_image_has_one_object_with_keypoint_confidences_and_nothing_else(
    tmp_path: Path, fake: FakeLitposeFrames
) -> None:
    fake.points = lambda _image: [(1.0, 2.0, 0.7), (3.0, 4.0, 0.6)]

    result = _predict(tmp_path, _images(tmp_path, "s1"))

    assert [len(objects) for objects in result.objects] == [1, 1]
    (obj,) = result.objects[0]
    assert obj.keypoints == (
        Keypoint(x=1.0, y=2.0, visibility=2, score=0.7),
        Keypoint(x=3.0, y=4.0, visibility=2, score=0.6),
    )
    assert (obj.score, obj.track_id, obj.bbox) == (None, None, None)


def test_the_tool_pose_is_the_configs_keypoint_names_without_edges(
    tmp_path: Path, fake: FakeLitposeFrames
) -> None:
    fake.keypoint_names = ("snout", "left_ear", "right_ear")

    result = _predict(tmp_path, _images(tmp_path, "s1"))

    assert result.tool_pose == KeypointSchema(names=("snout", "left_ear", "right_ear"))


def _drop_the_first(results: list[LitposeFrameResult]) -> list[LitposeFrameResult]:
    return [result for result in results if result.position != 0]


def _repeat_the_first(results: list[LitposeFrameResult]) -> list[LitposeFrameResult]:
    first = next(result for result in results if result.position == 0)
    return [*results, first]


def _add_another(results: list[LitposeFrameResult]) -> list[LitposeFrameResult]:
    other = results[0].model_copy(update={"position": 7})
    return [*results, other]


def _drop_a_keypoint(results: list[LitposeFrameResult]) -> list[LitposeFrameResult]:
    first, *rest = results
    short = first.model_copy(
        update={"keypoints": first.keypoints[:1], "confidence": first.confidence[:1]}
    )
    return [short, *rest]


@pytest.mark.parametrize(
    ("rewrite", "message"),
    [
        (_drop_the_first, "did not answer the image at position 0"),
        (_repeat_the_first, "position 0 twice"),
        (_add_another, "position 7, which no image of the request has"),
        (_drop_a_keypoint, "with 1 point and 1 confidence, and the model's config"),
    ],
    ids=["missing", "repeated", "another", "short"],
)
def test_an_answer_that_does_not_fit_the_request_is_refused(
    tmp_path: Path,
    fake: FakeLitposeFrames,
    rewrite: Rewrite,
    message: str,
) -> None:
    fake.rewrite = rewrite

    with pytest.raises(ValueError, match=message):
        _ = _predict(tmp_path, _images(tmp_path, "s1"))


def test_an_answer_without_keypoint_names_is_refused(
    tmp_path: Path, fake: FakeLitposeFrames
) -> None:
    fake.keypoint_names = ()
    fake.points = lambda _image: [(1.0, 2.0, 0.5)]

    with pytest.raises(ValueError, match="keypoint_names"):
        _ = _predict(tmp_path, _images(tmp_path, "s1"))


def test_the_adapter_writes_only_into_its_work_directory(
    tmp_path: Path, fake: FakeLitposeFrames
) -> None:
    _ = _predict(tmp_path, _images(tmp_path, "s1"))

    written = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    assert written
    assert all(path.parts[0] == "work" for path in written)
    assert len(fake.predictions) == 1


# --- the snippets in an interpreter --------------------------------------------------

_FAKE_LIGHTNING_POSE: Final = """
import os

__version__ = os.environ["FAKE_LITPOSE_VERSION"]
"""
"""The stand-in ``lightning_pose``, whose release is ``FAKE_LITPOSE_VERSION``."""

_FAKE_LIGHTNING_POSE_API: Final = """
import json
import os

import numpy as np


class _Config:
    def __init__(self):
        self.data = json.loads(os.environ["FAKE_LITPOSE_DATA"])


class Model:
    def __init__(self):
        self._config = _Config()

    @staticmethod
    def from_dir(model_dir, precision="fp32"):
        with open(os.environ["FAKE_LOG"], "w", encoding="utf-8") as fh:
            json.dump({"model_dir": str(model_dir), "precision": precision}, fh)
        return Model()

    @property
    def cfg(self):
        return self._config

    def predict_frame(self, frame_rgb, bbox=None):
        if frame_rgb.dtype != np.uint8 or frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
            raise ValueError(f"frame_rgb must be (H, W, 3) uint8, got {frame_rgb.shape}")
        red, green, blue = (float(value) for value in frame_rgb[0, 0])
        height = float(frame_rgb.shape[0])
        return {
            "keypoints": np.array([[red, green], [blue, height]], dtype=np.float32),
            "confidence": np.array([0.5, 0.25], dtype=np.float32),
        }
"""
"""The stand-in ``lightning_pose.api``.

``Model.from_dir`` records its arguments in ``FAKE_LOG``. The config's ``data``
section is the JSON object ``FAKE_LITPOSE_DATA``. ``predict_frame`` returns the
arrays of Lightning Pose 2.3.1 for two keypoints: the red and green values of
the image's top-left pixel, then its blue value and the image's height.
"""


@pytest.fixture
def model_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Run both snippets in this interpreter, with a stand-in ``lightning_pose``.

    The stand-in reports release :data:`LITPOSE_PREDICT_FRAME_MIN` and a config
    naming ``nose`` and ``tail``.

    Returns:
        The file in which ``Model.from_dir`` records the model that it loads.
    """
    fakes = tmp_path / "fakes"
    package = fakes / "lightning_pose"
    package.mkdir(parents=True)
    _ = (package / "__init__.py").write_text(_FAKE_LIGHTNING_POSE)
    _ = (package / "api.py").write_text(_FAKE_LIGHTNING_POSE_API)
    record = tmp_path / "loaded.json"
    monkeypatch.setenv("FAKE_LOG", str(record))
    monkeypatch.setenv("FAKE_LITPOSE_VERSION", LITPOSE_PREDICT_FRAME_MIN)
    data = json.dumps({"keypoint_names": ["nose", "tail"]})
    monkeypatch.setenv("FAKE_LITPOSE_DATA", data)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    _ = install_fake_tool_python(
        monkeypatch, LITPOSE_ENV, tmp_path / "bin", imports=fakes
    )
    return record


def _painted(
    tmp_path: Path, colors: Sequence[tuple[int, int, int]]
) -> list[FrameImage]:
    """Write one 6x4 PNG of each RGB color of *colors*, and return them in order."""
    images: list[FrameImage] = []
    for position, (red, green, blue) in enumerate(colors):
        key = f"frames/s1/frame_{position:06d}.png"
        path = tmp_path / key
        path.parent.mkdir(parents=True, exist_ok=True)
        pixels = np.empty((4, 6, 3), np.uint8)
        # OpenCV writes an array's channels in BGR order.
        pixels[...] = (blue, green, red)
        assert cv2.imwrite(str(path), pixels), path
        image = FrameImage(position=position, key=key, path=path, width=6, height=4)
        images.append(image)
    return images


def test_the_probe_snippet_prints_the_installed_release(
    model_record: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ = model_record
    monkeypatch.setenv("FAKE_LITPOSE_VERSION", "2.3.4")

    assert probe_litpose_version() == "2.3.4"


def test_the_snippet_predicts_each_image_read_as_rgb_in_request_order(
    tmp_path: Path, model_record: Path
) -> None:
    lines: list[str] = []

    result = _predict(
        tmp_path, _painted(tmp_path, [(10, 20, 30), (40, 50, 60)]), lines=lines
    )

    found = [
        [(point.x, point.y, point.score) for point in obj.keypoints]
        for (obj,) in result.objects
    ]
    assert found == [
        [(10.0, 20.0, 0.5), (30.0, 4.0, 0.25)],
        [(40.0, 50.0, 0.5), (60.0, 4.0, 0.25)],
    ]
    assert result.tool_pose == KeypointSchema(names=("nose", "tail"))
    loaded = json.loads(model_record.read_text())
    assert loaded == {"model_dir": str(tmp_path / "model"), "precision": "fp32"}
    assert "predicted 2 of 2 images" in [line.strip() for line in lines]


@pytest.mark.parametrize("version", ["2.3.0", "2.2.9.post1", "unknown"])
def test_the_snippet_guard_refuses_an_older_release_before_the_model_loads(
    tmp_path: Path,
    model_record: Path,
    monkeypatch: pytest.MonkeyPatch,
    version: str,
) -> None:
    monkeypatch.setenv("FAKE_LITPOSE_VERSION", version)

    with pytest.raises(PredictFramesUnavailable) as caught:
        _ = _predict(tmp_path, _painted(tmp_path, [(10, 20, 30)]))

    message = str(caught.value)
    assert repr(version) in message
    assert LITPOSE_PREDICT_FRAME_MIN in message
    assert not model_record.exists()


@pytest.mark.parametrize("version", ["2.3.1.post1", "2.10.0", "10.0"])
def test_the_snippet_guard_compares_release_numbers(
    tmp_path: Path,
    model_record: Path,
    monkeypatch: pytest.MonkeyPatch,
    version: str,
) -> None:
    monkeypatch.setenv("FAKE_LITPOSE_VERSION", version)

    result = _predict(tmp_path, _painted(tmp_path, [(10, 20, 30)]))

    assert len(result.objects) == 1
    assert model_record.is_file()


def test_a_config_without_keypoint_names_is_refused(
    tmp_path: Path, model_record: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ = model_record
    monkeypatch.setenv("FAKE_LITPOSE_DATA", "{}")

    with pytest.raises(ValueError, match="does not set data.keypoint_names"):
        _ = _predict(tmp_path, _painted(tmp_path, [(10, 20, 30)]))


def test_an_image_that_opencv_cannot_read_fails_the_snippet(
    tmp_path: Path, model_record: Path
) -> None:
    _ = model_record
    images = _painted(tmp_path, [(10, 20, 30)])
    _ = images[0].path.write_bytes(b"not a png")

    with pytest.raises(LitposeError, match="cannot read the image"):
        _ = _predict(tmp_path, images)
