"""Stand in for Lightning Pose in the tests of its tracker, recording each call.

``run_litpose`` calls Lightning Pose through one module-level seam in
``litpose/dataset_runs.py``, its one inference phase. :func:`install_fake_litpose`
replaces it with a :class:`FakeLitpose`, whose prediction writes a small
DeepLabCut CSV, the layout that Lightning Pose exports. A test runs the whole
tracker protocol without a model or a GPU. :func:`write_litpose_model`
writes the model directory that a run resolves.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from mosaic.tracking.litpose.run import LitposePredictResult

from tests.helpers.tracks import write_dlc_csv

_BODYPARTS: tuple[str, ...] = ("nose", "tail")
"""The keypoints that :class:`FakeLitpose` reports.

The model directory of :func:`write_litpose_model` names them too.
"""


@dataclass
class FakeLitpose:
    """Stand in for the Lightning Pose inference phase, recording each call."""

    predicted: list[Path] = field(default_factory=list)
    frames: int = 6
    written: list[npt.NDArray[np.float64]] = field(default_factory=list)
    """The values that each prediction wrote, in call order.

    Each entry contains the ``[x, y, likelihood]`` values, shaped
    ``(frame, bodypart, 3)``.
    """

    def predict(
        self, video_path: Path, out_csv: Path, **_kwargs: object
    ) -> LitposePredictResult:
        self.predicted.append(Path(video_path))
        out = Path(out_csv)
        self.written.append(
            write_dlc_csv(
                out, _BODYPARTS, n_frames=self.frames, scorer="heatmap_tracker"
            )
        )
        return LitposePredictResult(csv_path=out, stdout="", stderr="")


def install_fake_litpose(
    monkeypatch: pytest.MonkeyPatch, fake: FakeLitpose | None = None
) -> FakeLitpose:
    """Replace the tracker's inference seam with *fake*, or a new fake."""
    import mosaic.tracking.litpose.dataset_runs as dataset_runs

    installed = fake if fake is not None else FakeLitpose()
    monkeypatch.setattr(dataset_runs, "run_litpose_predict", installed.predict)
    return installed


def write_litpose_model(
    model_dir: Path,
    *,
    weights: bytes = b"weights",
    model_type: str = "heatmap",
    keypoint_names: Sequence[str] = _BODYPARTS,
) -> Path:
    """Write a minimal Lightning Pose model directory, and return it.

    The directory contains a ``config.yaml`` declaring *model_type* and naming
    *keypoint_names*, by default the two keypoints of :class:`FakeLitpose`, and
    one checkpoint with *weights*. The model's identity is a digest of both
    files' bytes. Two directories written with the same arguments are one model.

    The config has no ``data`` section when *keypoint_names* is empty, which is
    the config a pinned digest was taken over.
    """
    checkpoint = model_dir / "tb_logs" / "m" / "version_0" / "checkpoints" / "best.ckpt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    _ = checkpoint.write_bytes(weights)
    config = f"model:\n  model_type: {model_type}\n"
    if keypoint_names:
        config += f"data:\n  keypoint_names: [{', '.join(keypoint_names)}]\n"
    _ = (model_dir / "config.yaml").write_text(config)
    return model_dir
