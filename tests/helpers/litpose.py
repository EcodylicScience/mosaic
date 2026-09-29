"""Stand in for Lightning Pose in the tests of its tracker, recording each call.

``run_litpose`` calls Lightning Pose through one module-level seam in
``litpose/dataset_runs.py``, its one inference phase. :func:`install_fake_litpose`
replaces it with a :class:`FakeLitpose`, whose prediction writes a small
DeepLabCut CSV, the layout that Lightning Pose exports. A test therefore runs the
whole tracker protocol without a model or a GPU. :func:`write_litpose_model`
writes the model directory that a run resolves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from mosaic.tracking.litpose.run import LitposePredictResult

from tests.helpers.tracks import write_dlc_csv

_BODYPARTS: tuple[str, ...] = ("nose", "tail")
"""The keypoints that the model of :func:`write_litpose_model` names."""


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


def write_litpose_model(model_dir: Path, *, weights: bytes = b"weights") -> Path:
    """Write a minimal Lightning Pose model directory, and return it.

    The directory contains a ``config.yaml`` naming the two keypoints of
    :class:`FakeLitpose`, and one checkpoint with *weights*. The model's identity
    is a digest of both. Two directories written with the same *weights* are
    therefore one model.
    """
    checkpoint = model_dir / "tb_logs" / "m" / "version_0" / "checkpoints" / "best.ckpt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    _ = checkpoint.write_bytes(weights)
    names = ", ".join(_BODYPARTS)
    _ = (model_dir / "config.yaml").write_text(
        f"model:\n  model_type: heatmap\ndata:\n  keypoint_names: [{names}]\n"
    )
    return model_dir
