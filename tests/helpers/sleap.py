"""Stand in for SLEAP in the tests of the SLEAP tracker, recording each call.

``run_sleap`` calls SLEAP through two module-level seams in
``sleap/dataset_runs.py``: ``sleap-nn track`` and ``sleap-convert``.
:func:`install_fake_sleap` replaces both with a :class:`FakeSleap`, whose
inference writes a ``.slp`` and whose export writes a small, converter-readable
analysis ``.h5``. A test runs the whole tracker protocol without a model.
:func:`write_sleap_model` writes the model directory that a run names.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from mosaic.tracking.sleap.run import SleapConvertResult, SleapTrackResult

from tests.helpers.tracks import write_sleap_analysis_h5


@dataclass
class FakeSleap:
    """Recording stand-ins for the two SLEAP phases."""

    tracked: list[Path] = field(default_factory=list)
    track_settings: list[dict[str, object]] = field(default_factory=list)
    converted: list[Path] = field(default_factory=list)
    frames: int = 6

    def track(
        self, video_path: Path, output_slp: Path, **kwargs: object
    ) -> SleapTrackResult:
        self.tracked.append(Path(video_path))
        self.track_settings.append(kwargs)
        out = Path(output_slp)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"slp")
        return SleapTrackResult(slp_path=out, stdout="", stderr="")

    def convert(
        self, slp_path: Path, output_h5: Path, **_kwargs: object
    ) -> SleapConvertResult:
        self.converted.append(Path(slp_path))
        out = Path(output_h5)
        tracks = np.random.default_rng(0).random((self.frames, 1, 1, 2))
        write_sleap_analysis_h5(out, tracks)
        return SleapConvertResult(analysis_h5_path=out, stdout="", stderr="")


def install_fake_sleap(
    monkeypatch: pytest.MonkeyPatch, fake: FakeSleap | None = None
) -> FakeSleap:
    """Replace the tracker's two seams with *fake*, or a new :class:`FakeSleap`."""
    import mosaic.tracking.sleap.dataset_runs as dataset_runs

    installed = fake if fake is not None else FakeSleap()
    monkeypatch.setattr(dataset_runs, "run_sleap_track", installed.track)
    monkeypatch.setattr(dataset_runs, "run_sleap_convert", installed.convert)
    return installed


def write_sleap_model(
    directory: Path, weights: bytes = b"weights", *, head: str = ""
) -> Path:
    """Write a SLEAP model directory at *directory*, and return it.

    The directory contains ``best.ckpt`` with *weights*, whose digest is the model's
    identity. Two directories with the same *weights* are one model. When *head* is
    given, a ``training_config.yaml`` selecting it under ``head_configs`` is written
    beside the checkpoint, as SLEAP's trainer leaves one. The config is provenance,
    read for the model type, and a model does not need one.
    """
    directory.mkdir(parents=True, exist_ok=True)
    _ = (directory / "best.ckpt").write_bytes(weights)
    if head:
        _ = (directory / "training_config.yaml").write_text(
            f"head_configs:\n  {head}: {{}}\n"
        )
    return directory
