"""Injected media facts bypass re-probing on the dataset-scoped read paths."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from mosaic_media import probe_media

import mosaic.core.media.video_io as video_io
from mosaic.tracking.frame_extraction import extract_frames_single


def _write_mp4(path: Path, nframes: int = 6) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 48)
    )
    for _ in range(nframes):
        writer.write(np.zeros((48, 64, 3), np.uint8))
    writer.release()


class _ProbeCalled(RuntimeError):
    """Raised by the stubbed probe so any re-probe is observable."""


def _forbid_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*_args: object, **_kwargs: object):
        raise _ProbeCalled("probe_media was called despite injected facts")

    monkeypatch.setattr(video_io, "probe_media", _boom)


def test_open_frame_reader_injected_facts_skips_probe(tmp_path, monkeypatch):
    video = tmp_path / "clip.mp4"
    _write_mp4(video)
    facts = probe_media(video)

    _forbid_probe(monkeypatch)

    with video_io.open_frame_reader(video, facts=facts) as reader:
        frame_idx, frame = next(iter(reader))
    assert frame is not None
    assert frame_idx == 0


def test_open_frame_reader_without_facts_probes(tmp_path, monkeypatch):
    video = tmp_path / "clip.mp4"
    _write_mp4(video)

    _forbid_probe(monkeypatch)

    # No injected facts: the reader falls back to probing, which now raises.
    with pytest.raises(_ProbeCalled):
        video_io.open_frame_reader(video)


def test_extract_candidate_features_injected_facts_skips_probe(tmp_path, monkeypatch):
    video = tmp_path / "clip.mp4"
    _write_mp4(video, nframes=8)
    facts = probe_media(video)

    _forbid_probe(monkeypatch)

    indices, features = video_io.extract_candidate_features(
        video_path=video,
        start_frame=0,
        end_frame=7,
        candidate_step=1,
        resize=(16, 16),
        grayscale=True,
        crop_rect=None,
        facts=facts,
    )
    assert indices.size > 0
    assert features.shape[0] == indices.size


@pytest.mark.parametrize("method", ["uniform", "kmeans"])
def test_extract_frames_single_injected_facts_skips_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    video = tmp_path / "clip.mp4"
    _write_mp4(video, nframes=12)
    facts = probe_media(video)

    _forbid_probe(monkeypatch)

    result = extract_frames_single(
        video_path=video,
        output_dir=tmp_path / f"out_{method}",
        n_frames=3,
        method=method,
        facts=facts,
    )
    assert result.n_extracted == 3
