"""Stand in for the Ultralytics environment in the tracker's tests, recording calls.

``run_ultralytics`` calls Ultralytics through two module-level seams in
``ultralytics_track/dataset_runs.py``: a probe that reports an environment's
contents, and a track call that writes a predictions parquet.
:func:`install_fake_ultralytics` replaces both with :class:`FakeUltralytics`. A
test therefore runs the whole tracker protocol without an Ultralytics environment,
weights or a GPU.

The predictions that it writes are fixed by :func:`write_ultralytics_predictions`.
A test can therefore compute the table that the bridge publishes from them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import pytest

from mosaic.core.track_library.ultralytics_tracks import raw_columns
from mosaic.tracking.external.runner.ultralytics_protocol import (
    ProbeResponse,
    TrackRequest,
)
from mosaic.tracking.ultralytics_track.run import UltralyticsTrackResult
from mosaic.tracking.ultralytics_track.tracker_defaults import TRACKER_NAMES

ULTRALYTICS_KEYPOINTS: Final = 2
"""The number of keypoints in the fake's weights."""


def write_ultralytics_predictions(
    path: Path, *, n_frames: int = 4, n_ids: int = 2
) -> None:
    """Write a predictions parquet in the shape that the runner writes.

    Track ``t`` in frame ``f`` has the box ``(10t, 20, 10t + 5, 25)`` and keypoint
    ``k`` at ``(10t + f + k, 20 + k)``.
    """
    rows: list[list[float]] = []
    for frame in range(n_frames):
        for track in range(1, n_ids + 1):
            box = [10.0 * track, 20.0, 10.0 * track + 5, 25.0]
            keypoints: list[float] = []
            for k in range(ULTRALYTICS_KEYPOINTS):
                keypoints += [10.0 * track + frame + k, 20.0 + k, 0.8]
            rows.append([float(frame), float(track), *box, 0.9, 0.0, *keypoints])
    table = pd.DataFrame(
        np.array(rows, dtype=float), columns=list(raw_columns(ULTRALYTICS_KEYPOINTS))
    )
    table = table.astype({"frame": "int64", "track_id": "int64", "cls": "int64"})
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path, index=False)


def ultralytics_probe_response(
    model_task: str = "pose",
    *,
    has_locate: bool = False,
    version: str = "8.4.63",
    n_keypoints: int = ULTRALYTICS_KEYPOINTS,
) -> ProbeResponse:
    """Return the report of a healthy environment for the fake's weights.

    ``installed_tracker_table`` is empty, so the merge mosaic writes is its own
    resolved table -- exactly the case a fresh Ultralytics with no extra settings
    produces, and the one that makes the written YAML assertable. A POLO
    environment defines ``locate``, which *has_locate* reports.
    """
    return ProbeResponse(
        has_ultralytics=True,
        has_lap=True,
        has_locate=has_locate,
        ultralytics_version=version,
        tracker_names=list(TRACKER_NAMES),
        model_task=model_task,
        n_keypoints=n_keypoints,
        model_load_error="",
        installed_tracker_table={},
    )


@dataclass
class FakeUltralytics:
    """Recording stand-in for the two runner seams."""

    events: list[tuple[str, str]] = field(default_factory=list)
    requests: list[TrackRequest] = field(default_factory=list)
    work_dirs: list[Path] = field(default_factory=list)
    tracked: list[Path] = field(default_factory=list)
    n_frames: int = 4
    n_ids: int = 2

    def probe(self, model_path: Path | str, **_kwargs: object) -> ProbeResponse:
        self.events.append((str(model_path), "probe"))
        return ultralytics_probe_response()

    def track(
        self, request: TrackRequest, *, work_dir: Path, **_kwargs: object
    ) -> UltralyticsTrackResult:
        self.requests.append(request)
        self.work_dirs.append(Path(work_dir))
        self.tracked.append(Path(request.video_path))
        out_parquet = Path(request.output_parquet)
        self.events.append((out_parquet.name, "track"))
        write_ultralytics_predictions(
            out_parquet, n_frames=self.n_frames, n_ids=self.n_ids
        )
        return UltralyticsTrackResult(
            predictions_path=out_parquet, n_frames=self.n_frames, n_ids=self.n_ids
        )


def install_fake_ultralytics(monkeypatch: pytest.MonkeyPatch) -> FakeUltralytics:
    """Replace the tracker's two runner seams with a new :class:`FakeUltralytics`."""
    import mosaic.tracking.ultralytics_track.dataset_runs as dataset_runs

    fake = FakeUltralytics()
    monkeypatch.setattr(dataset_runs, "probe_ultralytics", fake.probe)
    monkeypatch.setattr(dataset_runs, "run_ultralytics_tool", fake.track)
    return fake
