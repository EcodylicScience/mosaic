"""End-to-end ``run_ultralytics`` reuse and provenance, without Ultralytics.

The three module-level seams in ``run.py`` are replaced with a recording fake
that writes a small, converter-readable predictions parquet. That exercises the
whole run protocol -- content ``run_id``, phase-marker reuse, the bridge, both
index writers -- with no weights, no torch and no GPU, so this file runs in the
default CI environment.

The fake records into one shared event log, so the *order* of resets and tracks
is assertable: every entry must be reset before it is tracked, which is what
keeps a run's identities independent of what ran before it in the process.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import mosaic.tracking.ultralytics_track.dataset_runs as dr
from mosaic.core.dataset import Dataset, new_dataset_manifest
from mosaic.core.pipeline.markers import read_phase_marker
from mosaic.core.pipeline.tracks_index import read_tracks_index
from mosaic.core.track_library.ultralytics_tracks import raw_columns
from mosaic.tracking.ultralytics_track.dataset_runs import (
    ultralytics_index_path,
    ultralytics_run_root,
)
from mosaic.tracking.ultralytics_track.run import UltralyticsTrackResult

from .conftest import write_media_index

_N_KEYPOINTS = 2


# --- fixtures --------------------------------------------------------------


@pytest.fixture
def ds(tmp_path: Path) -> Dataset:
    manifest = new_dataset_manifest("t", base_dir=tmp_path)
    dataset = Dataset(manifest_path=manifest).load(ensure_roots=True)
    write_media_index(dataset, ["vid1"])
    return dataset


def _make_model(path: Path, *, weights: bytes = b"weights") -> Path:
    """A YOLO model is one weights file, which is the default artifact shape."""
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_bytes(weights)
    return path


@pytest.fixture
def model(tmp_path: Path) -> Path:
    return _make_model(tmp_path / "yolo" / "best.pt")


def _write_predictions(path: Path, *, n_frames: int = 4, n_ids: int = 2) -> None:
    """A predictions parquet in the shape the tracker writes."""
    rows: list[list[float]] = []
    for frame in range(n_frames):
        for track in range(1, n_ids + 1):
            box = [10.0 * track, 20.0, 10.0 * track + 5, 25.0]
            keypoints: list[float] = []
            for k in range(_N_KEYPOINTS):
                keypoints += [10.0 * track + frame + k, 20.0 + k, 0.8]
            rows.append([float(frame), float(track), *box, 0.9, 0.0, *keypoints])
    table = pd.DataFrame(
        np.array(rows, dtype=float), columns=list(raw_columns(_N_KEYPOINTS))
    )
    table = table.astype({"frame": "int64", "track_id": "int64", "cls": "int64"})
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path, index=False)


@dataclass
class FakeUltralytics:
    """Recording stand-in for the three Ultralytics seams."""

    events: list[tuple[str, str]] = field(default_factory=list)
    tracked: list[Path] = field(default_factory=list)
    n_frames: int = 4
    n_ids: int = 2

    def load(self, model_path: Path, **_kwargs: object) -> object:
        self.events.append((str(model_path), "load"))
        return object()

    def reset(self, _session: object) -> None:
        self.events.append((self._current, "reset"))

    def track(
        self, _session: object, video_path: Path, out_parquet: Path, **_kwargs: object
    ) -> UltralyticsTrackResult:
        self.tracked.append(Path(video_path))
        self.events.append((Path(out_parquet).name, "track"))
        _write_predictions(Path(out_parquet), n_frames=self.n_frames, n_ids=self.n_ids)
        return UltralyticsTrackResult(
            predictions_path=Path(out_parquet),
            n_frames=self.n_frames,
            n_ids=self.n_ids,
        )

    _current: str = ""


@pytest.fixture
def ultralytics(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeUltralytics]:
    fake = FakeUltralytics()
    monkeypatch.setattr(dr, "require_ultralytics", lambda _tracker: None)
    monkeypatch.setattr(dr, "effective_tracker_table", lambda _t, resolved: resolved)
    monkeypatch.setattr(
        dr, "write_tracker_yaml", lambda path, table: _write_yaml(path, table)
    )
    monkeypatch.setattr(dr, "load_tracking_model", fake.load)
    monkeypatch.setattr(dr, "reset_trackers", fake.reset)
    monkeypatch.setattr(dr, "run_ultralytics_track", fake.track)
    yield fake


def _write_yaml(path: Path, table: object) -> Path:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(json.dumps(table, sort_keys=True))
    return path


def _index(ds: Dataset) -> pd.DataFrame:
    return pd.read_csv(ultralytics_index_path(ds))


# --- a fresh run produces tracks and both index rows ------------------------


def test_a_fresh_run_tracks_and_bridges(
    ds: Dataset, model: Path, ultralytics: FakeUltralytics
) -> None:
    run_id = dr.run_ultralytics(ds, model_path=str(model))

    assert run_id.startswith("ultralytics.8.4-")
    assert len(ultralytics.tracked) == 1

    work_dir = ultralytics_run_root(ds, run_id) / "vid1"
    assert (work_dir / "vid1.predictions.parquet").exists()
    # The tracker configuration is run-wide, so it sits at the run root.
    assert (ultralytics_run_root(ds, run_id) / "tracker.yaml").exists()
    assert "_tracking/ultralytics" in str(ds.relative_to_root(work_dir))

    tracks = read_tracks_index(ds)
    assert len(tracks) == 1
    row = tracks.iloc[0]
    assert str(row["producer"]) == "ultralytics"
    assert str(row["producer_run_id"]) == run_id
    assert int(row["n_rows"]) == 8  # 4 frames x 2 tracks

    index = _index(ds)
    assert str(index.iloc[0]["tracker"]) == "bytetrack"
    assert str(index.iloc[0]["model_task"]) == "pose"
    assert int(index.iloc[0]["n_keypoints"]) == _N_KEYPOINTS
    assert int(index.iloc[0]["n_frames"]) == 4
    assert str(index.iloc[0]["model_id"]) != ""


# --- reuse ------------------------------------------------------------------


def test_a_completed_run_reuses_the_tracking(
    ds: Dataset, model: Path, ultralytics: FakeUltralytics
) -> None:
    first = dr.run_ultralytics(ds, model_path=str(model))
    second = dr.run_ultralytics(ds, model_path=str(model))

    assert second == first
    assert len(ultralytics.tracked) == 1  # the marker proves the phase done

    # Everything the phase would have reported is re-derived from disk, so the
    # reuse run does not overwrite a good row with zeros.
    index = _index(ds)
    assert len(index) == 1
    assert int(index.iloc[0]["n_ids"]) == 2
    assert int(index.iloc[0]["n_frames"]) == 4
    assert int(index.iloc[0]["n_keypoints"]) == _N_KEYPOINTS
    assert str(index.iloc[0]["model_task"]) == "pose"


def test_overwrite_forces_a_recompute(
    ds: Dataset, model: Path, ultralytics: FakeUltralytics
) -> None:
    _ = dr.run_ultralytics(ds, model_path=str(model))
    _ = dr.run_ultralytics(ds, model_path=str(model), overwrite=True)
    assert len(ultralytics.tracked) == 2


# --- what makes a different run --------------------------------------------


def test_different_weights_are_a_different_run(
    ds: Dataset, tmp_path: Path, ultralytics: FakeUltralytics
) -> None:
    a = _make_model(tmp_path / "a" / "best.pt", weights=b"weights-A")
    b = _make_model(tmp_path / "b" / "best.pt", weights=b"weights-B")
    assert dr.run_ultralytics(ds, model_path=str(a)) != dr.run_ultralytics(
        ds, model_path=str(b)
    )


def test_a_different_backend_is_a_different_run(
    ds: Dataset, model: Path, ultralytics: FakeUltralytics
) -> None:
    assert dr.run_ultralytics(
        ds, model_path=str(model), tracker="bytetrack"
    ) != dr.run_ultralytics(ds, model_path=str(model), tracker="ocsort")


def test_an_override_restating_a_default_is_the_same_run(
    ds: Dataset, model: Path, ultralytics: FakeUltralytics
) -> None:
    """The payoff of declaring the defaults rather than reading them."""
    plain = dr.run_ultralytics(ds, model_path=str(model), tracker="bytetrack")
    restated = dr.run_ultralytics(
        ds,
        model_path=str(model),
        tracker="bytetrack",
        tracker_overrides={"track_buffer": 30},
    )
    assert restated == plain


def test_a_changed_override_is_a_different_run(
    ds: Dataset, model: Path, ultralytics: FakeUltralytics
) -> None:
    plain = dr.run_ultralytics(ds, model_path=str(model), tracker="bytetrack")
    tuned = dr.run_ultralytics(
        ds,
        model_path=str(model),
        tracker="bytetrack",
        tracker_overrides={"track_buffer": 90},
    )
    assert tuned != plain


def test_execution_knobs_do_not_move_the_run(
    ds: Dataset, model: Path, ultralytics: FakeUltralytics
) -> None:
    """Where and how it ran must not name what it produced."""
    base = dr.run_ultralytics(ds, model_path=str(model))
    for kwargs in (
        {"device": "cpu"},
        {"precision": "fp16"},
        {"batch_size": 1},
        {"prefetch": False},
    ):
        assert dr.run_ultralytics(ds, model_path=str(model), **kwargs) == base


# --- a killed run leaves nothing trusted ------------------------------------


def test_an_interrupted_track_is_not_trusted(
    ds: Dataset, model: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(dr, "require_ultralytics", lambda _tracker: None)
    monkeypatch.setattr(dr, "effective_tracker_table", lambda _t, resolved: resolved)
    monkeypatch.setattr(dr, "write_tracker_yaml", _write_yaml)
    monkeypatch.setattr(dr, "load_tracking_model", lambda *a, **k: object())
    monkeypatch.setattr(dr, "reset_trackers", lambda _s: None)

    def dying_track(
        _session: object, _video: Path, out_parquet: Path, **_kwargs: object
    ) -> UltralyticsTrackResult:
        _write_predictions(Path(out_parquet), n_frames=1, n_ids=1)  # a partial file
        raise RuntimeError("killed mid-video")

    monkeypatch.setattr(dr, "run_ultralytics_track", dying_track)
    with pytest.raises(RuntimeError):
        _ = dr.run_ultralytics(ds, model_path=str(model))

    root = ds.get_root("ultralytics")
    for parquet in root.rglob("*.predictions.parquet"):
        assert read_phase_marker(parquet.parent, "track") is None
    assert len(read_tracks_index(ds)) == 0

    # A working run then does the work: the partial file is not reused.
    fake = FakeUltralytics()
    monkeypatch.setattr(dr, "run_ultralytics_track", fake.track)
    _ = dr.run_ultralytics(ds, model_path=str(model))
    assert len(fake.tracked) == 1
    assert len(read_tracks_index(ds)) == 1


# --- the video, by content rather than by location --------------------------


def test_a_video_replaced_in_place_forces_a_recompute(
    ds: Dataset, model: Path, ultralytics: FakeUltralytics
) -> None:
    write_media_index(ds, ["vid1"], uids={"vid1": "uid-aaa"})
    first = dr.run_ultralytics(ds, model_path=str(model))
    write_media_index(ds, ["vid1"], uids={"vid1": "uid-bbb"})
    second = dr.run_ultralytics(ds, model_path=str(model))

    assert second == first  # the recipe did not change
    assert len(ultralytics.tracked) == 2  # but the bytes did


def test_the_same_video_under_a_new_name_is_not_a_recompute(
    ds: Dataset, model: Path, ultralytics: FakeUltralytics
) -> None:
    write_media_index(ds, ["vid1"], uids={"vid1": "uid-aaa"})
    _ = dr.run_ultralytics(ds, model_path=str(model))
    write_media_index(
        ds, ["vid1"], filenames={"vid1": "renamed.mp4"}, uids={"vid1": "uid-aaa"}
    )
    _ = dr.run_ultralytics(ds, model_path=str(model))
    assert len(ultralytics.tracked) == 1


# --- every entry starts from a clean tracker --------------------------------


def test_every_entry_is_reset_before_it_is_tracked(
    ds: Dataset, model: Path, ultralytics: FakeUltralytics
) -> None:
    """Ordering, which is what a fake can prove.

    That the reset *works* -- that identity numbering restarts -- is pinned in
    ``test_ultralytics_preflight.py`` against the real backends, because a fake
    asserting "ids start at 1" would only be asserting that the fake does.
    """
    write_media_index(ds, ["vid1", "vid2"])
    _ = dr.run_ultralytics(ds, model_path=str(model))

    kinds = [kind for _name, kind in ultralytics.events if kind in {"reset", "track"}]
    assert kinds == ["reset", "track", "reset", "track"]
    assert len(ultralytics.tracked) == 2
    assert len(read_tracks_index(ds)) == 2


def test_an_empty_scope_still_returns_the_run_it_minted(
    ds: Dataset, model: Path, ultralytics: FakeUltralytics
) -> None:
    """And loads no model on the way -- a matched-nothing run pays for nothing."""
    run_id = dr.run_ultralytics(ds, model_path=str(model), sequences=["absent"])
    assert run_id.startswith("ultralytics.8.4-")
    assert ultralytics.events == []
