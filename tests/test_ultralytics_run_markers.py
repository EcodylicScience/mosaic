"""End-to-end ``run_ultralytics`` reuse and provenance, without Ultralytics.

The two module-level seams in ``run.py`` are replaced with a recording fake: a
probe reporting what an environment holds, and a track call writing a small,
converter-readable predictions parquet. That exercises the whole run protocol --
content ``run_id``, phase-marker reuse, the bridge, both index writers -- with no
Ultralytics environment, no weights and no GPU, so this file runs in the default
CI environment.

Everything between the two seams is the real thing: the refusals are decided
from the reported probe, the merged tracker table is written as YAML, and each
entry's request is captured, which is what lets the facts the tool would open its
file with be asserted.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import mosaic.tracking.ultralytics_track.dataset_runs as dr
from mosaic.core.dataset import Dataset, new_dataset_manifest
from mosaic.core.pipeline.markers import read_phase_marker
from mosaic.core.pipeline.tracking_roots import TRACKING_ROOTS
from mosaic.core.pipeline.tracks_index import read_tracks_index
from mosaic.core.track_library.ultralytics_tracks import raw_columns
from mosaic.tracking.common.scope import TrackerWorkItem
from mosaic.tracking.common.tool_input import StoreExportMissingError
from mosaic.tracking.external.runner.ultralytics_protocol import (
    ProbeResponse,
    TrackRequest,
)
from mosaic.tracking.ultralytics_track.dataset_runs import (
    ultralytics_index_path,
    ultralytics_run_root,
)
from mosaic.tracking.ultralytics_track.run import (
    TRACK_REQUEST_NAME,
    TRACK_RESPONSE_NAME,
    UltralyticsTrackResult,
)
from mosaic.tracking.ultralytics_track.tracker_defaults import TRACKER_NAMES

from tests.helpers import write_media_index

# Selected by CI's `tracking` job with `-m tracker` rather than by a filename
# list in the workflow, so a new file here is covered the day it lands.
pytestmark = pytest.mark.tracker

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
    """A predictions parquet in the shape the runner writes."""
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


def _probe_response(model_task: str = "pose") -> ProbeResponse:
    """What a healthy environment reports for the fixture's weights.

    ``installed_tracker_table`` is empty, so the merge mosaic writes is its own
    resolved table -- exactly the case a fresh Ultralytics with no extra settings
    produces, and the one that makes the written YAML assertable.
    """
    return ProbeResponse(
        has_ultralytics=True,
        has_lap=True,
        ultralytics_version="8.4.63",
        tracker_names=list(TRACKER_NAMES),
        model_task=model_task,
        n_keypoints=_N_KEYPOINTS,
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
        return _probe_response()

    def track(
        self, request: TrackRequest, *, work_dir: Path, **_kwargs: object
    ) -> UltralyticsTrackResult:
        self.requests.append(request)
        self.work_dirs.append(Path(work_dir))
        self.tracked.append(Path(request.video_path))
        out_parquet = Path(request.output_parquet)
        self.events.append((out_parquet.name, "track"))
        _write_predictions(out_parquet, n_frames=self.n_frames, n_ids=self.n_ids)
        return UltralyticsTrackResult(
            predictions_path=out_parquet, n_frames=self.n_frames, n_ids=self.n_ids
        )


@pytest.fixture
def ultralytics(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeUltralytics]:
    fake = FakeUltralytics()
    monkeypatch.setattr(dr, "probe_ultralytics", fake.probe)
    monkeypatch.setattr(dr, "run_ultralytics_tool", fake.track)
    yield fake


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

    # The facts travel with the request, describing the file the tool opens --
    # here the source itself, so they are the ones the index row measured.
    request = ultralytics.requests[0]
    assert Path(request.video_path) == ds.get_root("media_raw") / "vid1.mp4"
    assert request.media_facts["width"] == 640

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
    assert dr.run_ultralytics(ds, model_path=str(model), device="cpu") == base
    assert dr.run_ultralytics(ds, model_path=str(model), precision="fp16") == base
    assert dr.run_ultralytics(ds, model_path=str(model), batch_size=1) == base
    assert dr.run_ultralytics(ds, model_path=str(model), prefetch=False) == base


# --- a killed run leaves nothing trusted ------------------------------------


def test_an_interrupted_track_is_not_trusted(
    ds: Dataset, model: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def steady_probe(_model_path: Path | str, **_kwargs: object) -> ProbeResponse:
        return _probe_response()

    monkeypatch.setattr(dr, "probe_ultralytics", steady_probe)

    def dying_track(
        request: TrackRequest, *, work_dir: Path, **_kwargs: object
    ) -> UltralyticsTrackResult:
        # A partial file, as a killed runner would leave behind.
        _write_predictions(Path(request.output_parquet), n_frames=1, n_ids=1)
        raise RuntimeError("killed mid-video")

    monkeypatch.setattr(dr, "run_ultralytics_tool", dying_track)
    with pytest.raises(RuntimeError):
        _ = dr.run_ultralytics(ds, model_path=str(model))

    root = ds.get_root("ultralytics")
    for parquet in root.rglob("*.predictions.parquet"):
        assert read_phase_marker(parquet.parent, "track") is None
    assert len(read_tracks_index(ds)) == 0

    # A working run then does the work: the partial file is not reused.
    fake = FakeUltralytics()
    monkeypatch.setattr(dr, "run_ultralytics_tool", fake.track)
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


# --- one entry, one process, one working directory --------------------------


def test_every_entry_is_tracked_once_into_its_own_directory(
    ds: Dataset, model: Path, ultralytics: FakeUltralytics
) -> None:
    """Ordering and addressing, which is what a fake can prove.

    Each entry is tracked exactly once, in scope order, into the working
    directory it was claimed under. Track identity restarting per entry needs no
    assertion here and no reset call: the runner tracks one video per process, so
    the counter every backend numbers from starts at zero by construction.
    """
    write_media_index(ds, ["vid1", "vid2"])
    run_id = dr.run_ultralytics(ds, model_path=str(model))

    run_root = ultralytics_run_root(ds, run_id)
    assert ultralytics.work_dirs == [run_root / "vid1", run_root / "vid2"]
    assert [Path(request.output_parquet) for request in ultralytics.requests] == [
        run_root / "vid1" / "vid1.predictions.parquet",
        run_root / "vid2" / "vid2.predictions.parquet",
    ]
    assert [kind for _name, kind in ultralytics.events] == ["probe", "track", "track"]
    assert len(read_tracks_index(ds)) == 2


def test_an_empty_scope_still_returns_the_run_it_minted(
    ds: Dataset, model: Path, ultralytics: FakeUltralytics
) -> None:
    """And spawns no tracking process -- a matched-nothing run tracks nothing.

    It does probe. The probe is what validates the weights and the backend, and
    that has to happen before the run is named, whatever the scope turns out to
    hold.
    """
    run_id = dr.run_ultralytics(ds, model_path=str(model), sequences=["absent"])
    assert run_id.startswith("ultralytics.8.4-")
    assert ultralytics.tracked == []
    assert [kind for _name, kind in ultralytics.events] == ["probe"]


def test_a_re_run_clears_the_previous_attempts_request_and_response(
    ds: Dataset, model: Path, ultralytics: FakeUltralytics
) -> None:
    """The two JSON names are spelled twice, and this is what ties them.

    ``core`` cannot import ``tracking``, so the phase's clear globs in
    ``TRACKING_ROOTS`` restate the strings ``run.py`` declares. Rename one side
    and nothing else notices: the glob stops matching, and a re-run runs beside
    the previous attempt's request rather than replacing it.
    """
    globs = TRACKING_ROOTS["ultralytics"].clear_globs("track")
    assert TRACK_REQUEST_NAME in globs
    assert TRACK_RESPONSE_NAME in globs

    write_media_index(ds, ["vid1"], uids={"vid1": "uid-aaa"})
    run_id = dr.run_ultralytics(ds, model_path=str(model))
    work_dir = ultralytics_run_root(ds, run_id) / "vid1"
    stale = [work_dir / TRACK_REQUEST_NAME, work_dir / TRACK_RESPONSE_NAME]
    for path in stale:
        _ = path.write_text("{}")

    # Replaced bytes under the same path, which is what invalidates the marker
    # and sends the entry back through the phase.
    write_media_index(ds, ["vid1"], uids={"vid1": "uid-bbb"})
    _ = dr.run_ultralytics(ds, model_path=str(model))

    assert [path for path in stale if path.exists()] == []


# --- what path the tool is given, and what facts describe it ----------------


def _point_at_a_store(ds: Dataset, sequence: str, store: Path) -> Path:
    """Re-address *sequence*'s indexed media at an imgstore recording.

    A store is a directory holding a ``metadata.yaml`` naming ``__store``, which
    is all ``is_imgstore`` reads -- so this needs no chunk files and no imgstore
    package.
    """
    store.mkdir(parents=True, exist_ok=True)
    _ = (store / "metadata.yaml").write_text("__store: {}\n")
    index_path = ds.get_root(ds.resolve_media_root()) / "index.csv"
    table = pd.read_csv(index_path)
    is_entry = table["sequence"] == sequence
    table.loc[is_entry, "abs_path"] = ds.relative_to_root(store)
    table.loc[is_entry, "media_type"] = "imgstore"
    table.to_csv(index_path, index=False)
    return store


def test_a_store_with_no_export_refuses_and_names_the_command(
    ds: Dataset, model: Path, ultralytics: FakeUltralytics
) -> None:
    """The tool opens a path, and no tool opens a directory of chunk files."""
    _ = _point_at_a_store(ds, "vid1", ds.get_root("media_raw") / "vid1.store")

    with pytest.raises(StoreExportMissingError, match="export-store"):
        _ = dr.run_ultralytics(ds, model_path=str(model))
    assert ultralytics.tracked == []


def test_the_facts_describe_the_file_the_tool_will_open(
    ds: Dataset,
    model: Path,
    ultralytics: FakeUltralytics,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    write_cfr_mp4: Callable[..., None],
) -> None:
    """A resolved export is a different file from the one the index measured.

    ``resolve_tool_input`` answers with the export a store was written out to, so
    passing the row's facts across would describe the store. The indexed row says
    640x480; the export here is 64x48, which is what tells the two apart.
    """
    export = tmp_path / "exports" / "vid1.mp4"
    write_cfr_mp4(export, frames=6, size=(64, 48))

    def resolved_export(_ds: Dataset, _item: TrackerWorkItem, *, kind: str) -> Path:
        return export

    monkeypatch.setattr(dr, "resolve_tool_input", resolved_export)

    _ = dr.run_ultralytics(ds, model_path=str(model))

    request = ultralytics.requests[0]
    assert Path(request.video_path) == export
    assert request.media_facts["width"] == 64
    assert request.media_facts["height"] == 48
