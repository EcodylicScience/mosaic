"""A tracker reuses an entry's output only for the clips that the output was made from.

Every tracker reads the whole entry: its one clip, the join of its clips, or for
Ultralytics, whose runner reads them in order, the clips themselves. The reuse
gate compares the identity of that whole input, the ordered composition of the
clips (``TrackerWorkItem.source_uid``), so a clip replaced after the first is
noticed. A marker that cannot name the clip
set, because it records only the first clip's uuid or no uuid at all, proves
nothing for several clips. One clip's identity is its own uuid, so a single-clip
entry reuses what it reused before.

The tools are the recording fakes. The clips are stubs, and their join is a small
real video.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import pytest

import mosaic.tracking.litpose.dataset_runs as litpose_runs
import mosaic.tracking.sleap.dataset_runs as sleap_runs
import mosaic.tracking.trex.dataset_runs as trex_runs
import mosaic.tracking.ultralytics_track.dataset_runs as ultralytics_runs
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.joined_export import JoinedExportMissingError
from mosaic.core.pipeline.markers import (
    PHASE_NAMES,
    read_phase_marker,
    write_phase_marker,
)
from mosaic.tracking.common.entry import AdoptEvidence, adopt_completed_directory
from mosaic.tracking.common.mint import tracker_run_root
from mosaic.tracking.common.scope import TrackerWorkItem
from mosaic.tracking.litpose.params import LitposeParams
from mosaic.tracking.sleap.params import SleapParams
from mosaic.tracking.trex.params import TrexParams
from mosaic.tracking.ultralytics_track.params import UltralyticsParams
from tests.helpers import (
    MediaClip,
    install_fake_litpose,
    install_fake_sleap,
    install_fake_trex,
    install_fake_ultralytics,
    make_dataset,
    stub_join,
    write_h264_mp4,
    write_litpose_model,
    write_media_index,
    write_sleap_model,
)

TrackerKind = Literal["sleap", "litpose", "ultralytics", "trex"]
TRACKERS: tuple[TrackerKind, ...] = ("sleap", "litpose", "ultralytics", "trex")
_ENTRY = "sess"


@dataclass(frozen=True, slots=True)
class Tracker:
    """One tracker with its tool faked.

    Attributes:
        kind: The tracker's kind, which names its root.
        run: Runs the tracker over the dataset, and returns the run id.
        calls: How many times the tool has tracked an entry.
    """

    kind: TrackerKind
    run: Callable[[Dataset], str]
    calls: Callable[[], int]


def _tracker(
    kind: TrackerKind, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Tracker:
    """Install the fake tool of *kind*, and return its tracker."""
    if kind == "sleap":
        sleap = install_fake_sleap(monkeypatch)
        sleap_params = SleapParams(
            model_paths=[str(write_sleap_model(tmp_path / "sleap_model"))]
        )
        return Tracker(
            kind,
            run=lambda ds: sleap_runs.run_sleap(ds, sleap_params),
            calls=lambda: len(sleap.tracked),
        )
    if kind == "litpose":
        litpose = install_fake_litpose(monkeypatch)
        litpose_params = LitposeParams(
            model_path=str(write_litpose_model(tmp_path / "litpose_model"))
        )
        return Tracker(
            kind,
            run=lambda ds: litpose_runs.run_litpose(ds, litpose_params),
            calls=lambda: len(litpose.predicted),
        )
    if kind == "ultralytics":
        ultralytics = install_fake_ultralytics(monkeypatch)
        weights = tmp_path / "yolo" / "best.pt"
        weights.parent.mkdir(parents=True)
        _ = weights.write_bytes(b"weights")
        ultralytics_params = UltralyticsParams.model_validate(
            {"model_path": str(weights)}
        )
        return Tracker(
            kind,
            run=lambda ds: ultralytics_runs.run_ultralytics(ds, ultralytics_params),
            calls=lambda: len(ultralytics.tracked),
        )
    trex = install_fake_trex(monkeypatch)
    return Tracker(
        kind,
        run=lambda ds: trex_runs.run_trex(ds, TrexParams()),
        calls=lambda: len(trex.tracked),
    )


class IndexSession(Protocol):
    """Index the one entry as one clip per uid, in order."""

    def __call__(self, *uids: str) -> None: ...


@pytest.fixture
def session(ds: Dataset, requires_ffmpeg: None) -> IndexSession:
    """Index the entry's clips, and write the join of several.

    The clips keep their filenames whatever their uids, so a changed uid is a clip
    replaced in place. The clips are stubs with stored facts, and the join that a
    one-file tool is handed is a real H.264 video. No join is written when a uid is
    empty, because clips of which one is unidentified have no join.
    """

    def index(*uids: str) -> None:
        write_media_index(
            ds,
            [
                MediaClip(
                    sequence=_ENTRY,
                    filename=f"c{order}.mp4",
                    video_order=order,
                    video_uuid=uid,
                )
                for order, uid in enumerate(uids)
            ],
        )
        if len(uids) > 1 and all(uids):
            write_h264_mp4(stub_join(ds, uids))

    return index


def _work_dir(ds: Dataset, tracker: Tracker, run_id: str) -> Path:
    return tracker_run_root(ds, tracker.kind, run_id) / _ENTRY


@pytest.fixture
def ds(tmp_path: Path) -> Dataset:
    return make_dataset(tmp_path / "ds")


@pytest.mark.parametrize("kind", TRACKERS)
def test_a_later_clip_replaced_is_tracked_again(
    kind: TrackerKind,
    ds: Dataset,
    session: IndexSession,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The first clip is unchanged, so only the whole clip set can notice."""
    tracker = _tracker(kind, monkeypatch, tmp_path)
    session("uid-a", "uid-b")
    first = tracker.run(ds)
    session("uid-a", "uid-c")
    second = tracker.run(ds)

    assert second == first, "the settings did not change, so neither does the run"
    assert tracker.calls() == 2, "the replaced clip was not tracked"


@pytest.mark.parametrize("kind", TRACKERS)
def test_an_unchanged_clip_set_is_reused(
    kind: TrackerKind,
    ds: Dataset,
    session: IndexSession,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tracker = _tracker(kind, monkeypatch, tmp_path)
    session("uid-a", "uid-b")
    run_id = tracker.run(ds)
    _ = tracker.run(ds)

    assert tracker.calls() == 1
    marker = read_phase_marker(_work_dir(ds, tracker, run_id), "track")
    assert marker is not None
    assert marker.source_uid not in ("", "uid-a"), "the marker names the clip set"


@pytest.mark.parametrize("kind", TRACKERS)
def test_one_clip_is_reused_under_its_own_uuid(
    kind: TrackerKind,
    ds: Dataset,
    session: IndexSession,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A single-clip marker records the clip's uuid, as every one on disk does."""
    tracker = _tracker(kind, monkeypatch, tmp_path)
    session("uid-a")
    run_id = tracker.run(ds)
    _ = tracker.run(ds)

    assert tracker.calls() == 1
    marker = read_phase_marker(_work_dir(ds, tracker, run_id), "track")
    assert marker is not None
    assert marker.source_uid == "uid-a"


@pytest.mark.parametrize("kind", TRACKERS)
@pytest.mark.parametrize("recorded", ["uid-a", ""], ids=["first-clip", "none"])
def test_a_marker_that_cannot_name_the_clip_set_is_tracked_again(
    kind: TrackerKind,
    recorded: str,
    ds: Dataset,
    session: IndexSession,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A marker naming the first clip, or no clip, says nothing of the others.

    A marker for several clips written by an earlier gate records the first
    clip's uuid, and a marker for media indexed without uuids records none. The
    path it also records is the first clip's, so the path cannot prove the rest
    either.
    """
    tracker = _tracker(kind, monkeypatch, tmp_path)
    session("uid-a", "uid-b")
    run_id = tracker.run(ds)
    work_dir = _work_dir(ds, tracker, run_id)
    for phase in PHASE_NAMES:
        marker = read_phase_marker(work_dir, phase)
        if marker is not None:
            write_phase_marker(
                work_dir, marker.model_copy(update={"source_uid": recorded})
            )

    _ = tracker.run(ds)
    assert tracker.calls() == 2


@pytest.mark.parametrize("kind", TRACKERS)
def test_several_clips_one_unidentified_are_refused_before_any_tool_runs(
    kind: TrackerKind,
    ds: Dataset,
    session: IndexSession,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Refused, not reused: the path of the first clip cannot vouch for the second.

    No join of such clips can be addressed, so none can be tracked either. The
    refusal names the command that mints the missing identity, and leaves the
    earlier run's marker and outputs in place.
    """
    tracker = _tracker(kind, monkeypatch, tmp_path)
    session("uid-a", "uid-b")
    run_id = tracker.run(ds)
    work_dir = _work_dir(ds, tracker, run_id)
    marker = read_phase_marker(work_dir, "track")
    outputs = sorted(path.relative_to(ds.base_dir) for path in ds.base_dir.rglob("*"))
    session("uid-a", "")

    with pytest.raises(JoinedExportMissingError, match="reprobe-media"):
        _ = tracker.run(ds)
    assert tracker.calls() == 1
    assert read_phase_marker(work_dir, "track") == marker, "the marker was cleared"
    kept = {path.relative_to(ds.base_dir) for path in ds.base_dir.rglob("*")}
    lost = [path for path in outputs if path not in kept]
    assert lost == [], "the earlier run's outputs were cleared"


@pytest.mark.parametrize("kind", TRACKERS)
def test_a_run_refused_for_its_media_records_no_tracks_variant(
    kind: TrackerKind,
    ds: Dataset,
    session: IndexSession,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The entries are refused before the run is minted, as the inference ops do."""
    tracker = _tracker(kind, monkeypatch, tmp_path)
    session("uid-a", "")

    with pytest.raises(JoinedExportMissingError):
        _ = tracker.run(ds)

    assert list(ds.get_root("tracks").rglob("params.json")) == []
    assert list(ds.base_dir.rglob("run_params.json")) == []


@pytest.mark.parametrize(("clips", "adopted"), [(1, True), (2, False)])
def test_a_directory_that_predates_markers_is_adopted_for_one_clip_only(
    clips: int, adopted: bool, ds: Dataset, tmp_path: Path
) -> None:
    """Its outputs look the same whether they cover one clip or all of them."""
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    _ = (work_dir / "sess.predictions.slp").write_bytes(b"slp")
    item = TrackerWorkItem(
        group="",
        sequence=_ENTRY,
        key=_ENTRY,
        video_paths=tuple(Path(f"c{order}.mp4") for order in range(clips)),
        fps=30.0,
    )

    adopt_completed_directory(
        ds,
        work_dir,
        "sleap.1.6-0123456789",
        item=item,
        required=("*.predictions.slp",),
        record=(AdoptEvidence("track", "*.predictions.slp"),),
    )
    assert (read_phase_marker(work_dir, "track") is not None) is adopted


def test_sleap_tracks_several_clips_again_rather_than_adopt_them(
    ds: Dataset,
    session: IndexSession,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A SLEAP directory whose markers are gone is tracked, not adopted."""
    tracker = _tracker("sleap", monkeypatch, tmp_path)
    session("uid-a", "uid-b")
    run_id = tracker.run(ds)
    work_dir = _work_dir(ds, tracker, run_id)
    for marker in work_dir.glob(".mosaic-*.json"):
        marker.unlink()

    _ = tracker.run(ds)
    assert tracker.calls() == 2
    marker = read_phase_marker(work_dir, "track")
    assert marker is not None
    assert not marker.backfilled
