"""An inference op covers the whole entry, not its first clip.

A recorder that splits a session into clips leaves an entry whose frames are one
axis over several files. The two Ultralytics inference ops hand their runner one
path, so a multi-clip entry resolves to its join, as the trackers' entries do.
The localizer reads in this process, so it reads the clips on the entry's frame
axis, and their join only when their frame rates differ. Either way the
published table's ``frame`` is the entry's frame.

The clips and the joins are real. The model runners are the recording fakes from
``tests.helpers``, and the localizer's fake reports a detection per frame of
what it is handed.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest
from mosaic_media import MediaFacts, probe_media

from mosaic.core.dataset import Dataset
from mosaic.core.media.video_io import read_entry_frames
from mosaic.core.pipeline.joined_export import (
    JOINED_KIND_DIRECTORY,
    JoinedExportMissingError,
    current_joined_recipes,
    joined_source_uid,
)
from mosaic.core.pipeline.markers import read_phase_marker
from mosaic.core.pipeline.ops import run_op
from mosaic.core.pipeline.sequence_index import decode_consumed_roots
from mosaic.core.pipeline.tracks_index import read_tracks_index
from mosaic.core.scope import Scope
from mosaic.tracking.common.scope import JoinedSourceMismatchError, build_work_items
from mosaic.tracking.common.tool_input import StoreExportMissingError
from mosaic.tracking.ops.infer import infer_run_root
from mosaic.tracking.pose_training.localizer_inference import (
    LocalizerDetection,
    LocalizerFrame,
)
from tests.helpers import (
    MediaClip,
    gray_level,
    install_fake_point_inference,
    install_fake_pose_inference,
    make_dataset,
    paint_gray,
    point_at_a_store,
    pose_per_frame,
    published_table,
    write_media_index,
    write_painted_entry,
)

pytestmark = pytest.mark.media

_SIZE = (64, 48)
_CLIP_FRAMES = 10
_ENTRY = ("", "sess")


def _dataset(
    tmp_path: Path, rates: Sequence[float], frames: int = _CLIP_FRAMES
) -> tuple[Dataset, list[Path]]:
    """One entry, ``sess``, of one clip of *frames* frames per rate in *rates*."""
    ds = make_dataset(tmp_path / "ds")
    clips = write_painted_entry(
        ds, "sess", [(frames, rate) for rate in rates], paint_gray, size=_SIZE
    )
    return ds, clips


@pytest.fixture
def model(tmp_path: Path) -> Path:
    path = tmp_path / "weights" / "best.pt"
    path.parent.mkdir(parents=True)
    _ = path.write_bytes(b"weights")
    return path


def _join(ds: Dataset) -> Path:
    _ = run_op(ds, "export-joined", {}, scope=Scope(entries=[_ENTRY]))
    (joined,) = sorted((ds.get_root("media") / "joined").glob("*.joined.mp4"))
    return joined


def _infer(ds: Dataset, kind: str, model: Path) -> str:
    return run_op(ds, kind, {"model": str(model)}, scope=Scope(entries=[_ENTRY]))


def _recorded_source_uid(ds: Dataset, kind: str, run_id: str) -> str:
    """The identity that *kind*'s completion marker records for the entry."""
    marker = read_phase_marker(infer_run_root(ds, kind, run_id) / "sess", "infer")
    assert marker is not None
    return marker.source_uid


def _tracker_source_uid(ds: Dataset) -> str:
    """The identity that a tracker's work item gives the entry."""
    (item,) = build_work_items(ds, ds.resolve_media_scope(None), kind="trex").items
    return item.source_uid


class TestAnOpThatHandsItsRunnerAPath:
    def test_it_is_handed_the_join_and_its_table_spans_both_clips(
        self,
        tmp_path: Path,
        model: Path,
        monkeypatch: pytest.MonkeyPatch,
        requires_ffmpeg: None,
    ) -> None:
        ds, _clips = _dataset(tmp_path, [30.0, 30.0])
        joined = _join(ds)
        fake = install_fake_pose_inference(monkeypatch, pose_per_frame)

        _ = _infer(ds, "infer-pose", model)

        assert fake.videos == [joined]
        frames = published_table(ds, "infer-pose")["frame"]
        assert (int(frames.min()), int(frames.max())) == (0, 2 * _CLIP_FRAMES - 1)
        (row,) = [row for _, row in read_tracks_index(ds).iterrows()]
        roots = decode_consumed_roots(str(row["consumed_source_roots"]))
        assert roots == ("media_raw",), "the clips, not the join the model read"

    def test_it_records_the_clips_identity_not_the_join_s(
        self,
        tmp_path: Path,
        model: Path,
        monkeypatch: pytest.MonkeyPatch,
        requires_ffmpeg: None,
    ) -> None:
        ds, _clips = _dataset(tmp_path, [30.0, 30.0])
        joined = _join(ds)
        _ = install_fake_pose_inference(monkeypatch, pose_per_frame)

        run_id = _infer(ds, "infer-pose", model)

        recorded = _recorded_source_uid(ds, "infer-pose", run_id)
        assert recorded == _tracker_source_uid(ds)
        assert recorded != probe_media(joined).video_uuid

    def test_a_missing_join_is_refused_naming_the_command(
        self,
        tmp_path: Path,
        model: Path,
        monkeypatch: pytest.MonkeyPatch,
        requires_ffmpeg: None,
    ) -> None:
        ds, _clips = _dataset(tmp_path, [30.0, 30.0])
        fake = install_fake_pose_inference(monkeypatch, pose_per_frame)

        with pytest.raises(JoinedExportMissingError, match="--kind export-joined"):
            _ = _infer(ds, "infer-pose", model)
        assert fake.videos == []
        assert fake.probed == [], "the model's environment was probed first"


@pytest.mark.parametrize("kind", ["infer-pose", "infer-points", "infer-localizer"])
@pytest.mark.parametrize(
    ("second", "reason"),
    [
        (MediaClip(filename="c1.mp4", video_order=1, width=1280), "c1.mp4 has width"),
        (MediaClip(filename="c1.mp4", video_order=1, fps=0.0), "no frame rate"),
    ],
    ids=["geometry", "rate"],
)
def test_clips_that_cannot_be_one_video_are_refused_as_a_tracker_refuses_them(
    tmp_path: Path, model: Path, kind: str, second: MediaClip, reason: str
) -> None:
    """Named for what is wrong with them, before a join or a model is looked for."""
    ds = make_dataset(tmp_path / "ds")
    write_media_index(ds, [MediaClip(filename="c0.mp4"), second])

    with pytest.raises(JoinedSourceMismatchError, match=reason):
        _ = _infer(ds, kind, model)


@pytest.mark.parametrize("kind", ["infer-localizer", "infer-pose"])
def test_stores_at_two_rates_are_refused_before_a_model_loads(
    tmp_path: Path, model: Path, kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stores are not joined, so no reader places their frames on one axis."""
    ds = make_dataset(tmp_path / "ds")
    write_media_index(
        ds,
        [
            MediaClip(filename="a.mp4", video_uuid="uid-a", fps=30.0),
            MediaClip(filename="b.mp4", video_order=1, video_uuid="uid-b", fps=31.0),
        ],
    )
    for order, name in enumerate(("a", "b")):
        _ = point_at_a_store(
            ds, "sess", ds.get_root("media_raw") / f"{name}.store", video_order=order
        )
    localizer = _install_fake_localizer(monkeypatch)
    runner = install_fake_pose_inference(monkeypatch, pose_per_frame)

    with pytest.raises(JoinedSourceMismatchError) as refused:
        _ = _infer(ds, kind, model)

    message = str(refused.value)
    assert "b.store" in message
    assert "--kind preprocess" in message
    assert localizer == [] and runner.videos == []


@pytest.mark.parametrize("kind", ["infer-pose", "infer-points"])
def test_stores_at_one_rate_are_refused_for_a_runner_handed_a_path(
    tmp_path: Path, model: Path, kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The runner is handed one file, and export-joined does not join stores."""
    ds = make_dataset(tmp_path / "ds")
    write_media_index(
        ds,
        [
            MediaClip(filename="a.mp4", video_uuid="uid-a", frame_count=40),
            MediaClip(
                filename="b.mp4", video_order=1, video_uuid="uid-b", frame_count=60
            ),
        ],
    )
    for order, name in enumerate(("a", "b")):
        _ = point_at_a_store(
            ds, "sess", ds.get_root("media_raw") / f"{name}.store", video_order=order
        )
    pose = install_fake_pose_inference(monkeypatch, pose_per_frame)
    points = install_fake_point_inference(monkeypatch)

    with pytest.raises(JoinedSourceMismatchError) as refused:
        _ = _infer(ds, kind, model)

    message = str(refused.value)
    assert "export-joined does not join stores" in message
    assert '"stop":100' in message
    assert pose.videos == [] and points.videos == []
    assert pose.probed == [] and points.probed == []


def _two_clips(sequence: str) -> list[MediaClip]:
    return [
        MediaClip(
            sequence=sequence,
            filename=f"{sequence}-{order}.mp4",
            video_order=order,
            video_uuid=f"{sequence}-uid-{order}",
        )
        for order in range(2)
    ]


class TestEntriesWithNoFileBuilt:
    """Every entry whose join or store export cannot be read is named at once."""

    def test_every_missing_join_is_named_before_a_model_runs(
        self, tmp_path: Path, model: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ds = make_dataset(tmp_path / "ds")
        write_media_index(ds, [*_two_clips("a"), *_two_clips("b")])
        fake = install_fake_pose_inference(monkeypatch, pose_per_frame)

        with pytest.raises(JoinedExportMissingError) as refused:
            _ = run_op(ds, "infer-pose", {"model": str(model)})

        message = str(refused.value)
        assert message.startswith("[infer-pose] 2 entries cannot be read yet")
        for sequence in ("a", "b"):
            assert f'--kind export-joined --entries ":{sequence}"' in message
        assert fake.videos == []
        assert fake.probed == [], "the model's environment was probed first"

    def test_every_missing_store_export_is_named_as_one(
        self, tmp_path: Path, model: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Refused as a missing store export, the error that a caller catches."""
        ds = make_dataset(tmp_path / "ds")
        write_media_index(ds, [MediaClip(sequence="s"), MediaClip(sequence="t")])
        for sequence in ("s", "t"):
            _ = point_at_a_store(
                ds, sequence, ds.get_root("media_raw") / f"{sequence}.store"
            )
        fake = install_fake_pose_inference(monkeypatch, pose_per_frame)

        with pytest.raises(StoreExportMissingError) as refused:
            _ = run_op(ds, "infer-pose", {"model": str(model)})

        message = str(refused.value)
        for sequence in ("s", "t"):
            assert f'--kind export-store --entries ":{sequence}"' in message
        assert fake.videos == [] and fake.probed == []

    def test_two_current_joins_are_named_beside_a_missing_one(
        self, tmp_path: Path, model: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One entry needs a join built, and the other needs one of two deleted."""
        ds = make_dataset(tmp_path / "ds")
        write_media_index(ds, [*_two_clips("a"), *_two_clips("b")])
        root = ds.get_root("media") / JOINED_KIND_DIRECTORY
        root.mkdir(parents=True)
        uid = joined_source_uid(ds.resolve_media("", "b").facts)
        for recipe in sorted(current_joined_recipes()):
            _ = (root / f"{uid}.{recipe}.joined.mp4").write_bytes(b"join")
        fake = install_fake_pose_inference(monkeypatch, pose_per_frame)

        with pytest.raises(JoinedExportMissingError) as refused:
            _ = run_op(ds, "infer-pose", {"model": str(model)})

        message = str(refused.value)
        assert message.startswith("[infer-pose] 2 entries cannot be read yet")
        assert "Each is named below with what to do" in message
        assert '--kind export-joined --entries ":a"' in message
        assert "2 current joins" in message and "delete the rest" in message
        assert fake.videos == [] and fake.probed == []

    def test_a_missing_join_and_a_missing_export_are_named_together(
        self, tmp_path: Path, model: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ds = make_dataset(tmp_path / "ds")
        write_media_index(ds, [*_two_clips("a"), MediaClip(sequence="s")])
        _ = point_at_a_store(ds, "s", ds.get_root("media_raw") / "s.store")
        fake = install_fake_pose_inference(monkeypatch, pose_per_frame)

        with pytest.raises(FileNotFoundError) as refused:
            _ = run_op(ds, "infer-pose", {"model": str(model)})

        assert not isinstance(
            refused.value, (JoinedExportMissingError, StoreExportMissingError)
        )
        message = str(refused.value)
        assert '--kind export-joined --entries ":a"' in message
        assert '--kind export-store --entries ":s"' in message
        assert fake.videos == [] and fake.probed == []


type Handed = list[tuple[Path, ...]]


def _install_fake_localizer(monkeypatch: pytest.MonkeyPatch) -> Handed:
    """Stand in for the localizer: one detection per frame of what it reads."""
    import mosaic.tracking.pose_training.localizer_inference as localizer

    handed: Handed = []

    def run(
        _model_path: str,
        video_paths: Sequence[Path],
        *,
        facts: Sequence[MediaFacts] | None = None,
        **_kwargs: object,
    ) -> list[LocalizerFrame]:
        handed.append(tuple(video_paths))
        assert facts is not None
        frames = sum(clip.frame_count for clip in facts)
        detection: LocalizerDetection = {
            "x": 1.0,
            "y": 4.0,
            "confidence": 0.9,
            "class_id": 0,
        }
        return [LocalizerFrame(frame, (detection,)) for frame in range(frames)]

    monkeypatch.setattr(localizer, "run_localizer_inference", run)
    return handed


class TestTheLocalizer:
    def test_it_reads_clips_of_one_rate_without_a_join(
        self,
        tmp_path: Path,
        model: Path,
        monkeypatch: pytest.MonkeyPatch,
        requires_ffmpeg: None,
    ) -> None:
        ds, clips = _dataset(tmp_path, [30.0, 30.0])
        handed = _install_fake_localizer(monkeypatch)

        _ = _infer(ds, "infer-localizer", model)

        assert handed == [tuple(clips)]
        frames = published_table(ds, "infer-localizer")["frame"]
        assert (int(frames.min()), int(frames.max())) == (0, 2 * _CLIP_FRAMES - 1)

    def test_it_reads_clips_of_two_rates_through_their_join(
        self,
        tmp_path: Path,
        model: Path,
        monkeypatch: pytest.MonkeyPatch,
        requires_ffmpeg: None,
    ) -> None:
        # Thirty frames, because over ten the two rates drift apart by less
        # than the half frame that the reader tolerates.
        ds, _clips = _dataset(tmp_path, [30.0, 31.0], frames=30)
        handed = _install_fake_localizer(monkeypatch)
        with pytest.raises(JoinedExportMissingError, match="--kind export-joined"):
            _ = _infer(ds, "infer-localizer", model)

        joined = _join(ds)
        run_id = _infer(ds, "infer-localizer", model)

        assert handed == [(joined,)]
        recorded = _recorded_source_uid(ds, "infer-localizer", run_id)
        assert recorded == _tracker_source_uid(ds)
        assert recorded != probe_media(joined).video_uuid


class TestReadingAnEntrysClips:
    """The reader behind the localizer, over real clips."""

    @staticmethod
    def _frames_and_levels(
        clips: Sequence[Path], *, start: int = 0, step: int = 1
    ) -> list[tuple[int, int]]:
        return [
            (frame, round(float(image.mean())))
            for frame, image in read_entry_frames(
                clips, start_frame=start, frame_step=step, target="analysis"
            )
        ]

    def test_frame_i_is_entry_frame_i_across_the_boundary(
        self, tmp_path: Path, requires_ffmpeg: None
    ) -> None:
        _ds, clips = _dataset(tmp_path, [30.0, 30.0])

        read = self._frames_and_levels(clips)

        assert [frame for frame, _ in read] == list(range(2 * _CLIP_FRAMES))
        for frame, level in read:
            assert level == pytest.approx(gray_level(frame), abs=4), frame

    def test_one_file_may_be_a_bare_path(
        self, tmp_path: Path, requires_ffmpeg: None
    ) -> None:
        """Not a sequence of the path's characters."""
        _ds, (clip,) = _dataset(tmp_path, [30.0])

        for bare in (clip, str(clip)):
            read = [frame for frame, _ in read_entry_frames(bare, target="analysis")]
            assert read == list(range(_CLIP_FRAMES)), bare

    def test_a_window_counts_entry_frames(
        self, tmp_path: Path, requires_ffmpeg: None
    ) -> None:
        _ds, clips = _dataset(tmp_path, [30.0, 30.0])

        read = self._frames_and_levels(clips, start=7, step=3)

        assert [frame for frame, _ in read] == [7, 10, 13, 16, 19]
        for frame, level in read:
            assert level == pytest.approx(gray_level(frame), abs=4), frame
