"""An inference table's ``time`` is seconds, and its ``frame`` is a source frame.

No model runner reports a time. The bridge times each frame by the rate of the
file that the model read, as a single clip is timed, and the shared bridge then
retimes a table from several clips by each clip's own rate.

Under a frame window, a row's ``frame`` is the frame that the model read, not its
place among the frames read. The Ultralytics runner's own loop is tested for that
in ``test_ultralytics_wire_contract.py``, and the localizer's here.

The clips and the join are real. The model runners are the recording fakes from
``tests.helpers``. The localizer runs its own loop over mosaic's reader, with its
network and peak finder stood in for.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from mosaic_media import probe_media

from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.ops import run_op
from mosaic.core.pipeline.tracks_axis import BackfillReads, recorded_media_frames
from mosaic.core.pipeline.tracks_index import (
    frame_axis_verdict,
    media_drift,
    read_tracks_index,
)
from mosaic.core.scope import Scope
from mosaic.tracking.pose_training.localizer_inference import (
    LocalizerDetection,
    run_localizer_inference,
)
from tests.helpers import (
    index_media_sequence,
    install_fake_point_inference,
    install_fake_pose_inference,
    make_dataset,
    paint_frame_code,
    pose_per_frame,
    published_table,
    write_painted_entry,
    write_painted_recording,
)

pytestmark = pytest.mark.media

_SIZE = (64, 48)
_ENTRY = ("", "sess")
_GROUPED = ("rec", "s2")


def _dataset(tmp_path: Path, clips: list[tuple[int, float]]) -> Dataset:
    """One entry, ``sess``, of one clip per ``(frames, fps)`` in *clips*."""
    ds = make_dataset(tmp_path / "ds")
    _ = write_painted_entry(ds, "sess", clips, paint_frame_code, size=_SIZE)
    return ds


@pytest.fixture
def model(tmp_path: Path) -> Path:
    path = tmp_path / "weights" / "best.pt"
    path.parent.mkdir(parents=True)
    _ = path.write_bytes(b"weights")
    return path


def _times_by_frame(table: pd.DataFrame) -> dict[int, float]:
    frames: list[int] = [int(frame) for frame in table["frame"].tolist()]
    times: list[float] = [float(time) for time in table["time"].tolist()]
    return dict(zip(frames, times, strict=True))


def _point_per_frame(video: Path) -> pd.DataFrame:
    """One point per frame of *video*, in the point runner's layout."""
    frames = probe_media(video).frame_count
    return pd.DataFrame(
        {
            "frame": range(frames),
            "detection_id": [0] * frames,
            "x": [1.0] * frames,
            "y": [4.0] * frames,
            "confidence": [0.9] * frames,
            "class_id": [0] * frames,
            "class_name": ["bee"] * frames,
        }
    )


class TestTime:
    @pytest.mark.parametrize("kind", ["infer-pose", "infer-points"])
    def test_one_clip_is_timed_in_seconds_by_its_rate(
        self,
        tmp_path: Path,
        model: Path,
        monkeypatch: pytest.MonkeyPatch,
        kind: str,
        requires_ffmpeg: None,
    ) -> None:
        ds = _dataset(tmp_path, [(12, 25.0)])
        if kind == "infer-pose":
            _ = install_fake_pose_inference(monkeypatch, pose_per_frame)
        else:
            _ = install_fake_point_inference(monkeypatch, _point_per_frame)

        _ = run_op(ds, kind, {"model": str(model)}, scope=Scope(entries=[_ENTRY]))

        times = _times_by_frame(published_table(ds, kind))
        assert sorted(times) == list(range(12))
        for frame, time in times.items():
            assert time == pytest.approx(frame / 25.0), frame

    def test_a_join_of_two_rates_is_timed_by_each_clip_once(
        self,
        tmp_path: Path,
        model: Path,
        monkeypatch: pytest.MonkeyPatch,
        requires_ffmpeg: None,
    ) -> None:
        """The first clip's thirty frames take one second, and the rest are at 31.

        Thirty frames a clip, because over ten the two rates drift apart by less
        than the half frame that the reader tolerates.
        """
        ds = _dataset(tmp_path, [(30, 30.0), (30, 31.0)])
        _ = run_op(ds, "export-joined", {}, scope=Scope(entries=[_ENTRY]))
        _ = install_fake_pose_inference(monkeypatch, pose_per_frame)

        _ = run_op(
            ds, "infer-pose", {"model": str(model)}, scope=Scope(entries=[_ENTRY])
        )

        times = _times_by_frame(published_table(ds, "infer-pose"))
        assert sorted(times) == list(range(60))
        assert times[29] == pytest.approx(29 / 30.0)
        assert times[30] == pytest.approx(1.0)
        assert times[45] == pytest.approx(1.0 + 15 / 31.0)
        assert times[59] == pytest.approx(1.0 + 29 / 31.0)


def _install_localizer_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stand in for the localizer's network, which finds one location per frame."""
    import mosaic.tracking.pose_training.localizer_inference as localizer

    def load(_model_path: object, **_kwargs: object) -> object:
        return object()

    def detect(
        _encoder: object, _image: object, *_args: object, **_kwargs: object
    ) -> list[LocalizerDetection]:
        return [{"x": 1.0, "y": 4.0, "confidence": 0.9, "class_id": 0}]

    monkeypatch.setattr(localizer, "_load_encoder", load)
    monkeypatch.setattr(localizer, "detect_locations", detect)


class TestAFrameWindow:
    def test_the_localizer_publishes_the_frames_that_it_read(
        self,
        tmp_path: Path,
        model: Path,
        monkeypatch: pytest.MonkeyPatch,
        requires_ffmpeg: None,
    ) -> None:
        """``start_frame`` 5 and ``frame_step`` 2 publish frames 5, 7, 9 and so on."""
        ds = _dataset(tmp_path, [(20, 25.0)])
        _install_localizer_network(monkeypatch)

        _ = run_op(
            ds,
            "infer-localizer",
            {"model": str(model), "start_frame": 5, "frame_step": 2},
            scope=Scope(entries=[_ENTRY]),
        )

        times = _times_by_frame(published_table(ds, "infer-localizer"))
        read = list(range(5, 20, 2))
        assert list(times) == read
        for frame, time in times.items():
            assert time == pytest.approx(frame / 25.0), frame


def test_the_localizer_reads_one_video_given_as_a_bare_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, requires_ffmpeg: None
) -> None:
    """Not a sequence of the path's characters."""
    ds = _dataset(tmp_path, [(6, 30.0)])
    (clip,) = sorted((ds.get_root("media_raw") / "sess").glob("*.mp4"))
    _install_localizer_network(monkeypatch)

    for bare in (clip, str(clip)):
        found = run_localizer_inference("unused.pt", bare, save_images=False)
        assert [frame.frame for frame in found] == list(range(6)), bare


class TestAContinuousGroup:
    """A sequence of a continuous group is inferred on its media, at its place.

    Its rows are on the recording's frame axis, which starts the sequence at the
    frames of the members before it. A row made when the sequence's media contained
    the whole group reads as made from other media, and the next run replaces
    it with rows placed on the sequence's frames.
    """

    def test_the_rows_start_at_the_sequence_s_first_frame(
        self,
        tmp_path: Path,
        model: Path,
        monkeypatch: pytest.MonkeyPatch,
        requires_ffmpeg: None,
    ) -> None:
        ds = make_dataset(tmp_path / "ds")
        _ = write_painted_recording(
            ds, "rec", [("s1", [(5, 25.0)]), ("s2", [(4, 25.0)])], size=_SIZE
        )
        ds.set_continuous_groups(["rec"])
        _ = install_fake_pose_inference(monkeypatch, pose_per_frame)

        _ = run_op(
            ds, "infer-pose", {"model": str(model)}, scope=Scope(entries=[_GROUPED])
        )

        times = _times_by_frame(published_table(ds, "infer-pose"))
        assert sorted(times) == [5, 6, 7, 8]
        assert times[5] == pytest.approx(5 / 25.0)

    def test_rows_from_a_media_variant_start_at_the_sequence_s_first_frame(
        self,
        tmp_path: Path,
        model: Path,
        monkeypatch: pytest.MonkeyPatch,
        requires_ffmpeg: None,
    ) -> None:
        ds = make_dataset(tmp_path / "ds")
        _ = write_painted_recording(
            ds, "rec", [("s1", [(5, 25.0)]), ("s2", [(4, 25.0)])], size=_SIZE
        )
        ds.set_continuous_groups(["rec"])
        crop = {"step": "crop", "x": 8, "y": 4, "width": 32, "height": 24}
        scope = Scope(entries=[_GROUPED])
        variant = run_op(
            ds, "preprocess", {"steps": [crop], "codec": "h264"}, scope=scope
        )
        _ = install_fake_pose_inference(monkeypatch, pose_per_frame)

        _ = run_op(
            ds, "infer-pose", {"model": str(model), "media": variant}, scope=scope
        )

        times = _times_by_frame(published_table(ds, "infer-pose"))
        assert sorted(times) == [5, 6, 7, 8]
        assert times[5] == pytest.approx(5 / 25.0)

    def test_a_row_made_from_the_whole_group_reads_as_stale_and_is_replaced(
        self,
        tmp_path: Path,
        model: Path,
        monkeypatch: pytest.MonkeyPatch,
        requires_ffmpeg: None,
    ) -> None:
        ds = make_dataset(tmp_path / "ds")
        whole = write_painted_entry(
            ds, "s2", [(5, 25.0), (4, 25.0)], paint_frame_code, size=_SIZE, group="rec"
        )
        _ = run_op(ds, "export-joined", {}, scope=Scope(entries=[_GROUPED]))
        _ = install_fake_pose_inference(monkeypatch, pose_per_frame)
        scope = Scope(entries=[_GROUPED])
        _ = run_op(ds, "infer-pose", {"model": str(model)}, scope=scope)
        (made,) = [row for _, row in read_tracks_index(ds).iterrows()]
        assert int(made["media_frames"]) == 9

        # The first clip becomes the sequence s1, and s2 keeps the rest.
        first = ds.get_root("media_raw") / "rec" / "s1" / "clip0.mp4"
        first.parent.mkdir(parents=True)
        _ = whole[0].rename(first)
        index_media_sequence(ds, "s2", ["clip1.mp4"], group="rec")
        index_media_sequence(ds, "s1", ["clip0.mp4"], group="rec")
        ds.set_continuous_groups(["rec"])

        rows = read_tracks_index(ds)
        assert media_drift(ds, rows) == (_GROUPED,)
        reads = BackfillReads(ds, [_GROUPED])
        recorded = recorded_media_frames(
            ds,
            producer="infer-pose",
            variant=str(made["run_id"]),
            group="rec",
            sequence="s2",
            consumed_media=str(made["consumed_media_composition"]),
            reads=reads,
        )
        assert recorded == "not-established"
        media = ds.resolve_media(*_GROUPED).frame_count
        assert media == 4
        verdict = frame_axis_verdict(
            "infer-pose",
            read=int(made["frames_read"]),
            media=media,
            known_tail_loss=None,
        )
        assert verdict == "mismatch"

        _ = run_op(ds, "infer-pose", {"model": str(model)}, scope=scope)

        (replaced,) = [row for _, row in read_tracks_index(ds).iterrows()]
        assert int(replaced["media_frames"]) == 4
        assert media_drift(ds, read_tracks_index(ds)) == ()
        assert sorted(_times_by_frame(published_table(ds, "infer-pose"))) == [
            5,
            6,
            7,
            8,
        ]
