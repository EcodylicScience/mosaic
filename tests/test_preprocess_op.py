"""The ``preprocess`` op: one entry's media in, one encoded variant file out.

Each entry's clips are read one at a time, or the upstream variant's file when
``media`` names one, the steps are applied to every selected frame, and the
frames are encoded to a partial file that is counted before it is published. The
row beside it records where the file sits in its entry's source, so a consumer
never probes it again. Every refusal a recipe earns is raised before any entry
is decoded.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from mosaic_media import probe_media
from mosaic_media.hwaccel import encoder_available
from mosaic_media.transcode import TranscodeError

from mosaic.core.dataset import Dataset
from mosaic.core.media.preprocess import FrameMap, Placement
from mosaic.core.media.video_io import open_frame_reader
from mosaic.core.pipeline import preprocess
from mosaic.core.pipeline.graph.lanes import TRANSCODE_LANE, lane_for_step
from mosaic.core.pipeline.job import CancelToken, Cancelled
from mosaic.core.pipeline.markers import new_inflight, read_inflight, write_inflight
from mosaic.core.pipeline.ops import run_op
from mosaic.core.pipeline.preprocess import (
    H264PipeWriter,
    PreprocessOp,
    PreprocessParams,
    PreprocessRefused,
    VariantWriter,
    preprocess_identity,
)
from mosaic.core.pipeline.preprocess_index import (
    media_variant_row,
    variant_facts,
    variant_placement,
    variant_row,
    write_media_variant_row,
)
from mosaic.core.pipeline.preprocess_layout import (
    media_variant_path,
    media_variant_run_root,
    media_variant_work_root,
    media_variants_root,
)
from mosaic.core.pipeline.run import AllEntriesFailed
from mosaic.core.pipeline.tracks_index import media_composition_for
from mosaic.core.scope import Scope
from mosaic.runlog import reduce_run_log, run_log_path

from tests.helpers import (
    MediaClip,
    index_media_sequence,
    make_dataset,
    write_h264_mp4,
    write_media_index,
)

_KIND = "preprocess"
_SIZE = (64, 48)

type Image = npt.NDArray[np.uint8]
type Step = dict[str, object]


def _crop(x: int, y: int, width: int, height: int) -> Step:
    return {"step": "crop", "x": x, "y": y, "width": width, "height": height}


def _trim(start: int, stop: int) -> Step:
    return {"step": "trim", "start": start, "stop": stop}


def _decimate(every: int) -> Step:
    return {"step": "decimate", "every": every}


_GRAYSCALE: Step = {"step": "grayscale"}


_LEVEL_TOLERANCE = 5
"""How far a decoded flat frame may sit from the level it was painted with.

Two encodes, the source's and the variant's, move a flat frame by up to about 4
levels. Painted levels are 12 apart, so a frame within this tolerance of its
level is that frame and not a neighbor.
"""


def _level(frame: int, base: int) -> int:
    """The gray level source frame *frame* of an entry is painted with.

    Eighteen levels 12 apart, repeating every 18 frames, so any two frames closer
    together than that are told apart after decoding.
    """
    return 16 + (base + 12 * frame) % 216


def _entry(
    ds: Dataset,
    sequence: str,
    clips: Sequence[tuple[int, float]],
    *,
    base: int = 0,
) -> None:
    """Write and index *sequence*'s clips, each ``(frames, fps)``, in order.

    Every frame is flat at :func:`_level` of its frame number across the whole
    entry, so the clips continue one another.
    """
    directory = ds.get_root("media_raw") / sequence
    names: list[str] = []
    offset = 0
    width, height = _SIZE
    for position, (frames, fps) in enumerate(clips):
        name = f"clip{position}.mp4"

        def paint(frame: int, first: int = offset) -> Image:
            return np.full((height, width, 3), _level(first + frame, base), np.uint8)

        write_h264_mp4(
            directory / name, frames=frames, fps=fps, size=_SIZE, paint=paint
        )
        names.append(name)
        offset += frames
    index_media_sequence(ds, sequence, names)


def _params(
    steps: list[Step], *, media: str = "", fps: float | None = None
) -> PreprocessParams:
    return PreprocessParams.model_validate({"steps": steps, "media": media, "fps": fps})


def _run(
    ds: Dataset,
    steps: list[Step],
    sequences: Sequence[str] = ("s",),
    *,
    media: str = "",
    fps: float | None = None,
    codec: str = "av1",
    overwrite: bool = False,
    execution_id: str | None = None,
) -> str:
    return run_op(
        ds,
        _KIND,
        {"steps": steps, "media": media, "fps": fps, "codec": codec},
        scope=Scope(entries=[("", sequence) for sequence in sequences]),
        overwrite=overwrite,
        execution_id=execution_id,
    )


def _row(ds: Dataset, run_id: str, sequence: str = "s") -> dict[str, str]:
    row = variant_row(ds, run_id, "", sequence, "")
    assert row is not None, f"no row for {sequence} under {run_id}"
    return row


def _means(path: Path) -> list[float]:
    """The mean level of each frame of *path*, decoded for analysis."""
    with open_frame_reader(path, target="analysis") as reader:
        return [float(np.mean(frame)) for _, frame in reader]


def _error_lines(ds: Dataset, execution_id: str) -> list[str]:
    log = run_log_path(ds.base_dir, execution_id).read_text()
    return [line for line in log.splitlines() if '"entry_error"' in line]


def _entries_written(ds: Dataset, execution_id: str) -> int:
    snapshot = reduce_run_log(run_log_path(ds.base_dir, execution_id))
    assert snapshot is not None
    return snapshot["entries_written"]


def _variant_files(ds: Dataset) -> list[Path]:
    root = media_variants_root(ds)
    return sorted(root.rglob("*.mp4")) if root.exists() else []


class _WriterSpy:
    """Counts the variant writers the op opens, and may wrap each one."""

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        wrap: Callable[[VariantWriter], VariantWriter] | None = None,
        params_for: Callable[[PreprocessParams], PreprocessParams] | None = None,
    ) -> None:
        self.opened: list[Path] = []
        real = preprocess.open_variant_writer

        def spy(
            params: PreprocessParams, path: Path, width: int, height: int, fps: float
        ) -> VariantWriter:
            self.opened.append(path)
            chosen = params if params_for is None else params_for(params)
            writer = real(chosen, path, width, height, fps)
            return writer if wrap is None else wrap(writer)

        monkeypatch.setattr(preprocess, "open_variant_writer", spy)


class _DroppingWriter:
    """Writes every frame but the first, while claiming to have written them all."""

    def __init__(self, inner: VariantWriter) -> None:
        self._inner = inner
        self._seen = 0

    def write(self, frame: Image) -> None:
        self._seen += 1
        if self._seen > 1:
            self._inner.write(frame)

    def close(self) -> None:
        self._inner.close()

    @property
    def frames_written(self) -> int:
        return self._seen

    @property
    def encoder_name(self) -> str:
        return self._inner.encoder_name


# --- one entry's media ------------------------------------------------------


@pytest.mark.media
def test_a_single_clip_crop_is_the_cropped_size_count_and_codec(
    tmp_path: Path,
) -> None:
    ds = make_dataset(tmp_path / "ds")
    _entry(ds, "s", [(20, 30.0)])

    run_id = _run(ds, [_crop(8, 8, 32, 24)])

    path = media_variant_path(ds, run_id, "", "s", "")
    facts = probe_media(path)
    assert (facts.width, facts.height, facts.codec_name) == (32, 24, "av1")
    assert facts.frame_count == 20
    row = _row(ds, run_id)
    placement = variant_placement(row)
    assert placement.fps == pytest.approx(30.0)
    assert placement == Placement(
        offset_x=8,
        offset_y=8,
        width=32,
        height=24,
        source_width=64,
        source_height=48,
        source_frame_count=20,
        frames=FrameMap(0, 1, 20),
        fps=placement.fps,
    )
    assert int(row["frame_count"]) == variant_placement(row).frames.count
    assert row["encoder"] == "libsvtav1"
    assert row["upstream"] == ""
    assert row["upstream_video_uuid"] == ""
    assert row["video_uuid"] == facts.video_uuid
    assert row["consumed_media_composition"] == media_composition_for(ds, "", "s")
    assert row["consumed_media_composition"] != ""
    assert not path.with_name("s.partial.mp4").exists()


@pytest.mark.media
def test_trim_and_decimate_across_clips_keep_the_mapped_source_frames(
    tmp_path: Path,
) -> None:
    """Frame ``i`` of the variant is source frame ``7 + 3 * i``, across both clips."""
    ds = make_dataset(tmp_path / "ds")
    _entry(ds, "s", [(30, 30.0), (30, 30.0)])

    run_id = _run(ds, [_trim(7, 50), _decimate(3)])

    kept = list(range(7, 50, 3))
    means = _means(media_variant_path(ds, run_id, "", "s", ""))
    assert len(means) == len(kept) == 15
    for position, (mean, source) in enumerate(zip(means, kept, strict=True)):
        assert mean == pytest.approx(_level(source, 0), abs=_LEVEL_TOLERANCE), position
    row = _row(ds, run_id)
    assert variant_placement(row).frames == FrameMap(7, 3, 15)
    assert variant_placement(row).fps == pytest.approx(10.0)


@pytest.mark.media
@pytest.mark.parametrize(("fps", "labeled"), [(None, 30.0), (25.0, 25.0)])
def test_a_mixed_rate_entry_is_labeled_at_the_first_clips_rate_or_at_fps(
    tmp_path: Path, fps: float | None, labeled: float
) -> None:
    """Clips at 30 and 31 fps, read one at a time, on one uniform grid.

    Three hundred frames each, because rate uniformity is judged on the drift
    accumulated over a clip: 30 beside 31 fps reads as uniform over ten frames.
    """
    ds = make_dataset(tmp_path / "ds")
    _entry(ds, "s", [(300, 30.0), (300, 31.0)])
    run_id = _run(ds, [_trim(290, 320)], fps=fps)

    path = media_variant_path(ds, run_id, "", "s", "")
    facts = probe_media(path)
    assert facts.fps == pytest.approx(labeled)
    assert facts.frame_count == 30
    assert variant_placement(_row(ds, run_id)).fps == pytest.approx(labeled)
    means = _means(path)
    assert means[0] == pytest.approx(_level(290, 0), abs=_LEVEL_TOLERANCE)
    assert means[10] == pytest.approx(_level(300, 0), abs=_LEVEL_TOLERANCE)


@pytest.mark.media
def test_an_imgstore_entry_is_read_without_an_export(
    tmp_path: Path, make_imgstore: Callable[..., tuple[Path, list[Image]]]
) -> None:
    ds = make_dataset(tmp_path / "ds", roots=["media", "tracks"])
    search = tmp_path / "raw"
    _ = make_imgstore(name="rec", nframes=12, parent=search, fill=True)
    ds.index_media([search])
    (entry,) = ds.resolve_media_scope(None)

    run_id = run_op(
        ds,
        _KIND,
        {"steps": [_crop(16, 12, 32, 24)]},
        scope=Scope(entries=[(entry.group, entry.sequence)]),
    )

    path = media_variant_path(ds, run_id, entry.group, entry.sequence, entry.camera)
    facts = probe_media(path)
    assert (facts.width, facts.height, facts.frame_count) == (32, 24, 12)


# --- a variant read by another ----------------------------------------------


_UPSTREAM_STEPS: list[Step] = [_crop(8, 8, 48, 32), _trim(2, 18)]


@pytest.mark.media
def test_a_chained_variant_composes_its_placement(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds")
    _entry(ds, "s", [(20, 30.0)])
    upstream = _run(ds, _UPSTREAM_STEPS)

    run_id = _run(ds, [_crop(16, 12, 24, 16), _decimate(2)], media=upstream)

    row = _row(ds, run_id)
    placement = variant_placement(row)
    assert placement.fps == pytest.approx(15.0)
    assert placement == Placement(
        offset_x=16,
        offset_y=12,
        width=24,
        height=16,
        source_width=64,
        source_height=48,
        source_frame_count=20,
        frames=FrameMap(2, 2, 8),
        fps=placement.fps,
    )
    assert row["upstream"] == upstream
    assert row["upstream_video_uuid"] == _row(ds, upstream)["video_uuid"]
    # Source frames 2, 4, ..., 16 are the upstream file's frames 0, 2, ..., 14.
    # Compared with that file rather than the painted levels, one encode apart.
    means = _means(media_variant_path(ds, run_id, "", "s", ""))
    upstream_means = _means(media_variant_path(ds, upstream, "", "s", ""))
    assert means == pytest.approx(upstream_means[0:16:2], abs=_LEVEL_TOLERANCE)


@pytest.mark.media
def test_a_chained_variant_is_encoded_again_when_its_upstream_is_rewritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = make_dataset(tmp_path / "ds")
    _entry(ds, "s", [(20, 30.0)])
    upstream = _run(ds, _UPSTREAM_STEPS)
    downstream: list[Step] = [_crop(16, 12, 24, 16)]
    run_id = _run(ds, downstream, media=upstream)
    before = _row(ds, upstream)["video_uuid"]

    # The same recipe written again with other bytes, as another machine's
    # encoder would write it.
    with monkeypatch.context() as patched:
        _ = _WriterSpy(
            patched, params_for=lambda p: p.model_copy(update={"quality": 40})
        )
        _ = _run(ds, _UPSTREAM_STEPS, overwrite=True)
    after = _row(ds, upstream)["video_uuid"]
    assert after != before
    spy = _WriterSpy(monkeypatch)

    _ = _run(ds, downstream, media=upstream)

    assert len(spy.opened) == 1
    assert _row(ds, run_id)["upstream_video_uuid"] == after


@pytest.mark.media
def test_a_chained_variant_over_a_drifted_upstream_fails_the_entry(
    tmp_path: Path,
) -> None:
    ds = make_dataset(tmp_path / "ds")
    _entry(ds, "s", [(20, 30.0)])
    upstream = _run(ds, _UPSTREAM_STEPS)
    # The entry's media changes after the upstream variant was written from it.
    _entry(ds, "s", [(20, 30.0)], base=100)
    assert (
        media_composition_for(ds, "", "s")
        != _row(ds, upstream)["consumed_media_composition"]
    )

    with pytest.raises(AllEntriesFailed):
        _ = _run(ds, [_crop(16, 12, 24, 16)], media=upstream, execution_id="drift")

    (line,) = _error_lines(ds, "drift")
    assert "MediaVariantDriftedError" in line
    assert upstream in line


@pytest.mark.media
@pytest.mark.parametrize(
    ("steps", "named"),
    [
        ([_crop(0, 0, 24, 16)], "not inside the current image"),
        ([_trim(0, 10)], "outside the frame map"),
    ],
    ids=["a crop outside the upstream image", "a trim before the upstream frames"],
)
def test_a_chained_recipe_the_upstream_cannot_hold_is_refused_for_the_run(
    tmp_path: Path, steps: list[Step], named: str
) -> None:
    """Refused before any entry is encoded, not recorded as a failed entry.

    ``plan_identity`` does not check a chained recipe, whose upstream may not be
    written yet when a graph plans, so this refusal is raised by ``run`` alone.
    """
    ds = make_dataset(tmp_path / "ds")
    _entry(ds, "s", [(20, 30.0)])
    _entry(ds, "t", [(20, 30.0)], base=50)
    upstream = _run(ds, _UPSTREAM_STEPS, ("s", "t"))

    with pytest.raises(PreprocessRefused, match=named) as refused:
        _ = _run(ds, steps, ("s", "t"), media=upstream)

    assert "preprocess cannot make a variant of s" in str(refused.value)
    downstream = preprocess_identity(_params(steps, media=upstream)).run_id
    assert not media_variant_run_root(ds, downstream).exists()


def test_a_chained_variant_without_its_upstream_row_fails_the_entry(
    three_entry_dataset: Dataset,
) -> None:
    upstream = "preprocess.0.1-0123456789"

    with pytest.raises(AllEntriesFailed):
        _ = run_op(
            three_entry_dataset,
            _KIND,
            {"steps": [_crop(0, 0, 32, 24)], "media": upstream},
            scope=Scope(entries=[("A", "one")]),
            execution_id="missing",
        )

    (line,) = _error_lines(three_entry_dataset, "missing")
    assert "MediaVariantMissingError" in line
    assert upstream in line


# --- reuse ------------------------------------------------------------------


@pytest.mark.media
def test_a_second_run_reuses_the_variant_and_reports_it_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = make_dataset(tmp_path / "ds")
    _entry(ds, "s", [(12, 30.0)])
    spy = _WriterSpy(monkeypatch)

    run_id = _run(ds, [_GRAYSCALE], execution_id="first")
    again = _run(ds, [_GRAYSCALE], execution_id="second")

    assert again == run_id
    assert len(spy.opened) == 1
    assert _entries_written(ds, "first") == 1
    assert _entries_written(ds, "second") == 1


@pytest.mark.media
def test_a_changed_composition_encodes_the_entry_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = make_dataset(tmp_path / "ds")
    _entry(ds, "s", [(12, 30.0)])
    run_id = _run(ds, [_GRAYSCALE])
    recorded = _row(ds, run_id)["consumed_media_composition"]
    _entry(ds, "s", [(12, 30.0)], base=100)
    spy = _WriterSpy(monkeypatch)

    _ = _run(ds, [_GRAYSCALE])

    assert len(spy.opened) == 1
    current = media_composition_for(ds, "", "s")
    assert current != recorded
    assert _row(ds, run_id)["consumed_media_composition"] == current


@pytest.mark.media
def test_a_blank_recorded_composition_is_not_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A blank cell means unknown, and unknown is not a change."""
    ds = make_dataset(tmp_path / "ds")
    _entry(ds, "s", [(12, 30.0)])
    run_id = _run(ds, [_GRAYSCALE])
    row = _row(ds, run_id)
    write_media_variant_row(
        ds,
        media_variant_row(
            ds,
            path=media_variant_path(ds, run_id, "", "s", ""),
            run_id=run_id,
            group="",
            sequence="s",
            camera="",
            upstream="",
            upstream_video_uuid="",
            placement=variant_placement(row),
            facts=variant_facts(row),
            encoder=row["encoder"],
            consumed_media_composition="",
        ),
    )
    spy = _WriterSpy(monkeypatch)

    _ = _run(ds, [_GRAYSCALE])

    assert spy.opened == []


@pytest.mark.media
def test_overwrite_encodes_a_current_variant_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = make_dataset(tmp_path / "ds")
    _entry(ds, "s", [(12, 30.0)])
    _ = _run(ds, [_GRAYSCALE])
    spy = _WriterSpy(monkeypatch)

    _ = _run(ds, [_GRAYSCALE], overwrite=True)

    assert len(spy.opened) == 1


# --- publishing -------------------------------------------------------------


@pytest.mark.media
def test_a_short_encode_is_refused_and_its_partial_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = make_dataset(tmp_path / "ds")
    _entry(ds, "s", [(12, 30.0)])
    _ = _WriterSpy(monkeypatch, wrap=_DroppingWriter)

    with pytest.raises(AllEntriesFailed):
        _ = _run(ds, [_GRAYSCALE], execution_id="short")

    run_id = preprocess_identity(_params([_GRAYSCALE])).run_id
    path = media_variant_path(ds, run_id, "", "s", "")
    assert not path.exists()
    assert path.with_name("s.partial.mp4").is_file()
    assert variant_row(ds, run_id, "", "s", "") is None
    (line,) = _error_lines(ds, "short")
    assert "11" in line
    assert "12" in line


@pytest.mark.media
def test_one_failing_entry_does_not_stop_the_others(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds")
    _entry(ds, "broken", [(12, 30.0)])
    _entry(ds, "fine", [(12, 30.0)], base=50)
    (ds.get_root("media_raw") / "broken" / "clip0.mp4").unlink()

    run_id = _run(ds, [_GRAYSCALE], ("broken", "fine"), execution_id="mixed")

    assert media_variant_path(ds, run_id, "", "fine", "").is_file()
    assert not media_variant_path(ds, run_id, "", "broken", "").exists()
    (line,) = _error_lines(ds, "mixed")
    assert '"broken"' in line
    assert _entries_written(ds, "mixed") == 1


class _CancellingWriter:
    """Asks the run to stop once the first frame is written."""

    def __init__(self, inner: VariantWriter, token: CancelToken) -> None:
        self._inner = inner
        self._token = token

    def write(self, frame: Image) -> None:
        self._inner.write(frame)
        self._token.cancel()

    def close(self) -> None:
        self._inner.close()

    @property
    def frames_written(self) -> int:
        return self._inner.frames_written

    @property
    def encoder_name(self) -> str:
        return self._inner.encoder_name


@pytest.mark.media
def test_a_cancel_during_an_encode_stops_the_run_and_leaves_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The encode heartbeats and checks for a cancel between frames.

    The partial is removed, no row is written, the entry's claim is released and
    the next entry is not started.
    """
    ds = make_dataset(tmp_path / "ds")
    _entry(ds, "s", [(12, 30.0)])
    _entry(ds, "t", [(12, 30.0)], base=50)
    token = CancelToken()
    spy = _WriterSpy(monkeypatch, wrap=lambda inner: _CancellingWriter(inner, token))

    with pytest.raises(Cancelled):
        _ = run_op(
            ds,
            _KIND,
            {"steps": [_GRAYSCALE]},
            scope=Scope(entries=[("", "s"), ("", "t")]),
            cancel_token=token,
            execution_id="cancelled",
        )

    run_id = preprocess_identity(_params([_GRAYSCALE])).run_id
    assert len(spy.opened) == 1
    assert _variant_files(ds) == []
    assert variant_row(ds, run_id, "", "s", "") is None
    assert read_inflight(media_variant_work_root(ds, run_id) / "s") is None
    events = [
        json.loads(line)["ev"]
        for line in run_log_path(ds.base_dir, "cancelled").read_text().splitlines()
    ]
    assert events[-1] == "cancelled"
    assert "heartbeat" in events


def test_an_entry_held_by_another_execution_is_skipped(
    three_entry_dataset: Dataset, capsys: pytest.CaptureFixture[str]
) -> None:
    params = _params([_crop(0, 0, 32, 24)])
    run_id = preprocess_identity(params).run_id
    held = media_variant_work_root(three_entry_dataset, run_id) / "A__one"
    held.mkdir(parents=True)
    write_inflight(
        held,
        new_inflight(
            execution_id="someone-else",
            host="other-host",
            pid=1,
            phase=None,
            idle_seconds=3600.0,
        ),
    )

    assert (
        run_op(
            three_entry_dataset,
            _KIND,
            params,
            scope=Scope(entries=[("A", "one")]),
        )
        == run_id
    )

    assert "held by another execution" in capsys.readouterr().err
    assert _variant_files(three_entry_dataset) == []


# --- refusals ---------------------------------------------------------------


_REFUSALS: dict[str, tuple[MediaClip, list[Step], str]] = {
    "a crop outside the frame": (
        MediaClip(filename="b.mp4", group="B", sequence="one", width=320, height=240),
        [_crop(400, 0, 100, 100)],
        "not inside the current image",
    ),
    "an odd frame size without a crop": (
        MediaClip(filename="b.mp4", group="B", sequence="one", width=641),
        [_GRAYSCALE],
        "641x480",
    ),
    "an empty frame map": (
        MediaClip(filename="b.mp4", group="B", sequence="one", frame_count=0),
        [_GRAYSCALE],
        "no frame",
    ),
}


@pytest.mark.parametrize("case", sorted(_REFUSALS))
def test_a_refused_recipe_is_refused_before_any_entry_is_written(
    tmp_path: Path, case: str
) -> None:
    """The first entry is valid and sorts first; nothing of it is written either."""
    clip, steps, named = _REFUSALS[case]
    ds = make_dataset(tmp_path / "ds")
    write_media_index(
        ds, [MediaClip(filename="a.mp4", group="A", sequence="one"), clip]
    )

    with pytest.raises(ValueError, match=named) as refused:
        _ = run_op(
            ds,
            _KIND,
            {"steps": steps},
            scope=Scope(entries=[("A", "one"), ("B", "one")]),
        )

    assert "B__one" in str(refused.value)
    run_id = preprocess_identity(_params(steps)).run_id
    assert not media_variant_run_root(ds, run_id).exists()


def test_the_odd_size_refusal_suggests_a_crop(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds")
    write_media_index(ds, [MediaClip(filename="a.mp4", sequence="s", height=479)])

    with pytest.raises(ValueError, match="crop"):
        _ = _run(ds, [_GRAYSCALE])


def test_plan_identity_refuses_a_crop_outside_the_frame(
    three_entry_dataset: Dataset,
) -> None:
    params = _params([_crop(600, 0, 100, 100)])
    scope = three_entry_dataset.resolve_scope(Scope(entries=[("A", "one")]))

    with pytest.raises(ValueError, match="not inside the current image"):
        _ = PreprocessOp().plan_identity(three_entry_dataset, params, scope)


def test_plan_identity_names_a_chained_variant_before_its_upstream_exists(
    three_entry_dataset: Dataset,
) -> None:
    params = _params([_crop(600, 0, 100, 100)], media="preprocess.0.1-0123456789")
    scope = three_entry_dataset.resolve_scope(Scope(entries=[("A", "one")]))

    identity = PreprocessOp().plan_identity(three_entry_dataset, params, scope)

    assert identity == preprocess_identity(params)


def test_plan_identity_passes_over_an_entry_awaiting_a_transcode(
    tmp_path: Path,
) -> None:
    """A rotated original needs an analysis transcode, which a graph may run first."""
    ds = make_dataset(tmp_path / "ds")
    write_media_index(ds, [MediaClip(filename="a.mp4", sequence="s", rotation=90)])
    params = _params([_crop(600, 0, 100, 100)])
    scope = ds.resolve_scope(Scope(entries=[("", "s")]))

    identity = PreprocessOp().plan_identity(ds, params, scope)

    assert identity == preprocess_identity(params)


# --- H.264 ------------------------------------------------------------------


@pytest.mark.media
def test_an_h264_variant_is_written_by_libx264(tmp_path: Path) -> None:
    if not encoder_available("libx264"):
        pytest.skip("the ffmpeg on PATH has no libx264")
    ds = make_dataset(tmp_path / "ds")
    _entry(ds, "s", [(12, 30.0)])

    run_id = _run(ds, [_crop(8, 8, 32, 24)], codec="h264")

    facts = probe_media(media_variant_path(ds, run_id, "", "s", ""))
    assert (facts.codec_name, facts.pixel_format) == ("h264", "yuv420p")
    assert (facts.width, facts.height, facts.frame_count) == (32, 24, 12)
    assert _row(ds, run_id)["encoder"] == "libx264"


def test_an_h264_variant_is_refused_by_name_without_libx264(
    three_entry_dataset: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    def only_av1(name: str) -> bool:
        return name == "libsvtav1"

    monkeypatch.setattr(preprocess, "encoder_available", only_av1)

    with pytest.raises(TranscodeError, match="missing or built without it"):
        _ = run_op(
            three_entry_dataset,
            _KIND,
            {"steps": [_crop(0, 0, 32, 24)], "codec": "h264"},
            scope=Scope(entries=[("A", "one")]),
        )

    assert _variant_files(three_entry_dataset) == []


@pytest.mark.media
def test_the_h264_writer_reports_what_ffmpeg_said(tmp_path: Path) -> None:
    path = tmp_path / "absent" / "variant.mp4"
    frame = np.zeros((48, 64, 3), np.uint8)

    with pytest.raises(TranscodeError, match="No such file or directory"):
        writer = H264PipeWriter(path, 64, 48, 30.0, crf=16)
        try:
            writer.write(frame)
        finally:
            writer.close()


@pytest.mark.media
def test_the_h264_writer_refuses_a_frame_of_another_size(tmp_path: Path) -> None:
    writer = H264PipeWriter(tmp_path / "variant.mp4", 64, 48, 30.0, crf=16)
    try:
        with pytest.raises(ValueError, match="does not fit"):
            writer.write(np.zeros((48, 32, 3), np.uint8))
    finally:
        writer.close()


# --- declaration ------------------------------------------------------------


def test_the_op_runs_in_the_transcode_lane() -> None:
    assert lane_for_step(_KIND) == TRANSCODE_LANE


def test_the_run_target_names_one_entry_or_counts_several(
    three_entry_dataset: Dataset,
) -> None:
    params = _params([_GRAYSCALE])
    one = three_entry_dataset.resolve_scope(Scope(entries=[("A", "one")]))
    three = three_entry_dataset.resolve_scope(Scope(groups=["A", "B"]))

    assert PreprocessOp().target(params, one) == "A/one"
    assert PreprocessOp().target(params, three) == "3 entries"


def test_the_identity_is_the_parameters_identity(
    three_entry_dataset: Dataset,
) -> None:
    params = _params([_crop(0, 0, 32, 24)], fps=12.5)
    scope = three_entry_dataset.resolve_scope(Scope(groups=["A"]))

    identity = PreprocessOp().plan_identity(three_entry_dataset, params, scope)

    assert identity == preprocess_identity(params)
    assert identity.tracks_variant == ""
