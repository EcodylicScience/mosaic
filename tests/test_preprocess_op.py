"""Test the ``preprocess`` op, which turns one entry's media into one variant file.

Each entry's clips are read one at a time, or the upstream variant's file when
``media`` names one, the steps are applied to every selected frame, and the
frames are encoded to a partial file that is counted before it is published. The
row beside it records the file's placement in its entry's source, and a consumer
never probes the file again. Every refusal of a recipe is raised before
any entry is decoded.
"""

from __future__ import annotations

import json
import shlex
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from mosaic_media import probe_media
from mosaic_media.hwaccel import encoder_available
from mosaic_media.transcode import TranscodeError
from typer.testing import CliRunner

from mosaic.cli import app
from mosaic.core.dataset import Dataset
from mosaic.core.media.preprocess import FrameMap, Placement
from mosaic.core.media.video_io import open_frame_reader
from mosaic.core.pipeline import preprocess
from mosaic.core.pipeline.graph.lanes import TRANSCODE_LANE, lane_for_step
from mosaic.core.pipeline.identity_scheme import read_identity_scheme
from mosaic.core.pipeline.job import CancelToken, Cancelled
from mosaic.core.pipeline.markers import new_inflight, read_inflight, write_inflight
from mosaic.core.pipeline.op_identity import OP_IDENTITY_SCHEME
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
    MediaVariantDriftedError,
    MediaVariantMissingError,
    build_media_variant_row,
    media_variant_facts,
    media_variant_placement,
    media_variant_rows,
    write_media_variant_row,
)
from mosaic.core.pipeline.preprocess_layout import (
    media_variant_path,
    media_variant_recipe_path,
    media_variant_run_root,
    media_variant_work_root,
    media_variants_root,
)
from mosaic.core.pipeline.run import AllEntriesFailed
from mosaic.core.pipeline.tracks_index import media_composition_for
from mosaic.core.pipeline.variant_source import VariantLookup
from mosaic.core.scope import Scope
from mosaic.runlog import reduce_run_log, run_log_path

from tests.helpers import (
    IndexReads,
    MediaClip,
    count_index_reads,
    entry_error_lines,
    index_media_sequence,
    make_dataset,
    write_h264_mp4,
    write_media_index,
    write_painted_entry,
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
"""The largest distance of a decoded flat frame from the level it was painted with.

Two encodes, the source's and the variant's, move a flat frame by up to about 4
levels. Painted levels are 12 apart. A frame within this tolerance of its level
is that frame and not a neighbor.
"""


def _level(frame: int, base: int) -> int:
    """Return the gray level that source frame *frame* of an entry is painted with.

    There are eighteen levels 12 apart, repeating every 18 frames. Any two frames
    closer together than that are told apart after decoding.
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
    entry. The clips continue one another.
    """
    width, height = _SIZE

    def paint(frame: int) -> Image:
        return np.full((height, width, 3), _level(frame, base), np.uint8)

    _ = write_painted_entry(ds, sequence, clips, paint, size=_SIZE)


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
    row = media_variant_rows(ds, run_id).get(("", sequence, ""))
    assert row is not None, f"no row for {sequence} under {run_id}"
    return row


def _forget_composition(ds: Dataset, run_id: str, sequence: str = "s") -> None:
    """Record *sequence*'s variant row again with a blank media composition."""
    row = _row(ds, run_id, sequence)
    write_media_variant_row(
        ds,
        build_media_variant_row(
            ds,
            path=media_variant_path(ds, run_id, "", sequence, ""),
            run_id=run_id,
            group="",
            sequence=sequence,
            camera="",
            upstream="",
            upstream_video_uuid="",
            placement=media_variant_placement(row),
            facts=media_variant_facts(row),
            encoder=row["encoder"],
            consumed_media_composition="",
        ),
    )


def _means(path: Path) -> list[float]:
    """Return the mean level of each frame of *path*, decoded for analysis."""
    with open_frame_reader(path, target="analysis") as reader:
        return [float(np.mean(frame)) for _, frame in reader]


def _entries_written(ds: Dataset, execution_id: str) -> int:
    snapshot = reduce_run_log(run_log_path(ds.base_dir, execution_id))
    assert snapshot is not None
    return snapshot["entries_written"]


def _variant_files(ds: Dataset) -> list[Path]:
    """Return every ``.mp4`` under the variants root, work directories included."""
    root = media_variants_root(ds)
    return sorted(root.rglob("*.mp4")) if root.exists() else []


def _partial(ds: Dataset, run_id: str, sequence: str) -> Path:
    """Return the path where the op encodes *sequence*'s variant before publishing."""
    return media_variant_work_root(ds, run_id) / sequence / f"{sequence}.partial.mp4"


class _WriterSpy:
    """Counts the variant writers that the op opens, and may wrap each one."""

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
    """Writes every frame but the first, and reports all of them as written."""

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
    placement = media_variant_placement(row)
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
    assert int(row["frame_count"]) == media_variant_placement(row).frames.count
    assert row["encoder"] == "libsvtav1"
    assert row["upstream"] == ""
    assert row["upstream_video_uuid"] == ""
    assert row["video_uuid"] == facts.video_uuid
    assert row["consumed_media_composition"] == media_composition_for(ds, "", "s")
    assert row["consumed_media_composition"] != ""
    assert _variant_files(ds) == [path]


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
    assert media_variant_placement(row).frames == FrameMap(7, 3, 15)
    assert media_variant_placement(row).fps == pytest.approx(10.0)


@pytest.mark.media
@pytest.mark.parametrize(("fps", "labeled"), [(None, 30.0), (25.0, 25.0)])
def test_a_mixed_rate_entry_is_labeled_at_the_first_clips_rate_or_at_fps(
    tmp_path: Path, fps: float | None, labeled: float
) -> None:
    """Clips at 30 and 31 fps are read one at a time onto one uniform grid.

    Each clip has three hundred frames, because rate uniformity is judged on the
    drift accumulated over a clip. Over ten frames, 30 beside 31 fps reads as
    uniform.
    """
    ds = make_dataset(tmp_path / "ds")
    _entry(ds, "s", [(300, 30.0), (300, 31.0)])
    run_id = _run(ds, [_trim(290, 320)], fps=fps)

    path = media_variant_path(ds, run_id, "", "s", "")
    facts = probe_media(path)
    assert facts.fps == pytest.approx(labeled)
    assert facts.frame_count == 30
    assert media_variant_placement(_row(ds, run_id)).fps == pytest.approx(labeled)
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
    placement = media_variant_placement(row)
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
    # The test compares with that file rather than the painted levels, which are
    # one encode further away.
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

    # The same recipe is written again with other bytes, as another machine's
    # encoder writes it.
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

    (line,) = entry_error_lines(ds, "drift")
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
    """The recipe is refused before any entry is encoded, not as a failed entry.

    ``plan_identity`` does not check a chained recipe, whose upstream may not be
    written yet when a graph plans. This refusal is raised by ``run`` alone.
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

    (line,) = entry_error_lines(three_entry_dataset, "missing")
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
    _forget_composition(ds, run_id)
    spy = _WriterSpy(monkeypatch)

    _ = _run(ds, [_GRAYSCALE])

    assert spy.opened == []


@pytest.mark.media
def test_a_placement_that_no_longer_fits_is_encoded_again_and_then_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The entry now has 20 frames where its variant was made from 12.

    Its recorded composition is blank. Only the placement then shows that the
    file is out of date. A consumer refuses the file until a plain run of the same
    recipe writes it again.
    """
    ds = make_dataset(tmp_path / "ds")
    _entry(ds, "s", [(12, 30.0)])
    run_id = _run(ds, [_GRAYSCALE])
    _forget_composition(ds, run_id)
    _entry(ds, "s", [(20, 30.0)])
    (entry,) = ds.resolve_media_scope(None)
    with pytest.raises(MediaVariantDriftedError):
        _ = VariantLookup.read(ds, run_id, [("", "s")]).resolve(ds, entry)
    spy = _WriterSpy(monkeypatch)

    _ = _run(ds, [_GRAYSCALE])

    assert len(spy.opened) == 1
    source = VariantLookup.read(ds, run_id, [("", "s")]).resolve(ds, entry)
    assert source.placement.frames.count == 20
    assert source.facts.frame_count == 20


@pytest.mark.media
def test_a_run_records_its_recipe_and_identity_scheme(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds")
    _entry(ds, "s", [(12, 30.0)])
    given: dict[str, object] = {
        "steps": [_crop(8, 8, 32, 24), _GRAYSCALE],
        "fps": 12.5,
        "quality": 20,
    }

    run_id = run_op(ds, _KIND, given, scope=Scope(entries=[("", "s")]))

    recorded = json.loads(media_variant_recipe_path(ds, run_id).read_text())
    assert recorded == PreprocessParams.model_validate(given).model_dump(mode="json")
    run_root = media_variant_run_root(ds, run_id)
    assert read_identity_scheme(run_root) == OP_IDENTITY_SCHEME


@pytest.mark.media
def test_the_printed_rewrite_command_runs_the_recorded_recipe(tmp_path: Path) -> None:
    """A consumer's remedy for a missing file, run as printed, writes it again."""
    ds = make_dataset(tmp_path / "ds")
    _entry(ds, "s", [(12, 30.0)])
    run_id = _run(ds, [_crop(8, 8, 32, 24), _decimate(2)], fps=12.5)
    dest = media_variant_path(ds, run_id, "", "s", "")
    dest.unlink()
    (entry,) = ds.resolve_media_scope(None)
    with pytest.raises(MediaVariantMissingError) as missing:
        _ = VariantLookup.read(ds, run_id, [("", "s")]).resolve(ds, entry)
    (line,) = [text for text in str(missing.value).splitlines() if "mosaic run" in text]
    manifest = str(ds.manifest_path)
    argv = [manifest if word == "<manifest>" else word for word in shlex.split(line)]
    assert argv[:2] == ["mosaic", "run"]

    result = CliRunner().invoke(app, argv[1:])

    assert result.exit_code == 0, result.output
    assert run_id in result.output
    assert dest.is_file()
    assert _row(ds, run_id)["frame_count"] == "6"


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
    assert _variant_files(ds) == [_partial(ds, run_id, "s")]
    assert ("", "s", "") not in media_variant_rows(ds, run_id)
    (line,) = entry_error_lines(ds, "short")
    assert "11" in line
    assert "12" in line


@pytest.mark.media
def test_an_encode_never_writes_where_another_entry_publishes(tmp_path: Path) -> None:
    """Entry ``a`` encoding again leaves entry ``a.partial``'s variant file alone.

    The partial is written inside the entry's work directory. Beside the
    destination it was ``a.partial.mp4``, the file that ``a.partial`` publishes, and
    renaming it into ``a.mp4`` took that file away.
    """
    ds = make_dataset(tmp_path / "ds")
    _entry(ds, "a", [(12, 30.0)])
    _entry(ds, "a.partial", [(12, 30.0)], base=50)
    run_id = _run(ds, [_GRAYSCALE], ("a", "a.partial"))
    neighbor = media_variant_path(ds, run_id, "", "a.partial", "")
    published = neighbor.read_bytes()

    _ = _run(ds, [_GRAYSCALE], ("a",), overwrite=True)

    assert neighbor.read_bytes() == published
    assert media_variant_path(ds, run_id, "", "a", "").is_file()
    assert _variant_files(ds) == sorted(
        media_variant_path(ds, run_id, "", sequence, "")
        for sequence in ("a", "a.partial")
    )


@pytest.mark.media
def test_one_failing_entry_does_not_stop_the_others(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds")
    _entry(ds, "broken", [(12, 30.0)])
    _entry(ds, "fine", [(12, 30.0)], base=50)
    (ds.get_root("media_raw") / "broken" / "clip0.mp4").unlink()

    run_id = _run(ds, [_GRAYSCALE], ("broken", "fine"), execution_id="mixed")

    assert media_variant_path(ds, run_id, "", "fine", "").is_file()
    assert not media_variant_path(ds, run_id, "", "broken", "").exists()
    (line,) = entry_error_lines(ds, "mixed")
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

    The partial is removed, the row is not written, the entry's claim is
    released, and the next entry is not started.
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
    assert ("", "s", "") not in media_variant_rows(ds, run_id)
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
        "do not select a frame",
    ),
}


@pytest.mark.parametrize("case", sorted(_REFUSALS))
def test_a_refused_recipe_is_refused_before_any_entry_is_written(
    tmp_path: Path, case: str
) -> None:
    """The first entry is valid and sorts first, and its files are not written."""
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


def test_plan_identity_resolves_its_scope_in_one_read(
    three_entry_dataset: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    scope = three_entry_dataset.resolve_scope(Scope(groups=["A", "B"]))
    reads = count_index_reads(monkeypatch)

    _ = PreprocessOp().plan_identity(
        three_entry_dataset, _params([_crop(0, 0, 32, 24)]), scope
    )

    assert reads.media_scopes == 1


def test_an_entry_awaiting_a_transcode_costs_planning_no_read_of_its_own(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = make_dataset(tmp_path / "ds")
    write_media_index(
        ds,
        [
            MediaClip(filename="a.mp4", sequence="a", rotation=90),
            MediaClip(filename="b.mp4", sequence="b"),
            MediaClip(filename="c.mp4", sequence="c"),
        ],
    )
    scope = ds.resolve_scope(Scope(entries=[("", "a"), ("", "b"), ("", "c")]))
    reads = count_index_reads(monkeypatch)

    _ = PreprocessOp().plan_identity(ds, _params([_crop(0, 0, 32, 24)]), scope)

    assert reads.media_scopes == 1


def test_planning_without_a_media_index_reads_for_it_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = make_dataset(tmp_path / "ds")
    params = _params([_crop(0, 0, 32, 24)])
    scope = ds.resolve_scope(Scope(entries=[("", "a"), ("", "b"), ("", "c")]))
    reads = count_index_reads(monkeypatch)

    identity = PreprocessOp().plan_identity(ds, params, scope)

    assert identity == preprocess_identity(params)
    assert reads.media_scopes == 1


def test_an_entry_awaiting_a_transcode_leaves_the_others_checked(
    tmp_path: Path,
) -> None:
    """One entry's refusal to resolve does not end the check of the rest.

    The scope's one read records the rotated original's refusal against that
    entry alone. The rotated one is passed over, and the other is refused for a
    crop outside its frame.
    """
    ds = make_dataset(tmp_path / "ds")
    write_media_index(
        ds,
        [
            MediaClip(filename="a.mp4", sequence="a", rotation=90),
            MediaClip(filename="b.mp4", sequence="b", width=64, height=48),
        ],
    )
    scope = ds.resolve_scope(Scope(entries=[("", "a"), ("", "b")]))

    with pytest.raises(PreprocessRefused, match="variant of b"):
        _ = PreprocessOp().plan_identity(ds, _params([_crop(600, 0, 100, 100)]), scope)


def test_an_entry_with_no_name_is_checked_under_its_first_files_stem(
    tmp_path: Path,
) -> None:
    ds = make_dataset(tmp_path / "ds")
    write_media_index(
        ds,
        [
            MediaClip(filename="clip.mp4", sequence=""),
            MediaClip(filename="b.mp4", sequence="b", width=64, height=48),
        ],
    )
    scope = ds.resolve_scope(Scope(entries=[("", ""), ("", "b")]))

    with pytest.raises(PreprocessRefused, match="variant of clip"):
        _ = PreprocessOp().plan_identity(ds, _params([_crop(600, 0, 100, 100)]), scope)


def test_a_skipped_camera_is_reported_once_by_a_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Planning does not print the line, and the run prints it once.

    The stub files cannot be decoded, and the run's one entry fails.
    """
    ds = make_dataset(tmp_path / "ds")
    write_media_index(
        ds,
        [
            MediaClip(filename=f"{camera}.mp4", sequence="s", camera=camera)
            for camera in ("left", "right")
        ],
    )
    scope = ds.resolve_scope(Scope(entries=[("", "s")]))

    _ = PreprocessOp().plan_identity(ds, _params([_GRAYSCALE]), scope)
    assert "skipping it" not in capsys.readouterr().err

    with pytest.raises(AllEntriesFailed):
        _ = _run(ds, [_GRAYSCALE])
    assert capsys.readouterr().err.count("skipping it") == 1


@pytest.mark.media
def test_a_run_reads_each_index_once_for_its_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A chained re-run reads the upstream's rows and its own, once each."""
    ds = make_dataset(tmp_path / "ds")
    sequences = ("s", "t", "u")
    for sequence in sequences:
        _entry(ds, sequence, [(6, 30.0)])
    upstream = _run(ds, [_crop(8, 8, 32, 24)], sequences)
    _ = _run(ds, [_GRAYSCALE], sequences, media=upstream)
    reads = count_index_reads(monkeypatch)

    _ = _run(ds, [_GRAYSCALE], sequences, media=upstream)

    assert reads == IndexReads(media_scopes=1, variant_indexes=2, compositions=1)


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


def _decoded(path: Path) -> npt.NDArray[np.float64]:
    """Return every frame of *path*, decoded for analysis, as one float array."""
    with open_frame_reader(path, target="analysis") as reader:
        return np.stack([np.asarray(frame, np.float64) for _, frame in reader])


@pytest.mark.media
def test_an_h264_variant_keeps_the_colors_of_its_source(tmp_path: Path) -> None:
    """Each channel of the variant is within 1.5 levels of the source, on average.

    ffmpeg converts the BGR frames to yuv420p before libx264 encodes them. With
    swscale's default rounding the conversion darkens every channel, blue by 3
    levels for this color. The source is lossless RGB, and its frames decode to
    the painted color.
    """
    if not (encoder_available("libx264") and encoder_available("libx264rgb")):
        pytest.skip("the ffmpeg on PATH has no libx264")
    ds = make_dataset(tmp_path / "ds")
    width, height = _SIZE
    painted = np.full((6, height, width, 3), (60, 120, 200), np.uint8)
    clip = ds.get_root("media_raw") / "s" / "clip0.mp4"
    write_h264_mp4(
        clip, frames=6, size=_SIZE, paint=lambda frame: painted[frame], lossless=True
    )
    index_media_sequence(ds, "s", [clip.name])

    run_id = _run(ds, [_trim(0, 6)], codec="h264")

    assert np.array_equal(_decoded(clip), painted), "the lossless source premise"
    variant = _decoded(media_variant_path(ds, run_id, "", "s", ""))
    error = np.mean(variant - painted, axis=(0, 1, 2))
    assert np.all(np.abs(error) <= 1.5), f"mean error per B, G, R channel: {error}"


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
