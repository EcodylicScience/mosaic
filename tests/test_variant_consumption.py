"""Test tracking a media variant and publishing its results in source space.

A tracker or inference op whose ``media`` names a variant hands its tool the
variant's file for each entry, and maps the table that the tool reports back onto
the entry's pixels and frames. Every reuse gate compares the file that the tool
reads. A run over a variant therefore never reuses output made from the entry
media, and it recomputes when the variant file is rewritten. An entry whose
variant is missing or out of date fails alone.

The variants are real. The ``preprocess`` op writes them from flat clips. The
trackers and the inference runners are the recording fakes from
``tests.helpers``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

import mosaic.tracking.litpose.dataset_runs as litpose_runs
import mosaic.tracking.sleap.dataset_runs as sleap_runs
import mosaic.tracking.trex.dataset_runs as trex_runs
import mosaic.tracking.ultralytics_track.dataset_runs as ultralytics_runs
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.markers import new_inflight, write_inflight
from mosaic.core.pipeline.ops import run_op
from mosaic.core.pipeline.preprocess_index import media_variant_rows
from mosaic.core.pipeline.promotion import promote_correction
from mosaic.core.pipeline.provenance import reached_by
from mosaic.core.pipeline.preprocess_layout import (
    media_variant_path,
    media_variant_recipe_path,
)
from mosaic.core.pipeline.run import AllEntriesFailed
from mosaic.core.pipeline.sequence_index import decode_consumed_roots
from mosaic.core.pipeline.tracks_index import (
    backfill_media_frames,
    read_media_frames,
    read_tracks_index,
)
from mosaic.core.scope import Scope
from mosaic.runlog import reduce_run_log, run_log_path
from mosaic.tracking.common.scope import build_work_items
from mosaic.tracking.common.tool_input import ToolCodecError
from mosaic.tracking.litpose.params import LitposeParams
from mosaic.tracking.ops.infer import infer_run_root
from mosaic.tracking.pose_training.localizer_inference import LocalizerDetection
from mosaic.tracking.sleap.params import SleapParams
from mosaic.tracking.trex.conversion_cache import conversion_slot
from mosaic.tracking.trex.params import TrexParams
from mosaic.tracking.ultralytics_track.params import UltralyticsParams

from tests.helpers import (
    FakeLitpose,
    FakeSleap,
    FakeTrex,
    FakeUltralytics,
    count_index_reads,
    entry_error_lines,
    install_fake_litpose,
    install_fake_point_inference,
    install_fake_pose_inference,
    install_fake_sleap,
    install_fake_trex,
    install_fake_ultralytics,
    make_dataset,
    scope_over,
    write_litpose_model,
    write_painted_entry,
    write_sleap_model,
)

pytestmark = pytest.mark.media

_SIZE = (64, 48)
_FRAMES = 20
_FPS = 30.0

_STEPS: list[dict[str, object]] = [
    {"step": "crop", "x": 8, "y": 4, "width": 32, "height": 24},
    {"step": "trim", "start": 2, "stop": 14},
    {"step": "decimate", "every": 2},
]
"""A 32x24 window at (8, 4), with source frames 2, 4, ..., 12 at 15 fps."""

_OFFSET_X, _OFFSET_Y = 8, 4
_FIRST, _STEP = 2, 2


def _source_frame(variant_frame: int) -> int:
    return _FIRST + _STEP * variant_frame


# --- the dataset --------------------------------------------------------------


_CLIP = [(_FRAMES, _FPS)]
"""Each entry's one clip, as ``(frames, fps)``."""


def _flat(shade: int) -> Callable[[int], npt.NDArray[np.uint8]]:
    """Return a painter that gives every frame the gray level *shade*."""

    def paint(_frame: int) -> npt.NDArray[np.uint8]:
        width, height = _SIZE
        return np.full((height, width, 3), shade, np.uint8)

    return paint


def _dataset(tmp_path: Path, sequences: Sequence[str] = ("s",)) -> Dataset:
    ds = make_dataset(tmp_path / "ds")
    for position, sequence in enumerate(sequences):
        _ = write_painted_entry(
            ds, sequence, _CLIP, _flat(40 + 30 * position), size=_SIZE
        )
    return ds


def _variant(
    ds: Dataset,
    sequences: Sequence[str] = ("s",),
    *,
    codec: str = "av1",
) -> str:
    """Write the variant of :data:`_STEPS` for *sequences*, and return its run id."""
    return run_op(
        ds,
        "preprocess",
        {"steps": _STEPS, "codec": codec},
        scope=Scope(entries=[("", sequence) for sequence in sequences]),
    )


def _variant_uuid(ds: Dataset, run_id: str, sequence: str = "s") -> str:
    row = media_variant_rows(ds, run_id).get(("", sequence, ""))
    assert row is not None
    return row["video_uuid"]


def _tracks(ds: Dataset, producer: str) -> pd.DataFrame:
    rows = read_tracks_index(ds)
    return rows[rows["producer"] == producer].reset_index(drop=True)


def _table(ds: Dataset, row: pd.Series) -> pd.DataFrame:
    return pd.read_parquet(ds.resolve_path(str(row["abs_path"])))


# --- the trackers -------------------------------------------------------------


@pytest.fixture
def model(tmp_path: Path) -> Path:
    path = tmp_path / "yolo" / "best.pt"
    path.parent.mkdir(parents=True)
    _ = path.write_bytes(b"weights")
    return path


@pytest.fixture
def sleap_model(tmp_path: Path) -> Path:
    return write_sleap_model(tmp_path / "sleap_model")


@pytest.fixture
def litpose_model(tmp_path: Path) -> Path:
    return write_litpose_model(tmp_path / "litpose_model")


@pytest.fixture
def ultralytics(monkeypatch: pytest.MonkeyPatch) -> FakeUltralytics:
    return install_fake_ultralytics(monkeypatch)


@pytest.fixture
def sleap(monkeypatch: pytest.MonkeyPatch) -> FakeSleap:
    return install_fake_sleap(monkeypatch)


@pytest.fixture
def trex(monkeypatch: pytest.MonkeyPatch) -> FakeTrex:
    return install_fake_trex(monkeypatch)


@pytest.fixture
def litpose(monkeypatch: pytest.MonkeyPatch) -> FakeLitpose:
    return install_fake_litpose(monkeypatch)


def _ultralytics_params(model: Path, media: str) -> UltralyticsParams:
    return UltralyticsParams.model_validate({"model_path": str(model), "media": media})


# --- the work item ------------------------------------------------------------


def test_a_variant_item_describes_the_file_the_tool_reads(tmp_path: Path) -> None:
    ds = _dataset(tmp_path)
    variant = _variant(ds)

    (plain,) = build_work_items(ds, ds.resolve_media_scope(None), kind="trex").items
    built = build_work_items(
        ds, ds.resolve_media_scope(None), kind="trex", media=variant
    )

    assert built.failures == ()
    (item,) = built.items
    path = media_variant_path(ds, variant, "", "s", "")
    assert item.video_paths == (path,)
    assert item.source_uid == item.video_uid == _variant_uuid(ds, variant)
    assert item.source_uid != plain.source_uid
    assert item.fps == pytest.approx(_FPS / _STEP)
    assert item.facts is not None and (item.facts.width, item.facts.height) == (32, 24)
    assert item.media == variant
    assert item.consumed_media == (path, plain.video_path)
    assert plain.media == ""
    assert plain.source_mapping is None


# --- Ultralytics --------------------------------------------------------------


def test_ultralytics_tracks_the_variant_and_publishes_in_source_space(
    tmp_path: Path, model: Path, ultralytics: FakeUltralytics
) -> None:
    ds = _dataset(tmp_path)
    variant = _variant(ds)

    run_id = ultralytics_runs.run_ultralytics(ds, _ultralytics_params(model, variant))

    (request,) = ultralytics.requests
    assert Path(request.video_path) == media_variant_path(ds, variant, "", "s", "")
    assert (request.media_facts["width"], request.media_facts["height"]) == (32, 24)

    (row,) = [row for _, row in _tracks(ds, "ultralytics").iterrows()]
    table = _table(ds, row)
    # The fake reports track t in variant frame f with its body center at
    # (10t + f + 0.5, 20.5) and its box's left edge at 10t.
    variant_frame = (table["frame"] - _FIRST) // _STEP
    assert sorted(set(table["frame"])) == [_source_frame(f) for f in range(4)]
    assert (
        table["X"] == 10 * table["source_track_id"] + variant_frame + 0.5 + _OFFSET_X
    ).all()
    assert (table["Y"] == 20.5 + _OFFSET_Y).all()
    assert (table["bbox_x1"] == 10 * table["source_track_id"] + _OFFSET_X).all()
    assert table["time"].to_numpy() == pytest.approx(table["frame"] / _FPS)

    assert "media_raw" in decode_consumed_roots(str(row["consumed_source_roots"]))
    assert read_media_frames(row) is None
    runs = ultralytics_runs.list_ultralytics_runs(ds)
    assert runs["run_id"].tolist() == [run_id]
    assert runs["media"].tolist() == [variant]


def test_ultralytics_reuses_a_variant_run_and_recomputes_a_rewritten_variant(
    tmp_path: Path, model: Path, ultralytics: FakeUltralytics
) -> None:
    ds = _dataset(tmp_path)
    variant = _variant(ds)
    params = _ultralytics_params(model, variant)
    _ = ultralytics_runs.run_ultralytics(ds, params)
    _ = ultralytics_runs.run_ultralytics(ds, params)
    assert len(ultralytics.tracked) == 1

    before = _variant_uuid(ds, variant)
    _ = write_painted_entry(ds, "s", _CLIP, _flat(200), size=_SIZE)
    assert _variant(ds) == variant
    assert _variant_uuid(ds, variant) != before

    _ = ultralytics_runs.run_ultralytics(ds, params)

    assert len(ultralytics.tracked) == 2
    assert ultralytics.tracked[1] == media_variant_path(ds, variant, "", "s", "")


def test_a_missing_variant_fails_only_its_entry(
    tmp_path: Path, model: Path, ultralytics: FakeUltralytics
) -> None:
    ds = _dataset(tmp_path, ("s", "t"))
    variant = _variant(ds, ("s",))

    _ = ultralytics_runs.run_ultralytics(
        ds, _ultralytics_params(model, variant), execution_id="missing"
    )

    assert ultralytics.tracked == [media_variant_path(ds, variant, "", "s", "")]
    (line,) = entry_error_lines(ds, "missing")
    assert "MediaVariantMissingError" in line
    assert '"t"' in line
    assert "--kind preprocess" in line
    assert _tracks(ds, "ultralytics")["sequence"].tolist() == ["s"]


def test_a_drifted_variant_fails_only_its_entry(
    tmp_path: Path, model: Path, ultralytics: FakeUltralytics
) -> None:
    ds = _dataset(tmp_path, ("s", "t"))
    variant = _variant(ds, ("s", "t"))
    # The entry's media changes after the variant was written from it.
    _ = write_painted_entry(ds, "t", _CLIP, _flat(200), size=_SIZE)

    _ = ultralytics_runs.run_ultralytics(
        ds, _ultralytics_params(model, variant), execution_id="drifted"
    )

    assert ultralytics.tracked == [media_variant_path(ds, variant, "", "s", "")]
    (line,) = entry_error_lines(ds, "drifted")
    assert "MediaVariantDriftedError" in line
    assert variant in line
    assert _tracks(ds, "ultralytics")["sequence"].tolist() == ["s"]


def test_a_run_whose_every_variant_is_missing_fails_naming_the_variant(
    tmp_path: Path, model: Path, ultralytics: FakeUltralytics
) -> None:
    """The tool did not run. The refusal names the variant and omits tool output."""
    ds = _dataset(tmp_path)
    missing = "preprocess.0.1-0123456789"

    with pytest.raises(AllEntriesFailed) as refused:
        _ = ultralytics_runs.run_ultralytics(
            ds, _ultralytics_params(model, missing), execution_id="unread"
        )

    message = str(refused.value)
    expected = (
        f"every entry that it attempted lacks a readable file of the media "
        f"variant {missing}"
    )
    assert expected in message
    assert f"Run the preprocess step that made {missing}" in message
    assert "--kind preprocess --entries :s --params" in message
    assert f"{missing} does not record a recipe" in message
    assert "tool output" not in message
    assert ultralytics.tracked == []
    (line,) = entry_error_lines(ds, "unread")
    assert "MediaVariantMissingError" in line


# --- SLEAP --------------------------------------------------------------------


def test_sleap_recomputes_after_the_variant_is_rewritten(
    tmp_path: Path, sleap_model: Path, sleap: FakeSleap
) -> None:
    ds = _dataset(tmp_path)
    variant = _variant(ds, codec="h264")
    params = SleapParams(model_paths=[str(sleap_model)], media=variant)
    run_id = sleap_runs.run_sleap(ds, params)
    _ = sleap_runs.run_sleap(ds, params)
    assert sleap.tracked == [media_variant_path(ds, variant, "", "s", "")]

    _ = write_painted_entry(ds, "s", _CLIP, _flat(200), size=_SIZE)
    _ = _variant(ds, codec="h264")
    _ = sleap_runs.run_sleap(ds, params)

    assert len(sleap.tracked) == 2
    runs = sleap_runs.list_sleap_runs(ds)
    assert runs["run_id"].tolist() == [run_id]
    assert runs["media"].tolist() == [variant]
    (row,) = [row for _, row in _tracks(ds, "sleap").iterrows()]
    assert set(_table(ds, row)["frame"]) <= {_source_frame(f) for f in range(6)}


def test_an_av1_variant_handed_to_sleap_names_the_h264_remedy(
    tmp_path: Path,
    sleap_model: Path,
    sleap: FakeSleap,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SLEAP's OpenCV cannot read AV1, and a variant can be made in H.264 instead."""
    monkeypatch.delenv("MOSAIC_ALLOW_TOOL_CODECS", raising=False)
    ds = _dataset(tmp_path)
    variant = _variant(ds)

    params = SleapParams(model_paths=[str(sleap_model)], media=variant)
    with pytest.raises(ToolCodecError) as refused:
        _ = sleap_runs.run_sleap(ds, params)

    message = str(refused.value)
    assert "which is av1" in message
    assert "py-opencv" in message
    assert f'the recipe of {variant} with "codec" set to "h264"' in message
    recipe = media_variant_recipe_path(ds, variant).absolute()
    assert f"--entries :s --params @{recipe}" in message
    assert sleap.tracked == []


# --- Lightning Pose -----------------------------------------------------------


def test_lightning_pose_tracks_the_variant_and_publishes_in_source_space(
    tmp_path: Path, litpose_model: Path, litpose: FakeLitpose
) -> None:
    """The variant is H.264, because Lightning Pose decodes AV1 only on newer GPUs."""
    ds = _dataset(tmp_path)
    variant = _variant(ds, codec="h264")

    run_id = litpose_runs.run_litpose(
        ds, LitposeParams(model_path=str(litpose_model), media=variant)
    )

    assert litpose.predicted == [media_variant_path(ds, variant, "", "s", "")]
    (row,) = [row for _, row in _tracks(ds, "litpose").iterrows()]
    table = _table(ds, row).sort_values("frame")
    # The fake predicts one row per variant frame, and row f is variant frame f.
    (written,) = litpose.written
    assert table["frame"].tolist() == [_source_frame(f) for f in range(len(written))]
    for keypoint in range(written.shape[1]):
        assert table[f"poseX{keypoint}"].to_numpy() == pytest.approx(
            written[:, keypoint, 0] + _OFFSET_X, abs=1e-5
        )
        assert table[f"poseY{keypoint}"].to_numpy() == pytest.approx(
            written[:, keypoint, 1] + _OFFSET_Y, abs=1e-5
        )
    assert table["time"].to_numpy() == pytest.approx(table["frame"] / _FPS)
    assert "media_raw" in decode_consumed_roots(str(row["consumed_source_roots"]))
    runs = litpose_runs.list_litpose_runs(ds)
    assert runs["run_id"].tolist() == [run_id]
    assert runs["media"].tolist() == [variant]


# --- TREx ---------------------------------------------------------------------


def test_a_trex_variant_run_does_not_reuse_the_entry_conversion(
    tmp_path: Path, trex: FakeTrex
) -> None:
    """A slot is keyed by the file converted. The variant gets a separate slot.

    A reuse of the entry's conversion tracks the uncropped frames and then shifts
    every position by the crop's offset.
    """
    ds = _dataset(tmp_path)
    variant = _variant(ds)
    _ = trex_runs.run_trex(ds, TrexParams(), scope_over(("", "s")))

    _ = trex_runs.run_trex(ds, TrexParams(media=variant), scope_over(("", "s")))

    variant_file = media_variant_path(ds, variant, "", "s", "")
    assert trex.sources[1] == [variant_file]
    scope = ds.resolve_media_scope(None)
    (plain,) = build_work_items(ds, scope, kind="trex").items
    (item,) = build_work_items(ds, scope, kind="trex", media=variant).items
    settings = trex_runs.phase_settings(
        trex_runs.trex_settings(TrexParams(), detect_model_id=None, vi_model_id=None),
        "convert",
    )
    variant_slot = conversion_slot(ds, settings, item)
    assert variant_slot is not None and variant_slot.name == _variant_uuid(ds, variant)
    assert variant_slot != conversion_slot(ds, settings, plain)
    assert variant_slot.is_dir()


def test_a_trex_variant_table_is_mapped_into_source_space(
    tmp_path: Path, trex: FakeTrex
) -> None:
    ds = _dataset(tmp_path)
    variant = _variant(ds)

    run_id = trex_runs.run_trex(ds, TrexParams(media=variant), scope_over(("", "s")))

    (row,) = [row for _, row in _tracks(ds, "trex").iterrows()]
    table = _table(ds, row)
    # The fake exports the body center at (f, f) in variant frame f.
    variant_frame = (table["frame"] - _FIRST) // _STEP
    assert sorted(set(table["frame"])) == [_source_frame(f) for f in range(4)]
    assert (table["X"] == variant_frame + _OFFSET_X).all()
    assert (table["Y"] == variant_frame + _OFFSET_Y).all()
    assert table["time"].to_numpy() == pytest.approx(table["frame"] / _FPS)
    assert "media_raw" in decode_consumed_roots(str(row["consumed_source_roots"]))
    assert read_media_frames(row) is None
    runs = trex_runs.list_trex_runs(ds)
    assert runs["run_id"].tolist() == [run_id]
    assert runs["media"].tolist() == [variant]
    assert runs["n_source_videos"].tolist() == [1]


# --- inference ----------------------------------------------------------------


def _install_fake_localizer(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Stand in for the localizer, which runs in this process.

    It reports one detection per frame, at ``(1, 4)`` in frame 0 and ``(3, 6)``
    in frame 1, and records each video that it is handed.
    """
    import mosaic.tracking.pose_training.localizer_inference as localizer

    videos: list[Path] = []

    def run(
        _model_path: str, video_path: Path, **_kwargs: object
    ) -> list[list[LocalizerDetection]]:
        videos.append(Path(video_path))
        return [
            [{"x": 1.0, "y": 4.0, "confidence": 0.9, "class_id": 0}],
            [{"x": 3.0, "y": 6.0, "confidence": 0.8, "class_id": 0}],
        ]

    monkeypatch.setattr(localizer, "run_localizer_inference", run)
    return videos


def _pose_videos(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    return install_fake_pose_inference(monkeypatch).videos


def _point_videos(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    return install_fake_point_inference(monkeypatch).videos


_INFERENCE: dict[
    str,
    tuple[
        Callable[[pytest.MonkeyPatch], list[Path]],
        set[tuple[int, float, float]],
    ],
] = {
    # The fake pose runner reports keypoints (1, 2) and (5, 8) in frames 0-3,
    # whose mean is the body center.
    "infer-pose": (
        _pose_videos,
        {(_source_frame(f), 3.0 + _OFFSET_X, 5.0 + _OFFSET_Y) for f in range(4)},
    ),
    # The fake point runner reports (1, 4) and (2, 5) in frame 0, (3, 6) in 1.
    "infer-points": (
        _point_videos,
        {
            (_source_frame(0), 1.0 + _OFFSET_X, 4.0 + _OFFSET_Y),
            (_source_frame(0), 2.0 + _OFFSET_X, 5.0 + _OFFSET_Y),
            (_source_frame(1), 3.0 + _OFFSET_X, 6.0 + _OFFSET_Y),
        },
    ),
    "infer-localizer": (
        _install_fake_localizer,
        {
            (_source_frame(0), 1.0 + _OFFSET_X, 4.0 + _OFFSET_Y),
            (_source_frame(1), 3.0 + _OFFSET_X, 6.0 + _OFFSET_Y),
        },
    ),
}


def _infer(
    ds: Dataset,
    kind: str,
    model: Path,
    media: str,
    sequences: Sequence[str] = ("s",),
    *,
    execution_id: str | None = None,
) -> str:
    return run_op(
        ds,
        kind,
        {"model": str(model), "media": media},
        scope=Scope(entries=[("", sequence) for sequence in sequences]),
        execution_id=execution_id,
    )


@pytest.mark.parametrize("kind", sorted(_INFERENCE))
def test_an_inference_op_publishes_a_variant_table_in_source_space(
    tmp_path: Path, model: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    install, expected = _INFERENCE[kind]
    videos = install(monkeypatch)
    ds = _dataset(tmp_path)
    variant = _variant(ds)

    run_id = _infer(ds, kind, model, variant)

    assert videos == [media_variant_path(ds, variant, "", "s", "")]
    (row,) = [row for _, row in _tracks(ds, kind).iterrows()]
    assert str(row["producer_run_id"]) == run_id
    table = _table(ds, row)
    published = {
        (int(frame), float(x), float(y))
        for frame, x, y in zip(table["frame"], table["X"], table["Y"], strict=True)
    }
    assert published == expected
    assert table["time"].to_numpy() == pytest.approx(table["frame"] / _FPS)
    assert "media_raw" in decode_consumed_roots(str(row["consumed_source_roots"]))


def test_a_missing_variant_fails_only_its_inference_entry(
    tmp_path: Path, model: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = install_fake_point_inference(monkeypatch)
    ds = _dataset(tmp_path, ("s", "t"))
    variant = _variant(ds, ("s",))

    _ = _infer(ds, "infer-points", model, variant, ("s", "t"), execution_id="points")

    assert fake.videos == [media_variant_path(ds, variant, "", "s", "")]
    (line,) = entry_error_lines(ds, "points")
    assert "MediaVariantMissingError" in line
    assert _tracks(ds, "infer-points")["sequence"].tolist() == ["s"]
    snapshot = reduce_run_log(run_log_path(ds.base_dir, "points"))
    assert snapshot is not None
    assert (snapshot["entries_written"], snapshot["entries_failed"]) == (1, 1)


def test_an_inference_run_reads_the_variant_index_once(
    tmp_path: Path, model: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ = install_fake_point_inference(monkeypatch)
    sequences = ("s", "t", "u")
    ds = _dataset(tmp_path, sequences)
    variant = _variant(ds, sequences)
    reads = count_index_reads(monkeypatch)

    _ = _infer(ds, "infer-points", model, variant, sequences)

    assert len(_tracks(ds, "infer-points")) == len(sequences)
    assert (reads.media_scopes, reads.variant_indexes) == (1, 1)


def test_an_inference_run_with_no_readable_variant_runs_no_model(
    tmp_path: Path, model: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = install_fake_point_inference(monkeypatch)
    ds = _dataset(tmp_path)

    with pytest.raises(
        AllEntriesFailed, match="every entry that it attempted lacks a readable file"
    ):
        _ = _infer(ds, "infer-points", model, "preprocess.0.1-0123456789")

    assert fake.videos == []
    assert _tracks(ds, "infer-points").empty


def test_an_inference_run_whose_readable_entries_are_held_names_the_variant(
    tmp_path: Path, model: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one readable entry is held by another execution, the other unreadable.

    A model did not run for any entry that this run attempted. The refusal
    therefore names the variant and the command that writes it, as the tracker
    driver's does.
    """
    fake = install_fake_point_inference(monkeypatch)
    ds = _dataset(tmp_path, ("s", "t"))
    variant = _variant(ds, ("s",))
    run_id = _infer(ds, "infer-points", model, variant)
    held = infer_run_root(ds, "infer-points", run_id) / "s"
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
    ran = len(fake.videos)

    with pytest.raises(AllEntriesFailed) as refused:
        _ = _infer(ds, "infer-points", model, variant, ("s", "t"))

    message = str(refused.value)
    assert "every entry that it attempted lacks a readable file" in message
    assert "--kind preprocess --entries :t --params" in message
    assert len(fake.videos) == ran


def test_an_inference_run_reports_the_entries_it_holds(
    tmp_path: Path, model: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The count includes cache hits, and a re-run reports a first run's coverage."""
    _ = install_fake_point_inference(monkeypatch)
    ds = _dataset(tmp_path, ("s", "t"))

    for attempt in ("first", "again"):
        _ = _infer(ds, "infer-points", model, "", ("s", "t"), execution_id=attempt)
        snapshot = reduce_run_log(run_log_path(ds.base_dir, attempt))
        assert snapshot is not None
        assert snapshot["entries_written"] == 2


# --- provenance ---------------------------------------------------------------


def test_backfill_leaves_a_variant_table_media_frames_blank(
    tmp_path: Path, model: Path, ultralytics: FakeUltralytics
) -> None:
    """A variant table does not span its source axis, and its length stays blank.

    The entry-media table beside it is filled as before.
    """
    ds = _dataset(tmp_path)
    variant = _variant(ds)
    on_variant = ultralytics_runs.run_ultralytics(
        ds, _ultralytics_params(model, variant)
    )
    on_entry = ultralytics_runs.run_ultralytics(ds, _ultralytics_params(model, ""))

    filled = backfill_media_frames(ds)

    assert filled["producer_run_id"].tolist() == [on_entry]
    recorded = {
        str(row["producer_run_id"]): read_media_frames(row)
        for _, row in _tracks(ds, "ultralytics").iterrows()
    }
    assert recorded == {on_variant: None, on_entry: _FRAMES}


def _predictions(ds: Dataset, run_id: str) -> Path:
    return (
        ultralytics_runs.ultralytics_run_root(ds, run_id)
        / "s"
        / ("s" + ultralytics_runs.PREDICTIONS_SUFFIX)
    )


def test_promoting_a_correction_of_a_variant_run_is_refused(
    tmp_path: Path, model: Path, ultralytics: FakeUltralytics
) -> None:
    """The output is in variant pixels and frames, and promotion converts it unmapped.

    A correction of a run over the entry media is not refused.
    """
    ds = _dataset(tmp_path)
    variant = _variant(ds)
    on_variant = ultralytics_runs.run_ultralytics(
        ds, _ultralytics_params(model, variant)
    )
    on_entry = ultralytics_runs.run_ultralytics(ds, _ultralytics_params(model, ""))

    with pytest.raises(ValueError, match=f"media variant {variant}"):
        _ = promote_correction(
            ds,
            "",
            "s",
            _predictions(ds, on_variant),
            src_format="ultralytics_tracks",
            derived_from=on_variant,
        )

    report = promote_correction(
        ds,
        "",
        "s",
        _predictions(ds, on_entry),
        src_format="ultralytics_tracks",
        derived_from=on_entry,
    )
    assert report.derived_from == on_entry


def test_a_variant_table_is_reached_from_its_source_media(
    tmp_path: Path, model: Path, ultralytics: FakeUltralytics
) -> None:
    ds = _dataset(tmp_path)
    variant = _variant(ds)
    _ = ultralytics_runs.run_ultralytics(ds, _ultralytics_params(model, variant))
    (row,) = [row for _, row in _tracks(ds, "ultralytics").iterrows()]

    reached = reached_by(ds, [("", "s")], "media_raw")

    tracks = reached[reached["kind"] == "tracks"]
    assert tracks["run_id"].tolist() == [str(row["run_id"])]
