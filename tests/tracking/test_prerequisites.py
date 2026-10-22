"""The runs an op needs before it reads each entry, and the entries no run clears.

The plan comes from the media index, the stores' metadata, the media variants'
rows and recipes, and the files on disk. It decodes no frame and writes no file.
Each entry is put through the checks of the op's own run, or of the ``preprocess``
run that writes the media variant it reads.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from mosaic.core.dataset import Dataset
from mosaic.core.entry import Entry
from mosaic.core.pipeline.joined_export import (
    JOINED_KIND_DIRECTORY,
    JoinedExportParams,
    current_joined_recipes,
    joined_run_id,
    joined_source_uid,
)
from mosaic.core.pipeline.preprocess import PreprocessParams, preprocess_identity
from mosaic.core.pipeline.preprocess_layout import media_variant_recipe_path
from mosaic.core.pipeline.store_export import StoreExportParams, export_run_id
from mosaic.tracking.frame_extraction.dataset_runs import ListedFramesRefused
from mosaic.tracking.prerequisites import (
    PrerequisitePlan,
    PrerequisiteRun,
    RefusalCause,
    plan_prerequisites,
)
from mosaic.tracking.ultralytics_track.params import UltralyticsParams
from tests.helpers import (
    MediaClip,
    add_media_variant,
    assert_tree_unchanged,
    export_entry_store,
    make_dataset,
    point_at_a_store,
    store_dataset,
    stub_join,
    tree_snapshot,
    video_store_maker,
    write_media_index,
)

MakeMediaDataset = Callable[[Path], Dataset]

_ENTRY: Entry = ("", "sess")

_EXTRACT = {"n_frames": 5}
"""The smallest extract-frames params that validate."""

_CODEC_RECIPE = PreprocessParams(steps=[], codec="h264")
"""The variant that re-encodes the entry media in H.264."""

_CODEC_VARIANT = preprocess_identity(_CODEC_RECIPE).run_id

_GRAYSCALE = PreprocessParams.model_validate({"steps": [{"step": "grayscale"}]})
"""A variant in AV1, the codec of a variant with steps by default."""

_GRAYSCALE_AT_20 = PreprocessParams.model_validate(
    {"steps": [{"step": "grayscale"}], "quality": 20}
)
"""A variant other than :data:`_GRAYSCALE`."""


def _clips(
    sequence: str = "sess",
    *,
    camera: str = "",
    rates: tuple[float, float] = (30.0, 30.0),
    uids: tuple[str, str] = ("uid-0", "uid-1"),
    widths: tuple[int, int] = (640, 640),
    codecs: tuple[str, str] = ("h264", "h264"),
    frame_count: int = 100,
) -> list[MediaClip]:
    """Two clips of one camera of *sequence*, each identified by its uid and camera."""
    return [
        MediaClip(
            sequence=sequence,
            camera=camera,
            filename=f"{sequence}{camera}-c{order}.mp4",
            video_order=order,
            video_uuid=f"{sequence}{camera}-{uid}" if uid else "",
            fps=rate,
            width=width,
            codec=codec,
            frame_count=frame_count,
        )
        for order, (rate, uid, width, codec) in enumerate(
            zip(rates, uids, widths, codecs, strict=True)
        )
    ]


def _join_run(entry: Entry = _ENTRY, *, reencode: bool = False) -> PrerequisiteRun:
    params = JoinedExportParams(reencode=reencode)
    return PrerequisiteRun("export-joined", params, (entry,), joined_run_id(params), ())


def _store_run(entry: Entry = _ENTRY) -> PrerequisiteRun:
    params = StoreExportParams()
    return PrerequisiteRun("export-store", params, (entry,), export_run_id(params), ())


def _preprocess_run(
    entries: tuple[Entry, ...] = (_ENTRY,),
    recipe: PreprocessParams = _CODEC_RECIPE,
    *,
    after: tuple[int, ...] = (),
) -> PrerequisiteRun:
    run_id = preprocess_identity(recipe).run_id
    return PrerequisiteRun("preprocess", recipe, entries, run_id, after)


def _record(ds: Dataset, recipe: PreprocessParams) -> str:
    """Save *recipe* where its variant's runs save it, and return the variant."""
    variant = preprocess_identity(recipe).run_id
    path = media_variant_recipe_path(ds, variant)
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(recipe.model_dump_json(indent=2))
    return variant


def _refused(plan: PrerequisitePlan) -> list[tuple[str, str, str, RefusalCause]]:
    """Each refusal of *plan*, without its reason."""
    return [
        (refusal.group, refusal.sequence, refusal.camera, refusal.cause)
        for refusal in plan.refusals
    ]


def _reason(plan: PrerequisitePlan) -> str:
    """The reason of *plan*'s one refusal."""
    (refusal,) = plan.refusals
    return refusal.reason


# --- joins ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kind", "joined"),
    [
        ("trex", True),
        ("sleap", True),
        ("litpose", True),
        ("ultralytics", False),
        ("infer-pose", False),
        ("infer-localizer", False),
    ],
)
def test_several_clips_need_a_join_only_for_a_tool_that_opens_one_file(
    tmp_path: Path, kind: str, joined: bool
) -> None:
    ds = make_dataset(tmp_path / "ds")
    write_media_index(ds, _clips())

    plan = plan_prerequisites(ds, kind)

    assert plan.runs == ((_join_run(),) if joined else ())
    assert (plan.kind, plan.media, plan.refusals) == (kind, "", ())
    assert plan.entries == (_ENTRY,)

    _ = stub_join(ds, ["sess-uid-0", "sess-uid-1"])
    assert plan_prerequisites(ds, kind).runs == ()


def test_a_join_of_clips_in_two_codecs_is_planned_with_reencode(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds")
    write_media_index(ds, _clips(codecs=("h264", "av1")))

    assert plan_prerequisites(ds, "trex").runs == (_join_run(reencode=True),)
    assert plan_prerequisites(ds, "ultralytics").runs == ()


def test_clips_at_two_rates_need_a_join_for_the_localizer_alone(
    tmp_path: Path,
) -> None:
    ds = make_dataset(tmp_path / "ds")
    write_media_index(ds, _clips(rates=(30.0, 31.0)))

    assert plan_prerequisites(ds, "infer-localizer").runs == (_join_run(),)
    assert plan_prerequisites(ds, "trex").runs == (_join_run(),)
    assert plan_prerequisites(ds, "ultralytics").runs == ()
    assert plan_prerequisites(ds, "infer-pose").runs == ()


@pytest.mark.parametrize(
    "kind", ["trex", "ultralytics", "infer-pose", "infer-localizer"]
)
def test_clips_of_two_sizes_are_unjoinable_for_every_tool(
    tmp_path: Path, kind: str
) -> None:
    ds = make_dataset(tmp_path / "ds")
    write_media_index(ds, _clips(widths=(640, 1280)))

    plan = plan_prerequisites(ds, kind)

    assert _refused(plan) == [("", "sess", "", "unjoinable")]
    assert "c1.mp4 has width" in _reason(plan)
    assert (plan.runs, plan.entries) == ((), ())


@pytest.mark.parametrize(
    ("kind", "refused"),
    [
        ("trex", True),
        ("ultralytics", True),
        ("infer-pose", False),
        ("infer-localizer", False),
    ],
)
def test_clips_without_an_identity_are_refused_by_a_tracker(
    tmp_path: Path, kind: str, refused: bool
) -> None:
    """A tracker's reuse gate cannot name them. An inference op reads them as they are."""
    ds = make_dataset(tmp_path / "ds")
    write_media_index(ds, _clips(uids=("uid-0", "")))

    plan = plan_prerequisites(ds, kind)

    assert plan.runs == ()
    if refused:
        assert _refused(plan) == [("", "sess", "", "unidentified_clips")]
        assert "reprobe-media --apply" in _reason(plan)
        assert "--kind export-joined --entries :sess" in _reason(plan)
    else:
        assert plan.refusals == ()


def test_several_current_joins_are_the_users_to_choose_between(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds")
    write_media_index(ds, _clips())
    root = ds.get_root("media") / JOINED_KIND_DIRECTORY
    root.mkdir(parents=True)
    uid = joined_source_uid(ds.resolve_media(*_ENTRY).facts)
    for recipe in sorted(current_joined_recipes()):
        _ = (root / f"{uid}.{recipe}.joined.mp4").write_bytes(b"join")

    plan = plan_prerequisites(ds, "trex")

    assert _refused(plan) == [("", "sess", "", "several_joins")]
    assert "delete the rest" in _reason(plan)
    assert plan.runs == ()


def test_a_superseded_join_is_built_again(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds")
    write_media_index(ds, _clips())
    root = ds.get_root("media") / JOINED_KIND_DIRECTORY
    root.mkdir(parents=True)
    uid = joined_source_uid(ds.resolve_media(*_ENTRY).facts)
    _ = (root / f"{uid}.0000000000.joined.mp4").write_bytes(b"old join")

    plan = plan_prerequisites(ds, "trex")

    assert (plan.runs, plan.refusals) == ((_join_run(),), ())


def test_a_several_camera_entry_is_refused_a_join(tmp_path: Path) -> None:
    """``export-joined`` joins an entry of one camera. A tracker reads the first."""
    ds = make_dataset(tmp_path / "ds")
    write_media_index(ds, [*_clips(camera="a"), *_clips(camera="b")])

    plan = plan_prerequisites(ds, "trex")

    assert _refused(plan) == [("", "sess", "a", "multi_camera_join")]
    assert (plan.runs, plan.entries) == ((), ())


_JOINING_OPS = pytest.mark.parametrize(
    ("kind", "params", "rates"),
    [("trex", None, (30.0, 30.0)), ("extract-frames", _EXTRACT, (30.0, 31.0))],
    ids=["trex", "extract-frames"],
)
"""Two ops that read clips as their join, each with clip rates that it joins."""


@_JOINING_OPS
def test_the_join_of_a_nameless_entry_is_scoped_by_its_indexed_key(
    tmp_path: Path,
    kind: str,
    params: Mapping[str, object] | None,
    rates: tuple[float, float],
) -> None:
    """An unscoped plan names the entry by its first file's stem, which no scope names.

    The op finds a join by the clips that it holds. The join that the run builds
    under the indexed key is therefore the one that the op reads.
    """
    ds = make_dataset(tmp_path / "ds")
    write_media_index(
        ds,
        [
            MediaClip(
                sequence="",
                filename=f"rec-c{order}.mp4",
                video_order=order,
                video_uuid=f"rec-{order}",
                fps=rate,
            )
            for order, rate in enumerate(rates)
        ],
    )

    plan = plan_prerequisites(ds, kind, params)

    assert plan.runs == (_join_run(("", "")),)
    assert (plan.refusals, plan.entries) == ((), (("", "rec-c0"),))
    joined = [path.name for path in ds.resolve_media("", "").paths]
    assert joined == ["rec-c0.mp4", "rec-c1.mp4"]
    _ = stub_join(ds, ["rec-0", "rec-1"])
    assert plan_prerequisites(ds, kind, params).runs == ()


@_JOINING_OPS
def test_every_camera_of_a_several_camera_nameless_entry_is_refused_a_join(
    tmp_path: Path,
    kind: str,
    params: Mapping[str, object] | None,
    rates: tuple[float, float],
) -> None:
    """An unscoped plan labels each camera by its first file's stem.

    Each label then holds one camera, and the run that joins the indexed key
    resolves both cameras under it.
    """
    ds = make_dataset(tmp_path / "ds")
    write_media_index(
        ds, [*_clips("", camera="a", rates=rates), *_clips("", camera="b", rates=rates)]
    )

    plan = plan_prerequisites(ds, kind, params)

    assert _refused(plan) == [
        ("", "a-c0", "a", "multi_camera_join"),
        ("", "b-c0", "b", "multi_camera_join"),
    ]
    assert (plan.runs, plan.entries) == ((), ())


def test_clips_that_report_no_frame_counts_are_refused_a_join(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds")
    write_media_index(ds, _clips(frame_count=0))

    plan = plan_prerequisites(ds, "trex")

    assert _refused(plan) == [("", "sess", "", "needs_media_reprobe")]
    assert "reprobe-media --apply" in _reason(plan)
    assert plan.runs == ()


# --- stores -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("kind", "exported"),
    [("trex", True), ("ultralytics", False), ("infer-localizer", False)],
)
def test_a_video_store_needs_an_export_only_for_a_tool_that_opens_one_file(
    tmp_path: Path, make_media_dataset: MakeMediaDataset, kind: str, exported: bool
) -> None:
    ds = store_dataset(tmp_path, make_media_dataset, video_store_maker([5, 5, 2]))
    (media,) = ds.resolve_media_scope(None)
    entry = (media.group, media.sequence)

    plan = plan_prerequisites(ds, kind)

    assert plan.runs == ((_store_run(entry),) if exported else ())
    assert (plan.refusals, plan.entries) == ((), (entry,))


@pytest.mark.parametrize(
    ("kind", "exported"),
    [("trex", True), ("ultralytics", True), ("infer-localizer", False)],
)
def test_a_store_whose_chunks_are_not_video_needs_an_export_for_every_runner(
    tmp_path: Path, kind: str, exported: bool
) -> None:
    ds = make_dataset(tmp_path / "ds")
    write_media_index(ds, [MediaClip(video_uuid="uid-store")])
    _ = point_at_a_store(ds, "sess", ds.get_root("media_raw") / "sess.store")

    plan = plan_prerequisites(ds, kind)

    assert plan.runs == ((_store_run(),) if exported else ())
    assert (plan.refusals, plan.entries) == ((), (_ENTRY,))


def test_a_built_export_needs_nothing(
    tmp_path: Path, make_media_dataset: MakeMediaDataset
) -> None:
    ds = store_dataset(tmp_path, make_media_dataset, video_store_maker([5, 5, 2]))
    (media,) = ds.resolve_media_scope(None)
    _ = export_entry_store(ds, media.group, media.sequence)

    assert plan_prerequisites(ds, "trex").runs == ()


def test_a_store_without_an_identity_is_refused_an_export(tmp_path: Path) -> None:
    """``export-store`` names an export by the store's ``video_uuid``."""
    ds = make_dataset(tmp_path / "ds")
    write_media_index(ds, [MediaClip()])
    _ = point_at_a_store(ds, "sess", ds.get_root("media_raw") / "sess.store")

    plan = plan_prerequisites(ds, "trex")

    assert _refused(plan) == [("", "sess", "", "needs_media_reprobe")]
    assert "reprobe-media --apply" in _reason(plan)
    assert (plan.runs, plan.entries) == ((), ())


def test_the_store_of_every_camera_needs_an_identity_for_an_export(
    tmp_path: Path,
) -> None:
    """``export-store`` exports every camera of the entry, the one not read too."""
    ds = make_dataset(tmp_path / "ds")
    write_media_index(
        ds,
        [
            MediaClip(camera="a", filename="a.mp4", video_uuid="uid-a"),
            MediaClip(camera="b", filename="b.mp4"),
        ],
    )
    _ = point_at_a_store(ds, "sess", ds.get_root("media_raw") / "sess.store")

    plan = plan_prerequisites(ds, "trex")

    assert _refused(plan) == [("", "sess", "a", "needs_media_reprobe")]
    assert plan.runs == ()


def test_a_dataset_without_media_raw_cannot_hold_a_store_export(
    tmp_path: Path,
) -> None:
    """Its originals index is ``media/index.csv``, where an export is never read."""
    ds = make_dataset(tmp_path / "ds", roots=("media", "tracks"))
    write_media_index(ds, [MediaClip(video_uuid="uid-store")])
    _ = point_at_a_store(ds, "sess", ds.get_root("media") / "sess.store")

    plan = plan_prerequisites(ds, "trex")

    assert _refused(plan) == [("", "sess", "", "no_media_raw")]
    assert (plan.runs, plan.entries) == ((), ())


@pytest.mark.parametrize("kind", ["trex", "ultralytics", "litpose"])
def test_the_store_of_a_nameless_entry_is_refused_as_unresolved(
    tmp_path: Path, kind: str
) -> None:
    """Its export is looked up by the entry's label, which matches no index row."""
    ds = make_dataset(tmp_path / "ds")
    write_media_index(ds, [MediaClip(sequence="", filename="rec.mp4")])
    _ = point_at_a_store(ds, "", ds.get_root("media_raw") / "rec.store")

    plan = plan_prerequisites(ds, kind)

    assert _refused(plan) == [("", "rec", "", "unresolved_media")]
    assert "No media file found" in _reason(plan)
    assert (plan.runs, plan.entries) == ((), ())


# --- extract-frames -----------------------------------------------------------


def test_mixed_rate_clips_are_joined_before_frames_are_extracted(
    tmp_path: Path,
) -> None:
    ds = make_dataset(tmp_path / "ds")
    write_media_index(ds, [*_clips("mixed", rates=(30.0, 31.0)), *_clips("even")])

    plan = plan_prerequisites(ds, "extract-frames", _EXTRACT)

    assert plan.runs == (_join_run(("", "mixed")),)
    assert plan.refusals == ()
    assert plan.entries == (("", "even"), ("", "mixed"))


def test_mixed_rate_clips_that_cannot_be_one_video_are_refused_for_extract_frames(
    tmp_path: Path,
) -> None:
    ds = make_dataset(tmp_path / "ds")
    write_media_index(ds, _clips(rates=(30.0, 31.0), widths=(640, 1280)))

    plan = plan_prerequisites(ds, "extract-frames", _EXTRACT)

    assert _refused(plan) == [("", "sess", "", "unjoinable")]
    assert (plan.runs, plan.entries) == ((), ())


def test_every_camera_of_a_several_camera_entry_is_refused_a_join(
    tmp_path: Path,
) -> None:
    """``extract-frames`` reads every camera, and a join covers an entry of one."""
    ds = make_dataset(tmp_path / "ds")
    mixed = (30.0, 31.0)
    write_media_index(
        ds, [*_clips(camera="a", rates=mixed), *_clips(camera="b", rates=mixed)]
    )

    plan = plan_prerequisites(ds, "extract-frames", _EXTRACT)

    assert _refused(plan) == [
        ("", "sess", "a", "multi_camera_join"),
        ("", "sess", "b", "multi_camera_join"),
    ]
    assert (plan.runs, plan.entries) == ((), ())


def test_listed_frames_narrow_the_entries_of_extract_frames(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds")
    mixed = (30.0, 31.0)
    write_media_index(ds, [*_clips("a", rates=mixed), *_clips("b", rates=mixed)])
    listed = {"method": "list", "frames": [{"sequence": "a", "indices": [0]}]}

    plan = plan_prerequisites(ds, "extract-frames", listed)

    assert plan.runs == (_join_run(("", "a")),)
    assert plan.entries == (("", "a"),)
    with pytest.raises(ListedFramesRefused):
        _ = plan_prerequisites(ds, "extract-frames", listed, entries=[("", "b")])


def test_extract_frames_params_that_do_not_validate_are_refused(
    tmp_path: Path,
) -> None:
    ds = make_dataset(tmp_path / "ds")
    write_media_index(ds, _clips())

    with pytest.raises(ValidationError, match="n_frames"):
        _ = plan_prerequisites(ds, "extract-frames")


# --- one entry's failure is that entry's refusal ----------------------------


def test_an_entry_awaiting_its_analysis_transcode_is_refused_alone(
    tmp_path: Path,
) -> None:
    ds = make_dataset(tmp_path / "ds")
    write_media_index(ds, [*_clips("a"), MediaClip(sequence="b", codec="mpeg4")])

    plan = plan_prerequisites(ds, "trex")

    assert plan.runs == (_join_run(("", "a")),)
    assert _refused(plan) == [("", "b", "", "needs_analysis_transcode")]
    assert plan.entries == (("", "a"),)


@pytest.mark.parametrize("kind", ["ultralytics", "litpose"])
def test_a_store_whose_metadata_does_not_read_is_refused_alone(
    tmp_path: Path, kind: str
) -> None:
    ds = make_dataset(tmp_path / "ds")
    write_media_index(ds, [*_clips("a"), MediaClip(sequence="b", filename="b.mp4")])
    store = point_at_a_store(ds, "b", ds.get_root("media_raw") / "b.store")
    _ = (store / "metadata.yaml").write_text(
        "__store: {class: DirectoryImgStore, format: npy, imgshape: [48]}\n"
    )

    plan = plan_prerequisites(ds, kind)

    assert _refused(plan) == [("", "b", "", "unresolved_media")]
    assert "imgshape" in _reason(plan)
    assert plan.entries == (("", "a"),)


def test_an_entry_whose_stored_facts_do_not_read_is_refused_alone(
    tmp_path: Path,
) -> None:
    ds = make_dataset(tmp_path / "ds")
    write_media_index(ds, ["a", "b"])
    index = ds.get_root("media_raw") / "index.csv"
    rows = pd.read_csv(index, dtype=str, keep_default_na=False)
    rows.loc[rows["sequence"] == "b", "media_facts"] = ""
    rows.to_csv(index, index=False)

    plan = plan_prerequisites(ds, "trex")

    assert _refused(plan) == [("", "b", "", "unresolved_media")]
    assert "reprobe-media" in _reason(plan)
    assert plan.entries == (("", "a"),)


def test_an_entry_without_media_is_refused_alone(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds")
    write_media_index(ds, _clips())

    plan = plan_prerequisites(ds, "trex", entries=[("", "gone"), _ENTRY])

    assert plan.runs == (_join_run(),)
    assert _refused(plan) == [("", "gone", "", "no_media")]
    assert plan.entries == (_ENTRY,)


def test_without_a_media_index_every_named_entry_is_refused(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds")

    assert plan_prerequisites(ds, "trex") == PrerequisitePlan(
        kind="trex", media="", runs=(), refusals=(), entries=()
    )
    plan = plan_prerequisites(ds, "trex", entries=[_ENTRY])
    assert _refused(plan) == [("", "sess", "", "no_media")]


def test_entries_of_one_key_are_reduced_as_a_tracker_reduces_them(
    tmp_path: Path,
) -> None:
    """``(a, b__c)`` and ``(a__b, c)`` write one output, so the first indexed is read."""
    ds = make_dataset(tmp_path / "ds")
    write_media_index(
        ds,
        [
            MediaClip(group="a", sequence="b__c", filename="x.mp4"),
            MediaClip(group="a__b", sequence="c", filename="y.mp4"),
        ],
    )

    plan = plan_prerequisites(ds, "trex")

    assert _refused(plan) == [("a__b", "c", "", "no_media")]
    assert "shares the output name" in _reason(plan)
    assert plan.entries == (("a", "b__c"),)


# --- the codec variant --------------------------------------------------------


def test_av1_clips_are_read_through_an_h264_variant_by_lightning_pose(
    tmp_path: Path,
) -> None:
    """``preprocess`` reads the clips one at a time, and no join is planned."""
    ds = make_dataset(tmp_path / "ds")
    write_media_index(ds, _clips(codecs=("av1", "av1")))

    plan = plan_prerequisites(ds, "litpose")

    assert plan.media == _CODEC_VARIANT
    assert plan.runs == (_preprocess_run(),)
    assert (plan.refusals, plan.entries) == ((), (_ENTRY,))


@pytest.mark.parametrize(
    ("kind", "codec", "allowed"),
    [("litpose", "h264", ""), ("litpose", "av1", "av1"), ("sleap", "av1", "")],
    ids=["h264", "av1-allowed", "sleap"],
)
def test_no_variant_is_planned_for_a_codec_that_the_tool_may_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str, codec: str, allowed: str
) -> None:
    """SLEAP's decoder is tested in its environment, and a plan runs no test."""
    monkeypatch.setenv("MOSAIC_ALLOW_TOOL_CODECS", allowed)
    ds = make_dataset(tmp_path / "ds")
    write_media_index(ds, _clips(codecs=(codec, codec)))

    plan = plan_prerequisites(ds, kind)

    assert (plan.media, plan.runs, plan.refusals) == ("", (_join_run(),), ())


def test_an_h264_store_is_exported_for_lightning_pose_without_a_variant(
    tmp_path: Path, make_media_dataset: MakeMediaDataset
) -> None:
    """The export copies the H.264 chunks, which Lightning Pose reads."""
    ds = store_dataset(tmp_path, make_media_dataset, video_store_maker([5, 5, 2]))
    (media,) = ds.resolve_media_scope(None)
    entry = (media.group, media.sequence)

    plan = plan_prerequisites(ds, "litpose")

    assert (plan.media, plan.runs) == ("", (_store_run(entry),))
    assert (plan.refusals, plan.entries) == ((), (entry,))


def test_a_store_whose_export_codec_does_not_read_is_refused_for_lightning_pose(
    tmp_path: Path, make_media_dataset: MakeMediaDataset
) -> None:
    """The export's codec is read from the header of the store's first chunk.

    A chunk file that is gone is no chunk of the store, so the first chunk is
    overwritten instead.
    """
    ds = store_dataset(tmp_path, make_media_dataset, video_store_maker([5, 5, 2]))
    (media,) = ds.resolve_media_scope(None)
    (store,) = media.resolved.paths
    _ = (store / "000000.mp4").write_bytes(b"not video")

    plan = plan_prerequisites(ds, "litpose")

    assert _refused(plan) == [(media.group, media.sequence, "", "unresolved_media")]
    assert "000000.mp4" in _reason(plan)
    assert (plan.media, plan.runs, plan.entries) == ("", (), ())


def test_av1_clips_of_two_sizes_are_unjoinable_for_lightning_pose(
    tmp_path: Path,
) -> None:
    """Neither the tool nor ``preprocess`` reads them as one source."""
    ds = make_dataset(tmp_path / "ds")
    write_media_index(ds, _clips(codecs=("av1", "av1"), widths=(640, 1280)))

    plan = plan_prerequisites(ds, "litpose")

    assert _refused(plan) == [("", "sess", "", "unjoinable")]
    assert (plan.media, plan.runs, plan.entries) == ("", (), ())


def test_the_codec_variant_refuses_clips_of_two_sizes_beside_a_readable_entry(
    tmp_path: Path,
) -> None:
    """``media`` names one variant for the whole run, which every entry reads."""
    ds = make_dataset(tmp_path / "ds")
    av1 = ("av1", "av1")
    write_media_index(
        ds, [*_clips("a", codecs=av1), *_clips("b", codecs=av1, widths=(640, 1280))]
    )

    plan = plan_prerequisites(ds, "litpose")

    assert plan.media == _CODEC_VARIANT
    assert plan.runs == (_preprocess_run((("", "a"),)),)
    assert _refused(plan) == [("", "b", "", "unjoinable")]
    assert "has width 1280" in _reason(plan)
    assert plan.entries == (("", "a"),)


def test_an_h264_entry_reads_the_codec_variant_beside_an_av1_entry(
    tmp_path: Path,
) -> None:
    """``media`` names one variant for the whole run, which every entry reads."""
    ds = make_dataset(tmp_path / "ds")
    write_media_index(ds, [*_clips("a", codecs=("av1", "av1")), *_clips("h")])

    plan = plan_prerequisites(ds, "litpose")

    assert plan.media == _CODEC_VARIANT
    assert plan.runs == (_preprocess_run((("", "a"), ("", "h"))),)
    assert (plan.refusals, plan.entries) == ((), (("", "a"), ("", "h")))


def test_an_av1_entry_that_the_codec_variant_does_not_fit_is_refused_in_any_scope(
    tmp_path: Path,
) -> None:
    """Lightning Pose reads no AV1, and a variant of an odd frame width is not written."""
    ds = make_dataset(tmp_path / "ds")
    write_media_index(
        ds,
        [
            MediaClip(sequence="a", filename="a.mp4", codec="av1", frame_count=12),
            MediaClip(
                sequence="b", filename="b.mp4", codec="av1", width=641, frame_count=12
            ),
        ],
    )

    every = plan_prerequisites(ds, "litpose")
    alone = plan_prerequisites(ds, "litpose", entries=[("", "b")])

    assert every.runs == (_preprocess_run((("", "a"),)),)
    assert _refused(every) == _refused(alone) == [("", "b", "", "preprocess_refused")]
    assert "av1" in _reason(alone)
    assert "641x480" in _reason(alone)
    assert (alone.media, alone.runs, alone.entries) == ("", (), ())


def test_the_variant_run_of_a_nameless_entry_is_scoped_by_its_indexed_key(
    tmp_path: Path,
) -> None:
    """The run writes the file under the entry's label, where the tool looks it up."""
    ds = make_dataset(tmp_path / "ds")
    write_media_index(
        ds,
        [
            MediaClip(sequence="", filename="rec.mp4", codec="av1", frame_count=12),
        ],
    )

    plan = plan_prerequisites(ds, "litpose")

    assert plan.runs == (_preprocess_run((("", ""),)),)
    assert plan.entries == (("", "rec"),)
    _ = add_media_variant(ds, _CODEC_VARIANT, "rec")
    assert plan_prerequisites(ds, "litpose").runs == ()


# --- named variants -------------------------------------------------------------


def test_a_variant_that_lacks_an_entry_is_planned_from_its_recorded_recipe(
    tmp_path: Path,
) -> None:
    """A file written from other media than the entry's is written again."""
    ds = make_dataset(tmp_path / "ds")
    write_media_index(
        ds,
        [
            MediaClip(sequence="current", filename="a.mp4", frame_count=12),
            MediaClip(sequence="drifted", filename="b.mp4", frame_count=20),
            MediaClip(sequence="lacking", filename="c.mp4", frame_count=12),
        ],
    )
    variant = _record(ds, _GRAYSCALE)
    for sequence in ("current", "drifted"):
        _ = add_media_variant(ds, variant, sequence)

    plan = plan_prerequisites(ds, "trex", {"media": variant})

    assert plan.media == variant
    assert plan.runs == (
        _preprocess_run((("", "drifted"), ("", "lacking")), _GRAYSCALE),
    )
    assert plan.refusals == ()
    assert plan.entries == (("", "current"), ("", "drifted"), ("", "lacking"))


@pytest.mark.parametrize(
    ("recorded", "says"),
    [
        (None, "records no readable recipe"),
        (b'{"steps": 3}', "does not validate"),
        (b'{"steps": [\xff]}', "does not validate"),
        (_GRAYSCALE_AT_20.model_dump_json().encode(), "no longer"),
    ],
    ids=["absent", "invalid", "not-utf-8", "stale"],
)
def test_a_variant_without_a_recipe_that_makes_it_is_refused(
    tmp_path: Path, recorded: bytes | None, says: str
) -> None:
    """A recipe that hashes to another variant, as after a version bump, is stale."""
    ds = make_dataset(tmp_path / "ds")
    write_media_index(ds, [MediaClip(frame_count=12)])
    variant = preprocess_identity(_GRAYSCALE).run_id
    if recorded is not None:
        recipe = media_variant_recipe_path(ds, variant)
        recipe.parent.mkdir(parents=True)
        _ = recipe.write_bytes(recorded)

    plan = plan_prerequisites(ds, "trex", {"media": variant})

    assert _refused(plan) == [("", "sess", "", "no_variant_recipe")]
    assert says in _reason(plan)
    assert (plan.media, plan.runs, plan.entries) == (variant, (), ())


def test_the_codec_variant_is_planned_without_a_recipe(tmp_path: Path) -> None:
    """A caller names the codec variant before its first run records one."""
    ds = make_dataset(tmp_path / "ds")
    write_media_index(ds, [MediaClip(frame_count=12)])

    plan = plan_prerequisites(ds, "trex", {"media": _CODEC_VARIANT})

    assert (plan.media, plan.runs) == (_CODEC_VARIANT, (_preprocess_run(),))
    assert (plan.refusals, plan.entries) == ((), (_ENTRY,))


@pytest.mark.parametrize("source", ["recipe", "rows"])
def test_an_av1_variant_is_refused_to_lightning_pose(
    tmp_path: Path, source: str
) -> None:
    """The codec comes from the recipe, or from the files of a variant without one."""
    ds = make_dataset(tmp_path / "ds")
    write_media_index(ds, [MediaClip(frame_count=12)])
    variant = preprocess_identity(_GRAYSCALE).run_id
    if source == "recipe":
        _ = _record(ds, _GRAYSCALE)
    else:
        _ = add_media_variant(ds, variant, "sess")

    plan = plan_prerequisites(ds, "litpose", {"media": variant})

    assert _refused(plan) == [("", "sess", "", "undecodable_codec")]
    assert "av1" in _reason(plan)
    assert (plan.runs, plan.entries) == ((), ())
    assert plan_prerequisites(ds, "trex", {"media": variant}).refusals == ()


@pytest.mark.parametrize("drifted", [False, True], ids=["lacking", "drifted"])
def test_a_chained_variant_is_planned_after_the_upstream_that_it_reads(
    tmp_path: Path, drifted: bool
) -> None:
    """The upstream lacks the entry, or has a file written from other media.

    The plan reads both variants' rows and recipes, and writes nothing.
    """
    ds = make_dataset(tmp_path / "ds")
    write_media_index(ds, [MediaClip(frame_count=20)])
    upstream = _record(ds, _GRAYSCALE)
    if drifted:
        _ = add_media_variant(ds, upstream, "sess")
    chained = PreprocessParams.model_validate(
        {"steps": [{"step": "decimate", "every": 2}], "media": upstream}
    )
    variant = _record(ds, chained)
    before = tree_snapshot(ds.base_dir)

    plan = plan_prerequisites(ds, "trex", {"media": variant})

    assert plan.runs == (
        _preprocess_run(recipe=_GRAYSCALE),
        _preprocess_run(recipe=chained, after=(0,)),
    )
    assert (plan.refusals, plan.entries) == ((), (_ENTRY,))
    assert_tree_unchanged(ds.base_dir, before)


def test_frames_of_an_odd_size_are_refused_a_variant(tmp_path: Path) -> None:
    """A variant is encoded as yuv420p, whose frame sides are even."""
    ds = make_dataset(tmp_path / "ds")
    write_media_index(ds, [MediaClip(width=641, frame_count=12)])

    plan = plan_prerequisites(ds, "trex", {"media": _CODEC_VARIANT})

    assert _refused(plan) == [("", "sess", "", "preprocess_refused")]
    assert "641x480" in _reason(plan)
    assert (plan.runs, plan.entries) == ((), ())


# --- preprocess ---------------------------------------------------------------


def test_preprocess_needs_nothing_for_an_entry_that_its_recipe_fits(
    tmp_path: Path,
) -> None:
    ds = make_dataset(tmp_path / "ds")
    write_media_index(
        ds,
        [
            MediaClip(sequence="even", filename="a.mp4"),
            MediaClip(sequence="odd", filename="b.mp4", width=641),
        ],
    )

    plan = plan_prerequisites(ds, "preprocess", {"steps": [], "codec": "h264"})

    assert (plan.media, plan.runs) == ("", ())
    assert _refused(plan) == [("", "odd", "", "preprocess_refused")]
    assert plan.entries == (("", "even"),)


def test_preprocess_over_a_variant_plans_the_variant_first(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds")
    write_media_index(ds, [MediaClip(frame_count=12)])
    recipe = {"steps": [{"step": "grayscale"}], "media": _CODEC_VARIANT}

    plan = plan_prerequisites(ds, "preprocess", recipe)

    assert (plan.media, plan.runs) == (_CODEC_VARIANT, (_preprocess_run(),))
    assert (plan.refusals, plan.entries) == ((), (_ENTRY,))


def test_preprocess_params_that_do_not_validate_are_refused(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds")

    with pytest.raises(ValidationError, match="steps"):
        _ = plan_prerequisites(ds, "preprocess")


# --- the call -----------------------------------------------------------------


def test_entries_narrow_the_plan(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds")
    write_media_index(
        ds,
        [*_clips("a"), *_clips("b"), MediaClip(sequence="c", filename="c.mp4")],
    )

    assert plan_prerequisites(ds, "trex").runs == (
        _join_run(("", "a")),
        _join_run(("", "b")),
    )
    plan = plan_prerequisites(ds, "trex", entries=[("", "c"), ("", "b")])
    assert plan.runs == (_join_run(("", "b")),)
    assert plan.entries == (("", "b"), ("", "c"))


def test_runs_are_ordered_by_kind_then_by_entry(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds")
    write_media_index(
        ds,
        [MediaClip(sequence="a", filename="a.mp4", video_uuid="uid-a"), *_clips("b")],
    )
    _ = point_at_a_store(ds, "a", ds.get_root("media_raw") / "a.store")

    plan = plan_prerequisites(ds, "trex")

    assert plan.runs == (_join_run(("", "b")), _store_run(("", "a")))


def test_the_plan_is_the_same_for_entries_in_any_order(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds")
    write_media_index(
        ds,
        [
            *_clips("a"),
            *_clips("b", widths=(640, 1280)),
            *_clips("c", codecs=("h264", "av1")),
            MediaClip(sequence="d", filename="d.mp4"),
        ],
    )
    entries: list[Entry] = [("", "a"), ("", "b"), ("", "c"), ("", "d"), ("", "e")]

    forward = plan_prerequisites(ds, "trex", entries=entries)

    assert plan_prerequisites(ds, "trex", entries=entries[::-1]) == forward
    assert [run.entries for run in forward.runs] == [(("", "a"),), (("", "c"),)]
    assert _refused(forward) == [
        ("", "b", "", "unjoinable"),
        ("", "e", "", "no_media"),
    ]


@pytest.mark.parametrize("kind", ["trex-convert", "export-joined", "speed"])
def test_a_kind_without_an_answer_is_refused(tmp_path: Path, kind: str) -> None:
    ds = make_dataset(tmp_path / "ds")

    with pytest.raises(ValueError, match="extract-frames, infer-localizer"):
        _ = plan_prerequisites(ds, kind)


_VARIANT = "preprocess.0.1-0123456789"


@pytest.mark.parametrize(
    "params",
    [
        {"media": _VARIANT},
        UltralyticsParams.model_validate({"model_path": "best.pt", "media": _VARIANT}),
    ],
    ids=["mapping", "model"],
)
def test_the_media_variant_of_the_params_is_the_one_read(
    tmp_path: Path, params: Mapping[str, object] | UltralyticsParams
) -> None:
    ds = make_dataset(tmp_path / "ds")
    write_media_index(ds, _clips())

    plan = plan_prerequisites(ds, "ultralytics", params)

    assert plan.media == _VARIANT
    assert _refused(plan) == [("", "sess", "", "no_variant_recipe")]


def test_a_media_that_is_not_a_run_id_is_refused(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds")

    with pytest.raises(ValueError, match="media names a media variant"):
        _ = plan_prerequisites(ds, "trex", {"media": 3})
