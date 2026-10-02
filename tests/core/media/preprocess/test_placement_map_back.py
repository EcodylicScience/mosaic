"""Test mapping a table read from a media variant back into its source space.

A tracker run on a variant reports positions in the variant's pixels and frames
in the variant's frame axis. ``to_source_space`` shifts the positions by the
variant's offset, maps each frame through its frame map, retimes the table on
the source timeline, and drops the columns that the mapping cannot correct. The
tests build a table in source space, derive a tracker's report on the variant by
hand, and check that the mapping recovers the source table.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from mosaic_media import MediaFacts

from mosaic.core.media.facts_columns import store_facts
from mosaic.core.media.preprocess import (
    CropStep,
    DecimateStep,
    MediaStep,
    Placement,
    TrimStep,
)
from mosaic.core.media.timeline import ConcatenatedTimeline, concatenated_timeline
from mosaic.core.pipeline.placement import (
    RATE_DEPENDENT_BASES,
    EntryAxis,
    SourceMapping,
    UnclassifiedColumnError,
    classify_column,
    to_source_space,
)
from mosaic.core.track_converter import EntryHints
from mosaic.core.track_library.deeplabcut import DlcConverter, DlcParams
from mosaic.core.track_library.helpers import column_array, column_names
from mosaic.core.track_library.sleap import (
    SleapAnalysisH5Converter,
    SleapConvertParams,
)
from mosaic.core.track_library.trex import (
    DIMENSIONLESS_FIELDS,
    LENGTH_FIELDS,
    PIXEL_PREFIXES,
)
from mosaic.core.track_library.ultralytics_tracks import (
    UltralyticsTracksConverter,
    UltralyticsTracksParams,
    raw_columns,
)
from mosaic.tracking.external.runner.ultralytics_protocol import (
    POINT_COLUMNS,
    pose_columns,
)
from mosaic.tracking.pose_training.localizer_inference import (
    LocalizerDetection,
    LocalizerFrame,
    localizer_detections_to_dataframe,
)

from tests.helpers import clip_facts, write_dlc_csv, write_sleap_analysis_h5

_WIDTH, _HEIGHT = 640, 480
_CLIP = 300


def _clip(fps: float, frame_count: int = _CLIP) -> MediaFacts:
    return clip_facts(fps=fps, frame_count=frame_count, width=_WIDTH, height=_HEIGHT)


_ONE_CLIP = concatenated_timeline([_clip(30.0, 2 * _CLIP)])
_TWO_CLIPS = concatenated_timeline([_clip(30.0), _clip(30.0)])
_TWO_RATES = concatenated_timeline([_clip(30.0), _clip(31.0)])


def _mapping(
    timeline: ConcatenatedTimeline,
    steps: list[MediaStep],
    *,
    fps: float = 30.0,
) -> SourceMapping:
    """Return the mapping of a variant built by *steps*, labeled at *fps*."""
    placement = Placement.identity(_WIDTH, _HEIGHT, timeline.total_frames, 30.0)
    for step in steps:
        placement = step.place(placement)
    return SourceMapping(dataclasses.replace(placement, fps=fps), timeline)


def _source_table(frame_count: int) -> pd.DataFrame:
    """Return a table in source space: one individual on every source frame."""
    frames = np.arange(frame_count, dtype=np.int64)
    x = 100.0 + 0.5 * frames
    x[7] = np.nan
    y = 50.0 + 0.25 * frames
    return pd.DataFrame(
        {
            "frame": frames,
            "id": np.zeros(frame_count, dtype=np.int64),
            "X": x,
            "Y": y,
            "poseX0": x + 3.0,
            "poseY0": y - 2.0,
            "poseP0": np.full(frame_count, 0.9),
            "bbox_x1": x - 10.0,
            "bbox_y1": y - 8.0,
            "bbox_x2": x + 10.0,
            "bbox_y2": y + 8.0,
            "ANGLE": np.full(frame_count, 0.25),
            "group": "g",
            "sequence": "s",
        }
    )


def _as_tracked_on(source: pd.DataFrame, mapping: SourceMapping) -> pd.DataFrame:
    """Return a tracker's report on the variant, the hand-built inverse of map-back."""
    placement = mapping.placement
    kept = placement.frames.start + placement.frames.step * np.arange(
        placement.frames.count, dtype=np.int64
    )
    variant = source.iloc[kept].reset_index(drop=True)
    variant["frame"] = np.arange(placement.frames.count, dtype=np.int64)
    variant["time"] = column_array(variant, "frame") / placement.fps
    for name in ("X", "poseX0", "bbox_x1", "bbox_x2"):
        variant[name] = column_array(variant, name) - placement.offset_x
    for name in ("Y", "poseY0", "bbox_y1", "bbox_y2"):
        variant[name] = column_array(variant, name) - placement.offset_y
    return variant


# --- round trip --------------------------------------------------------------


@pytest.mark.parametrize(
    "steps",
    [
        [CropStep(x=120, y=40, width=320, height=240)],
        [TrimStep(start=100, stop=400)],
        [DecimateStep(every=3)],
        [
            CropStep(x=120, y=40, width=320, height=240),
            TrimStep(start=100, stop=400),
            DecimateStep(every=3),
        ],
    ],
    ids=["offset", "trim", "decimate", "composition"],
)
def test_a_table_maps_back_to_the_source_it_was_tracked_from(
    steps: list[MediaStep],
) -> None:
    mapping = _mapping(_ONE_CLIP, steps)
    source = _source_table(_ONE_CLIP.total_frames)
    kept = mapping.placement.frames.source_frames(
        np.arange(mapping.placement.frames.count, dtype=np.int64)
    )

    mapped = to_source_space(_as_tracked_on(source, mapping), mapping).frame

    expected = source.iloc[kept].reset_index(drop=True)
    pd.testing.assert_frame_equal(mapped[column_names(expected)], expected)
    assert np.allclose(column_array(mapped, "time"), kept / 30.0)


def test_a_position_keeps_its_nan() -> None:
    mapping = _mapping(_ONE_CLIP, [CropStep(x=120, y=40, width=320, height=240)])
    tracked = _as_tracked_on(_source_table(_ONE_CLIP.total_frames), mapping)

    mapped = to_source_space(tracked, mapping).frame

    assert np.isnan(column_array(mapped, "X")[7])
    assert column_array(mapped, "X").dtype == np.float64


def test_a_frame_past_the_variant_is_still_mapped() -> None:
    """The frame map is applied without a check of the tool's frame count."""
    mapping = _mapping(
        _ONE_CLIP, [TrimStep(start=100, stop=400), DecimateStep(every=3)]
    )
    count = mapping.placement.frames.count
    tracked = pd.DataFrame(
        {"frame": [count - 1, count], "frames": [count - 1, count], "X": [1.0, 2.0]}
    )

    mapped = to_source_space(tracked, mapping).frame

    assert column_array(mapped, "frame").tolist() == [
        100 + 3 * (count - 1),
        100 + 3 * count,
    ]
    assert np.array_equal(column_array(mapped, "frames"), column_array(mapped, "frame"))


def test_a_whole_number_frame_stored_as_a_float_is_mapped() -> None:
    """TRex writes frame numbers as floats, and a whole one is a frame number."""
    mapping = _mapping(_ONE_CLIP, [TrimStep(start=100, stop=400)])
    tracked = pd.DataFrame({"frame": [0.0, 1.0, 2.0], "X": [1.0, 2.0, 3.0]})

    mapped = to_source_space(tracked, mapping).frame

    assert column_array(mapped, "frame").tolist() == [100, 101, 102]


@pytest.mark.parametrize(
    ("name", "values", "count"),
    [
        ("frame", [0.0, np.nan, 2.0], 1),
        ("frame", [0.5, 1.0, 2.0], 1),
        ("frames", [np.inf, 1.5, 2.0], 2),
        ("tracklet_start", [0.0, np.nan, 2.0], 1),
    ],
    ids=["nan", "fractional", "frames", "float-tracklet-start"],
)
def test_a_frame_that_is_not_a_whole_number_refuses_the_table(
    name: str, values: list[float], count: int
) -> None:
    """NaN and 0.5 are refused, because casting turns each into a plausible frame."""
    mapping = _mapping(_ONE_CLIP, [TrimStep(start=100, stop=400)])
    tracked = pd.DataFrame({"frame": [0, 1, 2], "X": [1.0, 2.0, 3.0]})
    tracked[name] = values

    with pytest.raises(ValueError, match=rf"'{name}'.* {count} of its 3 values"):
        _ = to_source_space(tracked, mapping)


def test_a_tracklet_start_maps_like_a_frame_and_keeps_its_na() -> None:
    """TRex's tracklet key is a frame number, and NA there means no tracklet.

    A nullable integer column says so explicitly, unlike a float NaN, which may be
    padding. So its present values are mapped as ``frame`` is and NA stays NA.
    """
    mapping = _mapping(
        _ONE_CLIP, [TrimStep(start=100, stop=400), DecimateStep(every=3)]
    )
    tracked = pd.DataFrame(
        {
            "frame": np.arange(4, dtype=np.int64),
            "tracklet_start": pd.array([0, 0, None, 3], dtype="Int64"),
            "X": [1.0, 2.0, 3.0, 4.0],
        }
    )

    mapped = to_source_space(tracked, mapping).frame

    assert mapped["tracklet_start"].dtype == "Int64"
    assert mapped["tracklet_start"].tolist() == [100, 100, pd.NA, 109]
    assert column_array(mapped, "frame").tolist() == [100, 103, 106, 109]


# --- time and frame rate -----------------------------------------------------


def _rate_table(frames: npt.NDArray[np.int64], fps: float) -> pd.DataFrame:
    """Return a table timed as a tool times it, by frame index over one rate."""
    return pd.DataFrame(
        {
            "frame": frames,
            "time": frames / fps,
            "frame_rate": np.full(len(frames), fps),
            "X": np.zeros(len(frames)),
        }
    )


def test_a_trim_of_a_two_rate_timeline_is_timed_by_each_clip() -> None:
    mapping = _mapping(_TWO_RATES, [TrimStep(start=250, stop=350)])
    tracked = _rate_table(np.arange(100, dtype=np.int64), 30.0)

    mapped = to_source_space(tracked, mapping).frame

    frames = column_array(mapped, "frame")
    assert frames.tolist() == list(range(250, 350))
    expected_time = np.where(frames < 300, frames / 30.0, 10.0 + (frames - 300) / 31.0)
    assert np.allclose(column_array(mapped, "time"), expected_time)
    assert np.array_equal(
        column_array(mapped, "frame_rate"), np.where(frames < 300, 30.0, 31.0)
    )


def test_a_crop_of_a_two_rate_timeline_is_retimed() -> None:
    """The frame map is the identity, and the timeline's second clip is not at 30."""
    mapping = _mapping(_TWO_RATES, [CropStep(x=120, y=40, width=320, height=240)])
    tracked = _rate_table(np.arange(2 * _CLIP, dtype=np.int64), 30.0)

    mapped = to_source_space(tracked, mapping).frame

    assert column_array(mapped, "frame_rate")[_CLIP] == 31.0
    assert column_array(mapped, "time")[_CLIP + 31] == pytest.approx(11.0)


def test_a_crop_of_a_two_clip_timeline_is_retimed_and_keeps_its_rates() -> None:
    mapping = _mapping(_TWO_CLIPS, [CropStep(x=120, y=40, width=320, height=240)])
    tracked = _rate_table(np.arange(2 * _CLIP, dtype=np.int64), 30.0)
    tracked["timestamp"] = column_array(tracked, "frame") * (1e6 / 30.0)
    tracked["SPEED"] = 1.0

    result = to_source_space(tracked, mapping)

    assert result.dropped == ("timestamp",)
    assert np.allclose(column_array(result.frame, "time"), np.arange(2 * _CLIP) / 30.0)


# --- drop rules --------------------------------------------------------------


def _kinematic_table(frame_count: int) -> pd.DataFrame:
    frames = np.arange(frame_count, dtype=np.int64)
    return pd.DataFrame(
        {
            "frame": frames,
            "time": frames / 30.0,
            "timestamp": frames * (1e6 / 30.0),
            "X": np.zeros(frame_count),
            "Y": np.zeros(frame_count),
            "SPEED#wcentroid": np.ones(frame_count),
            "BORDER_DISTANCE": np.ones(frame_count),
            "VX": np.ones(frame_count),
            "video_size_0": np.full(frame_count, float(_WIDTH)),
            "video_size_1": np.full(frame_count, float(_HEIGHT)),
            "ACCELERATION": np.ones(frame_count),
        }
    )


def test_a_relabeled_rate_drops_the_rate_dependent_columns() -> None:
    mapping = _mapping(_ONE_CLIP, [], fps=15.0)
    tracked = _kinematic_table(_ONE_CLIP.total_frames)
    tracked["time"] = column_array(tracked, "frame") / 15.0

    result = to_source_space(tracked, mapping)

    assert result.dropped == ("timestamp", "SPEED#wcentroid", "VX", "ACCELERATION")
    assert np.allclose(
        column_array(result.frame, "time"), np.arange(_ONE_CLIP.total_frames) / 30.0
    )


def test_a_two_rate_timeline_drops_the_rate_dependent_columns() -> None:
    mapping = _mapping(_TWO_RATES, [TrimStep(start=0, stop=400)])

    result = to_source_space(_kinematic_table(400), mapping)

    assert result.dropped == ("timestamp", "SPEED#wcentroid", "VX", "ACCELERATION")


_JITTER_FRAMES = 8_000
"""Long enough that 30.0 and 30.002 fps place a frame more than half a frame apart."""


def _jittered(clip: Callable[[float], MediaFacts]) -> list[MediaFacts]:
    """Two recordings of one rate, estimated at 30.0 and 30.002 fps."""
    return [clip(30.0), clip(30.002)]


def _store(fps: float) -> MediaFacts:
    duration = _JITTER_FRAMES / fps
    return store_facts(_WIDTH, _HEIGHT, fps, _JITTER_FRAMES, "h264", duration, "", "")


def _plain(fps: float) -> MediaFacts:
    return _clip(fps, _JITTER_FRAMES)


def test_stores_of_one_rate_measured_twice_keep_the_rate_dependent_columns() -> None:
    """The store rule reads them as one rate, as the store reader does.

    Plain clips at the same two rates drift apart by more than half a frame over
    their length, which the plain rule reads as two rates.
    """
    trim: list[MediaStep] = [TrimStep(start=0, stop=400)]
    stores = _mapping(concatenated_timeline(_jittered(_store)), trim)
    plain = _mapping(concatenated_timeline(_jittered(_plain)), trim)

    assert stores.true_rate == 30.0
    assert to_source_space(_kinematic_table(400), stores).dropped == ("timestamp",)
    assert plain.true_rate is None
    assert to_source_space(_kinematic_table(400), plain).dropped == (
        "timestamp",
        "SPEED#wcentroid",
        "VX",
        "ACCELERATION",
    )


def test_stores_of_one_rate_read_as_they_are_keep_the_rate_dependent_columns() -> None:
    """A tool that reads the stores themselves, as the localizer does."""
    frames = 2 * _JITTER_FRAMES
    stores = EntryAxis.of_entry_media(_jittered(_store), windowed=False)
    plain = EntryAxis.of_entry_media(_jittered(_plain), windowed=False)

    assert stores.place(_kinematic_table(frames)).dropped == ("timestamp",)
    assert plain.place(_kinematic_table(frames)).dropped == (
        "timestamp",
        "SPEED#wcentroid",
        "VX",
        "ACCELERATION",
    )


def test_a_decimation_labeled_at_its_true_rate_keeps_the_rate_dependent_columns() -> (
    None
):
    mapping = _mapping(_ONE_CLIP, [DecimateStep(every=2)], fps=15.0)

    result = to_source_space(_kinematic_table(_CLIP), mapping)

    assert result.dropped == ("timestamp",)


def test_every_per_second_trex_field_is_rate_dependent() -> None:
    """TRex's output annotations give each of these in cm/s or cm/s2."""
    assert RATE_DEPENDENT_BASES == {
        "VX",
        "VY",
        "AX",
        "AY",
        "SPEED",
        "SPEED_SMOOTH",
        "SPEED_OLD",
        "ACCELERATION",
        "ACCELERATION_SMOOTH",
        "ANGULAR_V",
        "ANGULAR_A",
    }


def test_a_crop_drops_the_extent_dependent_columns_and_a_trim_does_not() -> None:
    cropped = _mapping(_ONE_CLIP, [CropStep(x=120, y=40, width=320, height=240)])
    trimmed = _mapping(_ONE_CLIP, [TrimStep(start=100, stop=400)])

    crop_dropped = to_source_space(_kinematic_table(600), cropped).dropped
    trim_dropped = to_source_space(_kinematic_table(300), trimmed).dropped

    assert crop_dropped == ("BORDER_DISTANCE", "video_size_0", "video_size_1")
    assert trim_dropped == ("timestamp",)


def test_every_rule_drops_in_table_order() -> None:
    mapping = _mapping(
        _ONE_CLIP,
        [CropStep(x=120, y=40, width=320, height=240), TrimStep(start=0, stop=300)],
        fps=25.0,
    )

    result = to_source_space(_kinematic_table(300), mapping)

    assert result.dropped == (
        "timestamp",
        "SPEED#wcentroid",
        "BORDER_DISTANCE",
        "VX",
        "video_size_0",
        "video_size_1",
        "ACCELERATION",
    )
    assert not set(result.dropped) & set(column_names(result.frame))


# --- refusal and identity ----------------------------------------------------


def test_an_unclassified_numeric_column_refuses_the_table_naming_every_one() -> None:
    mapping = _mapping(_ONE_CLIP, [TrimStep(start=0, stop=10)])
    tracked = pd.DataFrame(
        {
            "frame": np.arange(10, dtype=np.int64),
            "X": np.zeros(10),
            "wingbeat": np.ones(10),
            "label": ["a"] * 10,
            "flagged": np.zeros(10, dtype=bool),
            "blob_count": np.ones(10, dtype=np.int64),
        }
    )

    with pytest.raises(UnclassifiedColumnError) as refusal:
        _ = to_source_space(tracked, mapping)

    message = str(refusal.value)
    assert "'wingbeat'" in message
    assert "'blob_count'" in message
    assert "'label'" not in message
    assert "'flagged'" not in message
    assert "mosaic.core.pipeline.placement" in message


def test_a_string_and_a_categorical_column_pass_through() -> None:
    mapping = _mapping(_ONE_CLIP, [TrimStep(start=0, stop=10)])
    tracked = pd.DataFrame(
        {
            "frame": np.arange(10, dtype=np.int64),
            "class_name": ["bee"] * 10,
            "caste": pd.Categorical([1, 2] * 5),
        }
    )

    mapped = to_source_space(tracked, mapping).frame

    assert column_array(mapped, "class_name").tolist() == ["bee"] * 10
    assert column_array(mapped, "caste").tolist() == [1, 2] * 5


def test_an_identity_mapping_returns_the_table_itself() -> None:
    mapping = _mapping(_ONE_CLIP, [])
    tracked = pd.DataFrame({"frame": [0, 1], "wingbeat": [1.0, 2.0]})

    result = to_source_space(tracked, mapping)

    assert mapping.is_identity
    assert result.frame is tracked
    assert result.dropped == ()


@pytest.mark.parametrize(
    ("timeline", "fps"),
    [(_ONE_CLIP, 15.0), (_TWO_CLIPS, 30.0)],
    ids=["relabeled", "two-clips"],
)
def test_an_identity_placement_is_not_an_identity_mapping(
    timeline: ConcatenatedTimeline, fps: float
) -> None:
    mapping = _mapping(timeline, [], fps=fps)

    assert mapping.placement.is_identity
    assert not mapping.is_identity


def test_a_mapping_refuses_a_timeline_of_another_length() -> None:
    placement = Placement.identity(_WIDTH, _HEIGHT, 100, 30.0)

    with pytest.raises(ValueError, match="600"):
        _ = SourceMapping(placement, _ONE_CLIP)


# --- classification completeness ---------------------------------------------

_POSE_NAMES = ("poseX0", "poseY0", "poseP0", "poseX12", "poseY12", "poseP12")


def test_every_pixel_prefix_trex_knows_names_a_pose_column() -> None:
    """TRex emits poseX and poseY columns, and other producers add poseP."""
    assert all(
        any(name.startswith(prefix) for name in _POSE_NAMES)
        for prefix in PIXEL_PREFIXES
    )


@pytest.mark.parametrize(
    "name",
    [
        *sorted(LENGTH_FIELDS | DIMENSIONLESS_FIELDS),
        *_POSE_NAMES,
        "X#wcentroid",
        "X#head",
        "SPEED#pcentroid",
        "midline_x_3",
        "video_size_1",
    ],
)
def test_every_trex_field_is_classified(name: str) -> None:
    assert classify_column(name) is not None


def _crop_and_trim(frame_count: int) -> SourceMapping:
    timeline = concatenated_timeline([_clip(30.0, frame_count)])
    return _mapping(
        timeline,
        [CropStep(x=120, y=40, width=320, height=240), TrimStep(start=0, stop=2)],
    )


def _assert_maps(table: pd.DataFrame, *, labels: tuple[str, ...] = ()) -> None:
    """Every column of *table* but the text *labels* is classified, and it maps."""
    unclassified = [
        name for name in column_names(table) if classify_column(name) is None
    ]
    mapped = to_source_space(table, _crop_and_trim(10)).frame

    assert unclassified == list(labels)
    assert len(mapped) == len(table)


def test_every_ultralytics_column_is_classified(tmp_path: Path) -> None:
    columns = raw_columns(2)
    raw = pd.DataFrame(
        {name: np.array([0, 1], dtype=np.float64) for name in columns}
    ).astype({"frame": "int64", "track_id": "int64", "cls": "int64"})
    path = tmp_path / "predictions.parquet"
    raw.to_parquet(path)

    table = UltralyticsTracksConverter().convert(
        path, UltralyticsTracksParams(), EntryHints(group="g", sequence="s")
    )

    assert "source_track_id" in column_names(table)
    _assert_maps(table)


def test_every_sleap_column_is_classified(tmp_path: Path) -> None:
    tracks = np.arange(2 * 2 * 3 * 2, dtype=np.float64).reshape(2, 2, 3, 2)
    scores = np.full((2, 2, 3), 0.9)
    path = tmp_path / "predictions.analysis.h5"
    write_sleap_analysis_h5(path, tracks, scores)

    table = SleapAnalysisH5Converter().convert(
        path, SleapConvertParams(), EntryHints(group="g", sequence="s")
    )

    assert {"poseX2", "poseY2", "poseP2"} <= set(column_names(table))
    _assert_maps(table)


def test_every_deeplabcut_and_lightning_pose_column_is_classified(
    tmp_path: Path,
) -> None:
    path = tmp_path / "predictions.csv"
    _ = write_dlc_csv(path, ["nose", "tail"], n_frames=2)

    table = DlcConverter().convert(
        path, DlcParams(), EntryHints(group="g", sequence="s")
    )

    assert {"poseX1", "poseY1", "poseP1"} <= set(column_names(table))
    _assert_maps(table)


def _bridged(table: pd.DataFrame) -> pd.DataFrame:
    """Return *table* with the columns that the inference bridge adds to it."""
    named = table.rename(columns={"x": "X", "y": "Y"})
    named["group"] = "g"
    named["sequence"] = "s"
    named["time"] = column_array(named, "frame").astype(np.float64)
    if "id" not in column_names(named):
        named["id"] = 0
    return named


def test_every_pose_inference_column_is_classified() -> None:
    columns = pose_columns(2)
    table = pd.DataFrame({name: np.array([0, 1]) for name in columns})
    table["X"] = 1.0
    table["Y"] = 1.0

    _assert_maps(_bridged(table))


def test_every_point_inference_column_is_classified() -> None:
    table = pd.DataFrame({name: np.array([0, 1]) for name in POINT_COLUMNS})
    table["class_name"] = ["bee", "bee"]

    _assert_maps(_bridged(table), labels=("class_name",))


def test_every_localizer_column_is_classified() -> None:
    detection: LocalizerDetection = {
        "x": 4.0,
        "y": 5.0,
        "confidence": 0.9,
        "class_id": 0,
    }
    table = localizer_detections_to_dataframe(
        [LocalizerFrame(0, (detection,)), LocalizerFrame(1, (detection,))]
    )

    _assert_maps(_bridged(table), labels=("class_name",))
