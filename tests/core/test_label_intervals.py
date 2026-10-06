"""Segment labels keep what a per-frame array cannot: overlap, nesting, and bounds.

The conversion from annotated seconds to frames is the midpoint rule, chosen so
that back-to-back segments tile -- no frame in both, none in neither -- at any
frame rate. Painting segments onto frames is a policy the caller names, with a
documented winner for every overlap and every tie. And the sidecar that carries
segments in a converted label file is invisible to every reader that predates it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# Both helpers take an untyped argument, so their signatures are partly Unknown.
from mosaic.core.helpers import (
    detect_label_format,  # pyright: ignore[reportUnknownVariableType]
    load_labels_for_feature_frames,  # pyright: ignore[reportUnknownVariableType]
)
from mosaic.core.label_intervals import (
    OVERLAP_POLICIES,
    SEG_END_FRAME,
    SEG_LABEL,
    SEG_START_FRAME,
    LabelSegments,
    OverlapPolicy,
    SegmentOverlapError,
    intervals_to_dense,
    label_vocabulary,
    seconds_to_frames,
    segments_from_npz,
)
from mosaic.core.pipeline._utils import atomic_savez

# --- seconds_to_frames -------------------------------------------------------


def test_a_frame_belongs_to_the_segment_holding_its_midpoint() -> None:
    """At 10 fps frame 2's midpoint is 0.25 s, which opens the second segment."""
    start, end = seconds_to_frames([0.0, 0.25], [0.25, 1.0], fps=10.0)
    assert start.tolist() == [0, 2]
    assert end.tolist() == [2, 10]
    assert start.dtype == np.int64
    assert end.dtype == np.int64


@pytest.mark.parametrize("fps", [10.0, 25.0, 29.97, 30.0, 59.94, 120.0])
def test_back_to_back_segments_tile_without_gap_or_overlap(fps: float) -> None:
    rng = np.random.default_rng(7)
    boundaries = np.concatenate(([0.0], np.sort(rng.uniform(0.0, 60.0, 200))))
    start, end = seconds_to_frames(boundaries[:-1], boundaries[1:], fps)
    # Each segment ends exactly where the next begins ...
    assert (end[:-1] == start[1:]).all()
    # ... so the frames they cover are every frame of the span, once.
    covered = np.concatenate(
        [np.arange(int(s), int(e), dtype=np.int64) for s, e in zip(start, end)]
    )
    assert covered.tolist() == list(range(int(start[0]), int(end[-1])))
    # And each frame's midpoint lies in its own segment.
    owner = np.repeat(np.arange(start.size), end - start)
    midpoint = (covered + 0.5) / fps
    assert (boundaries[:-1][owner] <= midpoint).all()
    assert (midpoint < boundaries[1:][owner]).all()


def test_a_boundary_written_to_the_millisecond_lands_on_its_frame() -> None:
    """0.033 s at 30 fps is 0.99 frames: the boundary meant is 1, not 0."""
    start, end = seconds_to_frames([0.0, 0.033], [0.033, 0.067], fps=30.0)
    assert start.tolist() == [0, 1]
    assert end.tolist() == [1, 2]


def test_a_segment_shorter_than_a_frame_is_empty_unless_widened() -> None:
    """[0.40, 0.42) s at 10 fps holds no midpoint; frame 4 is on screen at 0.40 s."""
    start, end = seconds_to_frames([0.40], [0.42], fps=10.0)
    assert (start.tolist(), end.tolist()) == ([4], [4])
    start, end = seconds_to_frames([0.40, 0.0], [0.42, 0.3], fps=10.0, widen_empty=True)
    assert start.tolist() == [4, 0]
    assert end.tolist() == [5, 3]


def test_a_point_event_widens_to_the_frame_on_screen() -> None:
    start, end = seconds_to_frames([1.26], [1.26], fps=10.0, widen_empty=True)
    assert (start.tolist(), end.tolist()) == ([12], [13])


@pytest.mark.parametrize("fps", [0.0, -30.0, float("nan"), float("inf")])
def test_an_unusable_frame_rate_is_refused(fps: float) -> None:
    with pytest.raises(ValueError, match="fps"):
        _ = seconds_to_frames([0.0], [1.0], fps)


@pytest.mark.parametrize(
    ("start_s", "end_s", "match"),
    [
        ([0.0, 1.0], [1.0], "differ in length"),
        ([2.0], [1.0], "before its start"),
        ([-1.0], [1.0], "negative"),
        ([float("nan")], [1.0], "finite"),
    ],
)
def test_malformed_times_are_refused(
    start_s: list[float], end_s: list[float], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        _ = seconds_to_frames(start_s, end_s, 30.0)


# --- intervals_to_dense ------------------------------------------------------


def test_disjoint_segments_paint_onto_a_background() -> None:
    dense = intervals_to_dense([0, 5], [3, 7], [1, 2], 8, policy="refuse", background=9)
    assert dense.tolist() == [1, 1, 1, 9, 9, 2, 2, 9]
    assert dense.dtype == np.int32


def test_touching_segments_do_not_overlap() -> None:
    """``[0, 3)`` and ``[3, 5)`` share a boundary, not a frame."""
    dense = intervals_to_dense([0, 3], [3, 5], [1, 2], 5, policy="refuse")
    assert dense.tolist() == [1, 1, 1, 2, 2]


def test_segments_are_clipped_to_the_frame_count() -> None:
    dense = intervals_to_dense([2, 9], [6, 12], [1, 2], 5, policy="refuse")
    assert dense.tolist() == [0, 0, 1, 1, 1]


def test_an_overlap_is_refused_naming_both_segments() -> None:
    with pytest.raises(SegmentOverlapError, match=r"segments 0 .* and 1 "):
        _ = intervals_to_dense([0, 4], [6, 8], [1, 2], 10, policy="refuse")


def test_only_an_overlap_inside_the_output_is_refused() -> None:
    """[0, 3) and [6, 9) overlap [8, 12) from frame 8; an 8-frame output never shows it."""
    starts, ends, labels = [0, 6, 8], [3, 9, 12], [1, 2, 3]
    dense = intervals_to_dense(starts, ends, labels, 8, policy="refuse")
    assert dense.tolist() == [1, 1, 1, 0, 0, 0, 2, 2]
    with pytest.raises(SegmentOverlapError, match=r"segments 1 .* and 2 "):
        _ = intervals_to_dense(starts, ends, labels, 9, policy="refuse")


def test_the_refusal_is_a_value_error() -> None:
    assert issubclass(SegmentOverlapError, ValueError)


# Nested: resting [0, 10) with grooming [3, 5) inside it.
_NESTED = ([0, 3], [10, 5], [1, 2])
# Partial: A [0, 4) (4 frames) and B [2, 10) (8 frames), B starting later.
_PARTIAL = ([0, 2], [4, 10], [1, 2])


@pytest.mark.parametrize(
    ("policy", "nested", "partial"),
    [
        (
            "shortest",
            [1, 1, 1, 2, 2, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
        ),
        (
            "longest",
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 2, 2, 2, 2, 2, 2, 2, 2],
        ),
        (
            "latest_start",
            [1, 1, 1, 2, 2, 1, 1, 1, 1, 1],
            [1, 1, 2, 2, 2, 2, 2, 2, 2, 2],
        ),
    ],
)
def test_each_policy_decides_an_overlap_its_own_way(
    policy: OverlapPolicy, nested: list[int], partial: list[int]
) -> None:
    assert policy in OVERLAP_POLICIES
    assert intervals_to_dense(*_NESTED, 10, policy=policy).tolist() == nested
    assert intervals_to_dense(*_PARTIAL, 10, policy=policy).tolist() == partial


@pytest.mark.parametrize("policy", ["shortest", "longest", "latest_start"])
def test_a_tie_goes_to_the_segment_listed_later(policy: OverlapPolicy) -> None:
    """Equal length and equal start: nothing but listing order tells them apart."""
    dense = intervals_to_dense([0, 0], [4, 4], [1, 2], 4, policy=policy)
    assert dense.tolist() == [2, 2, 2, 2]
    dense = intervals_to_dense([0, 0], [4, 4], [2, 1], 4, policy=policy)
    assert dense.tolist() == [1, 1, 1, 1]


def test_priority_is_measured_before_clipping() -> None:
    """Cut to three frames both segments are three long; uncut, the first is longer.

    Measured after the cut the two would tie and the later-listed one would win.
    """
    dense = intervals_to_dense([0, 0], [100, 4], [1, 2], 3, policy="longest")
    assert dense.tolist() == [1, 1, 1]


def test_the_result_does_not_depend_on_the_input_being_sorted() -> None:
    starts, ends, labels = [5, 0, 3], [9, 10, 4], [3, 1, 2]
    order = [2, 0, 1]
    dense = intervals_to_dense(starts, ends, labels, 10, policy="shortest")
    shuffled = intervals_to_dense(
        [starts[i] for i in order],
        [ends[i] for i in order],
        [labels[i] for i in order],
        10,
        policy="shortest",
    )
    assert dense.tolist() == shuffled.tolist() == [1, 1, 1, 2, 1, 3, 3, 3, 3, 1]


@pytest.mark.parametrize(
    ("starts", "ends", "labels", "match"),
    [
        ([0.0], [2.0], [1], "integers"),
        ([3], [2], [1], "before it starts"),
        ([-1], [2], [1], "numbered from 0"),
        ([0, 1], [2], [1, 1], "differ in length"),
    ],
)
def test_malformed_segments_are_refused(
    starts: list[float], ends: list[float], labels: list[int], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        _ = intervals_to_dense(starts, ends, labels, 5, policy="refuse")


def test_an_unknown_policy_is_refused() -> None:
    with pytest.raises(ValueError, match="policy"):
        _ = intervals_to_dense([0], [1], [1], 2, policy="first")  # pyright: ignore[reportArgumentType]


# --- the sidecar -------------------------------------------------------------


def _segments() -> LabelSegments:
    return LabelSegments(
        start_frame=np.array([0, 3, 12]),
        end_frame=np.array([10, 5, 14]),
        label=np.array([1, 2, 2]),
        weight=np.array([1.0, 0.5, 0.9]),
        individual=np.array([-1, 0, 1]),
        extra={"seg_annotator": np.array(["ann", "bob", "ann"])},
    )


def _write(path: Path, **arrays: object) -> Path:
    atomic_savez(path, **arrays)
    return path


def test_a_file_without_segment_keys_has_no_sidecar() -> None:
    assert segments_from_npz({"labels": np.zeros(3, dtype=int)}) is None


def test_the_sidecar_round_trips_through_a_label_file(tmp_path: Path) -> None:
    segments = _segments()
    path = _write(
        tmp_path / "labels.npz",
        labels=segments.to_dense(20, policy="shortest"),
        label_ids=np.array([0, 1, 2]),
        label_names=np.array(["none", "rest", "groom"], dtype=object),
        **segments.to_payload(),
    )
    with np.load(path, allow_pickle=True) as npz:
        read = segments_from_npz(npz)
    assert read is not None
    assert read.start_frame.tolist() == [0, 3, 12]
    assert read.end_frame.tolist() == [10, 5, 14]
    assert read.label.tolist() == [1, 2, 2]
    assert read.label.dtype == np.int32
    assert read.weight is not None
    assert read.weight.tolist() == [1.0, 0.5, 0.9]
    assert read.individual is not None
    assert read.individual.tolist() == [-1, 0, 1]
    assert read.extra["seg_annotator"].tolist() == ["ann", "bob", "ann"]
    assert read.lengths.tolist() == [10, 2, 2]
    assert len(read) == 3


def test_the_listing_order_is_kept_and_onset_order_is_derived() -> None:
    segments = LabelSegments(
        start_frame=np.array([5, 0, 0]),
        end_frame=np.array([6, 4, 2]),
        label=np.array([1, 1, 1]),
    )
    assert segments.start_frame.tolist() == [5, 0, 0]
    assert segments.onset_order().tolist() == [2, 1, 0]


def test_readers_that_predate_the_sidecar_ignore_it(tmp_path: Path) -> None:
    """The dense array is what they read, and the segment keys never reach them."""
    segments = _segments()
    dense = segments.to_dense(20, policy="shortest")
    path = _write(tmp_path / "dense.npz", labels=dense, **segments.to_payload())
    with np.load(path, allow_pickle=True) as npz:
        assert detect_label_format(npz) == "dense"
    frames = np.arange(20)
    assert load_labels_for_feature_frames(path, frames).tolist() == dense.tolist()


def test_an_event_file_keeps_its_format_beside_a_sidecar(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "events.npz",
        frames=np.array([3, 4]),
        labels=np.array([2, 2]),
        individual_ids=np.array([[-1, -1], [-1, -1]]),
        **_segments().to_payload(),
    )
    with np.load(path, allow_pickle=True) as npz:
        assert detect_label_format(npz) == "individual_pair_v1"
    assert load_labels_for_feature_frames(path, np.arange(6)).tolist() == [
        0,
        0,
        0,
        2,
        2,
        0,
    ]


def test_a_partial_sidecar_is_refused() -> None:
    with pytest.raises(ValueError, match=SEG_LABEL):
        _ = segments_from_npz(
            {SEG_START_FRAME: np.array([0]), SEG_END_FRAME: np.array([2])}
        )


def test_a_label_the_file_does_not_declare_is_refused() -> None:
    with pytest.raises(ValueError, match=r"\[7\]"):
        _ = segments_from_npz(
            {
                SEG_START_FRAME: np.array([0]),
                SEG_END_FRAME: np.array([2]),
                SEG_LABEL: np.array([7]),
                "label_ids": np.array([0, 1]),
            }
        )


@pytest.mark.parametrize(
    ("key", "values", "match"),
    [
        ("weight", np.array([1.0]), "seg_weight"),
        ("individual", np.array([0]), "seg_individual"),
        ("individual", np.array([0.0, 1.0]), "integers"),
    ],
)
def test_an_optional_array_of_the_wrong_length_or_kind_is_refused(
    key: str, values: np.ndarray, match: str
) -> None:
    arrays: dict[str, np.ndarray] = {
        "start_frame": np.array([0, 2]),
        "end_frame": np.array([2, 4]),
        "label": np.array([1, 1]),
        key: values,
    }
    with pytest.raises(ValueError, match=match):
        _ = LabelSegments(**arrays)  # pyright: ignore[reportArgumentType]


def test_an_extra_array_must_be_a_segment_key_of_the_right_length() -> None:
    with pytest.raises(ValueError, match="one value per segment"):
        _ = LabelSegments(
            start_frame=np.array([0]),
            end_frame=np.array([1]),
            label=np.array([1]),
            extra={"seg_note": np.array(["a", "b"])},
        )
    with pytest.raises(ValueError, match="must start with"):
        _ = LabelSegments(
            start_frame=np.array([0]),
            end_frame=np.array([1]),
            label=np.array([1]),
            extra={"note": np.array(["a"])},
        )


def test_the_vocabulary_maps_ids_to_names() -> None:
    names = np.array(["none", "rest"], dtype=object)
    assert label_vocabulary({"label_ids": np.array([0, 1]), "label_names": names}) == {
        0: "none",
        1: "rest",
    }
    assert label_vocabulary({"label_ids": np.array([3, 5])}) == {3: "3", 5: "5"}
    assert label_vocabulary({"labels": np.zeros(2)}) == {}
    with pytest.raises(ValueError, match="label_names"):
        _ = label_vocabulary({"label_ids": np.array([0, 1]), "label_names": names[:1]})
