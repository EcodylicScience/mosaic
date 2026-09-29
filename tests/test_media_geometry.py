"""Where a media variant's pixels and frames sit in its entry's source media.

A variant's tracks are mapped back to source space through its placement, so a
wrong frame map or offset publishes a table whose frames and positions name the
wrong moment and the wrong pixel. Each test builds the maps the pre-processing
steps build (``trim`` is ``within``, ``decimate`` is ``every``) and checks the
source frames and file indices they answer with.
"""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Callable

import numpy as np
import pytest

from mosaic.core.media.preprocess.geometry import FrameMap, Placement

# --- FrameMap ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("start", "step", "count"),
    [(-1, 1, 10), (0, 0, 10), (0, -2, 10), (0, 1, -1)],
)
def test_a_frame_map_refuses_an_impossible_grid(
    start: int, step: int, count: int
) -> None:
    with pytest.raises(ValueError):
        _ = FrameMap(start=start, step=step, count=count)


@pytest.mark.parametrize("field", ["start", "step", "count"])
def test_a_frame_map_refuses_a_fractional_field(field: str) -> None:
    """A float would be truncated where a placement is written, not refused."""
    with pytest.raises(TypeError, match=field):
        _ = dataclasses.replace(FrameMap(0, 1, 10), **{field: 1.5})


def test_source_frames_map_file_frames_onto_the_grid() -> None:
    frames = FrameMap(start=10, step=3, count=5)

    mapped = frames.source_frames(np.array([0, 1, 4], dtype=np.int64))

    assert mapped.tolist() == [10, 13, 22]
    assert mapped.dtype == np.int64


def test_within_keeps_the_frames_inside_the_range() -> None:
    assert FrameMap(0, 1, 100).within(10, 20) == FrameMap(10, 1, 10)


def test_within_accepts_the_whole_map_at_both_edges() -> None:
    whole = FrameMap(0, 1, 100)
    assert whole.within(0, 100) == whole

    decimated = FrameMap(5, 3, 10)
    assert decimated.within(5, 35) == decimated


@pytest.mark.parametrize(("start", "stop"), [(0, 101), (-1, 50), (99, 101)])
def test_within_refuses_a_range_past_the_map(start: int, stop: int) -> None:
    with pytest.raises(ValueError) as refusal:
        _ = FrameMap(0, 1, 100).within(start, stop)

    assert f"[{start}, {stop})" in str(refusal.value)
    assert "[0, 100)" in str(refusal.value)


def test_within_refuses_a_start_before_a_trimmed_map() -> None:
    with pytest.raises(ValueError, match=r"\[5, 20\).*\[10, 60\)"):
        _ = FrameMap(10, 1, 50).within(5, 20)


def test_within_refuses_a_range_selecting_no_frame() -> None:
    with pytest.raises(ValueError, match="no frame"):
        _ = FrameMap(0, 10, 10).within(1, 5)


@pytest.mark.parametrize(("start", "stop"), [(20, 20), (30, 20)])
def test_within_refuses_a_start_at_or_past_the_stop(start: int, stop: int) -> None:
    with pytest.raises(ValueError, match=rf"\[{start}, {stop}\)"):
        _ = FrameMap(0, 1, 100).within(start, stop)


def test_every_keeps_a_remainder_frame() -> None:
    assert FrameMap(0, 1, 10).every(3) == FrameMap(0, 3, 4)


def test_every_without_a_remainder() -> None:
    assert FrameMap(0, 1, 9).every(3) == FrameMap(0, 3, 3)


@pytest.mark.parametrize("n", [1, 0, -2])
def test_every_refuses_a_factor_below_two(n: int) -> None:
    with pytest.raises(ValueError):
        _ = FrameMap(0, 1, 10).every(n)


def test_within_after_every_begins_at_the_next_kept_frame() -> None:
    """A ``trim`` after a ``decimate`` starts on the grid, not at its own start."""
    decimated = FrameMap(0, 1, 100).every(4)

    trimmed = decimated.within(5, 30)

    assert trimmed == FrameMap(8, 4, 6)
    assert trimmed.source_frames(np.arange(6, dtype=np.int64)).tolist() == [
        8,
        12,
        16,
        20,
        24,
        28,
    ]


def test_file_indices_read_a_chained_map_from_its_upstream_file() -> None:
    upstream = FrameMap(0, 1, 1000).within(100, 500)
    chained = upstream.within(200, 300).every(5)

    first, stride, count = chained.file_indices(upstream)

    assert (first, stride, count) == (100, 5, 20)
    positions = first + stride * np.arange(count, dtype=np.int64)
    assert (
        upstream.source_frames(positions).tolist()
        == chained.source_frames(np.arange(count, dtype=np.int64)).tolist()
    )


def test_file_indices_over_a_decimated_upstream() -> None:
    upstream = FrameMap(0, 1, 1000).every(2)

    assert upstream.every(3).file_indices(upstream) == (0, 3, 167)
    assert upstream.within(101, 200).file_indices(upstream) == (51, 1, 49)


@pytest.mark.parametrize(
    ("chained", "upstream"),
    [
        (FrameMap(101, 2, 10), FrameMap(0, 2, 500)),
        (FrameMap(0, 3, 10), FrameMap(0, 2, 500)),
        (FrameMap(0, 2, 501), FrameMap(0, 2, 500)),
        (FrameMap(0, 1, 5), FrameMap(10, 1, 5)),
    ],
    ids=["start-off-grid", "step-not-a-multiple", "past-the-end", "before-the-start"],
)
def test_file_indices_refuse_a_map_off_the_upstream_grid(
    chained: FrameMap, upstream: FrameMap
) -> None:
    with pytest.raises(ValueError):
        _ = chained.file_indices(upstream)


def test_is_identity_over_a_frame_count() -> None:
    assert FrameMap(0, 1, 100).is_identity_over(100)
    assert not FrameMap(0, 1, 100).is_identity_over(101)
    assert not FrameMap(1, 1, 99).is_identity_over(100)
    assert not FrameMap(0, 2, 50).is_identity_over(100)


# --- Placement ---------------------------------------------------------------


def _placement(
    *,
    offset_x: int = 0,
    offset_y: int = 0,
    width: int = 640,
    height: int = 480,
    source_width: int = 640,
    source_height: int = 480,
    source_frame_count: int = 100,
    frames: FrameMap | None = None,
    fps: float = 30.0,
) -> Placement:
    return Placement(
        offset_x=offset_x,
        offset_y=offset_y,
        width=width,
        height=height,
        source_width=source_width,
        source_height=source_height,
        source_frame_count=source_frame_count,
        frames=frames if frames is not None else FrameMap(0, 1, source_frame_count),
        fps=fps,
    )


@pytest.mark.parametrize(
    "build",
    [
        lambda: _placement(offset_x=-1, width=100),
        lambda: _placement(offset_y=-1, height=100),
        lambda: _placement(width=0),
        lambda: _placement(height=0),
        lambda: _placement(offset_x=600, width=100),
        lambda: _placement(offset_y=400, height=100),
        lambda: _placement(width=641),
        lambda: _placement(fps=0.0),
        lambda: _placement(fps=-30.0),
        lambda: _placement(fps=math.nan),
        lambda: _placement(fps=math.inf),
        lambda: _placement(frames=FrameMap(0, 1, 101)),
        lambda: _placement(frames=FrameMap(0, 3, 35)),
    ],
    ids=[
        "negative-x",
        "negative-y",
        "zero-width",
        "zero-height",
        "past-the-right-edge",
        "past-the-bottom-edge",
        "wider-than-the-source",
        "zero-fps",
        "negative-fps",
        "nan-fps",
        "infinite-fps",
        "frames-past-the-source",
        "decimated-frames-past-the-source",
    ],
)
def test_a_placement_refuses_geometry_outside_its_source(
    build: Callable[[], Placement],
) -> None:
    with pytest.raises(ValueError):
        _ = build()


@pytest.mark.parametrize(
    "field",
    [
        "offset_x",
        "offset_y",
        "width",
        "height",
        "source_width",
        "source_height",
        "source_frame_count",
    ],
)
def test_a_placement_refuses_a_fractional_field(field: str) -> None:
    """A float would be truncated where the placement is written, not refused."""
    with pytest.raises(TypeError, match=field):
        _ = dataclasses.replace(_placement(), **{field: 0.5})


def test_identity_is_the_identity() -> None:
    placement = Placement.identity(640, 480, 100, 30.0)

    assert placement == _placement()
    assert placement.is_spatial_identity
    assert placement.is_frame_identity
    assert placement.is_identity


def test_a_crop_is_not_a_spatial_identity() -> None:
    cropped = _placement(offset_x=10, offset_y=20, width=320, height=240)

    assert not cropped.is_spatial_identity
    assert cropped.is_frame_identity
    assert not cropped.is_identity


def test_a_full_size_rectangle_at_an_offset_cannot_exist() -> None:
    with pytest.raises(ValueError):
        _ = _placement(offset_x=2)


@pytest.mark.parametrize(
    "frames",
    [FrameMap(10, 1, 50), FrameMap(0, 2, 50), FrameMap(0, 1, 99)],
    ids=["trimmed", "decimated", "short"],
)
def test_a_frame_selection_is_not_a_frame_identity(frames: FrameMap) -> None:
    selected = _placement(frames=frames)

    assert selected.is_spatial_identity
    assert not selected.is_frame_identity
    assert not selected.is_identity


def test_a_relabelled_rate_leaves_the_identity_alone() -> None:
    """What a relabelled rate means is decided by map-back, not here."""
    assert _placement(fps=15.0).is_identity


def test_json_round_trips() -> None:
    placement = _placement(
        offset_x=10, offset_y=20, width=320, height=240, frames=FrameMap(8, 4, 6)
    )

    assert Placement.from_json(placement.to_json()) == placement


def test_json_is_canonical() -> None:
    """Sorted keys and compact separators, whatever order the fields were given."""
    one = Placement(
        offset_x=10,
        offset_y=20,
        width=320,
        height=240,
        source_width=640,
        source_height=480,
        source_frame_count=100,
        frames=FrameMap(start=8, step=4, count=6),
        fps=30.0,
    )
    other = Placement(
        fps=30,
        frames=FrameMap(count=6, step=4, start=8),
        source_frame_count=100,
        source_height=480,
        source_width=640,
        height=240,
        width=320,
        offset_y=20,
        offset_x=10,
    )
    from_numpy = dataclasses.replace(
        one,
        offset_x=np.int64(10),
        width=np.int32(320),
        source_frame_count=np.int64(100),
        frames=dataclasses.replace(one.frames, start=np.int64(8), count=np.uint16(6)),
        fps=np.float64(30.0),
    )

    assert one.to_json() == other.to_json() == from_numpy.to_json()
    assert one.to_json() == (
        '{"fps":30.0,"frames":{"count":6,"start":8,"step":4},"height":240,'
        '"offset_x":10,"offset_y":20,"source_frame_count":100,"source_height":480,'
        '"source_width":640,"width":320}'
    )


def _edited(key: str, value: object) -> str:
    """A valid placement's JSON with one field set to *value*."""
    document: dict[str, object] = json.loads(_placement().to_json())
    document[key] = value
    return json.dumps(document)


def _without(key: str) -> str:
    """A valid placement's JSON with one field removed."""
    document: dict[str, object] = json.loads(_placement().to_json())
    del document[key]
    return json.dumps(document)


@pytest.mark.parametrize(
    "text",
    [
        "[]",
        "not json",
        _without("width"),
        _edited("rotation", 90),
        _edited("width", "640"),
        _edited("width", True),
        _edited("width", 640.0),
        _edited("width", -640),
        _edited("fps", "30"),
        _edited("frames", [0, 1, 100]),
        _edited("frames", {"start": 0, "step": 1}),
        _edited("frames", {"start": 0, "step": 1, "count": 100, "stop": 100}),
        _edited("frames", {"start": 0, "step": 1.0, "count": 100}),
    ],
    ids=[
        "not-an-object",
        "not-json",
        "missing-field",
        "unknown-field",
        "string-width",
        "bool-width",
        "float-width",
        "negative-width",
        "string-fps",
        "frames-not-an-object",
        "frames-missing-count",
        "frames-unknown-field",
        "frames-float-step",
    ],
)
def test_from_json_refuses_a_malformed_placement(text: str) -> None:
    with pytest.raises(ValueError):
        _ = Placement.from_json(text)
