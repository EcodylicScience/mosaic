"""Video-frame keys: one recorded frame is named by origin, group, sequence and camera."""

from __future__ import annotations

from mosaic.core.annotations.split import (
    VideoFrame,
    frame_key,
    recording_unit_key,
    sequence_key,
    unit_key,
)
from mosaic.core.entry import from_safe_name


def _frame(
    *,
    origin: str = "o1",
    group: str = "g1",
    sequence: str = "s1",
    camera: str = "c1",
    index: int = 7,
    stem: str = "",
) -> VideoFrame:
    return VideoFrame(origin, group, sequence, camera, index, stem)


def test_two_groups_with_one_sequence_name_give_two_sequence_keys() -> None:
    first = sequence_key(_frame(group="a"))
    second = sequence_key(_frame(group="b"))
    assert first == "o1:a:s1"
    assert first != second


def test_a_colon_inside_a_part_cannot_split_a_key() -> None:
    key = frame_key(_frame(origin="o:1", group="a:b", sequence="c:d", camera="e:f"))
    assert key == "o%3A1:a%3Ab:c%3Ad:e%3Af:7"
    assert len(key.split(":")) == 5
    decoded = tuple(from_safe_name(part) for part in key.split(":")[:4])
    assert decoded == ("o:1", "a:b", "c:d", "e:f")


def test_frame_key_uses_the_decimal_index() -> None:
    assert frame_key(_frame(index=12)) == "o1:g1:s1:c1:12"


def test_frame_key_uses_the_encoded_stem_when_the_index_is_unrecorded() -> None:
    key = frame_key(_frame(index=-1, stem="a:b@c"))
    assert key == "o1:g1:s1:c1:@a%3Ab%40c"


def test_frame_unit_drops_the_camera_so_two_cameras_share_a_unit() -> None:
    one = unit_key(_frame(camera="c1"), "frame", {}, {})
    two = unit_key(_frame(camera="c2"), "frame", {}, {})
    assert one == two == "o1:g1:s1:7"
    assert unit_key(_frame(index=8), "frame", {}, {}) != one


def test_frame_unit_of_an_unrecorded_index_uses_the_stem() -> None:
    assert unit_key(_frame(index=-1, stem="x"), "frame", {}, {}) == "o1:g1:s1:@x"


def test_sequence_unit_is_the_sequence_key() -> None:
    frame = _frame()
    assert unit_key(frame, "sequence", {}, {}) == sequence_key(frame)
    assert unit_key(_frame(camera="c2"), "sequence", {}, {}) == sequence_key(frame)


def test_group_unit_looks_up_the_sequence_key_first() -> None:
    labels = {"o1:g1:s1": "by-key", "s1": "by-name"}
    assert unit_key(_frame(), "group", labels, {}) == "by-key"


def test_group_unit_then_the_bare_sequence_name() -> None:
    assert unit_key(_frame(), "group", {"s1": "by-name"}, {}) == "by-name"


def test_group_unit_falls_back_to_the_sequence_key() -> None:
    assert unit_key(_frame(), "group", {"other": "x"}, {}) == "o1:g1:s1"


def test_a_recording_unit_key_leaves_the_sequence_part_empty() -> None:
    assert recording_unit_key(_frame()) == "o1:g1:"
    assert recording_unit_key(_frame(origin="o:1", group="a:b")) == "o%3A1:a%3Ab:"


def test_a_member_of_a_continuous_group_is_drawn_as_its_recording() -> None:
    units = {"o1:g1:s1": "o1:g1:", "o1:g1:s2": "o1:g1:"}

    first = unit_key(_frame(sequence="s1"), "recording", {}, units)
    second = unit_key(_frame(sequence="s2", camera="c2"), "recording", {}, units)

    assert first == second == "o1:g1:"


def test_a_sequence_outside_a_continuous_group_is_its_own_recording() -> None:
    units = {"o1:g1:s1": "o1:g1:"}
    frame = _frame(group="g2")

    assert unit_key(frame, "recording", {}, units) == sequence_key(frame)


def test_without_continuous_groups_a_recording_unit_is_the_sequence_unit() -> None:
    frame = _frame()

    assert unit_key(frame, "recording", {}, {}) == unit_key(frame, "sequence", {}, {})
