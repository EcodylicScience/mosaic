"""Keypoint references resolve by index or name in a standard-library leaf."""

from __future__ import annotations

import re

import pytest

from mosaic.core.keypoint_refs import (
    check_heading_pair,
    check_keypoint_refs,
    check_keypoints_distinct,
    keypoint_index,
    resolve_keypoint_refs,
)
from tests.helpers import modules_loaded_by


def test_references_resolve_by_name_and_by_index_in_the_order_given() -> None:
    names = ["nose", "left_wrist", "right_wrist"]
    assert resolve_keypoint_refs(["right_wrist", 0], names, 3, who="t") == [2, 0]


def test_an_unknown_name_names_the_known_ones() -> None:
    with pytest.raises(ValueError, match="left_wrist"):
        _ = resolve_keypoint_refs(["elbow"], ["nose", "left_wrist"], 2, who="t")


def test_a_name_without_a_name_list_is_refused() -> None:
    with pytest.raises(ValueError, match="not given keypoint names"):
        _ = resolve_keypoint_refs(["nose"], None, 3, who="t")


@pytest.mark.parametrize("index", [-1, 3])
def test_an_index_outside_the_table_is_refused(index: int) -> None:
    with pytest.raises(ValueError, match="outside"):
        _ = resolve_keypoint_refs([index], None, 3, who="t")


@pytest.mark.parametrize(
    ("refs", "names", "match"),
    [
        ([0, -2], None, "negative"),
        (["nose"], None, "not given keypoint names"),
        (["elbow"], ["nose", "tail"], "no keypoint named"),
        ([2], ["nose", "tail"], "outside"),
    ],
)
def test_a_reference_no_table_could_resolve_is_refused(
    refs: list[int | str], names: list[str] | None, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        check_keypoint_refs(refs, names, "t")


def test_an_index_is_unbounded_without_names() -> None:
    check_keypoint_refs([0, 99], None, "t")
    check_keypoint_refs(["tail", 1], ["nose", "tail"], "t")


def test_a_reference_is_one_keypoint_by_index_or_by_name() -> None:
    names = ["nose", "tail"]
    assert keypoint_index(1, names) == 1
    assert keypoint_index("tail", names) == 1
    assert keypoint_index(0, None) == 0


def test_a_keypoint_named_twice_is_refused_whichever_form_names_it() -> None:
    check_keypoints_distinct(None, None, "t")
    check_keypoints_distinct([0, 1], None, "t")
    message = "t: keypoints names a keypoint twice, [1, 1]."
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        check_keypoints_distinct([1, 1], None, "t")
    with pytest.raises(ValueError, match="names a keypoint twice"):
        check_keypoints_distinct([0, "nose"], ["nose", "tail"], "t")


def test_a_heading_of_one_keypoint_is_refused_whichever_form_names_it() -> None:
    check_heading_pair((0, 1), None, "t")
    message = "t: heading (1, 1) is a keypoint and itself, which points nowhere."
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        check_heading_pair((1, 1), None, "t")
    with pytest.raises(ValueError, match="itself"):
        check_heading_pair(("tail", 1), ["nose", "tail"], "t")


def test_the_leaf_loads_no_pose_columns_pandas_or_numpy() -> None:
    loaded = modules_loaded_by(
        "import mosaic.core.keypoint_refs",
        ["mosaic.core.pose_columns", "pandas", "numpy"],
    )
    assert loaded == []
