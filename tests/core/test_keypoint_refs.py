"""Keypoint references resolve by index or name in a standard-library leaf."""

from __future__ import annotations

import pytest

from mosaic.core.keypoint_refs import check_keypoint_refs, resolve_keypoint_refs
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


def test_the_leaf_loads_no_pose_columns_pandas_or_numpy() -> None:
    loaded = modules_loaded_by(
        "import mosaic.core.keypoint_refs",
        ["mosaic.core.pose_columns", "pandas", "numpy"],
    )
    assert loaded == []
