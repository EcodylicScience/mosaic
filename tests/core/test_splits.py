"""The roles a dataset gives its groups: the manifest edit and the readers."""

from __future__ import annotations

from pathlib import Path

import pytest

from mosaic.core.manifest import SplitRole, read_manifest
from mosaic.core.splits import groups_with_role, held_out_groups, role_of
from tests.helpers import make_dataset

ROLES: dict[str, SplitRole] = {"a": "train", "b": "test", "c": "sealed", "d": "test"}


def test_readers_accept_a_mapping_a_manifest_and_a_dataset(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds")
    ds.set_splits("train", ["a"])
    ds.set_splits("test", ["d", "b"])
    ds.set_splits("sealed", ["c"])

    for source in (ROLES, ds.manifest, ds):
        assert role_of(source, "a") == "train"
        assert role_of(source, "c") == "sealed"
        assert role_of(source, "absent") is None
        assert groups_with_role(source, "test") == ("b", "d")
        assert groups_with_role(source, "test", "sealed") == ("b", "c", "d")
        assert groups_with_role(source) == ("a", "b", "c", "d")
        assert held_out_groups(source) == ("b", "c", "d")


def test_the_empty_group_has_no_role() -> None:
    assert role_of(ROLES, "") is None  # pyright: ignore[reportArgumentType]


def test_a_dataset_sets_and_removes_roles_under_the_manifest_lock(
    tmp_path: Path,
) -> None:
    ds = make_dataset(tmp_path / "ds")
    ds.set_splits("train", ["a", "b"])
    ds.set_splits("test", ["b", "c"])
    assert read_manifest(ds.manifest_path).splits == {
        "a": "train",
        "b": "test",
        "c": "test",
    }
    assert ds.remove_splits(["a", "nope"]) == ["a"]
    assert ds.splits == {"b": "test", "c": "test"}
    assert read_manifest(ds.manifest_path).splits == ds.splits


def test_the_property_is_a_copy(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds")
    ds.set_splits("test", ["a"])
    ds.splits["a"] = "train"
    assert ds.splits == {"a": "test"}


def test_a_refused_role_leaves_the_manifest_as_it_was(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds")
    ds.set_splits("test", ["a"])
    with pytest.raises(ValueError, match="empty group"):
        ds.set_splits("train", ["b", ""])
    with pytest.raises(ValueError, match="splits"):
        ds.set_splits("validation", ["b"])  # pyright: ignore[reportArgumentType]
    assert ds.splits == {"a": "test"}
    assert read_manifest(ds.manifest_path).splits == {"a": "test"}


def test_a_save_keeps_roles_written_by_another_process(tmp_path: Path) -> None:
    """``mutate_manifest`` re-reads, so a stale in-memory copy does not clobber."""
    ds = make_dataset(tmp_path / "ds")
    other = make_dataset(tmp_path / "ds")
    other.set_splits("test", ["from-other"])
    ds.set_splits("train", ["from-ds"])
    assert read_manifest(ds.manifest_path).splits == {
        "from-other": "test",
        "from-ds": "train",
    }
