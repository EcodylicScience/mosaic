"""Resolving a list of stored paths gives the answers that resolving each one does."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from mosaic.core.stored_paths import (
    link_tree,
    resolve_stored_path,
    resolve_stored_paths,
    stored_form,
    tree_relative_cell,
)


def test_a_list_resolves_as_each_cell_does(tmp_path: Path) -> None:
    anchor = (tmp_path / "study").resolve()
    (anchor / "tracks" / "v1").mkdir(parents=True)
    (anchor / "elsewhere").mkdir()
    (anchor / "alias").symlink_to(anchor / "elsewhere", target_is_directory=True)
    _ = (anchor / "elsewhere" / "real.parquet").write_bytes(b"x")
    (anchor / "tracks" / "link.parquet").symlink_to(
        anchor / "elsewhere" / "real.parquet"
    )
    outside = tmp_path / "outside" / "a.parquet"
    outside.parent.mkdir()
    _ = outside.write_bytes(b"x")
    cells = [
        "tracks/v1/a.parquet",
        "tracks/v1/b.parquet",
        "alias/real.parquet",
        "alias/missing.parquet",
        "tracks/link.parquet",
        "tracks/../tracks/v1/c.parquet",
        "never/made/d.parquet",
        str(outside),
        str(tmp_path / "gone" / "e.parquet"),
        "a.parquet",
    ]

    resolved = resolve_stored_paths(cells, lambda: anchor)

    assert resolved == [resolve_stored_path(cell, anchor) for cell in cells]


def test_an_absent_path_is_remapped_as_a_single_one_is(tmp_path: Path) -> None:
    anchor = tmp_path / "study"
    moved = tmp_path / "moved"
    (moved / "tracks").mkdir(parents=True)
    path_map = [(tmp_path / "gone", moved)]
    cell = str(tmp_path / "gone" / "tracks" / "a.parquet")

    [resolved] = resolve_stored_paths([cell], lambda: anchor, path_map=path_map)

    assert resolved == resolve_stored_path(cell, anchor, path_map=path_map)
    assert resolved == moved / "tracks" / "a.parquet"


def test_the_anchor_is_asked_for_only_when_a_cell_is_relative(tmp_path: Path) -> None:
    asked: list[int] = []

    def anchor() -> Path:
        asked.append(1)
        return tmp_path

    _ = resolve_stored_paths([str(tmp_path / "a")], anchor)
    assert asked == []

    _ = resolve_stored_paths(["a", "b/c", "b/d"], anchor)
    assert asked == [1]


def test_a_path_under_the_base_is_stored_relative_to_it(tmp_path: Path) -> None:
    base = tmp_path / "study"

    assert stored_form(base / "tracks" / "a.parquet", base) == str(
        Path("tracks") / "a.parquet"
    )


def test_a_path_outside_the_base_is_stored_as_it_is(tmp_path: Path) -> None:
    base = tmp_path / "study"
    outside = tmp_path / "elsewhere" / "a.parquet"

    assert stored_form(outside, base) == str(outside)


def test_a_sibling_that_shares_the_base_prefix_is_outside_it(tmp_path: Path) -> None:
    base = tmp_path / "study"
    sibling = tmp_path / "study2" / "a.parquet"

    assert stored_form(sibling, base) == str(sibling)


def test_the_base_itself_is_stored_as_a_dot(tmp_path: Path) -> None:
    base = tmp_path / "study"

    assert stored_form(base, base) == str(Path("."))


def test_a_stored_form_resolves_back_to_its_path(tmp_path: Path) -> None:
    anchor = tmp_path.resolve() / "study"
    inside = anchor / "tracks" / "a.parquet"
    outside = tmp_path.resolve() / "elsewhere" / "a.parquet"

    assert resolve_stored_path(stored_form(inside, anchor), anchor) == inside
    assert resolve_stored_path(stored_form(outside, anchor), anchor) == outside


# --- the tree of a relative link ------------------------------------------------


def _raise_value_error(*_args: object) -> str:
    raise ValueError("paths on two drives")


@pytest.mark.parametrize(
    ("link", "tree"),
    [("../share", "data"), ("../../examples/calms21", ".")],
    ids=["sibling", "two-levels-up"],
)
def test_a_links_tree_is_where_its_parent_steps_lead(
    tmp_path: Path, link: str, tree: str
) -> None:
    base = tmp_path.resolve() / "data" / "fork"

    assert link_tree(base, link) == Path(os.path.normpath(tmp_path.resolve() / tree))


def test_a_links_tree_is_found_without_resolving_a_path(tmp_path: Path) -> None:
    base = tmp_path.resolve() / "data" / "fork"
    (tmp_path / "elsewhere").mkdir()
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "share").symlink_to(
        tmp_path / "elsewhere", target_is_directory=True
    )

    assert link_tree(base, "../share") == tmp_path.resolve() / "data"


def test_a_tree_at_the_file_system_root_is_none() -> None:
    base = Path(Path.cwd().anchor) / "fork"

    assert link_tree(base, "../share") is None


def test_a_tree_that_commonpath_refuses_is_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(os.path, "commonpath", _raise_value_error)

    assert link_tree(tmp_path.resolve() / "fork", "../share") is None


def _tree(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    """A tree holding a fork, a share and a video, as resolved paths."""
    tree = tmp_path.resolve() / "data"
    fork = tree / "fork"
    share = tree / "share"
    video = share / "media_raw" / "s1.mp4"
    fork.mkdir(parents=True)
    video.parent.mkdir(parents=True)
    _ = video.write_bytes(b"video")
    return tree, fork, share, video


def test_a_file_under_the_tree_is_stored_relative_to_the_anchor(
    tmp_path: Path,
) -> None:
    tree, fork, share, video = _tree(tmp_path)

    cell = tree_relative_cell(video, video, fork, tree, share)

    assert cell == "../share/media_raw/s1.mp4"
    assert cell == Path(os.path.relpath(video, fork)).as_posix()


def test_a_file_outside_the_tree_is_none(tmp_path: Path) -> None:
    tree, fork, share, _video = _tree(tmp_path)
    outside = tmp_path.resolve() / "outside" / "s3.npy"
    outside.parent.mkdir()
    _ = outside.write_bytes(b"s3")

    assert tree_relative_cell(outside, outside, fork, tree, share) is None


def test_a_file_reached_through_a_symlink_is_none(tmp_path: Path) -> None:
    tree, fork, share, video = _tree(tmp_path)
    alias = tree / "alias"
    alias.symlink_to(share, target_is_directory=True)
    lexical = alias / "media_raw" / "s1.mp4"

    assert tree_relative_cell(lexical, video, fork, tree, share) is None


def test_a_target_reached_through_a_symlink_is_none(tmp_path: Path) -> None:
    tree, fork, share, video = _tree(tmp_path)
    alias = tree / "alias"
    alias.symlink_to(share, target_is_directory=True)

    assert tree_relative_cell(video, video, fork, tree, alias) is None


def test_a_file_a_path_map_moved_is_none(tmp_path: Path) -> None:
    tree, fork, share, video = _tree(tmp_path)
    moved = tree / "moved" / "s1.mp4"
    moved.parent.mkdir()
    _ = moved.write_bytes(b"video")

    assert tree_relative_cell(video, moved, fork, tree, share) is None


def test_a_file_with_no_tree_is_none(tmp_path: Path) -> None:
    _tree_path, fork, share, video = _tree(tmp_path)

    assert tree_relative_cell(video, video, fork, None, share) is None


def test_a_cell_that_relpath_refuses_is_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tree, fork, share, video = _tree(tmp_path)
    monkeypatch.setattr(os.path, "relpath", _raise_value_error)

    assert tree_relative_cell(video, video, fork, tree, share) is None
