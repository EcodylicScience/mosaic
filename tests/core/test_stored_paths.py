"""Resolving a list of stored paths gives the answers that resolving each one does."""

from __future__ import annotations

from pathlib import Path

from mosaic.core.stored_paths import (
    resolve_stored_path,
    resolve_stored_paths,
    stored_form,
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
