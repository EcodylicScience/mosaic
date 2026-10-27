"""Arena layouts: their geometry, their key and the arenas index that stores them."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from mosaic.core.arenas import (
    Arena,
    ArenaKey,
    ArenaLayout,
    Rect,
    layout_lookup_order,
)
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.arena_index import (
    ArenaIndexError,
    ArenaLayoutRow,
    arena_index,
    arena_index_path,
)
from mosaic.core.recording import RecordingKey
from tests.helpers import add_track_sequences, make_dataset

PRINCETON_XS = (0, 506, 1012)
PRINCETON_YS = (0, 170, 346, 524, 702, 878, 1068)


def _two_arenas(width: int = 200) -> ArenaLayout:
    """A layout of a left and a right arena over a 200x100 frame."""
    return ArenaLayout(
        width=width,
        height=100,
        arenas=(
            Arena(name="left", region=Rect(x0=0, y0=0, x1=100, y1=100)),
            Arena(name="right", region=Rect(x0=100, y0=0, x1=200, y1=100)),
        ),
    )


def _one_arena() -> ArenaLayout:
    return ArenaLayout(
        width=200,
        height=100,
        arenas=(Arena(name="all", region=Rect(x0=0, y0=0, x1=200, y1=100)),),
    )


class TestGeometry:
    def test_the_princeton_grid_names_twelve_tanks_column_by_column(self) -> None:
        layout = ArenaLayout.grid(xs=PRINCETON_XS, ys=PRINCETON_YS)

        assert layout.names == (
            *(f"A{row}" for row in range(1, 7)),
            *(f"B{row}" for row in range(1, 7)),
        )
        assert (layout.width, layout.height) == (1012, 1068)
        assert layout.arenas[0].region == Rect(x0=0, y0=0, x1=506, y1=170)
        assert layout.arenas[11].region == Rect(x0=506, y0=878, x1=1012, y1=1068)
        assert {arena.capacity for arena in layout.arenas} == {1}

    def test_the_grid_excludes_the_rects_it_names(self) -> None:
        strip = Rect(x0=95, y0=0, x1=105, y1=170)
        layout = ArenaLayout.grid(
            xs=PRINCETON_XS, ys=PRINCETON_YS, exclude={"A1": [strip]}
        )

        assert layout.arenas[0].exclude == (strip,)
        with pytest.raises(ValueError, match="lacks"):
            _ = ArenaLayout.grid(xs=PRINCETON_XS, ys=PRINCETON_YS, exclude={"C1": []})

    def test_overlapping_arenas_are_refused(self) -> None:
        with pytest.raises(ValidationError, match="overlap"):
            _ = ArenaLayout(
                width=200,
                height=100,
                arenas=(
                    Arena(name="a", region=Rect(x0=0, y0=0, x1=101, y1=100)),
                    Arena(name="b", region=Rect(x0=100, y0=0, x1=200, y1=100)),
                ),
            )

    def test_touching_arenas_share_no_pixel(self) -> None:
        """Rects are half-open, so the edge at x 100 belongs to the right arena."""
        layout = _two_arenas()

        assert not layout.arenas[0].region.overlaps(layout.arenas[1].region)

    def test_an_arena_outside_the_frame_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="leaves the 150x100 frame"):
            _ = _two_arenas(width=150)

    def test_an_exclude_outside_its_region_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="outside its region"):
            _ = Arena(
                name="a",
                region=Rect(x0=0, y0=0, x1=10, y1=10),
                exclude=(Rect(x0=5, y0=5, x1=20, y1=10),),
            )

    @pytest.mark.parametrize("name", ["", " a", "a__b", "a:b", "a/b"])
    def test_an_arena_name_is_an_entry_name_token(self, name: str) -> None:
        with pytest.raises(ValidationError):
            _ = Arena(name=name, region=Rect(x0=0, y0=0, x1=10, y1=10))

    def test_repeated_names_and_empty_layouts_are_refused(self) -> None:
        region = Rect(x0=0, y0=0, x1=10, y1=10)
        with pytest.raises(ValidationError, match="repeat"):
            _ = ArenaLayout(
                width=30,
                height=10,
                arenas=(
                    Arena(name="a", region=region),
                    Arena(name="a", region=Rect(x0=10, y0=0, x1=20, y1=10)),
                ),
            )
        with pytest.raises(ValidationError):
            _ = ArenaLayout(width=30, height=10, arenas=())

    def test_a_rect_without_area_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="x0 < x1"):
            _ = Rect(x0=10, y0=0, x1=10, y1=5)


class TestDigest:
    def test_the_digest_is_pinned(self) -> None:
        """A layout's digest names it in the tracks it assigns, so it cannot move."""
        layout = ArenaLayout.grid(xs=PRINCETON_XS, ys=PRINCETON_YS)

        assert layout.digest() == (
            "9f955c44e22706564636823bb3a81fe3495be93c73e79afbde5d14f0f889d2c2"
        )

    def test_key_order_does_not_change_the_digest(self) -> None:
        layout = _two_arenas()
        dumped = layout.model_dump(mode="json")
        reordered = dict(reversed(list(dumped.items())))

        assert ArenaLayout.model_validate(reordered).digest() == layout.digest()

    def test_a_moved_edge_changes_the_digest(self) -> None:
        moved = ArenaLayout(
            width=200,
            height=100,
            arenas=(
                Arena(name="left", region=Rect(x0=0, y0=0, x1=99, y1=100)),
                Arena(name="right", region=Rect(x0=100, y0=0, x1=200, y1=100)),
            ),
        )

        assert moved.digest() != _two_arenas().digest()


def test_a_lookup_tries_the_entry_then_its_recording_then_every_camera() -> None:
    order = layout_lookup_order(
        RecordingKey("rec", "s1"), RecordingKey("rec", ""), "cam1"
    )

    assert order == (
        ArenaKey("rec", "s1", "cam1"),
        ArenaKey("rec", "", "cam1"),
        ArenaKey("rec", "s1", ""),
        ArenaKey("rec", "", ""),
    )
    undivided = RecordingKey("ns", "a")
    assert layout_lookup_order(undivided, undivided, "") == (ArenaKey("ns", "a", ""),)


@pytest.fixture
def dataset(tmp_path: Path) -> Dataset:
    """Tracks of the divided recording ``rec``, the empty-group ``solo`` and ``ns/a``."""
    ds = make_dataset(tmp_path / "ds")
    add_track_sequences(ds, ("rec", "s1"), ("rec", "s2"), ("", "solo"), ("ns", "a"))
    ds.set_continuous_groups(["rec"])
    return ds


class TestDatasetLayouts:
    def test_a_divided_recording_s_layout_applies_to_every_sequence(
        self, dataset: Dataset
    ) -> None:
        path = dataset.set_arena_layout(_two_arenas(), group="rec")

        assert path == dataset.get_root("arenas") / "rec.json"
        for sequence in ("s1", "s2"):
            found = dataset.arena_layout("rec", sequence)
            assert found is not None
            assert found.key == ArenaKey("rec", "", "")
            assert found.layout == _two_arenas()
        assert dataset.arena_layout("", "solo") is None

    def test_a_camera_s_layout_comes_before_every_camera_s(
        self, dataset: Dataset
    ) -> None:
        _ = dataset.set_arena_layout(_two_arenas(), group="rec")
        _ = dataset.set_arena_layout(_one_arena(), group="rec", camera="top")

        top = dataset.arena_layout("rec", "s1", "top")
        side = dataset.arena_layout("rec", "s1", "side")

        assert top is not None and top.layout == _one_arena()
        assert side is not None and side.layout == _two_arenas()
        assert (dataset.get_root("arenas") / "rec.top.json").is_file()

    def test_an_undivided_recording_is_keyed_by_its_entry(
        self, dataset: Dataset
    ) -> None:
        _ = dataset.set_arena_layout(_two_arenas(), group="ns", sequence="a")

        found = dataset.arena_layout("ns", "a")

        assert found is not None and found.key == ArenaKey("ns", "a", "")
        assert (dataset.get_root("arenas") / "ns__a.json").is_file()

    def test_setting_a_key_again_replaces_its_layout(self, dataset: Dataset) -> None:
        _ = dataset.set_arena_layout(_two_arenas(), group="rec")
        _ = dataset.set_arena_layout(_one_arena(), group="rec")

        assert list(dataset.arena_layouts()) == [ArenaKey("rec", "", "")]
        found = dataset.arena_layout("rec", "s1")
        assert found is not None and found.layout == _one_arena()

    def test_a_namespace_group_names_no_recording(self, dataset: Dataset) -> None:
        with pytest.raises(ValueError, match="not a continuous group"):
            _ = dataset.set_arena_layout(_two_arenas(), group="ns")

    def test_a_row_of_a_namespace_group_is_skipped_at_lookup(
        self, dataset: Dataset
    ) -> None:
        """A hand-written ``(ns, "")`` row matches no entry of ``ns``."""
        _ = dataset.set_arena_layout(_two_arenas(), group="rec")
        index = arena_index(arena_index_path(dataset))
        index.append(
            [
                ArenaLayoutRow(
                    abs_path=Path("arenas/rec.json"),
                    group="ns",
                    sequence="",
                    camera="",
                    digest=_two_arenas().digest(),
                    n_arenas=2,
                )
            ]
        )

        assert dataset.arena_layout("ns", "a") is None

    def test_a_divided_recording_s_sequence_is_reserved(self, dataset: Dataset) -> None:
        with pytest.raises(ValueError, match="once per recording"):
            _ = dataset.set_arena_layout(_two_arenas(), group="rec", sequence="s1")

    def test_a_recording_without_entries_is_refused(self, dataset: Dataset) -> None:
        with pytest.raises(ValueError, match="no entry"):
            _ = dataset.set_arena_layout(_two_arenas(), group="", sequence="ghost")

    def test_two_keys_cannot_share_a_file(self, tmp_path: Path) -> None:
        ds = make_dataset(tmp_path / "ds")
        add_track_sequences(ds, ("", "a.b"), ("", "a"))
        _ = ds.set_arena_layout(_two_arenas(), group="", sequence="a.b")

        with pytest.raises(ArenaIndexError, match="holds the layout of"):
            _ = ds.set_arena_layout(_two_arenas(), group="", sequence="a", camera="b")

    def test_two_rows_of_one_key_are_refused_at_the_read(
        self, dataset: Dataset
    ) -> None:
        _ = dataset.set_arena_layout(_two_arenas(), group="rec")
        path = arena_index_path(dataset)
        lines = path.read_text().splitlines()
        _ = path.write_text("\n".join([*lines, lines[-1]]) + "\n")

        with pytest.raises(ArenaIndexError, match="two layouts"):
            _ = dataset.arena_layouts()

    def test_a_changed_file_is_refused_at_the_read(self, dataset: Dataset) -> None:
        path = dataset.set_arena_layout(_two_arenas(), group="rec")
        _ = path.write_text(_one_arena().model_dump_json())

        with pytest.raises(ArenaIndexError, match="digest"):
            _ = dataset.arena_layouts()

    def test_a_missing_file_is_refused_at_the_read(self, dataset: Dataset) -> None:
        path = dataset.set_arena_layout(_two_arenas(), group="rec")
        path.unlink()

        with pytest.raises(ArenaIndexError, match="cannot be read as a layout"):
            _ = dataset.arena_layouts()

    def test_a_garbled_index_is_refused_at_the_read(self, dataset: Dataset) -> None:
        _ = dataset.set_arena_layout(_two_arenas(), group="rec")
        path = arena_index_path(dataset)
        _ = path.write_text(path.read_text() + '"rec,unterminated\n')

        with pytest.raises(ArenaIndexError, match="cannot be read"):
            _ = dataset.arena_layouts()

    @pytest.mark.parametrize(
        "text", ["{not json", '{"width": 200}'], ids=["garbled", "invalid"]
    )
    def test_an_unparsable_file_is_refused_at_the_read(
        self, dataset: Dataset, text: str
    ) -> None:
        path = dataset.set_arena_layout(_two_arenas(), group="rec")
        _ = path.write_text(text)

        with pytest.raises(ArenaIndexError, match="cannot be read as a layout"):
            _ = dataset.arena_layouts()


def test_a_read_creates_no_file(dataset: Dataset) -> None:
    root = dataset.get_root("arenas")
    before = sorted(root.rglob("*")) if root.exists() else []

    assert dataset.arena_layouts() == {}
    assert dataset.arena_layout("rec", "s1") is None
    assert (sorted(root.rglob("*")) if root.exists() else []) == before

    _ = dataset.set_arena_layout(_two_arenas(), group="rec")
    written = {path: path.stat().st_mtime_ns for path in root.rglob("*")}
    _ = dataset.arena_layouts()
    _ = dataset.arena_layout("rec", "s2", "cam")
    assert {path: path.stat().st_mtime_ns for path in root.rglob("*")} == written


def test_the_path_passes_visit_the_arenas_index(dataset: Dataset) -> None:
    """A layout written with an absolute path in the tree is made relative."""
    from mosaic.core.pipeline.dataset_indexes import (
        iter_dataset_indexes,
        reconcilable_index,
    )

    file = dataset.set_arena_layout(_two_arenas(), group="rec")
    path = arena_index_path(dataset)
    text = path.read_text()
    _ = path.write_text(text.replace("arenas/rec.json", str(file)))

    assert path in {index.path for index in iter_dataset_indexes(dataset)}
    assert reconcilable_index("arenas") is arena_index
    assert dataset.make_portable()[str(path)] == 1
    assert path.read_text() == text
    moved = dataset.rewrite_index_paths({"arenas": "moved"}, dry_run=True)
    assert moved.get(str(path)) == 1
