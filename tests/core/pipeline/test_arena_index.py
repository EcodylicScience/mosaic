"""The arenas index's one layout lookup, over index rows, shared by its callers."""

from __future__ import annotations

from pathlib import Path

import pytest

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
    entry_arena_row,
    media_cameras,
    read_arena_index_rows,
    read_arena_layout,
    read_arena_layouts,
)
from mosaic.core.recording import RecordingKey
from tests.helpers import (
    MediaClip,
    add_track_sequences,
    make_dataset,
    write_media_index,
)


def _two_arenas() -> ArenaLayout:
    return ArenaLayout(
        width=200,
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


@pytest.fixture
def dataset(tmp_path: Path) -> Dataset:
    """The divided recording ``rec``, the undivided ``solo`` and the group ``ns``."""
    ds = make_dataset(tmp_path / "ds")
    add_track_sequences(ds, ("rec", "s1"), ("rec", "s2"), ("", "solo"), ("ns", "a"))
    ds.set_continuous_groups(["rec"])
    return ds


def _declare_row(ds: Dataset, key: ArenaKey, like: ArenaKey) -> None:
    """Add a row for *key* that names the file and digest of *like*'s row."""
    model = read_arena_index_rows(ds)[like]
    arena_index(arena_index_path(ds)).append(
        [
            ArenaLayoutRow(
                abs_path=Path(model.stored_path),
                group=key.group,
                sequence=key.sequence,
                camera=key.camera,
                digest=model.digest,
                n_arenas=model.n_arenas,
            )
        ]
    )


class TestEntryArenaRow:
    def test_it_finds_the_row_that_the_dataset_lookup_finds_on_every_key(
        self, dataset: Dataset
    ) -> None:
        entry = ("rec", "s1")
        order = layout_lookup_order(
            RecordingKey(*entry), dataset.recording_of(*entry), "top"
        )
        last = order[-1]
        _ = dataset.set_arena_layout(_two_arenas(), group="rec")
        assert last == ArenaKey("rec", "", "")

        for declared in reversed(order[:-1]):
            _declare_row(dataset, declared, last)
            rows = read_arena_index_rows(dataset)

            found = entry_arena_row(dataset, entry, {"top"}, rows)
            looked_up = dataset.arena_layout(*entry, "top")

            assert found is not None and looked_up is not None
            assert found.key == declared == looked_up.key

    def test_an_undivided_recording_is_found_by_its_entry_key(
        self, dataset: Dataset
    ) -> None:
        _ = dataset.set_arena_layout(_two_arenas(), group="", sequence="solo")
        rows = read_arena_index_rows(dataset)

        found = entry_arena_row(dataset, ("", "solo"), set(), rows)

        assert found is not None and found.key == ArenaKey("", "solo", "")

    def test_it_returns_none_without_a_row(self, dataset: Dataset) -> None:
        assert entry_arena_row(dataset, ("rec", "s1"), {"top"}, {}) is None

    def test_a_camera_without_a_layout_is_passed_over(self, dataset: Dataset) -> None:
        _ = dataset.set_arena_layout(_two_arenas(), group="rec", camera="top")
        rows = read_arena_index_rows(dataset)

        found = entry_arena_row(dataset, ("rec", "s1"), {"top", "side"}, rows)

        assert found is not None and found.key == ArenaKey("rec", "", "top")

    def test_cameras_that_find_different_digests_are_refused(
        self, dataset: Dataset
    ) -> None:
        _ = dataset.set_arena_layout(_two_arenas(), group="rec", camera="top")
        _ = dataset.set_arena_layout(_one_arena(), group="rec", camera="side")
        rows = read_arena_index_rows(dataset)

        with pytest.raises(ValueError, match="find 2 arena layouts"):
            _ = entry_arena_row(dataset, ("rec", "s1"), {"top", "side"}, rows)

    def test_cameras_that_find_one_layout_are_not_refused(
        self, dataset: Dataset
    ) -> None:
        _ = dataset.set_arena_layout(_two_arenas(), group="rec")
        rows = read_arena_index_rows(dataset)

        found = entry_arena_row(dataset, ("rec", "s1"), {"top", "side"}, rows)

        assert found is not None and found.key == ArenaKey("rec", "", "")

    def test_a_sequence_less_row_of_a_group_that_is_not_divided_is_skipped(
        self, dataset: Dataset
    ) -> None:
        _ = dataset.set_arena_layout(_two_arenas(), group="rec")
        _declare_row(dataset, ArenaKey("ns", "", ""), ArenaKey("rec", "", ""))
        rows = read_arena_index_rows(dataset)

        assert entry_arena_row(dataset, ("ns", "a"), set(), rows) is None
        assert dataset.arena_layout("ns", "a") is None


class TestReadArenaIndexRows:
    def test_it_returns_each_key_s_row_without_opening_a_file(
        self, dataset: Dataset
    ) -> None:
        path = dataset.set_arena_layout(_two_arenas(), group="rec")
        path.unlink()

        rows = read_arena_index_rows(dataset)

        assert list(rows) == [ArenaKey("rec", "", "")]
        row = rows[ArenaKey("rec", "", "")]
        assert row.digest == _two_arenas().digest()
        assert row.n_arenas == 2

    def test_a_dataset_without_the_index_has_no_rows(self, dataset: Dataset) -> None:
        assert read_arena_index_rows(dataset) == {}

    def test_a_key_declared_twice_is_refused(self, dataset: Dataset) -> None:
        _ = dataset.set_arena_layout(_two_arenas(), group="rec")
        path = arena_index_path(dataset)
        lines = path.read_text().splitlines()
        _ = path.write_text("\n".join([*lines, lines[-1]]) + "\n")

        with pytest.raises(ArenaIndexError, match="two layouts"):
            _ = read_arena_index_rows(dataset)

    def test_an_unreadable_index_is_refused(self, dataset: Dataset) -> None:
        _ = dataset.set_arena_layout(_two_arenas(), group="rec")
        path = arena_index_path(dataset)
        _ = path.write_text(path.read_text() + '"rec,unterminated\n')

        with pytest.raises(ArenaIndexError, match="cannot be read"):
            _ = read_arena_index_rows(dataset)


class TestReadArenaLayout:
    def test_it_reads_the_file_a_row_names(self, dataset: Dataset) -> None:
        _ = dataset.set_arena_layout(_two_arenas(), group="rec")
        row = read_arena_index_rows(dataset)[ArenaKey("rec", "", "")]

        assert read_arena_layout(dataset, row) == _two_arenas()

    def test_a_file_edited_by_hand_is_refused_against_its_row(
        self, dataset: Dataset
    ) -> None:
        path = dataset.set_arena_layout(_two_arenas(), group="rec")
        row = read_arena_index_rows(dataset)[ArenaKey("rec", "", "")]
        _ = path.write_text(_one_arena().model_dump_json())

        with pytest.raises(ArenaIndexError, match="digest"):
            _ = read_arena_layout(dataset, row)

    def test_a_missing_file_is_refused(self, dataset: Dataset) -> None:
        path = dataset.set_arena_layout(_two_arenas(), group="rec")
        row = read_arena_index_rows(dataset)[ArenaKey("rec", "", "")]
        path.unlink()

        with pytest.raises(ArenaIndexError, match="cannot be read as a layout"):
            _ = read_arena_layout(dataset, row)

    def test_the_layouts_of_a_dataset_are_the_files_its_rows_name(
        self, dataset: Dataset
    ) -> None:
        _ = dataset.set_arena_layout(_two_arenas(), group="rec")
        _ = dataset.set_arena_layout(_one_arena(), group="", sequence="solo")

        declared = read_arena_layouts(dataset)

        assert {key: found.layout for key, found in declared.items()} == {
            ArenaKey("rec", "", ""): _two_arenas(),
            ArenaKey("", "solo", ""): _one_arena(),
        }


class TestAbsoluteStoredPaths:
    """An index row may name a file by the absolute path of another machine."""

    def _store_absolute(self, dataset: Dataset, stored: Path) -> Path:
        """Point the ``rec`` row at *stored*, and return the file it really is."""
        file = dataset.set_arena_layout(_two_arenas(), group="rec")
        index = arena_index_path(dataset)
        cell = f"arenas/{file.name}"
        assert cell in index.read_text()
        _ = index.write_text(index.read_text().replace(cell, str(stored)))
        return file

    def test_a_path_remapped_through_the_path_map_is_read(
        self, dataset: Dataset
    ) -> None:
        file = self._store_absolute(dataset, Path("/other-machine/arenas/rec.json"))
        dataset.remap_roots({"/other-machine/arenas": str(file.parent)})

        rows = read_arena_index_rows(dataset)
        found = dataset.arena_layout("rec", "s1")

        row = rows[ArenaKey("rec", "", "")]
        assert read_arena_layout(dataset, row) == _two_arenas()
        assert found is not None and found.path == file

    def test_a_missing_absolute_file_leaves_the_index_readable(
        self, dataset: Dataset
    ) -> None:
        _ = self._store_absolute(dataset, Path("/other-machine/arenas/rec.json"))

        rows = read_arena_index_rows(dataset)

        assert list(rows) == [ArenaKey("rec", "", "")]
        with pytest.raises(ArenaIndexError, match="cannot be read as a layout"):
            _ = read_arena_layout(dataset, rows[ArenaKey("rec", "", "")])


def test_media_cameras_lists_each_entry_s_cameras(dataset: Dataset) -> None:
    write_media_index(
        dataset,
        [
            MediaClip(group="rec", sequence="s1", camera="top", filename="a.mp4"),
            MediaClip(group="rec", sequence="s1", camera="side", filename="b.mp4"),
            MediaClip(group="rec", sequence="s2", camera="", filename="c.mp4"),
        ],
    )

    assert media_cameras(dataset) == {
        ("rec", "s1"): {"top", "side"},
        ("rec", "s2"): {""},
    }
