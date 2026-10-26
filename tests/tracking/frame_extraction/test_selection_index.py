"""The selections index, its inventory kind and the frames it hands to extraction.

``selections/select-frames/index.csv`` has one row per run and entry, each naming the
entry's ``selection.parquet``. ``selected_frames`` reads a run's finished rows into the
``frames`` argument of a ``list`` extraction, and writes nothing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mosaic.core.dataset import Dataset, open_dataset
from mosaic.core.pipeline.dataset_indexes import reconcilable_index
from mosaic.core.pipeline.inventory import FrameSelectionRef
from mosaic.core.scope import Scope
from mosaic.tracking import dataset_inventory
from mosaic.tracking.frame_extraction.selection_index import (
    FRAME_SELECTION_COLUMNS,
    FrameSelectionRow,
    frame_selection_index,
    selected_frames,
    selection_index_path,
    selections_kind_root,
)
from tests.helpers import (
    assert_tree_unchanged,
    make_dataset,
    record_entry_selection,
    tree_snapshot,
)

RUN_ID = "select-frames.0.1-0123456789"
OTHER_RUN_ID = "select-frames.0.1-abcdef0123"
VARIANT = "trex.0.1-0123456789"
REF = FrameSelectionRef(run_id=RUN_ID, source_tracks_run_id=VARIANT)


@pytest.fixture
def dataset(tmp_path: Path) -> Dataset:
    return make_dataset(tmp_path / "ds")


def _row(*, sequence: str, n_selected: int, run_id: str = RUN_ID) -> FrameSelectionRow:
    return FrameSelectionRow(
        run_id=run_id,
        abs_path=Path(f"selections/select-frames/{run_id}/G__{sequence}"),
        group="G",
        sequence=sequence,
        source_tracks_run_id=VARIANT,
        producer="trex",
        n_candidates=2 * n_selected,
        n_selected=n_selected,
        skipped_criteria="[]",
        body_length_px="",
        params_hash="h",
    )


def _select(
    ds: Dataset, sequence: str, frames: list[int], *, finished: bool = True
) -> Path:
    return record_entry_selection(
        ds,
        run_id=RUN_ID,
        group="G",
        sequence=sequence,
        frames=frames,
        finished=finished,
        source_tracks_run_id=VARIANT,
    )


def _reopened(ds: Dataset) -> Dataset:
    return open_dataset(ds.manifest_path, ensure_roots=False)


# --- the index -------------------------------------------------------------------


def test_a_selection_row_has_the_documented_columns() -> None:
    assert FRAME_SELECTION_COLUMNS == (
        "abs_path",
        "run_id",
        "started_at",
        "finished_at",
        "group",
        "sequence",
        "source_tracks_run_id",
        "producer",
        "n_candidates",
        "n_selected",
        "skipped_criteria",
        "body_length_px",
        "params_hash",
    )


def test_the_index_is_in_the_select_frames_directory_of_its_own_root(
    dataset: Dataset,
) -> None:
    """Outside the frames root, whose every child the inventory reads as a method."""
    root = dataset.get_root("selections")

    assert root == dataset.base_dir / "selections"
    assert selection_index_path(dataset) == root / "select-frames" / "index.csv"
    assert not selections_kind_root(dataset).is_relative_to(dataset.get_root("frames"))


def test_the_root_registers_the_opener_for_reconcile() -> None:
    assert reconcilable_index("selections") is frame_selection_index


def test_the_opener_refuses_an_index_in_any_other_directory(tmp_path: Path) -> None:
    stray = tmp_path / "selections" / "extract-frames"
    stray.mkdir(parents=True)
    _ = (stray / "index.csv").write_text("run_id\n")

    with pytest.raises(ValueError, match=r"extract-frames") as raised:
        _ = frame_selection_index(stray / "index.csv")

    assert str(stray) in str(raised.value)


def test_reindex_leaves_an_index_in_any_other_directory_unwritten(
    dataset: Dataset,
) -> None:
    stray = dataset.get_root("selections") / "extract-frames"
    stray.mkdir(parents=True)
    _ = (stray / "index.csv").write_text("run_id,abs_path\nr,gone.parquet\n")
    before = tree_snapshot(dataset.base_dir)

    with pytest.raises(ValueError, match=r"extract-frames"):
        _ = dataset.reindex("selections", dry_run=False)

    assert_tree_unchanged(dataset.base_dir, before)


def test_a_selection_row_replaces_the_row_of_its_run_and_entry(
    dataset: Dataset,
) -> None:
    index = frame_selection_index(selection_index_path(dataset))
    index.append([_row(sequence="S1", n_selected=3), _row(sequence="S2", n_selected=4)])
    index.append([_row(sequence="S1", n_selected=5)])
    index.append([_row(sequence="S1", n_selected=6, run_id=OTHER_RUN_ID)])

    rows = index.read().sort_values(["run_id", "sequence"])

    assert list(
        zip(rows["run_id"], rows["sequence"], rows["n_selected"], strict=True)
    ) == [
        (RUN_ID, "S1", 5),
        (RUN_ID, "S2", 4),
        (OTHER_RUN_ID, "S1", 6),
    ]
    assert rows["body_length_px"].tolist() == ["", "", ""]


def test_reindex_drops_the_row_of_a_missing_selection_file(dataset: Dataset) -> None:
    kept = _select(dataset, "S1", [4])
    _select(dataset, "S2", [9]).unlink()
    path = selection_index_path(dataset)

    dropped = dataset.reindex("selections", dry_run=False)

    assert dropped == {str(path): 1}
    assert frame_selection_index(path).read()["abs_path"].tolist() == [
        dataset.relative_to_root(kept)
    ]


# --- selected_frames -------------------------------------------------------------


def test_the_frames_of_each_entry_come_back_sorted_and_distinct(
    dataset: Dataset,
) -> None:
    _ = _select(dataset, "S2", [30, 5, 30, 12])
    _ = _select(dataset, "S1", [7])

    frames = selected_frames(dataset, RUN_ID)

    assert frames == {("G", "S1"): (7,), ("G", "S2"): (5, 12, 30)}
    assert list(frames) == [("G", "S1"), ("G", "S2")]


def test_an_entry_that_picked_no_frame_is_left_out(dataset: Dataset) -> None:
    """``EntryFrames.indices`` needs at least one frame."""
    _ = _select(dataset, "S1", [7])
    _ = _select(dataset, "S2", [])

    assert selected_frames(dataset, RUN_ID) == {("G", "S1"): (7,)}


def test_an_unfinished_row_is_left_out(dataset: Dataset) -> None:
    _ = _select(dataset, "S1", [7])
    _ = _select(dataset, "S2", [9], finished=False)

    assert selected_frames(dataset, RUN_ID) == {("G", "S1"): (7,)}


def test_only_the_named_run_is_read(dataset: Dataset) -> None:
    _ = _select(dataset, "S1", [7])
    _ = record_entry_selection(
        dataset, run_id=OTHER_RUN_ID, group="G", sequence="S2", frames=[9]
    )

    assert selected_frames(dataset, OTHER_RUN_ID) == {("G", "S2"): (9,)}


def test_an_absent_run_selects_nothing(dataset: Dataset) -> None:
    _ = _select(dataset, "S1", [7])

    assert selected_frames(dataset, OTHER_RUN_ID) == {}


def test_an_absent_or_undeclared_root_selects_nothing(tmp_path: Path) -> None:
    bare = make_dataset(tmp_path / "bare", ensure_roots=False)
    reopened = _reopened(bare)
    before = tree_snapshot(bare.base_dir)

    assert selected_frames(reopened, RUN_ID) == {}
    _ = reopened.roots.pop("selections")
    assert selected_frames(reopened, RUN_ID) == {}
    assert_tree_unchanged(bare.base_dir, before)


def test_a_read_only_selection_read_writes_nothing(dataset: Dataset) -> None:
    _ = _select(dataset, "S1", [7])
    reopened = _reopened(dataset)
    before = tree_snapshot(dataset.base_dir)

    frames = selected_frames(reopened, RUN_ID)

    assert frames == {("G", "S1"): (7,)}
    assert_tree_unchanged(dataset.base_dir, before)


def test_a_finished_row_without_its_file_raises(dataset: Dataset) -> None:
    gone = _select(dataset, "S1", [7])
    gone.unlink()

    with pytest.raises(FileNotFoundError, match=RUN_ID) as raised:
        _ = selected_frames(dataset, RUN_ID)

    assert str(gone) in str(raised.value)
    assert "sequence='S1'" in str(raised.value)


# --- the inventory ---------------------------------------------------------------


def test_a_selection_run_covers_the_entries_of_its_rows(dataset: Dataset) -> None:
    _ = _select(dataset, "S1", [7])
    _ = _select(dataset, "S2", [])

    found = dataset_inventory(dataset, kinds=["frame-selection"])

    record = found.record(REF)
    assert found.errors == ()
    assert record is not None
    assert (record.name, record.status) == ("select-frames", "complete")
    assert found.coverage(REF).present == {("G", "S1"), ("G", "S2")}


def test_a_selection_row_without_its_file_is_an_orphan_row(dataset: Dataset) -> None:
    _select(dataset, "S1", [7]).unlink()
    _ = _select(dataset, "S2", [9])

    record = dataset_inventory(dataset, kinds=["frame-selection"]).record(REF)

    assert record is not None
    assert record.orphan_rows == {("G", "S1")}
    assert record.status == "inconsistent"


def test_a_selection_row_without_a_finish_is_an_unfinished_row(
    dataset: Dataset,
) -> None:
    _ = _select(dataset, "S1", [7])
    _ = _select(dataset, "S2", [9], finished=False)

    record = dataset_inventory(dataset, kinds=["frame-selection"]).record(REF)

    assert record is not None
    assert record.unfinished_rows == {("G", "S2")}


def test_an_entry_scope_reads_only_its_rows(dataset: Dataset) -> None:
    _ = _select(dataset, "S1", [7])
    _ = _select(dataset, "S2", [9])

    found = dataset_inventory(
        dataset, kinds=["frame-selection"], scope=Scope(entries=[("G", "S2")])
    )

    record = found.record(REF)
    assert record is not None
    assert record.rows == {("G", "S2")}


def test_a_selection_is_not_read_as_a_frame_run(dataset: Dataset) -> None:
    _ = _select(dataset, "S1", [7])

    found = dataset_inventory(dataset, kinds=["frame-run", "frame-selection"])

    assert found.of_kind("frame-run") == ()
    assert [record.ref for record in found.of_kind("frame-selection")] == [REF]


def test_a_read_only_inventory_of_an_absent_root_writes_nothing(
    tmp_path: Path,
) -> None:
    bare = make_dataset(tmp_path / "bare", ensure_roots=False)
    before = tree_snapshot(bare.base_dir)

    found = dataset_inventory(_reopened(bare), kinds=["frame-selection"])

    assert found.of_kind("frame-selection") == ()
    assert found.unavailable_kinds == frozenset()
    assert_tree_unchanged(bare.base_dir, before)
