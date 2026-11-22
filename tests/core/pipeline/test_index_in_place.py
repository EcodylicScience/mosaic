"""Indexing the tables of a variant that are already under ``tracks/`` or ``labels/``.

The source is a dataset whose raw files were converted. Its variant directory is
copied into a fresh dataset with an empty index, and the rows that
``index_tracks_variant`` or ``index_labels_variant`` write there are compared with
the conversion's.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.entry import EntryKeyCollisionError
from mosaic.core.pipeline.graph import Plan, plan_pipeline
from mosaic.core.pipeline.labels_index import (
    index_labels_variant,
    labels_index_path,
    read_labels_index,
)
from mosaic.core.pipeline.tracks_index import (
    index_tracks_variant,
    read_carries_tracklets,
    read_tracks_index,
    tracks_index_path,
)
from tests.helpers import (
    CONVERTED_ENTRIES,
    MediaClip,
    assert_tree_unchanged,
    calms21_dataset,
    converted_source,
    make_dataset,
    speed_recipe,
    tree_snapshot,
    write_media_index,
)

CONVERSION_ONLY: Final = [
    "started_at",
    "finished_at",
    "source_abs_path",
    "source_md5",
    "consumed_source_roots",
    "consumed_composition",
    "first_frame",
    "producer",
]
"""The columns that a row of the conversion fills and an indexed row does not."""

BLANK: Final = [
    "source_abs_path",
    "source_md5",
    "consumed_source_roots",
    "consumed_composition",
    "first_frame",
]
"""The columns that an indexed row leaves blank."""


class Indexed:
    """A source, its one tracks variant and a fresh dataset holding a copy of it."""

    def __init__(self, tmp_path: Path) -> None:
        self.source: Dataset = converted_source(tmp_path / "source")
        rows = read_tracks_index(self.source)
        self.variant: str = str(rows["run_id"].iloc[0])
        self.source_rows: pd.DataFrame = rows
        self.std_format: str = str(rows["std_format"].iloc[0])
        self.carries_tracklets: bool | None = read_carries_tracklets(rows.iloc[0])
        self.dataset: Dataset = make_dataset(tmp_path / "indexed", name="indexed")
        self.root: Path = self.place(self.variant)

    def place(self, name: str) -> Path:
        """Copy the tables and sidecar of the source's variant to ``tracks/<name>``."""
        origin = self.source.get_root("tracks") / self.variant
        destination = self.dataset.get_root("tracks") / name
        destination.mkdir(parents=True)
        for path in origin.iterdir():
            if path.suffix in (".parquet", ".json"):
                _ = shutil.copy2(path, destination / path.name)
        return destination

    def index(self, variant: str | None = None) -> int:
        return index_tracks_variant(
            self.dataset,
            self.variant if variant is None else variant,
            std_format=self.std_format,
            carries_tracklets=self.carries_tracklets,
        )

    def rows(self) -> pd.DataFrame:
        return read_tracks_index(self.dataset)


@pytest.fixture
def indexed(tmp_path: Path) -> Indexed:
    return Indexed(tmp_path)


def _keyed(frame: pd.DataFrame, drop: list[str]) -> pd.DataFrame:
    """*frame* ordered by entry, without the columns in *drop*."""
    ordered = frame.sort_values(["group", "sequence"]).reset_index(drop=True)
    return ordered.drop(columns=drop)


def test_the_tables_are_indexed_as_the_conversion_indexed_them(
    indexed: Indexed,
) -> None:
    assert indexed.rows().empty

    written = indexed.index()

    assert written == len(CONVERTED_ENTRIES)
    rows = indexed.rows()
    assert rows["producer"].tolist() == [indexed.source_rows["producer"].iloc[0]] * 2
    for column in BLANK:
        assert rows[column].tolist() == ["", ""], column
    pd.testing.assert_frame_equal(
        _keyed(rows, CONVERSION_ONLY), _keyed(indexed.source_rows, CONVERSION_ONLY)
    )


def _facts(plan: Plan) -> list[tuple[str, str | None, str]]:
    return [(step.step_id, step.run_id, str(step.status)) for step in plan.steps]


def test_a_recipe_plans_over_the_indexed_dataset_as_over_the_source(
    indexed: Indexed,
) -> None:
    _ = indexed.index()

    on_source = plan_pipeline(
        indexed.source, speed_recipe(), tracks_run_id=indexed.variant
    )
    on_indexed = plan_pipeline(
        indexed.dataset, speed_recipe(), tracks_run_id=indexed.variant
    )

    assert _facts(on_indexed) == _facts(on_source)
    assert on_indexed.problems == on_source.problems


def test_a_rerun_leaves_the_rows_equal_apart_from_the_start_time(
    indexed: Indexed,
) -> None:
    _ = indexed.index()
    first = indexed.rows()

    written = indexed.index()

    assert written == len(CONVERTED_ENTRIES)
    pd.testing.assert_frame_equal(
        _keyed(indexed.rows(), ["started_at"]), _keyed(first, ["started_at"])
    )


def test_the_variant_of_a_conversion_is_refused_and_the_tree_is_left_alone(
    tmp_path: Path,
) -> None:
    source = converted_source(tmp_path / "source")
    rows = read_tracks_index(source)
    before = tree_snapshot(source.base_dir)

    with pytest.raises(ValueError, match="source"):
        _ = index_tracks_variant(
            source,
            str(rows["run_id"].iloc[0]),
            std_format=str(rows["std_format"].iloc[0]),
            carries_tracklets=read_carries_tracklets(rows.iloc[0]),
        )

    assert_tree_unchanged(source.base_dir, before)


def test_a_staged_table_and_a_stray_file_are_not_indexed(indexed: Indexed) -> None:
    _ = shutil.copy2(
        indexed.root / "g__a.parquet", indexed.root / ".stage-g__c.parquet"
    )
    _ = (indexed.root / "notes.txt").write_text("not a table", encoding="utf-8")

    assert indexed.index() == len(CONVERTED_ENTRIES)
    assert sorted(indexed.rows()["sequence"]) == ["a", "b"]


def _drop_sidecar(root: Path) -> None:
    (root / "params.json").unlink()


def _change_op(root: Path) -> None:
    sidecar = json.loads((root / "params.json").read_text())
    sidecar["op"] = "convert-other"
    _ = (root / "params.json").write_text(json.dumps(sidecar), encoding="utf-8")


def _rewrite(table: Path, edit: Callable[[pd.DataFrame], None]) -> None:
    frame = pd.read_parquet(table)
    edit(frame)
    frame.to_parquet(table)


def _other_group(root: Path) -> None:
    def edit(frame: pd.DataFrame) -> None:
        frame["group"] = "other"

    _rewrite(root / "g__a.parquet", edit)


def _other_sequence(root: Path) -> None:
    def edit(frame: pd.DataFrame) -> None:
        frame["sequence"] = "other"

    _rewrite(root / "g__a.parquet", edit)


def _two_sequences(root: Path) -> None:
    def edit(frame: pd.DataFrame) -> None:
        frame.loc[frame.index[0], "sequence"] = "a2"

    _rewrite(root / "g__a.parquet", edit)


def _missing_column(root: Path) -> None:
    def edit(frame: pd.DataFrame) -> None:
        _ = frame.pop("X")

    _rewrite(root / "g__a.parquet", edit)


@pytest.mark.parametrize(
    ("spoil", "reason"),
    [
        (_drop_sidecar, "params.json"),
        (_change_op, "op"),
        (_other_group, "g__a.parquet"),
        (_other_sequence, "g__a.parquet"),
        (_two_sequences, "g__a.parquet"),
        (_missing_column, "g__a.parquet"),
    ],
    ids=[
        "no-sidecar",
        "other-op",
        "other-group",
        "other-sequence",
        "two-sequences",
        "missing-column",
    ],
)
def test_a_refusal_raises_and_leaves_the_tree_unchanged(
    indexed: Indexed, spoil: Callable[[Path], None], reason: str
) -> None:
    spoil(indexed.root)
    before = tree_snapshot(indexed.dataset.base_dir)

    with pytest.raises(ValueError, match=reason):
        _ = indexed.index()

    assert_tree_unchanged(indexed.dataset.base_dir, before)


def test_an_id_that_is_not_an_op_run_id_is_refused(indexed: Indexed) -> None:
    _ = indexed.place("trex-0123456789")
    before = tree_snapshot(indexed.dataset.base_dir)

    with pytest.raises(ValueError, match="trex-0123456789"):
        _ = indexed.index("trex-0123456789")

    assert_tree_unchanged(indexed.dataset.base_dir, before)


def test_a_refusal_on_the_second_table_writes_neither_row(indexed: Indexed) -> None:
    def edit(frame: pd.DataFrame) -> None:
        _ = frame.pop("X")

    _rewrite(indexed.root / "g__b.parquet", edit)
    before = tree_snapshot(indexed.dataset.base_dir)

    with pytest.raises(ValueError, match=r"g__b\.parquet"):
        _ = indexed.index()

    assert_tree_unchanged(indexed.dataset.base_dir, before)
    assert not tracks_index_path(indexed.dataset).exists() or indexed.rows().empty


LABELS_ONLY: Final = [
    "started_at",
    "finished_at",
    "source_abs_path",
    "source_md5",
    "consumed_source_roots",
    "consumed_composition",
]
"""The columns that a conversion of labels fills and an indexed row does not."""

LABEL_KIND: Final = "behavior"


class IndexedLabels:
    """A source with converted labels and a fresh dataset holding a copy of them."""

    def __init__(self, source: Dataset, tmp_path: Path, kind: str = LABEL_KIND) -> None:
        self.source: Dataset = source
        self.kind: str = kind
        self.source_rows: pd.DataFrame = read_labels_index(source, kind)
        self.variant: str = str(self.source_rows["run_id"].iloc[0])
        self.dataset: Dataset = make_dataset(tmp_path / "indexed", name="indexed")
        self.root: Path = self.place(self.variant)

    def place(self, name: str) -> Path:
        """Copy the source's variant, files and sidecar, to ``labels/<kind>/<name>``."""
        origin = self.source.get_root("labels") / self.kind / self.variant
        destination = self.dataset.get_root("labels") / self.kind / name
        destination.mkdir(parents=True)
        for path in origin.iterdir():
            if path.suffix in (".npz", ".json"):
                _ = shutil.copy2(path, destination / path.name)
        return destination

    def index(self, variant: str | None = None) -> int:
        return index_labels_variant(
            self.dataset, self.kind, self.variant if variant is None else variant
        )

    def rows(self) -> pd.DataFrame:
        return read_labels_index(self.dataset, self.kind)


def _calms21_labels(tmp_path: Path) -> IndexedLabels:
    """Dense labels: CalMS21 annotations of 4 and 3 frames, as ``behavior``."""
    source = calms21_dataset(
        tmp_path / "source", {"train": {"ann": {"s1": 4, "s2": 3}}}
    )
    _ = source.convert_all_labels(kind=LABEL_KIND, source_format="calms21_npy")
    return IndexedLabels(source, tmp_path)


def _predictions_labels(tmp_path: Path) -> IndexedLabels:
    """Event labels: predictions at frames 0, 5 and 9 of one sequence, 2 of another."""
    source = make_dataset(tmp_path / "source", name="source")
    root = source.get_root("labels_raw") / "predictions"
    root.mkdir(parents=True)
    pd.DataFrame(
        {
            "sequence": ["s1", "s1", "s1", "s2"],
            "frame": [0, 5, 9, 2],
            "predicted_label": [0, 1, 0, 1],
        }
    ).to_parquet(root / "run.parquet", index=False)
    source.index_labels_raw(
        [root], patterns=["*.parquet"], src_format="predictions_table"
    )
    _ = source.convert_all_labels(kind="predictions", source_format="predictions_table")
    return IndexedLabels(source, tmp_path, "predictions")


@pytest.fixture
def indexed_labels(tmp_path: Path) -> IndexedLabels:
    return _calms21_labels(tmp_path)


@pytest.fixture
def indexed_predictions(tmp_path: Path) -> IndexedLabels:
    return _predictions_labels(tmp_path)


def test_the_label_files_are_indexed_as_the_conversion_indexed_them(
    indexed_labels: IndexedLabels,
) -> None:
    assert indexed_labels.rows().empty

    written = indexed_labels.index()

    assert written == 2
    rows = indexed_labels.rows()
    assert rows["producer"].tolist() == ["convert-labels-calms21_npy"] * 2
    for column in ("source_abs_path", "source_md5", "consumed_source_roots"):
        assert rows[column].tolist() == ["", ""], column
    assert rows["n_frames"].tolist() == ["4", "3"]
    pd.testing.assert_frame_equal(
        _keyed(rows, LABELS_ONLY), _keyed(indexed_labels.source_rows, LABELS_ONLY)
    )


def test_a_rerun_of_the_labels_leaves_the_rows_equal_apart_from_the_start_time(
    indexed_labels: IndexedLabels,
) -> None:
    _ = indexed_labels.index()
    first = indexed_labels.rows()

    written = indexed_labels.index()

    assert written == 2
    pd.testing.assert_frame_equal(
        _keyed(indexed_labels.rows(), ["started_at"]), _keyed(first, ["started_at"])
    )


def test_the_variant_of_a_labels_conversion_is_refused_and_the_tree_is_left_alone(
    tmp_path: Path,
) -> None:
    labels = _calms21_labels(tmp_path)
    before = tree_snapshot(labels.source.base_dir)

    with pytest.raises(ValueError, match="source"):
        _ = index_labels_variant(labels.source, LABEL_KIND, labels.variant)

    assert_tree_unchanged(labels.source.base_dir, before)


def test_a_staged_file_and_a_stray_file_are_not_indexed_as_labels(
    indexed_labels: IndexedLabels,
) -> None:
    _ = shutil.copy2(
        indexed_labels.root / "train__s1.npz", indexed_labels.root / ".stage-x.npz"
    )
    _ = (indexed_labels.root / "notes.txt").write_text("not labels", encoding="utf-8")

    assert indexed_labels.index() == 2
    assert sorted(indexed_labels.rows()["sequence"]) == ["s1", "s2"]


def _drop_label_sidecar(root: Path) -> None:
    (root / "params.json").unlink()


def _change_label_op(root: Path) -> None:
    sidecar = json.loads((root / "params.json").read_text())
    sidecar["op"] = "convert-labels-other"
    _ = (root / "params.json").write_text(json.dumps(sidecar), encoding="utf-8")


def _change_label_kind(root: Path) -> None:
    sidecar = json.loads((root / "params.json").read_text())
    sidecar["kind"] = "other"
    _ = (root / "params.json").write_text(json.dumps(sidecar), encoding="utf-8")


def _drop_label_key(root: Path, key: str) -> None:
    path = root / "train__s1.npz"
    with np.load(path, allow_pickle=True) as loaded:
        payload = {name: loaded[name] for name in loaded.files if name != key}
    np.savez(path, allow_pickle=True, **payload)


def _rewrite_array(path: Path, key: str, value: object) -> None:
    with np.load(path, allow_pickle=True) as loaded:
        payload = {name: loaded[name] for name in loaded.files}
    payload[key] = np.asarray(value)
    np.savez(path, allow_pickle=True, **payload)


def _other_label_group(root: Path) -> None:
    _rewrite_array(root / "train__s1.npz", "group", "other")


def _other_label_sequence(root: Path) -> None:
    _rewrite_array(root / "train__s1.npz", "sequence", "other")


@pytest.mark.parametrize(
    ("spoil", "reason"),
    [
        (_drop_label_sidecar, "params.json"),
        (_change_label_op, "op"),
        (_change_label_kind, "kind"),
        (_other_label_group, "train__s1.npz"),
        (_other_label_sequence, "train__s1.npz"),
        (partial(_drop_label_key, key="label_format"), "label_format"),
        (partial(_drop_label_key, key="frames"), "frames"),
        (partial(_drop_label_key, key="label_ids"), "label_ids"),
        (partial(_drop_label_key, key="label_names"), "label_names"),
        (partial(_drop_label_key, key="group"), "'group' scalar"),
        (partial(_drop_label_key, key="sequence"), "'sequence' scalar"),
    ],
    ids=[
        "no-sidecar",
        "other-op",
        "other-kind",
        "other-group",
        "other-sequence",
        "no-label-format",
        "no-frames",
        "no-label-ids",
        "no-label-names",
        "no-group",
        "no-sequence",
    ],
)
def test_a_labels_refusal_raises_and_leaves_the_tree_unchanged(
    indexed_labels: IndexedLabels, spoil: Callable[[Path], None], reason: str
) -> None:
    spoil(indexed_labels.root)
    before = tree_snapshot(indexed_labels.dataset.base_dir)

    with pytest.raises(ValueError, match=reason):
        _ = indexed_labels.index()

    assert_tree_unchanged(indexed_labels.dataset.base_dir, before)


def test_a_labels_id_that_is_not_an_op_run_id_is_refused(
    indexed_labels: IndexedLabels,
) -> None:
    _ = indexed_labels.place("trex-0123456789")
    before = tree_snapshot(indexed_labels.dataset.base_dir)

    with pytest.raises(ValueError, match="trex-0123456789"):
        _ = indexed_labels.index("trex-0123456789")

    assert_tree_unchanged(indexed_labels.dataset.base_dir, before)


def test_a_refusal_on_the_second_label_file_writes_neither_row(
    indexed_labels: IndexedLabels,
) -> None:
    _rewrite_array(indexed_labels.root / "train__s2.npz", "sequence", "other")
    before = tree_snapshot(indexed_labels.dataset.base_dir)

    with pytest.raises(ValueError, match=r"train__s2\.npz"):
        _ = indexed_labels.index()

    assert_tree_unchanged(indexed_labels.dataset.base_dir, before)
    assert not labels_index_path(indexed_labels.dataset, LABEL_KIND).exists()


def test_a_label_file_whose_entry_key_collides_with_a_media_entry_writes_no_row(
    indexed_labels: IndexedLabels,
) -> None:
    write_media_index(indexed_labels.dataset, [MediaClip(sequence="train__s1")])
    before = tree_snapshot(indexed_labels.dataset.base_dir)

    with pytest.raises(EntryKeyCollisionError):
        _ = indexed_labels.index()

    assert_tree_unchanged(indexed_labels.dataset.base_dir, before)
    assert not labels_index_path(indexed_labels.dataset, LABEL_KIND).exists()


def test_an_event_file_is_indexed_with_the_frame_count_of_its_conversion(
    indexed_predictions: IndexedLabels,
) -> None:
    written = indexed_predictions.index()

    assert written == 2
    rows = indexed_predictions.rows()
    assert rows["n_frames"].tolist() == ["10", "3"]
    pd.testing.assert_frame_equal(
        _keyed(rows, LABELS_ONLY), _keyed(indexed_predictions.source_rows, LABELS_ONLY)
    )


def test_a_file_without_frames_is_indexed_with_no_frames(
    indexed_labels: IndexedLabels,
) -> None:
    _rewrite_array(
        indexed_labels.root / "train__s1.npz", "frames", np.empty(0, dtype=np.int32)
    )

    assert indexed_labels.index() == 2

    rows = indexed_labels.rows().set_index("sequence")
    assert rows.loc["s1", "n_frames"] == "0"
    assert rows.loc["s2", "n_frames"] == "3"
