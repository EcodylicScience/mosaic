"""Indexing the tables of a tracks variant that are already under ``tracks/``.

The source is a dataset whose raw TREx files were converted. Its variant
directory is copied into a fresh dataset with an empty tracks index, and the
rows that ``index_tracks_variant`` writes there are compared with the
conversion's.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Final

import pandas as pd
import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.graph import Plan, plan_pipeline
from mosaic.core.pipeline.tracks_index import (
    index_tracks_variant,
    read_carries_tracklets,
    read_tracks_index,
    tracks_index_path,
)
from tests.helpers import (
    CONVERTED_ENTRIES,
    assert_tree_unchanged,
    converted_source,
    make_dataset,
    speed_recipe,
    tree_snapshot,
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
