"""``assign-arenas``: each row's arena from its body center, and ids kept per arena."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mosaic.core.arenas import ArenaLayout, Rect
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.arena_assignment import (
    ArenaCapacityError,
    assign_arenas,
)
from mosaic.core.pipeline.ops import run_op
from mosaic.core.pipeline.run_log import read_runs, run_log_dir, run_log_path
from mosaic.core.pipeline.tracks_identity import tracks_variant_root
from mosaic.core.pipeline.tracks_index import (
    read_tracks_index,
    tracks_compositions,
    write_tracks_row,
)
from mosaic.core.pipeline.writers import write_parquet_atomic
from mosaic.core.scope import Scope
from tests.helpers import MediaClip, make_dataset, write_media_index

XS = (0, 506, 1012)
YS = (0, 170, 346, 524, 702, 878, 1068)
STRIP = 95, 105
SOURCE = "convert-trex_npz.0.2-aaaaaaaaaa"


def _strip(row: int) -> Rect:
    """The mirrored strip of the left tank in grid row *row*, from 0."""
    return Rect(x0=STRIP[0], y0=YS[row], x1=STRIP[1], y1=YS[row + 1])


def princeton_layout() -> ArenaLayout:
    """The twelve tanks of a Princeton camera, the left ones without their strip.

    The frame is wider and taller than the grid, so a row can lie in no tank.
    """
    return ArenaLayout.grid(
        xs=XS,
        ys=YS,
        width=1024,
        height=1080,
        exclude={f"A{row + 1}": [_strip(row)] for row in range(6)},
    )


def _centers() -> list[tuple[float, float]]:
    """A point inside each tank, clear of the strip, in ordinal order."""
    return [
        (x0 + 300.0, (y0 + y1) / 2)
        for x0 in XS[:-1]
        for y0, y1 in zip(YS[:-1], YS[1:], strict=True)
    ]


def princeton_table(
    sequence: str, *, frames: range, id_of: dict[int, int], group: str = "rec"
) -> pd.DataFrame:
    """One fish per tank in every frame, its tracker id ``id_of[ordinal]``."""
    rows = [
        {
            "frame": frame,
            "time": frame / 20.0,
            "id": id_of[ordinal],
            "group": group,
            "sequence": sequence,
            "X": x,
            "Y": y,
            "confidence": 0.9,
        }
        for frame in frames
        for ordinal, (x, y) in enumerate(_centers())
    ]
    return pd.DataFrame(rows)


def _ids(offset: int) -> dict[int, int]:
    return {ordinal: offset + 11 - ordinal for ordinal in range(12)}


class TestAssignArenas:
    def test_each_fish_is_named_by_its_tank_and_its_tank_s_ordinal(self) -> None:
        table = princeton_table("s1", frames=range(3), id_of=_ids(40))

        assigned, report = assign_arenas(table, princeton_layout())

        first = assigned[assigned["frame"] == 0]
        assert first["arena"].tolist() == list(princeton_layout().names)
        assert first["id"].tolist() == list(range(12))
        assert report.rows == 36
        assert set(report.kept.values()) == {3}
        assert sum(report.dropped.values()) == 0
        assert assigned["frame"].tolist() == table["frame"].tolist()

    def test_the_mirrored_strip_is_dropped_and_counted(self) -> None:
        table = princeton_table("s1", frames=range(2), id_of=_ids(0))
        reflection = table.iloc[[0]].assign(X=100.0, Y=85.0, id=99)

        assigned, report = assign_arenas(
            pd.concat([table, reflection], ignore_index=True), princeton_layout()
        )

        assert len(assigned) == len(table)
        assert report.dropped["excluded"] == 1
        assert report.kept["A1"] == 2

    def test_a_row_in_no_tank_or_without_a_center_is_dropped_and_counted(
        self,
    ) -> None:
        table = princeton_table("s1", frames=range(1), id_of=_ids(0))
        strays = pd.concat(
            [
                table.iloc[[0]].assign(X=1018.0, id=70),
                table.iloc[[1]].assign(Y=1074.0, id=71),
                table.iloc[[2]].assign(X=np.nan, id=72),
            ],
            ignore_index=True,
        )

        assigned, report = assign_arenas(
            pd.concat([table, strays], ignore_index=True), princeton_layout()
        )

        assert len(assigned) == 12
        assert report.dropped == {
            "no_position": 1,
            "outside": 2,
            "excluded": 0,
            "over_capacity": 0,
        }

    def test_above_capacity_the_most_confident_row_stays(self) -> None:
        table = princeton_table("s1", frames=range(1), id_of=_ids(0))
        x, y = _centers()[3]
        rival = table.iloc[[3]].assign(X=x + 5, Y=y, id=50, confidence=0.95)
        weak = table.iloc[[3]].assign(X=x - 5, Y=y, id=51, confidence=np.nan)

        assigned, report = assign_arenas(
            pd.concat([table, weak, rival], ignore_index=True), princeton_layout()
        )

        kept = assigned[assigned["arena"] == "A4"]
        assert kept["X"].tolist() == [x + 5]
        assert kept["id"].tolist() == [3]
        assert report.dropped["over_capacity"] == 2

    def test_above_capacity_without_confidence_is_refused(self) -> None:
        table = princeton_table("s1", frames=range(1), id_of=_ids(0))
        table = pd.concat([table, table.iloc[[0]].assign(id=50)], ignore_index=True)

        with pytest.raises(ArenaCapacityError, match="'A1'.*frame 0"):
            _ = assign_arenas(table.drop(columns="confidence"), princeton_layout())

    def test_ids_of_a_shared_arena_number_its_individuals(self) -> None:
        layout = ArenaLayout.grid(xs=(0, 50, 100), ys=(0, 100), capacity=None)
        table = pd.DataFrame(
            {
                "frame": [0, 0, 0, 1, 1],
                "id": [7, 3, 9, 3, 7],
                "X": [10.0, 20.0, 70.0, 20.0, 60.0],
                "Y": [50.0] * 5,
            }
        )

        assigned, _ = assign_arenas(table, layout)

        assert assigned["arena"].tolist() == ["A1", "A1", "B1", "A1", "B1"]
        assert assigned["id"].tolist() == [1, 0, 3, 0, 2]

    def test_a_table_without_a_center_is_refused(self) -> None:
        with pytest.raises(ValueError, match="'X'"):
            _ = assign_arenas(
                pd.DataFrame({"frame": [0], "id": [0], "Y": [1.0]}), princeton_layout()
            )


def _seed(ds: Dataset, table: pd.DataFrame, first_frame: int | None) -> None:
    """Write *table*, one entry, into the source variant."""
    group = str(table["group"].iloc[0])
    sequence = str(table["sequence"].iloc[0])
    out = (
        tracks_variant_root(ds.get_root("tracks"), SOURCE)
        / f"{group}__{sequence}.parquet"
    )
    _ = write_parquet_atomic(table, out)
    write_tracks_row(
        ds,
        run_id=SOURCE,
        group=group,
        sequence=sequence,
        out_path=out,
        producer="convert-trex_npz",
        std_format="mosaic_v1",
        n_rows=len(table),
        consumed_source_roots=("tracks_raw",),
        first_frame=first_frame,
    )


@pytest.fixture
def dataset(tmp_path: Path) -> Dataset:
    """Two hourly sequences of one divided recording, tracked with other ids."""
    ds = make_dataset(tmp_path / "ds")
    _seed(ds, princeton_table("s1", frames=range(0, 4), id_of=_ids(0)), None)
    _seed(ds, princeton_table("s2", frames=range(4, 8), id_of=_ids(20)), 4)
    ds.set_continuous_groups(["rec"])
    _ = ds.set_arena_layout(princeton_layout(), group="rec")
    return ds


def _run_events(ds: Dataset, event: str) -> list[dict[str, object]]:
    """Return the *event* records of the latest assign-arenas run's log."""
    latest = max(
        read_runs(run_log_dir(ds.base_dir), kind="assign-arenas"),
        key=lambda run: run["started_at"],
    )
    lines = run_log_path(ds.base_dir, latest["execution_id"]).read_text().splitlines()
    records = [json.loads(line) for line in lines]
    return [record for record in records if record["ev"] == event]


def _assigned_rows(ds: Dataset) -> pd.DataFrame:
    index = read_tracks_index(ds)
    return index[index["producer"] == "assign-arenas"].reset_index(drop=True)


class TestAssignArenasOp:
    def test_ids_name_one_tank_in_both_sequences(self, dataset: Dataset) -> None:
        _ = run_op(dataset, "assign-arenas", {}, track=False)

        rows = _assigned_rows(dataset)
        assert sorted(rows["sequence"]) == ["s1", "s2"]
        assert rows["run_id"].iloc[0].startswith("assign-arenas.0.1-")
        tables = {
            row["sequence"]: pd.read_parquet(dataset.resolve_path(row["abs_path"]))
            for _, row in rows.iterrows()
        }
        for table in tables.values():
            by_arena = table.groupby("arena")["id"].unique()
            assert {arena: list(ids) for arena, ids in by_arena.items()} == {
                name: [ordinal] for ordinal, name in enumerate(princeton_layout().names)
            }

    def test_the_variant_keeps_its_source_s_frame_axis(self, dataset: Dataset) -> None:
        _ = run_op(dataset, "assign-arenas", {}, track=False)

        rows = _assigned_rows(dataset).set_index("sequence")
        assert rows.loc["s2", "first_frame"] == "4"
        assert rows.loc["s1", "first_frame"] == ""
        assert (rows.loc["s2", "frame_min"], rows.loc["s2", "frame_max"]) == ("4", "7")
        assert set(rows["consumed_media_composition"]) == {""}

    def test_the_layout_digest_is_in_each_table_s_composition(
        self, dataset: Dataset
    ) -> None:
        _ = run_op(dataset, "assign-arenas", {}, track=False)
        variant = str(_assigned_rows(dataset)["run_id"].iloc[0])
        digest = princeton_layout().digest()

        compositions = tracks_compositions(dataset, [variant])

        assert all(f"arenas={digest}" in cell for cell in compositions.values())

    def test_a_new_layout_recomputes_under_the_same_variant(
        self, dataset: Dataset
    ) -> None:
        first = run_op(dataset, "assign-arenas", {}, track=False)
        again = run_op(dataset, "assign-arenas", {}, track=False)
        before = _assigned_rows(dataset)
        moved = ArenaLayout.grid(xs=XS, ys=YS, width=1024, height=1080)
        _ = dataset.set_arena_layout(moved, group="rec")

        changed = run_op(dataset, "assign-arenas", {}, track=False)

        after = _assigned_rows(dataset)
        assert first == again != changed
        assert list(after["run_id"]) == list(before["run_id"])
        assert all(moved.digest() in cell for cell in after["consumed_composition"])

    def test_an_entry_without_a_layout_is_named(self, tmp_path: Path) -> None:
        ds = make_dataset(tmp_path / "ds")
        _seed(
            ds, princeton_table("lone", frames=range(2), id_of=_ids(0), group=""), None
        )

        with pytest.raises(RuntimeError, match="1 of 1 entries unwritten"):
            _ = run_op(ds, "assign-arenas", {}, track=False)

    def test_a_scoped_run_reads_only_its_entries(self, dataset: Dataset) -> None:
        _ = run_op(
            dataset,
            "assign-arenas",
            {},
            track=False,
            scope=Scope(entries=[("rec", "s2")]),
        )

        assert list(_assigned_rows(dataset)["sequence"]) == ["s2"]

    def test_an_entry_s_dropped_rows_are_reported_in_the_run_log(
        self, tmp_path: Path
    ) -> None:
        ds = make_dataset(tmp_path / "ds")
        clean = princeton_table("s1", frames=range(2), id_of=_ids(0))
        table = princeton_table("s2", frames=range(2), id_of=_ids(0))
        reflection = table.iloc[[0]].assign(X=100.0, Y=85.0, id=99)
        _seed(ds, clean, None)
        _seed(ds, pd.concat([table, reflection], ignore_index=True), None)
        ds.set_continuous_groups(["rec"])
        _ = ds.set_arena_layout(princeton_layout(), group="rec")

        _ = run_op(ds, "assign-arenas", {})

        events = _run_events(ds, "rows_dropped")
        assert [(event["key"], event["counts"]) for event in events] == [
            (
                "rec__s2",
                {
                    "no_position": 0,
                    "outside": 0,
                    "excluded": 1,
                    "over_capacity": 0,
                },
            )
        ]

    def test_cameras_that_find_different_layouts_fail_their_entry_by_name(
        self, dataset: Dataset
    ) -> None:
        write_media_index(
            dataset,
            [
                MediaClip(group="rec", sequence="s1", camera=camera, filename=name)
                for camera, name in (("top", "s1-top.mp4"), ("side", "s1-side.mp4"))
            ],
        )
        moved = ArenaLayout.grid(xs=XS, ys=YS, width=1024, height=1080)
        _ = dataset.set_arena_layout(moved, group="rec", camera="side")

        with pytest.raises(RuntimeError, match="1 of 2 entries unwritten"):
            _ = run_op(dataset, "assign-arenas", {})

        (failure,) = _run_events(dataset, "entry_error")
        assert failure["key"] == "rec__s1"
        assert "find 2 arena layouts" in str(failure["error"])
