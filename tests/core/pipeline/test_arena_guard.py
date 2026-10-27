"""``run_feature`` refuses a feature that relates individuals across arenas on several."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mosaic.core.arenas import ArenaLayout
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.arena_guard import (
    ArenaBlindFeature,
    entry_arena_count,
    refuse_arena_blind,
)
from mosaic.core.pipeline.run import AllEntriesFailed, run_feature
from tests.helpers import (
    add_track_sequences,
    add_tracks_table,
    make_dataset,
    track_table,
)

TWO = ArenaLayout.grid(xs=(0, 50, 100), ys=(0, 100))
ONE = ArenaLayout.grid(xs=(0, 100), ys=(0, 100))


def test_an_input_of_two_arenas_is_refused() -> None:
    table = pd.DataFrame({"id": [0, 1], "arena": ["A1", "B1"]})

    with pytest.raises(ArenaBlindFeature, match="holds 2 \\(A1, B1\\)"):
        refuse_arena_blind("nearest-neighbor", table, 2)
    refuse_arena_blind("nearest-neighbor", table.assign(arena="A1"), 2)


def test_an_input_without_arenas_is_refused_on_a_layout_of_several() -> None:
    table = pd.DataFrame({"id": [0, 1]})

    with pytest.raises(ArenaBlindFeature, match="layout of 2 arenas"):
        refuse_arena_blind("nearest-neighbor", table, 2)
    refuse_arena_blind("nearest-neighbor", table, 1)
    refuse_arena_blind("nearest-neighbor", table, 0)


def test_the_count_reads_the_layouts_of_the_entry_s_recording(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds")
    add_track_sequences(ds, ("rec", "s1"), ("ns", "a"), ("", "solo"))
    ds.set_continuous_groups(["rec"])
    _ = ds.set_arena_layout(ONE, group="rec")
    _ = ds.set_arena_layout(TWO, group="rec", camera="side")
    _ = ds.set_arena_layout(ONE, group="ns", sequence="a")
    declared = ds.arena_layouts()

    assert entry_arena_count(ds, "rec", "s1", declared) == 2
    assert entry_arena_count(ds, "ns", "a", declared) == 1
    assert entry_arena_count(ds, "", "solo", declared) == 0


def _two_tanks(ds: Dataset, *, arenas: bool) -> None:
    """One entry of two fish, one to a tank, with ``arena`` when *arenas*."""
    table = track_table("", "solo", n_ids=2)
    if arenas:
        table["arena"] = ["A1", "B1"] * (len(table) // 2)
    add_tracks_table(ds, "convert-x.0.1-aaaaaaaaaa", table, std_format="mosaic_v1")


@pytest.mark.parametrize("arenas", [True, False], ids=["assigned", "unassigned"])
def test_an_arena_blind_feature_fails_the_entry(tmp_path: Path, arenas: bool) -> None:
    from mosaic.behavior.feature_library.nearestneighbor import NearestNeighbor

    ds = make_dataset(tmp_path / "ds")
    _two_tanks(ds, arenas=arenas)
    _ = ds.set_arena_layout(TWO, group="", sequence="solo")

    with pytest.raises(AllEntriesFailed):
        _ = run_feature(ds, NearestNeighbor())


def test_an_arena_blind_feature_runs_on_one_arena(tmp_path: Path) -> None:
    from mosaic.behavior.feature_library.nearestneighbor import NearestNeighbor

    ds = make_dataset(tmp_path / "ds")
    _two_tanks(ds, arenas=False)
    _ = ds.set_arena_layout(ONE, group="", sequence="solo")

    result = run_feature(ds, NearestNeighbor())

    assert result.failed_entries == ()


def test_an_arena_aware_feature_runs_on_two_arenas(tmp_path: Path) -> None:
    from mosaic.behavior.feature_library.pair_position import PairPositionFeatures

    ds = make_dataset(tmp_path / "ds")
    table = track_table("", "solo", n_ids=2)
    table["ANGLE"] = 0.0
    table["arena"] = ["A1", "B1"] * (len(table) // 2)
    add_tracks_table(ds, "convert-x.0.1-aaaaaaaaaa", table, std_format="mosaic_v1")
    _ = ds.set_arena_layout(TWO, group="", sequence="solo")

    result = run_feature(ds, PairPositionFeatures())

    assert result.failed_entries == ()


def test_an_output_cached_before_a_layout_of_several_is_not_served(
    tmp_path: Path,
) -> None:
    """Declaring a layout leaves the tracks unchanged, so the cache cannot tell."""
    from mosaic.behavior.feature_library.nearestneighbor import NearestNeighbor

    ds = make_dataset(tmp_path / "ds")
    _two_tanks(ds, arenas=False)
    first = run_feature(ds, NearestNeighbor())
    _ = ds.set_arena_layout(TWO, group="", sequence="solo")

    with pytest.raises(AllEntriesFailed):
        _ = run_feature(ds, NearestNeighbor())
    assert first.failed_entries == ()


def test_an_output_cached_on_one_arena_is_served(tmp_path: Path) -> None:
    from mosaic.behavior.feature_library.nearestneighbor import NearestNeighbor

    ds = make_dataset(tmp_path / "ds")
    _two_tanks(ds, arenas=False)
    _ = ds.set_arena_layout(ONE, group="", sequence="solo")
    _ = run_feature(ds, NearestNeighbor())

    assert run_feature(ds, NearestNeighbor()).cache_hit
