"""Holding an inventory across calls, and noticing when it has moved.

Never on disk: a materialized inventory would sit beside the indexes looking
equally authoritative, and the first disagreement would need somebody to decide
which to believe. Stale is safe, so polling is enough -- and a filesystem
watcher would not work over NFS, which is in the portability story.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mosaic.cli._features import build_feature
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.inventory import InventoryCache
from mosaic.core.pipeline.inventory import cache as cache_module
from tests.helpers import (
    add_tracks_variant,
    assert_tree_unchanged,
    make_dataset,
    tree_snapshot,
)


def test_a_held_view_is_reused_rather_than_rebuilt(
    scenario_dataset: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    holder = InventoryCache(scenario_dataset)
    _ = holder.get(kinds=["feature"])

    builds: list[int] = []
    real = cache_module.inventory
    monkeypatch.setattr(
        cache_module,
        "inventory",
        lambda *a, **k: (builds.append(1), real(*a, **k))[1],
    )
    _ = holder.get()
    _ = holder.get()

    assert builds == []


def test_an_unchanged_dataset_costs_stats_and_no_reread(
    scenario_dataset: Dataset,
) -> None:
    """Tens of syscalls and no parsing, which is what makes a timer reasonable."""
    holder = InventoryCache(scenario_dataset)
    _ = holder.get(kinds=["feature"])

    report = holder.revalidate()

    assert not report.stale
    assert report.changed == ()
    assert report.stat_count > 0


def test_a_new_run_is_noticed(scenario_dataset: Dataset) -> None:
    holder = InventoryCache(scenario_dataset)
    before = holder.get(kinds=["feature"])
    assert before.records == ()

    _ = scenario_dataset.run_feature(build_feature("speed-angvel", None, None))
    report = holder.revalidate()

    assert report.stale
    assert holder.get().records


def test_only_the_index_that_moved_is_reported(scenario_dataset: Dataset) -> None:
    """A changed stamp names its own file, so a caller can see what moved."""
    holder = InventoryCache(scenario_dataset)
    _ = scenario_dataset.run_feature(build_feature("speed-angvel", None, None))
    _ = holder.get(kinds=["feature"])

    index_path = (
        scenario_dataset.get_root("features")
        / "speed-angvel__from__tracks"
        / "index.csv"
    )
    index_path.write_text(index_path.read_text() + "\n")
    report = holder.revalidate()

    assert index_path in report.changed


def test_an_identity_scheme_change_forces_a_full_rebuild(
    scenario_dataset: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Not staleness: the artifacts did not move, their names did. Refreshing
    only the files that changed would leave a view half in each scheme."""
    holder = InventoryCache(scenario_dataset)
    _ = holder.get(kinds=["feature"])

    monkeypatch.setattr(cache_module, "FEATURE_IDENTITY_SCHEME", "999")
    report = holder.revalidate()

    assert report.full_rebuild
    assert "identity scheme" in report.reason


def test_nothing_is_written_to_the_dataset(scenario_dataset: Dataset) -> None:
    """A view is never persisted where it could be mistaken for the record."""
    holder = InventoryCache(scenario_dataset)
    _ = scenario_dataset.run_feature(build_feature("speed-angvel", None, None))
    root = Path(scenario_dataset.base_dir)
    before = tree_snapshot(root)

    _ = holder.get()
    _ = holder.revalidate()

    assert_tree_unchanged(root, before)
    assert not (root / ".mosaic" / "inventory.json").exists()


def _scanned_kinds(monkeypatch: pytest.MonkeyPatch) -> list[frozenset[str]]:
    """Record the kinds of each scan that the holder makes."""
    scans: list[frozenset[str]] = []
    real = cache_module.inventory

    def recorded(*args: object, **kwargs: object) -> object:
        kinds = kwargs.get("kinds")
        scans.append(frozenset(kinds))  # pyright: ignore[reportArgumentType]
        return real(*args, **kwargs)  # pyright: ignore[reportArgumentType]

    monkeypatch.setattr(cache_module, "inventory", recorded)
    return scans


def test_a_new_run_rereads_the_feature_kind_alone(
    scenario_dataset: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    holder = InventoryCache(scenario_dataset)
    before = holder.get()
    variants = before.of_kind("tracks-variant")
    assert variants
    scans = _scanned_kinds(monkeypatch)

    _ = scenario_dataset.run_feature(build_feature("speed-angvel", None, None))
    report = holder.revalidate()

    assert report.stale
    assert scans == [frozenset({"feature"})]
    after = holder.get()
    assert after.of_kind("tracks-variant") == variants
    assert [record.name for record in after.of_kind("feature")] == [
        "speed-angvel__from__tracks"
    ]


def test_the_records_of_a_partial_reread_are_those_of_a_full_one(
    scenario_dataset: Dataset,
) -> None:
    holder = InventoryCache(scenario_dataset)
    _ = holder.get()
    _ = scenario_dataset.run_feature(build_feature("speed-angvel", None, None))

    _ = holder.revalidate()

    assert holder.get().records == InventoryCache(scenario_dataset).get().records


def test_a_moved_tracks_index_rereads_every_kind(
    scenario_dataset: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    holder = InventoryCache(scenario_dataset)
    _ = holder.get()
    scans = _scanned_kinds(monkeypatch)
    index_path = scenario_dataset.get_root("tracks") / "index.csv"
    _ = index_path.write_text(index_path.read_text() + "\n")

    assert holder.revalidate().stale

    [scanned] = scans
    assert {"feature", "tracks-variant"} <= scanned


def test_a_tracks_root_inside_the_features_root_rereads_every_kind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The index sits under ``features/``, and it is the tracks index all the same."""
    ds = make_dataset(tmp_path / "ds")
    ds.roots["tracks"] = "features/tracks"
    ds.get_root("tracks").mkdir(parents=True)
    add_tracks_variant(ds, "v1", "s1", "s2")
    holder = InventoryCache(ds)
    _ = holder.get()
    scans = _scanned_kinds(monkeypatch)
    index_path = ds.get_root("tracks") / "index.csv"
    _ = index_path.write_text(index_path.read_text() + "\n")

    assert holder.revalidate().stale

    [scanned] = scans
    assert {"feature", "tracks-variant"} <= scanned


def test_an_index_that_moves_during_a_scan_is_seen_by_the_next_revalidation(
    scenario_dataset: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    index_path = scenario_dataset.get_root("tracks") / "index.csv"
    real = cache_module.inventory

    def moving(*args: object, **kwargs: object) -> object:
        _ = index_path.write_text(index_path.read_text() + "\n")
        return real(*args, **kwargs)  # pyright: ignore[reportArgumentType]

    monkeypatch.setattr(cache_module, "inventory", moving)
    holder = InventoryCache(scenario_dataset)
    _ = holder.get()
    monkeypatch.setattr(cache_module, "inventory", real)

    assert holder.revalidate().stale
