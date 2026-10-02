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
    before = _tree(Path(scenario_dataset.base_dir))

    _ = holder.get()
    _ = holder.revalidate()

    assert _tree(Path(scenario_dataset.base_dir)) == before
    assert not (Path(scenario_dataset.base_dir) / ".mosaic" / "inventory.json").exists()


def _tree(root: Path) -> set[str]:
    return {str(p.relative_to(root)) for p in root.rglob("*")}
