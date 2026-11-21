"""Holding an inventory across calls, and noticing when it has moved.

Never on disk: a materialized inventory would sit beside the indexes looking
equally authoritative, and the first disagreement would need somebody to decide
which to believe. Stale is safe, so polling is enough -- and a filesystem
watcher would not work over NFS, which is in the portability story.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pytest

from mosaic.cli._features import build_feature
from mosaic.core.dataset import Dataset
from mosaic.core.manifest import DatasetLink
from mosaic.core.pipeline.artifact_kinds import LINKED_KINDS, LinkedKind
from mosaic.core.pipeline.index import feature_index_path
from mosaic.core.pipeline.inventory import InventoryCache
from mosaic.core.pipeline.inventory import cache as cache_module
from mosaic.core.pipeline.inventory._read import ParquetFooterMemo
from mosaic.core.pipeline.inventory.model import ArtifactKind
from mosaic.core.pipeline.label_series import TAGS_SERIES
from mosaic.core.pipeline.label_series_index import series_index_path
from mosaic.core.pipeline.labels_index import labels_index_path
from mosaic.core.scope import Scope
from tests.helpers import (
    copy_of,
    labelled_source,
    add_feature_run,
    add_tracks_variant,
    assert_tree_unchanged,
    make_dataset,
    observed,
    raw_source,
    tree_snapshot,
)


def test_a_held_view_is_reused_rather_than_rebuilt(
    scenario_dataset: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    holder = InventoryCache(scenario_dataset)
    _ = holder.get(kinds=["feature"])

    builds: list[int] = []

    def count_build(*_args: object, **_kwargs: object) -> None:
        builds.append(1)

    monkeypatch.setattr(
        cache_module, "inventory", observed(cache_module.inventory, count_build)
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


def _scanned_kinds(monkeypatch: pytest.MonkeyPatch) -> list[frozenset[ArtifactKind]]:
    """Record the kinds of each scan that the holder makes."""
    scans: list[frozenset[ArtifactKind]] = []

    def record(
        ds: Dataset,
        *,
        kinds: Iterable[ArtifactKind] | None = None,
        scope: Scope | None = None,
        tracks_run_id: str | None = None,
        footers: ParquetFooterMemo | None = None,
    ) -> None:
        assert kinds is not None
        scans.append(frozenset(kinds))

    monkeypatch.setattr(
        cache_module, "inventory", observed(cache_module.inventory, record)
    )
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

    def move_index(*_args: object, **_kwargs: object) -> None:
        _ = index_path.write_text(index_path.read_text() + "\n")

    monkeypatch.setattr(cache_module, "inventory", observed(real, move_index))
    holder = InventoryCache(scenario_dataset)
    _ = holder.get()
    monkeypatch.setattr(cache_module, "inventory", real)

    assert holder.revalidate().stale


# --- a dataset that links another ---------------------------------------------

STORAGE = "store"
RUN_ID = "0.1-1111111111"
LINKED_VARIANT = "linked-variant"


def _linking_copy(
    tmp_path: Path, *, provides: tuple[LinkedKind, ...] = LINKED_KINDS
) -> tuple[Dataset, Dataset]:
    """A source with a tracks variant and a feature run, and a copy that links it."""
    source = make_dataset(tmp_path / "source", name="source")
    add_tracks_variant(source, LINKED_VARIANT, "seq_a", std_format="mosaic_v1")
    _ = add_feature_run(source, STORAGE, RUN_ID, entries=[("", "seq_a")])
    copy = make_dataset(tmp_path / "copy", name="copy")
    _ = copy.add_link(
        DatasetLink(id="source", path=str(source.base_dir), provides=provides)
    )
    return source, copy


def _touch(path: Path) -> None:
    """Move *path*'s stamp without changing what it holds."""
    _ = path.write_text(path.read_text() + "\n")


def test_a_moved_linked_feature_index_rereads_the_feature_kind_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, copy = _linking_copy(tmp_path)
    holder = InventoryCache(copy)
    _ = holder.get()
    scans = _scanned_kinds(monkeypatch)

    _touch(feature_index_path(source, STORAGE))
    report = holder.revalidate()

    assert report.changed == (feature_index_path(source, STORAGE),)
    assert scans == [frozenset({"feature"})]


def test_a_moved_linked_tracks_index_rereads_every_kind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, copy = _linking_copy(tmp_path)
    holder = InventoryCache(copy)
    _ = holder.get()
    scans = _scanned_kinds(monkeypatch)

    _touch(source.get_root("tracks") / "index.csv")

    assert holder.revalidate().stale
    [scanned] = scans
    assert {"feature", "tracks-variant"} <= scanned


def test_a_new_linked_run_is_listed_after_a_revalidation(tmp_path: Path) -> None:
    source, copy = _linking_copy(tmp_path)
    holder = InventoryCache(copy)
    assert [record.run_id for record in holder.get().of_kind("feature")] == [RUN_ID]

    _ = add_feature_run(source, STORAGE, "0.1-2222222222", entries=[("", "seq_a")])

    assert holder.revalidate().stale
    assert [record.run_id for record in holder.get().of_kind("feature")] == [
        RUN_ID,
        "0.1-2222222222",
    ]


def test_a_models_only_links_tracks_index_moving_leaves_the_view_as_it_was(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, copy = _linking_copy(tmp_path, provides=("models",))
    holder = InventoryCache(copy)
    held = holder.get()
    generation = holder.generation
    scans = _scanned_kinds(monkeypatch)

    _touch(source.get_root("tracks") / "index.csv")
    _touch(feature_index_path(source, STORAGE))
    report = holder.revalidate()

    assert not report.stale
    assert scans == []
    assert holder.generation == generation
    assert holder.get() is held


def test_a_linked_series_index_moves_the_view_and_the_labels_root_index_does_not(
    tmp_path: Path,
) -> None:
    source = raw_source(tmp_path / "source")
    copy = make_dataset(tmp_path / "copy", name="copy")
    _ = copy.add_link(
        DatasetLink(id="source", path=str(source.base_dir), provides=("label-series",))
    )
    holder = InventoryCache(copy)
    assert holder.get().of_kind("label-series")

    _touch(source.get_root("labels_raw") / "index.csv")
    assert not holder.revalidate().stale

    series = series_index_path(source, TAGS_SERIES)
    _touch(series)
    assert holder.revalidate().changed == (series,)


@pytest.mark.parametrize("kind", ["models", "frames"])
def test_a_linked_index_of_each_remaining_kind_moves_the_view(
    tmp_path: Path, kind: LinkedKind
) -> None:
    source = make_dataset(tmp_path / "source", name="source")
    index = source.get_root(kind) / "kind" / "index.csv"
    index.parent.mkdir(parents=True)
    _ = index.write_text("\n")
    copy = make_dataset(tmp_path / "copy", name="copy")
    _ = copy.add_link(
        DatasetLink(id="source", path=str(source.base_dir), provides=(kind,))
    )
    holder = InventoryCache(copy)
    _ = holder.get()

    _touch(index)

    assert holder.revalidate().changed == (index,)


def test_a_moved_linked_labels_index_rereads_every_kind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = labelled_source(tmp_path / "source")
    holder = InventoryCache(copy_of(source, tmp_path / "copy"))
    _ = holder.get()
    scans = _scanned_kinds(monkeypatch)
    index = labels_index_path(source, "behavior")

    _touch(index)
    report = holder.revalidate()

    assert report.changed == (index,)
    [scanned] = scans
    assert {"feature", "labels-variant"} <= scanned


def test_a_models_only_links_labels_index_moving_leaves_the_view_as_it_was(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = labelled_source(tmp_path / "source")
    holder = InventoryCache(copy_of(source, tmp_path / "copy", provides=("models",)))
    held = holder.get()
    scans = _scanned_kinds(monkeypatch)

    _touch(labels_index_path(source, "behavior"))

    assert not holder.revalidate().stale
    assert scans == []
    assert holder.get() is held


def test_every_linked_kind_names_the_root_its_indexes_sit_in() -> None:
    assert set(cache_module.LINKED_ROOT_KEY) == set(LINKED_KINDS)


def test_a_link_becoming_reachable_and_unreachable_again_rebuilds_the_view(
    tmp_path: Path,
) -> None:
    source, copy = _linking_copy(tmp_path)
    away = tmp_path / "away"
    source.base_dir.rename(away)
    holder = InventoryCache(copy)
    assert holder.get().of_kind("feature") == ()

    away.rename(source.base_dir)
    assert holder.revalidate().stale
    assert [record.run_id for record in holder.get().of_kind("feature")] == [RUN_ID]
    assert not holder.revalidate().stale

    source.base_dir.rename(away)
    assert holder.revalidate().stale
    assert holder.get().of_kind("feature") == ()


def test_an_offline_link_warns_once_over_several_revalidations(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source, copy = _linking_copy(tmp_path)
    source.base_dir.rename(tmp_path / "away")
    holder = InventoryCache(copy)
    _ = holder.get()

    for _ in range(3):
        _ = holder.revalidate()

    assert capsys.readouterr().err.count("is not reachable") == 1


def _replace_by_another_dataset(source: Dataset) -> Dataset:
    """Make the path of *source* hold a different dataset, which gets a tracks index."""
    for child in source.base_dir.iterdir():
        if child.is_file():
            child.unlink()
    impostor = make_dataset(source.base_dir, name="impostor")
    add_tracks_variant(impostor, "impostor-variant", "seq_x", std_format="mosaic_v1")
    return impostor


def test_a_mismatched_link_is_named_in_the_errors_and_stamps_nothing(
    tmp_path: Path,
) -> None:
    source, copy = _linking_copy(tmp_path)
    impostor = _replace_by_another_dataset(source)
    own = Dataset(manifest_path=copy.manifest_path).load()
    holder = InventoryCache(own)

    held = holder.get(kinds=["feature", "tracks-variant"])

    assert [error.split(": ", 1)[0] for error in held.errors] == [
        "feature",
        "tracks-variant",
    ]
    assert all("was linked to dataset" in error for error in held.errors)
    _touch(impostor.get_root("tracks") / "index.csv")
    assert not holder.revalidate().stale


def test_an_offline_link_beside_a_mismatched_one_warns_once_over_several_revalidations(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source, copy = _linking_copy(tmp_path)
    away = make_dataset(tmp_path / "away", name="away")
    _ = copy.add_link(DatasetLink(id="away", path=str(away.base_dir)))
    away.base_dir.rename(tmp_path / "gone")
    _ = _replace_by_another_dataset(source)
    holder = InventoryCache(Dataset(manifest_path=copy.manifest_path).load())
    _ = holder.get(kinds=["feature"])

    for _ in range(3):
        _ = holder.revalidate()

    assert capsys.readouterr().err.count("is not reachable") == 1


def test_a_mismatched_link_leaves_the_stamps_of_the_other_links(
    tmp_path: Path,
) -> None:
    source, copy = _linking_copy(tmp_path)
    friend = make_dataset(tmp_path / "friend", name="friend")
    add_tracks_variant(friend, "friend-variant", "seq_f", std_format="mosaic_v1")
    _ = copy.add_link(DatasetLink(id="friend", path=str(friend.base_dir)))
    _ = _replace_by_another_dataset(source)
    holder = InventoryCache(Dataset(manifest_path=copy.manifest_path).load())
    _ = holder.get(kinds=["tracks-variant"])

    _touch(friend.get_root("tracks") / "index.csv")

    assert holder.revalidate().stale
