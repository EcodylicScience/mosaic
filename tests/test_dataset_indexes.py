"""The shared index enumeration, and the root-agnostic reconciler -- item 6.1.

Three passes used to enumerate "which files are indexes" separately. These pin
that there is now one answer, that the two path passes and the reconciler all
read it, and that the tracker roots -- reached by no pass at all before this --
are in it.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mosaic.core.dataset import Dataset, new_dataset_manifest
from mosaic.core.pipeline.dataset_indexes import (
    feature_storages,
    iter_dataset_indexes,
    label_kinds,
    reconcilable_index,
    root_subdirectories,
)
from mosaic.core.pipeline.tracking_roots import TRACKING_ROOTS


def _dataset(tmp_path: Path) -> Dataset:
    manifest = new_dataset_manifest(name="reconcile", base_dir=tmp_path / "ds")
    return Dataset(manifest_path=manifest).load(ensure_roots=True)


def _trex_row(ds: Dataset, run_id: str, sequence: str, output: Path) -> None:
    """One tracker index row naming *output*, written through the real writer."""
    from mosaic.tracking.trex.dataset_runs import (
        TRexIndexRow,
        trex_index,
        trex_index_path,
    )

    idx = trex_index(trex_index_path(ds))
    idx.ensure()
    idx.append(
        [
            TRexIndexRow(
                run_id=run_id,
                group="",
                sequence=sequence,
                abs_path=Path(ds.relative_to_root(output)),
                video_abs_path="",
                params_hash="",
            )
        ]
    )


# --- The enumeration ---------------------------------------------------------


def test_every_tracking_root_is_enumerated(tmp_path: Path) -> None:
    """The roots that no reindex, prune or portability pass reached before."""
    ds = _dataset(tmp_path)

    reached = {index.root_key for index in iter_dataset_indexes(ds)}

    assert set(TRACKING_ROOTS) <= reached


def test_a_root_with_both_shapes_yields_both(tmp_path: Path) -> None:
    """``features`` carries a root-level index *and* one per storage directory.

    A list of roots cannot express that, which is why the table is (root, shape)
    pairs. Losing the distinction drops one index or the other silently.
    """
    ds = _dataset(tmp_path)
    (ds.get_root("features") / "speed-angvel__from__tracks").mkdir(parents=True)

    paths = [i.path for i in iter_dataset_indexes(ds) if i.root_key == "features"]

    assert ds.get_root("features") / "index.csv" in paths
    assert ds.get_root("features") / "speed-angvel__from__tracks" / "index.csv" in paths


def test_an_unset_root_is_skipped_rather_than_raising(tmp_path: Path) -> None:
    """A manifest may declare fewer roots than the defaults, and passes must run."""
    base = tmp_path / "sparse"
    (base / "tracks").mkdir(parents=True)
    ds = Dataset(
        manifest_path=base / "dataset.yaml", roots={"tracks": str(base / "tracks")}
    )

    keys = {index.root_key for index in iter_dataset_indexes(ds)}

    assert keys == {"tracks"}


def test_the_enumeration_is_stable(tmp_path: Path) -> None:
    """Filesystem order is not, so a report built on this must sort."""
    ds = _dataset(tmp_path)
    for name in ("z-feature", "a-feature", "m-feature"):
        (ds.get_root("features") / name).mkdir(parents=True)

    first = [i.path for i in iter_dataset_indexes(ds)]
    second = [i.path for i in iter_dataset_indexes(ds)]

    assert first == second


def test_tracker_path_columns_come_from_the_registry(tmp_path: Path) -> None:
    """A path column added to a tracker row must not need a second edit.

    They used to live in a hand-written table beside ``default_roots``, which is
    what a new tracker forgets -- and a column missing from it silently stops
    being portable.
    """
    from mosaic.core.dataset import _INDEX_PATH_COLUMNS

    for key, root in TRACKING_ROOTS.items():
        assert _INDEX_PATH_COLUMNS.get(key, ()) == root.path_columns


# --- The reconciler ----------------------------------------------------------


def test_a_tracker_row_naming_a_deleted_directory_is_dropped(tmp_path: Path) -> None:
    """The gap item 6.1 leaves the sweeper: everything removed by hand.

    Before this, the ``_tracking`` indexes were reached by no reconcile pass, so
    a working directory deleted by hand left its row naming it forever.
    """
    ds = _dataset(tmp_path)
    gone = ds.get_root("trex") / "trex.1.0-aaaa" / "seq_a"
    gone.mkdir(parents=True)
    _trex_row(ds, "trex.1.0-aaaa", "seq_a", gone)

    import shutil

    shutil.rmtree(gone)
    dropped = ds.reindex("trex", dry_run=False)

    assert sum(dropped.values()) == 1
    assert reconcilable_index("trex") is not None


def test_a_present_directory_keeps_its_row(tmp_path: Path) -> None:
    """The negative half: reconciling is not "drop everything"."""
    ds = _dataset(tmp_path)
    kept = ds.get_root("trex") / "trex.1.0-bbbb" / "seq_a"
    kept.mkdir(parents=True)
    _trex_row(ds, "trex.1.0-bbbb", "seq_a", kept)

    assert ds.reindex("trex", dry_run=False) == {}

    from mosaic.tracking.trex.dataset_runs import trex_index_path

    assert len(pd.read_csv(trex_index_path(ds))) == 1


def test_a_dry_run_reports_without_writing(tmp_path: Path) -> None:
    """Dry-run is the default, and it must not be a dry-run in name only."""
    ds = _dataset(tmp_path)
    gone = ds.get_root("trex") / "trex.1.0-cccc" / "seq_a"
    gone.mkdir(parents=True)
    _trex_row(ds, "trex.1.0-cccc", "seq_a", gone)

    from mosaic.tracking.trex.dataset_runs import trex_index_path

    import shutil

    shutil.rmtree(gone)
    before = trex_index_path(ds).read_text()
    reported = ds.reindex("trex")

    assert sum(reported.values()) == 1
    assert trex_index_path(ds).read_text() == before


def test_restricting_to_one_root_leaves_the_others_alone(tmp_path: Path) -> None:
    """``--root`` is a filter on the walk, not on the report."""
    ds = _dataset(tmp_path)
    gone = ds.get_root("trex") / "trex.1.0-dddd" / "seq_a"
    gone.mkdir(parents=True)
    _trex_row(ds, "trex.1.0-dddd", "seq_a", gone)

    import shutil

    shutil.rmtree(gone)

    assert ds.reindex("tracks", dry_run=False) == {}
    assert sum(ds.reindex("trex", dry_run=False).values()) == 1


def test_a_root_with_no_registered_opener_is_left_alone(tmp_path: Path) -> None:
    """``media_raw`` is raw pandas, not an ``IndexCSV``; a wrong pass is worse."""
    ds = _dataset(tmp_path)

    assert reconcilable_index("media_raw") is None
    assert ds.reindex("media_raw", dry_run=False) == {}


def test_every_tracker_root_registers_a_reconcilable_index() -> None:
    """All of them, not just TREx -- an unregistered root is reclaimed by nothing.

    ``register_reconcilable_index`` is an import side effect of each tracker's
    ``dataset_runs``, so a new tracker that declares a ``TrackingRoot`` row and
    forgets the registration leaves its index reachable by no reindex or prune
    pass: a working directory deleted by hand keeps its row forever, and
    ``mosaic sweep-tracking`` silently reclaims nothing. Only TREx was asserted,
    which is exactly the assertion a copied integration passes without doing the
    thing.
    """
    from mosaic.tracking import register_ops

    # Registration happens as the import side effect; this is its explicit marker.
    register_ops()

    unregistered = sorted(
        key
        for key, root in TRACKING_ROOTS.items()
        if root.retention == "tracker" and reconcilable_index(key) is None
    )

    assert unregistered == []


# --- One answer to "what does this root hold" --------------------------------


def test_a_root_that_is_unset_lists_nothing(tmp_path: Path) -> None:
    """Not an error. A pass over a half-built dataset must not raise."""
    manifest = new_dataset_manifest(name="bare", base_dir=tmp_path / "ds")
    ds = Dataset(manifest_path=manifest).load(ensure_roots=False)
    ds.roots.pop("features", None)

    assert feature_storages(ds) == []
    assert root_subdirectories(ds, "features") == []


def test_a_root_declared_but_not_created_lists_nothing(tmp_path: Path) -> None:
    """Declared in the manifest, never made on disk -- the ordinary early state."""
    ds = _dataset(tmp_path)
    features_root = ds.get_root("features")
    if features_root.exists():
        import shutil

        shutil.rmtree(features_root)

    assert feature_storages(ds) == []


def test_children_come_back_sorted_and_files_are_not_children(tmp_path: Path) -> None:
    ds = _dataset(tmp_path)
    root = ds.get_root("features")
    for name in ("zebra", "alpha", "mid"):
        (root / name).mkdir(parents=True, exist_ok=True)
    (root / "index.csv").write_text("run_id\n")

    assert feature_storages(ds) == ["alpha", "mid", "zebra"]


def test_require_index_is_what_separates_a_kind_from_a_variant(
    tmp_path: Path,
) -> None:
    """A ``labels/<kind>/`` holds an index; the variant dirs below it do not.

    Without the filter a variant directory would be read as a kind, and
    ``read_labels_index`` would then be asked for an index that cannot exist.
    """
    ds = _dataset(tmp_path)
    labels_root = ds.get_root("labels")
    (labels_root / "behavior").mkdir(parents=True, exist_ok=True)
    (labels_root / "behavior" / "index.csv").write_text("run_id\n")
    (labels_root / "behavior" / "some-variant").mkdir(parents=True, exist_ok=True)
    (labels_root / "not-a-kind").mkdir(parents=True, exist_ok=True)

    assert label_kinds(ds) == ["behavior"]
    assert root_subdirectories(ds, "labels") == ["behavior", "not-a-kind"]


def test_the_listing_is_written_in_exactly_one_place() -> None:
    """The defect this retired is duplication, and duplication passes every test.

    Seven call sites spelled "list a dataset root's child directories" for
    themselves, under three mutually incompatible guards -- ``try``/``except
    KeyError``, ``has_root``, and a truthiness test on ``ds.roots``. Guards that
    disagree diverge silently: a root invisible to one pass and not another fails
    nothing. Asserted over the source, because a behavioural test cannot see a
    second copy that happens to agree today.

    Detected as a function naming both ``get_root`` and ``iterdir``, which is
    what reaching into a *dataset root's* children looks like; an ordinary
    directory walk names no root and is not matched.
    """
    import ast

    import mosaic

    allowed = {
        # The definition site.
        "core/pipeline/dataset_indexes.py",
        # Two levels deep, over run and entry directories inside one tracker
        # root rather than over the root's own children -- a different question,
        # and it wants Paths rather than names.
        "core/dataset.py",
        # Also a level deeper: the variant directories inside one label *kind*,
        # which this removes. ``root_subdirectories`` takes a root key and
        # cannot name a child of a child, so there is nothing here to call.
        "core/pipeline/labels_migration.py",
    }
    source_root = Path(mosaic.__file__).parent
    matched: set[str] = set()
    for source in sorted(source_root.rglob("*.py")):
        # A vendored virtualenv under the excluded workspace member.
        if "feature_library/external" in source.as_posix():
            continue
        tree = ast.parse(source.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            named = {
                inner.attr
                for inner in ast.walk(node)
                if isinstance(inner, ast.Attribute)
            }
            if {"get_root", "iterdir"} <= named:
                matched.add(str(source.relative_to(source_root)))

    assert sorted(matched - allowed) == [], (
        "these list a dataset root's children themselves instead of calling "
        f"root_subdirectories: {sorted(matched - allowed)}"
    )
    assert allowed - matched == set(), (
        "expected every allowlisted module to still match, but these did not: "
        f"{sorted(allowed - matched)} -- a rename may have made this guard scan "
        "for something that no longer exists"
    )


# --- labels_raw, which no pass reached ---------------------------------------


def test_labels_raw_is_enumerated(tmp_path: Path) -> None:
    """Written by ``index_labels_raw`` and read by ``convert_all_labels``, but
    invisible to every portability, reindex and reconcile pass before this."""
    ds = _dataset(tmp_path)

    reached = {index.root_key for index in iter_dataset_indexes(ds)}

    assert "labels_raw" in reached


def test_a_labels_raw_row_is_made_portable(tmp_path: Path) -> None:
    """The defect the omission caused: an absolute path surviving a move.

    Asserted through ``make_portable`` rather than the enumeration alone,
    because enumerating a root that no pass then rewrites would satisfy the
    test above and fix nothing.
    """
    ds = _dataset(tmp_path)
    raw_root = ds.get_root("labels_raw")
    raw_root.mkdir(parents=True, exist_ok=True)
    labels_file = raw_root / "scored.csv"
    labels_file.write_text("frame,label\n0,walk\n")
    index_path = raw_root / "index.csv"
    pd.DataFrame(
        [
            {
                "group": "",
                "sequence": "seq_a",
                "abs_path": str(labels_file.resolve()),
                "src_format": "boris_aggregated_csv",
            }
        ]
    ).to_csv(index_path, index=False)

    _ = ds.make_portable(dry_run=False)

    stored = str(pd.read_csv(index_path)["abs_path"].iloc[0])
    assert not Path(stored).is_absolute(), f"labels_raw row was left absolute: {stored}"
    assert ds.resolve_path(stored).resolve() == labels_file.resolve()
