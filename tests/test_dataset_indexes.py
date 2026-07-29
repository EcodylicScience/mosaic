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
    iter_dataset_indexes,
    reconcilable_index,
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
