"""The item 9.3 migrations: populate labels_raw, and the reverse break rollback."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mosaic.core.dataset import Dataset, new_dataset_manifest
from mosaic.core.pipeline.labels_index import read_labels_index
from mosaic.core.pipeline.labels_migration import migrate_labels_raw, revert_labels

_LEGACY_LABEL_COLUMNS = [
    "kind",
    "label_format",
    "group",
    "sequence",
    "group_safe",
    "sequence_safe",
    "abs_path",
    "source_abs_path",
    "source_md5",
    "n_frames",
    "label_ids",
    "label_names",
]


def _dataset_with_legacy_label_source(tmp_path: Path) -> tuple[Dataset, Path]:
    """A dataset with a label file registered into tracks_raw, as a pre-9.3 one has."""
    ds = Dataset(new_dataset_manifest("t", tmp_path / "ds")).load()
    src = tmp_path / "uploads"
    src.mkdir()
    np.save(
        src / "f1.npy",
        {"g": {"s1": {"annotations": [0, 1, 2, 3]}}},
        allow_pickle=True,
    )
    ds.index_tracks_raw([src], patterns=["*.npy"], src_format="calms21_npy")
    return ds, src / "f1.npy"


def test_forward_migration_populates_labels_raw_without_moving_files(
    tmp_path: Path,
) -> None:
    ds, source_file = _dataset_with_legacy_label_source(tmp_path)
    result = migrate_labels_raw(ds)
    assert result["rows"] == 1

    labels_raw = ds.get_root("labels_raw")
    index = pd.read_csv(labels_raw / "index.csv")
    assert list(index["sequence"]) == ["f1"]
    assert set(index["src_format"]) == {"calms21_npy"}
    assert (labels_raw / "sequences.csv").exists()
    # The file is referenced where it lies -- never copied or moved.
    assert source_file.exists()


def test_forward_migration_is_idempotent(tmp_path: Path) -> None:
    ds, _ = _dataset_with_legacy_label_source(tmp_path)
    first = migrate_labels_raw(ds)
    second = migrate_labels_raw(ds)
    assert first == second == {"rows": 1}


def test_migration_with_no_label_rows_writes_nothing(tmp_path: Path) -> None:
    ds = Dataset(new_dataset_manifest("t", tmp_path / "ds")).load()
    src = tmp_path / "uploads"
    src.mkdir()
    # A track-only format, not a label converter.
    (src / "t.csv").write_text("frame,x,y\n0,1,2\n")
    ds.index_tracks_raw([src], patterns=["*.csv"], src_format="deeplabcut")
    result = migrate_labels_raw(ds)
    assert result["rows"] == 0


def test_reverse_restores_flat_layout_and_untyped_index(tmp_path: Path) -> None:
    ds, _ = _dataset_with_legacy_label_source(tmp_path)
    migrate_labels_raw(ds)
    ds.convert_all_labels(kind="behavior", source_format="calms21_npy")

    idx = read_labels_index(ds, "behavior")
    variant = next(str(r) for r in idx["run_id"] if str(r))
    vdir = ds.get_root("labels") / "behavior" / variant
    assert vdir.exists()

    result = revert_labels(ds)
    assert result["labels_raw_removed"] == 1

    # Variant directory gone, flat npz restored, index in the legacy 12-column shape.
    assert not vdir.exists()
    flat = ds.get_root("labels") / "behavior" / "g__s1.npz"
    assert flat.exists()
    legacy = pd.read_csv(ds.get_root("labels") / "behavior" / "index.csv")
    assert list(legacy.columns) == _LEGACY_LABEL_COLUMNS
    assert set(legacy["kind"]) == {"behavior"}
    # labels_raw index and composition are gone.
    assert not (ds.get_root("labels_raw") / "index.csv").exists()
    assert not (ds.get_root("labels_raw") / "sequences.csv").exists()

    # The npz payload survived the round trip unchanged.
    with np.load(flat, allow_pickle=True) as data:
        assert data["labels"].tolist() == [0, 1, 2, 3]
