"""Which group a converted track table is filed under.

A CalMS21 ``.npy`` holds several sequences under in-file groups (annotators), and
feeds both the tracks and the labels root. A source that declares ``group_from``
has the scan record a file-level group, and both conversions must then file each
sequence under it, or a label lookup by ``(group, sequence)`` finds nothing and
every row reads the default label.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import mosaic.behavior.label_library  # noqa: F401  -- registers the label converters
import mosaic.core.track_library  # noqa: F401  -- registers the track converters
from mosaic.core.dataset import Dataset
from mosaic.core.manifest import LabelsScanSource, TracksScanSource
from mosaic.core.pipeline.tracks_index import read_tracks_index

from tests.helpers import make_dataset, write_calms21_npy


def _calms21_dataset(tmp_path: Path, *, declared: bool) -> Dataset:
    """A dataset reading one CalMS21 file as a tracks and a labels source."""
    ds = make_dataset((tmp_path / "ds").resolve())
    raw = ds.base_dir / "raw"
    write_calms21_npy(raw / "task1_test.npy", {"annotator0": {"seq_a": 4}})
    group_from = "filename" if declared else None
    ds.add_scan_source(
        TracksScanSource(
            id="calms21",
            path=str(raw),
            patterns=("*.npy",),
            src_format="calms21_npy",
            multi_sequences_per_file=True,
            group_from=group_from,
        )
    )
    ds.add_scan_source(
        LabelsScanSource(
            id="calms21-labels",
            path=str(raw),
            patterns=("*.npy",),
            src_format="calms21_npy",
            multi_sequences_per_file=True,
            group_from=group_from,
        )
    )
    ds.scan_tracks()
    ds.scan_labels()
    return ds


def _track_groups(ds: Dataset) -> set[str]:
    return {str(group) for group in read_tracks_index(ds)["group"]}


def _label_groups(ds: Dataset) -> set[str]:
    index = Path(ds.get_root("labels")) / "behavior" / "index.csv"
    frame = pd.read_csv(index, dtype=str, keep_default_na=False)
    return set(frame["group"].tolist())


def test_a_declared_grouping_files_tracks_and_labels_under_one_group(
    tmp_path: Path,
) -> None:
    """Neither conversion names a policy, and both take the source's rule."""
    ds = _calms21_dataset(tmp_path, declared=True)

    ds.convert_all_tracks()
    ds.convert_all_labels(kind="behavior", source_format="calms21_npy")

    assert _track_groups(ds) == {"task1_test"}
    assert _label_groups(ds) == _track_groups(ds)


def test_without_a_declared_grouping_the_group_inside_the_file_is_kept(
    tmp_path: Path,
) -> None:
    """The scan records no file-level group, so the in-file annotator names it."""
    ds = _calms21_dataset(tmp_path, declared=False)

    ds.convert_all_tracks()
    ds.convert_all_labels(kind="behavior", source_format="calms21_npy")

    assert _track_groups(ds) == {"annotator0"}
    assert _label_groups(ds) == _track_groups(ds)


def test_an_explicit_infile_policy_overrides_the_declared_grouping(
    tmp_path: Path,
) -> None:
    ds = _calms21_dataset(tmp_path, declared=True)

    ds.convert_all_tracks(group_from="infile")

    assert _track_groups(ds) == {"annotator0"}
