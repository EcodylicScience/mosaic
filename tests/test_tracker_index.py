"""The shared tracker run index, and what happens to one written before it.

The count column used to be spelled ``n_tracks`` by SLEAP and ``n_individuals``
by TREx and Lightning Pose. It is ``n_ids`` now, which means a user's existing
``_tracking/<tool>/index.csv`` holds a column this code no longer names. These
pin that the old counts are carried forward rather than read back as zero, and
that a caller sees the current columns whatever the file still spells them.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mosaic.core.dataset import Dataset, new_dataset_manifest
from mosaic.tracking.common.index import (
    TrackerRunRowBase,
    list_tracker_runs,
    tracker_index,
    tracker_index_path,
)
from mosaic.tracking.litpose.dataset_runs import LitposeIndexRow
from mosaic.tracking.sleap.dataset_runs import SleapIndexRow
from mosaic.tracking.trex.dataset_runs import TRexIndexRow


@pytest.fixture
def ds(tmp_path: Path) -> Dataset:
    manifest = new_dataset_manifest("idx", base_dir=tmp_path)
    return Dataset(manifest_path=manifest).load(ensure_roots=True)


def _legacy_index(path: Path, count_column: str) -> None:
    """An index as an older mosaic wrote it: the count under its old name."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "abs_path": "_tracking/sleap/run-a/vid1",
                "run_id": "run-a",
                "started_at": "2026-01-01T00:00:00+00:00",
                "finished_at": "",
                "group": "",
                "sequence": "vid1",
                "video_abs_path": "media_raw/vid1.mp4",
                "params_hash": "abc123",
                "model_id": "deadbeef",
                "model_type": "single_instance",
                count_column: 7,
                "slp_path": "_tracking/sleap/run-a/vid1/vid1.predictions.slp",
                "analysis_h5_path": "_tracking/sleap/run-a/vid1/vid1.analysis.h5",
            }
        ]
    ).to_csv(path, index=False)


# --- the rename ------------------------------------------------------------


def test_every_tracker_row_carries_the_shared_columns() -> None:
    """The four every tracker records, plus the count, come from one place."""
    for row_cls in (TRexIndexRow, SleapIndexRow, LitposeIndexRow):
        assert issubclass(row_cls, TrackerRunRowBase)


def test_an_old_count_column_is_carried_forward_not_zeroed(ds: Dataset) -> None:
    """The migration that matters: a user's recorded counts survive the rename."""
    path = tracker_index_path(ds, "sleap")
    _legacy_index(path, "n_tracks")

    index = tracker_index(path, SleapIndexRow)
    index.append(
        [
            SleapIndexRow(
                run_id="run-b",
                group="",
                sequence="vid2",
                abs_path=Path("_tracking/sleap/run-b/vid2"),
                video_abs_path="media_raw/vid2.mp4",
                params_hash="def456",
                n_ids=3,
            )
        ]
    )

    written = pd.read_csv(path).set_index("sequence")
    assert "n_tracks" not in written.columns
    # The pre-existing row keeps the count it recorded; the new row records its own.
    assert int(written.loc["vid1", "n_ids"]) == 7
    assert int(written.loc["vid2", "n_ids"]) == 3


def test_the_other_old_spelling_is_carried_forward_too(ds: Dataset) -> None:
    """TREx and Lightning Pose spelled it differently from SLEAP."""
    path = tracker_index_path(ds, "litpose")
    _legacy_index(path, "n_individuals")

    index = tracker_index(path, SleapIndexRow)
    index.append(
        [
            SleapIndexRow(
                run_id="run-b",
                group="",
                sequence="vid2",
                abs_path=Path("_tracking/litpose/run-b/vid2"),
                video_abs_path="",
                params_hash="",
                n_ids=1,
            )
        ]
    )

    written = pd.read_csv(path)
    assert "n_individuals" not in written.columns
    assert int(written.loc[written["sequence"] == "vid1", "n_ids"].iloc[0]) == 7


def test_reading_an_unmigrated_index_already_shows_the_new_name(ds: Dataset) -> None:
    """``list_*_runs`` reads through the projection, so it never has to migrate.

    A read must not depend on whether a write happened to have run first --
    otherwise the column a caller sees would depend on the file's write history.
    """
    _legacy_index(tracker_index_path(ds, "sleap"), "n_tracks")

    listed = list_tracker_runs(ds, "sleap", SleapIndexRow)

    assert "n_tracks" not in listed.columns
    assert int(listed.iloc[0]["n_ids"]) == 7


# --- the empty cases -------------------------------------------------------


def test_an_absent_root_lists_empty_but_typed(ds: Dataset) -> None:
    listed = list_tracker_runs(ds, "sleap", SleapIndexRow)

    assert len(listed) == 0
    assert list(listed.columns) == [
        "abs_path",
        "run_id",
        "started_at",
        "finished_at",
        "group",
        "sequence",
        "video_abs_path",
        "params_hash",
        "n_ids",
        "consumed_media_composition",
        "model_id",
        "model_type",
        "slp_path",
        "analysis_h5_path",
    ]
