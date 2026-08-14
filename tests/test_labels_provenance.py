"""The labels arms of the reverse-dependency walk (item 9.3 / 6.2)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.composition import SourceMember, labels_raw_composition
from mosaic.core.pipeline.labels_index import write_labels_row
from mosaic.core.pipeline.provenance import reached_by
from mosaic.core.pipeline.sequence_index import write_sequence_compositions
from tests.helpers import make_dataset


def _scored_label_row(ds: Dataset, digest: str) -> None:
    """A labels_raw composition for ("", "s") and a scored label row consuming it."""
    comp = labels_raw_composition(
        [SourceMember(name="s.csv", digest=digest, algo="md5")]
    )
    _ = write_sequence_compositions(ds, "labels_raw", compositions={("", "s"): comp})
    out_path = ds.get_root("labels") / "behavior" / "v1" / "s.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, labels=np.array([0, 1]))
    write_labels_row(
        ds,
        run_id="v1",
        group="",
        sequence="s",
        out_path=out_path,
        producer="convert-labels-boris_aggregated_csv",
        label_kind="behavior",
        label_format="individual_pair_v1",
        n_frames=2,
        source="s.csv",
        source_md5=digest,
        consumed_source_roots=("labels_raw",),
    )


def test_scored_label_reached_and_current_before_change(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds", save=False, ensure_roots=False)
    _scored_label_row(ds, "deadbeef")
    reached = reached_by(ds, [("", "s")], "labels_raw")
    label_rows = reached[reached["kind"] == "labels"]
    assert len(label_rows) == 1
    row = label_rows.iloc[0]
    assert row["run_id"] == "v1"
    assert row["consumed_roots"] == "labels_raw"
    # The composition still matches what the row recorded: current, not drifted.
    assert row["verdict"] == "current"


def test_scored_label_drifts_when_labels_raw_composition_changes(
    tmp_path: Path,
) -> None:
    ds = make_dataset(tmp_path / "ds", save=False, ensure_roots=False)
    _scored_label_row(ds, "deadbeef")
    # The source file's checksum changes (a re-score / re-upload): re-project the
    # labels_raw composition, leaving the recorded row as it was.
    new_comp = labels_raw_composition(
        [SourceMember(name="s.csv", digest="feedface", algo="md5")]
    )
    _ = write_sequence_compositions(
        ds, "labels_raw", compositions={("", "s"): new_comp}
    )

    reached = reached_by(ds, [("", "s")], "labels_raw")
    label_rows = reached[reached["kind"] == "labels"]
    assert len(label_rows) == 1
    assert label_rows.iloc[0]["verdict"] == "drifted"


def test_labels_raw_change_does_not_reach_tracks_only_entries(tmp_path: Path) -> None:
    # A change under labels_raw reaches only what consumed labels_raw. An entry
    # with no label row is not in the result.
    ds = make_dataset(tmp_path / "ds", save=False, ensure_roots=False)
    _scored_label_row(ds, "deadbeef")
    reached = reached_by(ds, [("", "other")], "labels_raw")
    assert reached.empty
