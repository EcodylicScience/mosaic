"""The labels arms of the reverse-dependency walk (item 9.3 / 6.2)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.composition import (
    SequenceComposition,
    SourceMember,
    labels_raw_composition,
)
from mosaic.core.pipeline.delete_set import delete_set
from mosaic.core.pipeline.labels_index import write_labels_row
from mosaic.core.pipeline.provenance import reached_by
from mosaic.core.pipeline.sequence_index import write_sequence_compositions
from mosaic.core.scope import Scope
from tests.helpers import (
    SOURCE_ENTRIES,
    assert_tree_unchanged,
    copy_of,
    labelled_source,
    make_dataset,
    tree_snapshot,
)


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


def test_a_linked_labels_row_names_its_link_and_is_declined_by_a_delete(
    tmp_path: Path,
) -> None:
    source = labelled_source(tmp_path / "source")
    copy = copy_of(source, tmp_path / "copy")
    link_id = f"source-{source.uuid}"
    entry = SOURCE_ENTRIES[0]
    _ = write_sequence_compositions(
        copy,
        "labels_raw",
        compositions={entry: SequenceComposition(digest="moved", member_count=1)},
    )
    before = tree_snapshot(source.base_dir)

    reached = reached_by(copy, [entry], "labels_raw")
    report = delete_set(copy, "labels_raw", scope=Scope(entries=[entry]), apply=True)

    labels = reached[reached["kind"] == "labels"]
    assert list(labels["link_id"]) == [link_id]
    assert list(labels["verdict"]) == ["drifted"]
    assert report.candidates == ()
    [declined] = [item for item in report.declined if item.kind == "labels"]
    assert link_id in declined.reason
    assert_tree_unchanged(source.base_dir, before)
