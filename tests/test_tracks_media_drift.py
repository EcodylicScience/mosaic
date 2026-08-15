"""A bridged tracks variant notices when the media under it moved.

The hazard an ordering-only edge creates. A tracker or an inference run reads an
entry's video, but its variant identity is params plus its model -- there is no
term for the pixels. So re-transcode the entry, re-link the derivative, and the
identifier does not move: a reader asking "is this run current" is told yes over
frames from a different encode, and a graph reports itself done.

The tracker *run* index has recorded and compared this since the inventory
landed. Inference has no run index of its own -- one existed, was written, never
read, and was removed -- so its cell belongs on the tracks row, which is already
per entry. That is what this covers, and it closes the last case in which
``validate`` would have had to refuse a media writer feeding a media reader.
"""

from __future__ import annotations


import pandas as pd
import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.tracks_index import (
    drifted_media_entries,
    read_tracks_index,
    write_tracks_row,
)
from mosaic.core.pipeline.writers import write_parquet_atomic


def _table(group: str, sequence: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "frame": [0, 1],
            "time": [0.0, 0.1],
            "id": [0, 0],
            "group": [group, group],
            "sequence": [sequence, sequence],
            "X": [1.0, 2.0],
            "Y": [3.0, 4.0],
        }
    )


def _write_row(
    ds: Dataset, variant: str, entry: tuple[str, str], *, bridged: bool
) -> None:
    group, sequence = entry
    out = ds.get_root("tracks") / variant / f"{sequence}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    _ = write_parquet_atomic(_table(group, sequence), out)
    write_tracks_row(
        ds,
        run_id=variant,
        group=group,
        sequence=sequence,
        out_path=out,
        producer="infer-pose" if bridged else "convert-calms21_npy",
        std_format="mosaic_v1",
        n_rows=2,
        producer_run_id="infer-pose.0.1-aaaaaaaaaa" if bridged else "",
        records_media=bridged,
    )


def test_a_bridged_row_records_what_its_media_was(scenario_dataset: Dataset) -> None:
    """The cell exists on the row a bridge writes, and is what drift compares."""
    _write_row(
        scenario_dataset, "infer-pose.0.1-aaaaaaaaaa", ("", "seq_a"), bridged=True
    )
    frame = read_tracks_index(scenario_dataset)
    assert "consumed_media_composition" in frame.columns


def test_a_conversion_records_no_media_composition(
    scenario_dataset: Dataset,
) -> None:
    """It opened no video, so a cell here would be a claim about nothing."""
    _write_row(
        scenario_dataset,
        "convert-calms21_npy.0.2-bbbbbbbbbb",
        ("", "seq_a"),
        bridged=False,
    )
    frame = read_tracks_index(scenario_dataset)
    recorded = {
        str(row["run_id"]): str(row.get("consumed_media_composition", ""))
        for _, row in frame.iterrows()
    }
    assert recorded["convert-calms21_npy.0.2-bbbbbbbbbb"] == ""


def test_an_unchanged_entry_does_not_read_as_drifted(
    scenario_dataset: Dataset,
) -> None:
    variant = "infer-pose.0.1-aaaaaaaaaa"
    _write_row(scenario_dataset, variant, ("", "seq_a"), bridged=True)
    assert drifted_media_entries(scenario_dataset, variant) == ()


def test_a_blank_cell_is_unknown_rather_than_changed(
    scenario_dataset: Dataset,
) -> None:
    """A row written before the column existed is not evidence of a change."""
    variant = "convert-calms21_npy.0.2-bbbbbbbbbb"
    _write_row(scenario_dataset, variant, ("", "seq_a"), bridged=False)
    assert drifted_media_entries(scenario_dataset, variant) == ()


def test_media_moving_under_a_bridged_variant_reads_as_drift(
    scenario_dataset: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole point: the identifier does not move, so the cell has to notice.

    The projection is moved rather than the videos, because what is being tested
    is the comparison: a re-transcode changes what the projection records for the
    entry, and the run identifier is unchanged by construction.
    """
    variant = "infer-pose.0.1-aaaaaaaaaa"
    entry = ("", "seq_a")

    import mosaic.core.pipeline.sequence_index as sequence_index

    recorded: dict[tuple[str, str], str] = {entry: "before"}

    def _projection(ds: Dataset, entries: object) -> dict[tuple[str, str], str]:
        return dict(recorded)

    monkeypatch.setattr(sequence_index, "media_compositions_for", _projection)
    _write_row(scenario_dataset, variant, entry, bridged=True)
    assert drifted_media_entries(scenario_dataset, variant) == ()

    recorded[entry] = "after"
    assert drifted_media_entries(scenario_dataset, variant) == (entry,)


def test_a_drifted_variant_is_complete_but_drifted_in_the_inventory(
    scenario_dataset: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Superseded is not invalid: it is reported, not refused."""
    import mosaic.core.pipeline.sequence_index as sequence_index
    from mosaic.core.pipeline.inventory import TracksVariantRef, inventory

    variant = "infer-pose.0.1-aaaaaaaaaa"
    entry = ("", "seq_a")
    recorded: dict[tuple[str, str], str] = {entry: "before"}

    def _projection(ds: Dataset, entries: object) -> dict[tuple[str, str], str]:
        return dict(recorded)

    monkeypatch.setattr(sequence_index, "media_compositions_for", _projection)
    _write_row(scenario_dataset, variant, entry, bridged=True)

    recorded[entry] = "after"
    held = inventory(scenario_dataset, kinds=["tracks-variant"])
    record = held.record(TracksVariantRef(run_id=variant))
    assert record is not None
    assert record.drift == (entry,)
    assert record.status == "complete-but-drifted"
