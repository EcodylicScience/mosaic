"""A tracker run records what its media was, and notices when that moves.

A tracker's identity is its settings, with no term for the media it read: which
file it opens is decided at run time from the routing verdict on the media
index. So re-transcoding an entry and re-linking the derivative leaves the run
identifier exactly where it was, and anything keyed on identity alone reports
the work done over pixels from a different encode.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mosaic.core.dataset import Dataset, new_dataset_manifest
from mosaic.core.pipeline.inventory import TrackerRunRef, inventory
from mosaic.core.pipeline.sequence_index import media_compositions_for
from mosaic.tracking.common.index import drifted_media_entries


@pytest.fixture(autouse=True)
def _producers_imported() -> None:
    from mosaic.tracking import register_ops

    register_ops()


def _write_trex_row(ds: Dataset, run_id: str, sequence: str, recorded: str) -> None:
    from mosaic.tracking.trex.dataset_runs import (
        TRexIndexRow,
        trex_index,
        trex_index_path,
    )

    work = ds.get_root("trex") / run_id / sequence
    work.mkdir(parents=True, exist_ok=True)
    idx = trex_index(trex_index_path(ds))
    idx.ensure()
    idx.append(
        [
            TRexIndexRow(
                run_id=run_id,
                group="",
                sequence=sequence,
                abs_path=Path(ds.relative_to_root(work)),
                video_abs_path="",
                params_hash="",
                consumed_media_composition=recorded,
            )
        ]
    )
    idx.mark_finished(run_id)


def test_the_column_is_on_every_tracker_not_just_trex() -> None:
    """It was recorded by TREx and by frame extraction and by nobody else, so
    every other producer that reads media recorded nothing at all."""
    import dataclasses

    from mosaic.tracking.litpose.dataset_runs import LitposeIndexRow
    from mosaic.tracking.sleap.dataset_runs import SleapIndexRow
    from mosaic.tracking.trex.dataset_runs import TRexIndexRow
    from mosaic.tracking.ultralytics_track.dataset_runs import UltralyticsIndexRow

    for row_cls in (TRexIndexRow, SleapIndexRow, LitposeIndexRow, UltralyticsIndexRow):
        names = {field.name for field in dataclasses.fields(row_cls)}
        assert "consumed_media_composition" in names, row_cls.__name__


def test_a_moved_source_reads_as_drift(
    scenario_dataset_with_media: Dataset,
) -> None:
    """The case the identity cannot see."""
    ds = scenario_dataset_with_media
    entry = ("", "seq_a")
    was = media_compositions_for(ds, [entry])[entry]
    assert was, "the fixture recorded no media composition to compare against"
    _write_trex_row(ds, "trex.1.0-aaaaaaaaaa", "seq_a", was)

    assert drifted_media_entries(ds, "trex", "trex.1.0-aaaaaaaaaa") == ()

    # A different arrangement of the same entry's media, recorded as though the
    # run had read it. Equivalent to a re-transcode that re-linked underneath.
    _write_trex_row(ds, "trex.1.0-bbbbbbbbbb", "seq_a", was + "-moved")

    assert drifted_media_entries(ds, "trex", "trex.1.0-bbbbbbbbbb") == (entry,)


def test_an_empty_cell_on_either_side_is_not_drift(
    scenario_dataset_with_media: Dataset,
) -> None:
    """The honest-empty rule every composition comparison here follows: a blank
    recorded cell predates the column and a blank current one is not
    establishable, and neither is evidence of change."""
    ds = scenario_dataset_with_media
    _write_trex_row(ds, "trex.1.0-cccccccccc", "seq_a", "")

    assert drifted_media_entries(ds, "trex", "trex.1.0-cccccccccc") == ()


def test_drift_reaches_the_inventory(
    scenario_dataset_with_media: Dataset,
) -> None:
    """Complete and loadable, but superseded: reported rather than refused."""
    ds = scenario_dataset_with_media
    entry = ("", "seq_a")
    was = media_compositions_for(ds, [entry])[entry]
    _write_trex_row(ds, "trex.1.0-dddddddddd", "seq_a", was + "-moved")

    record = inventory(ds, kinds=["tracker-run"]).record(
        TrackerRunRef(root_key="trex", run_id="trex.1.0-dddddddddd")
    )

    assert record is not None
    assert record.status == "complete-but-drifted"
    assert record.drift == (entry,)


def test_a_legacy_media_rooted_dataset_records_nothing(tmp_path: Path) -> None:
    """A derivative has no composition of its own, the carve-out the media
    composition writer already makes."""
    manifest = new_dataset_manifest(name="legacy", base_dir=tmp_path / "ds")
    ds = Dataset(manifest_path=manifest).load(ensure_roots=True)
    ds.roots.pop("media_raw", None)

    assert media_compositions_for(ds, [("", "seq_a")]) == {}
