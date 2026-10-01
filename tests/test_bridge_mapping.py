"""Test the shared bridge's placing of a table on its entry's axes.

``publish_tracks_table`` maps a table tracked on a media variant into its entry's
source space, and times a table tracked on the join of an entry's clips by the
clips, before the schema is checked. It reports the columns that either dropped.
``publish_or_record`` turns that report into a run-log event, and a table the
mapping refuses into a failed entry.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.media.facts_columns import store_facts
from mosaic.core.media.preprocess import CropStep, Placement, TrimStep
from mosaic.core.media.timeline import concatenated_timeline
from mosaic.core.pipeline.job import job_context
from mosaic.core.pipeline.placement import (
    EntryAxis,
    SourceMapping,
    UnclassifiedColumnError,
)
from mosaic.core.pipeline.run_log import read_run, run_log_dir, run_log_path
from mosaic.core.pipeline.tracks_index import read_tracks_index
from mosaic.core.schema import TrackSchemaError
from mosaic.tracking.common.bridge import (
    BridgeCounts,
    publish_or_record,
    publish_tracks_table,
)
from tests.helpers import clip_facts, make_dataset

_KIND = "sleap"
_VARIANT = "sleap.9.9-aaaaaaaaaa"
_WIDTH, _HEIGHT = 640, 480
_SOURCE_FRAMES = 300
_CROP_X, _CROP_Y = 120, 40
_TRIM_START = 100
_ROWS = 4


def _crop_and_trim() -> SourceMapping:
    """Return a 320x240 crop at (120, 40) of source frames 100 to 199, at 30 fps."""
    clip = store_facts(
        _WIDTH, _HEIGHT, 30.0, _SOURCE_FRAMES, "h264", _SOURCE_FRAMES / 30.0, "", ""
    )
    placement = Placement.identity(_WIDTH, _HEIGHT, _SOURCE_FRAMES, 30.0)
    for step in (
        CropStep(x=_CROP_X, y=_CROP_Y, width=320, height=240),
        TrimStep(start=_TRIM_START, stop=200),
    ):
        placement = step.place(placement)
    return SourceMapping(placement, concatenated_timeline([clip]))


def _variant_axis() -> EntryAxis:
    return EntryAxis.of_variant(_crop_and_trim())


def _joined_axis(*rates: float) -> EntryAxis:
    """Return the axis of a join of one clip of ``_SOURCE_FRAMES`` per rate."""
    clips = [clip_facts(fps=rate, frame_count=_SOURCE_FRAMES) for rate in rates]
    return EntryAxis.of_entry_media(clips, windowed=False)


def _variant_table() -> pd.DataFrame:
    """Return a table as a tracker reports it on the variant file.

    ``timestamp`` is minted from a frame index and one rate, and
    ``BORDER_DISTANCE`` measures to the border of the cropped image. The mapping
    drops both.
    """
    frames = np.arange(_ROWS, dtype=np.int64)
    return pd.DataFrame(
        {
            "frame": frames,
            "time": frames / 30.0,
            "timestamp": frames / 30.0,
            "id": np.zeros(_ROWS, dtype=np.int64),
            "group": "g",
            "sequence": "s",
            "X": 10.0 + frames,
            "Y": 20.0 + frames,
            "poseX0": 11.0 + frames,
            "poseY0": 21.0 + frames,
            "BORDER_DISTANCE": np.full(_ROWS, 5.0),
        }
    )


def _publish(
    ds: Dataset,
    table: pd.DataFrame,
    *,
    axis: EntryAxis | None = None,
    strict: bool = False,
) -> BridgeCounts:
    return publish_tracks_table(
        ds,
        table,
        kind=_KIND,
        group="g",
        sequence="s",
        tracks_variant=_VARIANT,
        producer_run_id="sleap.9.9-bbbbbbbbbb",
        source=ds.get_root("tracks"),
        consumed=[],
        axis=EntryAxis() if axis is None else axis,
        strict=strict,
    )


def _events(ds: Dataset, execution_id: str) -> list[dict[str, object]]:
    """Return the attempt's run-log records, one per line."""
    path = run_log_path(ds.base_dir, execution_id)
    return [json.loads(line) for line in path.read_text().splitlines()]


def _published(ds: Dataset) -> pd.DataFrame:
    rows = read_tracks_index(ds)
    assert len(rows) == 1
    return pd.read_parquet(ds.resolve_path(str(rows.iloc[0]["abs_path"])))


def test_a_mapped_table_is_published_in_source_space(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path)

    counts = _publish(ds, _variant_table(), axis=_variant_axis())

    table = _published(ds)
    source_frames = _TRIM_START + np.arange(_ROWS)
    assert table["frame"].tolist() == source_frames.tolist()
    assert table["time"].tolist() == pytest.approx((source_frames / 30.0).tolist())
    assert table["X"].tolist() == [10.0 + _CROP_X + i for i in range(_ROWS)]
    assert table["Y"].tolist() == [20.0 + _CROP_Y + i for i in range(_ROWS)]
    assert table["poseX0"].tolist() == [11.0 + _CROP_X + i for i in range(_ROWS)]
    assert table["poseY0"].tolist() == [21.0 + _CROP_Y + i for i in range(_ROWS)]
    assert not {"timestamp", "BORDER_DISTANCE"} & set(table.columns)
    assert counts.dropped == ("timestamp", "BORDER_DISTANCE")
    assert counts.n_rows == _ROWS
    assert counts.frame_span == (_TRIM_START, _TRIM_START + _ROWS - 1)


def test_without_a_mapping_the_table_is_published_unchanged(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path)
    table = _variant_table()

    counts = _publish(ds, table)

    pd.testing.assert_frame_equal(_published(ds), table)
    assert counts.dropped == ()


def test_validation_reads_the_mapped_table(tmp_path: Path) -> None:
    """The mapping runs first, and the table checked is the one published.

    A table without ``time`` is refused by strict validation alone, and
    publishes once the mapping retimes it on the source timeline.
    """
    ds = make_dataset(tmp_path)
    untimed = _variant_table().drop(columns=["time"])

    with pytest.raises(TrackSchemaError, match="time"):
        _ = _publish(ds, untimed, strict=True)
    counts = _publish(ds, untimed, axis=_variant_axis(), strict=True)

    assert counts.n_rows == _ROWS
    assert "time" in _published(ds).columns


def test_strict_validation_refuses_a_table_missing_a_required_column(
    tmp_path: Path,
) -> None:
    """Strict validation refuses before any write. Lenient publishes and reports."""
    ds = make_dataset(tmp_path)
    headless = _variant_table().drop(columns=["X"])

    with pytest.raises(TrackSchemaError, match="X"):
        _ = _publish(ds, headless, strict=True)
    assert read_tracks_index(ds).empty

    counts = _publish(ds, headless)
    assert counts.n_rows == _ROWS
    assert "X" not in _published(ds).columns


def test_dropped_columns_reach_the_run_log(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ds = make_dataset(tmp_path)
    with job_context(ds, kind=_KIND, target=_KIND) as ctx:
        counts = publish_or_record(
            ctx,
            "g__s",
            lambda: _publish(ds, _variant_table(), axis=_variant_axis()),
            kind=_KIND,
        )

    assert counts is not None
    assert counts.dropped == ("timestamp", "BORDER_DISTANCE")
    snapshot = read_run(run_log_dir(ds.base_dir), ctx.execution_id)
    assert snapshot is not None
    assert snapshot["entries_columns_dropped"] == 1
    assert snapshot["entries_failed"] == 0
    assert snapshot["status"] == "finished"
    assert [
        (record["key"], record["columns"])
        for record in _events(ds, ctx.execution_id)
        if record["ev"] == "columns_dropped"
    ] == [("g__s", ["timestamp", "BORDER_DISTANCE"])]
    assert capsys.readouterr().err == (
        f"[{_KIND}] g__s: published without timestamp, BORDER_DISTANCE, which do "
        "not map onto the source media's pixels, frames or clock.\n"
    )


def test_one_column_a_joined_retime_dropped_is_reported_in_the_singular(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A join's retiming drops ``timestamp``, which a tool mints from one rate."""
    ds = make_dataset(tmp_path)
    table = _variant_table().drop(columns=["BORDER_DISTANCE"])

    with job_context(ds, kind=_KIND, target=_KIND) as ctx:
        _ = publish_or_record(
            ctx,
            "g__s",
            lambda: _publish(ds, table, axis=_joined_axis(30.0, 30.0)),
            kind=_KIND,
        )

    # After the line that reports the four-row table against its 600 frames.
    assert capsys.readouterr().err.endswith(
        f"[{_KIND}] g__s: published without timestamp, which does not map onto the "
        "source media's pixels, frames or clock.\n"
    )


def test_a_join_of_two_rates_drops_what_one_rate_spoiled(tmp_path: Path) -> None:
    """Per-second columns computed against one rate go, in the table's order.

    ``mosaic_v1`` forbids ``SPEED`` and ``VX`` whatever else happens, so the
    table publishes only because the retiming dropped them before validation.
    """
    ds = make_dataset(tmp_path)
    table = (
        _variant_table()
        .drop(columns=["BORDER_DISTANCE"])
        .assign(SPEED=np.ones(_ROWS), VX=np.ones(_ROWS))
    )

    with job_context(ds, kind=_KIND, target=_KIND) as ctx:
        counts = publish_or_record(
            ctx,
            "g__s",
            lambda: _publish(ds, table, axis=_joined_axis(30.0, 31.0)),
            kind=_KIND,
        )

    assert counts is not None
    assert counts.dropped == ("timestamp", "SPEED", "VX")
    assert not {"timestamp", "SPEED", "VX"} & set(_published(ds).columns)
    assert [
        record["columns"]
        for record in _events(ds, ctx.execution_id)
        if record["ev"] == "columns_dropped"
    ] == [["timestamp", "SPEED", "VX"]]


def test_a_table_from_one_clip_keeps_its_columns(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path)

    counts = _publish(ds, _variant_table(), axis=_joined_axis(30.0))

    assert counts.dropped == ()
    pd.testing.assert_frame_equal(_published(ds), _variant_table())


def test_a_table_that_keeps_every_column_records_no_event(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ds = make_dataset(tmp_path)
    with job_context(ds, kind=_KIND, target=_KIND) as ctx:
        counts = publish_or_record(
            ctx, "g__s", lambda: _publish(ds, _variant_table()), kind=_KIND
        )

    assert counts is not None
    snapshot = read_run(run_log_dir(ds.base_dir), ctx.execution_id)
    assert snapshot is not None
    assert snapshot["entries_columns_dropped"] == 0
    assert capsys.readouterr().err == ""


def test_a_table_the_mapping_refuses_is_a_failed_entry(tmp_path: Path) -> None:
    """A table with an unclassified numeric column is refused and does not publish."""
    ds = make_dataset(tmp_path)
    table = _variant_table().assign(mystery=np.ones(_ROWS))
    with job_context(ds, kind=_KIND, target=_KIND) as ctx:
        counts = publish_or_record(
            ctx,
            "g__s",
            lambda: _publish(ds, table, axis=_variant_axis()),
            kind=_KIND,
        )

    assert counts is None
    assert ctx.failed_keys == ["g__s"]
    snapshot = read_run(run_log_dir(ds.base_dir), ctx.execution_id)
    assert snapshot is not None
    assert snapshot["entries_failed"] == 1
    assert snapshot["status"] == "finished"
    errors = [
        json.loads(str(record["error"]))
        for record in _events(ds, ctx.execution_id)
        if record["ev"] == "entry_error"
    ]
    assert [error["type"] for error in errors] == [UnclassifiedColumnError.__name__]
    assert read_tracks_index(ds).empty
