"""An uneven time axis is a run-log event a user can read, not only a log line.

A uniform-grid method warns when a table steps unevenly. ``run_feature`` collects each
warning per entry into a ``time_irregular`` event of the run-log, which the status
reads count and list. The event is neither a status nor a terminal event, and a reader
that does not know it ignores it.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from mosaic.behavior.feature_library.individual_wavelet import IndividualWavelet
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.graph import run_status_wire
from mosaic.core.pipeline.run import run_feature
from mosaic.core.timebase import collect_irregular_notices, uniform_grid_rate
from mosaic.runlog import (
    TERMINAL_STATUSES,
    JsonlRunLog,
    read_run,
    read_time_irregular,
    run_log_dir,
)

from tests.helpers import (
    add_tracks_table,
    dropped_frame_tracks,
    make_dataset,
)

VARIANT = "convert-tracktor_corrected_tsv.0.1-aaaaaaaaaa"
WAVELET: dict[str, object] = {"cols": ["X"], "f_min": 2.0, "f_max": 5.0, "n_freq": 4}


def _feature() -> IndividualWavelet:
    params: dict[str, object] = dict(WAVELET)
    return IndividualWavelet(IndividualWavelet.Inputs(("tracks",)), params=params)


def _dataset(tmp_path: Path, *, even: bool) -> tuple[Dataset, str]:
    ds = make_dataset(tmp_path / "ds")
    table = dropped_frame_tracks(ids=(0, 1), sequence="s1")
    if even:
        table["time"] = table["frame"] / 30.0
    add_tracks_table(ds, VARIANT, table, std_format="mosaic_v1")
    return ds, "s1"


def test_a_collector_receives_the_notice_and_the_log_still_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    table = dropped_frame_tracks()
    with caplog.at_level(logging.WARNING, logger="mosaic.core.timebase"):
        with collect_irregular_notices() as notices:
            _ = uniform_grid_rate(table, "individual-wavelet")

    [notice] = notices
    assert notice.who == "individual-wavelet"
    assert notice.largest_step == pytest.approx(1.45)
    assert (notice.first_frame, notice.last_frame) == (89, 90)
    assert "resample-tracks" in notice.message
    assert any("resample-tracks" in record.getMessage() for record in caplog.records)


def test_no_collector_means_no_notice_is_kept() -> None:
    with collect_irregular_notices() as notices:
        pass
    _ = uniform_grid_rate(dropped_frame_tracks(), "individual-wavelet")

    assert notices == []


def test_the_run_log_event_counts_into_the_snapshot_and_is_listed(
    tmp_path: Path,
) -> None:
    path = tmp_path / "01TESTEXECUTION.jsonl"
    log = JsonlRunLog(path, "01TESTEXECUTION")
    log.started(kind="feature", target="x", owner="o", host="h", pid=1)
    log.time_irregular(
        "g__a",
        who="pair-wavelet",
        largest_step=1.45,
        first_frame=89,
        last_frame=90,
        median_step=0.0189,
        rate=30.8,
    )
    log.time_irregular(
        "g__b",
        who="pair-wavelet",
        largest_step=0.4,
        first_frame=39,
        last_frame=40,
        median_step=0.0189,
        rate=40.0,
    )
    log.finished()
    log.close()

    snapshot = read_run(tmp_path, "01TESTEXECUTION")
    page = read_time_irregular(tmp_path, "01TESTEXECUTION", limit=1)

    assert snapshot is not None
    assert snapshot["entries_time_irregular"] == 2
    assert snapshot["status"] == "finished"
    assert page["total"] == 2 and page["has_more"]
    [first] = page["items"]
    assert first["key"] == "g__a"
    assert first["largest_step"] == pytest.approx(1.45)
    assert "resample-tracks" in first["message"]


def test_the_event_is_not_a_status() -> None:
    assert "time_irregular" not in TERMINAL_STATUSES
    assert TERMINAL_STATUSES == frozenset({"finished", "failed", "cancelled"})


def test_a_reader_that_does_not_know_the_event_folds_the_log_as_before(
    tmp_path: Path,
) -> None:
    """An unknown event kind advances liveness and changes nothing else."""
    path = tmp_path / "01OLD.jsonl"
    lines = [
        {
            "ev": "started",
            "t": "2026-11-11T00:00:00Z",
            "kind": "feature",
            "target": "x",
        },
        {"ev": "time_irregular", "t": "2026-11-11T00:00:01Z", "key": "a", "who": "w"},
        {"ev": "finished", "t": "2026-11-11T00:00:02Z"},
    ]
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n")

    snapshot = read_run(tmp_path, "01OLD")

    assert snapshot is not None
    assert snapshot["status"] == "finished"


@pytest.mark.parametrize("mode", ["thread", "process"])
def test_run_feature_records_one_event_per_irregular_entry(
    tmp_path: Path, mode: str
) -> None:
    ds, sequence = _dataset(tmp_path, even=False)

    result = run_feature(ds, _feature(), parallel_workers=2, parallel_mode=mode)

    logs = run_log_dir(ds.base_dir)
    snapshot = read_run(logs, str(result.execution_id))
    assert snapshot is not None
    assert snapshot["entries_time_irregular"] == 1
    [record] = read_time_irregular(logs, str(result.execution_id))["items"]
    assert record["key"].endswith(sequence)
    assert record["who"] == "individual-wavelet"
    assert record["largest_step"] == pytest.approx(1.45)
    assert snapshot["status"] == "finished"


def test_an_even_table_records_no_event(tmp_path: Path) -> None:
    ds, _ = _dataset(tmp_path, even=True)

    result = run_feature(ds, _feature())

    snapshot = read_run(run_log_dir(ds.base_dir), str(result.execution_id))
    assert snapshot is not None
    assert snapshot["entries_time_irregular"] == 0


def test_the_status_wire_counts_and_lists_the_notices(tmp_path: Path) -> None:
    ds, _ = _dataset(tmp_path, even=False)
    result = run_feature(ds, _feature())

    report = run_status_wire(ds.base_dir, str(result.execution_id))

    assert report is not None
    assert report.entries_time_irregular == 1
    [notice] = report.time_irregular
    assert notice.who == "individual-wavelet"
    assert "resample-tracks" in notice.message
    assert not report.time_irregular_has_more
    assert report.outcome == "finished"
