"""A run that applies its entries in phases, each with its own settings.

The phases are one attempt: one ``started``, one run id and one fit, a total of every
phase's entries and one final count of the entries written. Each phase measures its
own levels and peak and writes its own final ``resources`` event.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import TypeAdapter

from mosaic.core.dataset import Dataset
from mosaic.core.entry import Entry
from mosaic.core.pipeline.run import ApplyPhase, run_feature
from mosaic.runlog import read_run, run_log_dir, run_log_path
from tests.helpers import ParallelFeature, add_tracks_variant, make_dataset

VARIANT = "convert-trex.0.2-1111111111"
SEQUENCES = ("seq_a", "seq_b", "seq_c", "seq_d")
ROWS = {"seq_a": 40, "seq_b": 80, "seq_c": 120, "seq_d": 160}


def _entries(*sequences: str) -> tuple[Entry, ...]:
    return tuple(("", sequence) for sequence in sequences)


def _dataset(tmp_path: Path) -> Dataset:
    ds = make_dataset(tmp_path / "ds")
    for sequence, rows in ROWS.items():
        add_tracks_variant(ds, VARIANT, sequence, n_rows=rows, std_format="mosaic_v1")
    return ds


def _events(ds: Dataset, execution_id: str, kind: str) -> list[dict[str, object]]:
    lines = run_log_path(ds.base_dir, execution_id).read_text().splitlines()
    return [record for record in map(json.loads, lines) if record["ev"] == kind]


_LEVELS = TypeAdapter(list[dict[str, float]])


def _levels(report: dict[str, object]) -> list[dict[str, float]]:
    """The ``in_flight_levels`` of a ``resources`` event."""
    return _LEVELS.validate_python(report["in_flight_levels"])


def _input_rows(ds: Dataset, execution_id: str) -> dict[str, float]:
    """The input rows of each computed entry, by key."""
    return {
        str(event["key"]): float(str(event["input_rows"]))
        for event in _events(ds, execution_id, "entry_end")
        if "input_rows" in event
    }


def _phase(
    *sequences: str, workers: int = 1, mode: str = "thread", margin: float = 0.0
) -> ApplyPhase:
    return ApplyPhase(
        entries=_entries(*sequences),
        workers=workers,
        mode="process" if mode == "process" else "thread",
        admission_margin=margin,
        measure_worker_base=False,
    )


def test_each_phase_writes_its_own_final_resources_in_one_attempt(
    tmp_path: Path,
) -> None:
    ds = _dataset(tmp_path)

    result = run_feature(
        ds,
        ParallelFeature(),
        phases=(_phase("seq_d"), _phase("seq_a", "seq_b", "seq_c", workers=3)),
        execution_id="PHASED",
    )

    assert result.entries_written == 4
    assert result.failed_entries == ()
    assert len(_events(ds, "PHASED", "started")) == 1
    snapshot = read_run(run_log_dir(ds.base_dir), "PHASED")
    assert snapshot is not None
    assert snapshot["progress_total"] == 4
    assert snapshot["entries_written"] == 4
    reports = _events(ds, "PHASED", "resources")
    first = [report for report in reports if report["workers"] == 1][-1]
    last = reports[-1]
    assert last["workers"] == 3
    rows = _input_rows(ds, "PHASED")
    assert [level["rows"] for level in _levels(first)] == [rows["seq_d"]]
    assert sum(level["rows"] for level in _levels(last)) == pytest.approx(
        rows["seq_a"] + rows["seq_b"] + rows["seq_c"]
    )


def test_phases_that_run_alike_are_one_phase(tmp_path: Path) -> None:
    """One entry at a time in threads is one way to run, whatever mode was asked."""
    ds = _dataset(tmp_path)

    _ = run_feature(
        ds,
        ParallelFeature(),
        phases=(
            _phase("seq_d", "seq_c", workers=1, mode="thread"),
            _phase("seq_b", "seq_a", workers=1, mode="process"),
        ),
        execution_id="ALIKE",
    )

    reports = _events(ds, "ALIKE", "resources")
    levels = _levels(reports[-1])
    assert [level["in_flight"] for level in levels] == [1]
    assert levels[0]["rows"] == pytest.approx(sum(_input_rows(ds, "ALIKE").values()))


def test_an_entry_in_two_phases_is_refused(tmp_path: Path) -> None:
    ds = _dataset(tmp_path)

    with pytest.raises(ValueError, match="seq_b"):
        _ = run_feature(
            ds,
            ParallelFeature(),
            phases=(
                _phase("seq_a", "seq_b"),
                _phase("seq_b", "seq_c", "seq_d", workers=2),
            ),
        )


def test_an_entry_in_no_phase_is_refused(tmp_path: Path) -> None:
    ds = _dataset(tmp_path)

    with pytest.raises(ValueError, match="seq_d"):
        _ = run_feature(
            ds,
            ParallelFeature(),
            phases=(_phase("seq_a", "seq_b"), _phase("seq_c", workers=2)),
        )


def test_a_cache_hit_is_one_claim_of_the_whole_attempt(tmp_path: Path) -> None:
    ds = _dataset(tmp_path)
    phases = (_phase("seq_d", "seq_c"), _phase("seq_b", "seq_a", workers=2))
    first = run_feature(ds, ParallelFeature(), phases=phases, execution_id="FIRST")
    second = run_feature(ds, ParallelFeature(), phases=phases, execution_id="SECOND")

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert len(_events(ds, "FIRST", "cache_hit")) == 0
    assert len(_events(ds, "SECOND", "cache_hit")) == 1

    run_root = (
        ds.get_root("features") / "prov-parallel__from__tracks" / str(first.run_id)
    )
    (run_root / "seq_a.parquet").unlink()
    third = run_feature(ds, ParallelFeature(), phases=phases, execution_id="THIRD")

    assert third.cache_hit is False
    assert len(_events(ds, "THIRD", "cache_hit")) == 0
    assert third.entries_written == 4
