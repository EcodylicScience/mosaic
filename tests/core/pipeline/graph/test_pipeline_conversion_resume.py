"""A conversion that stopped part way is pending, and a resubmit converts the rest.

``convert-tracks`` takes no scope, so a plan that measured it against the tables
the dataset already held would read a variant with 6 of 10 raw files converted as
complete. The plan measures it against the raw index rows of its format.
"""

from __future__ import annotations

import socket
from pathlib import Path

from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.graph import (
    CoverageShort,
    Recipe,
    plan_pipeline,
    plan_wire,
    read_recipe,
    request_summary,
    run_pipeline,
    submit_request,
)
from mosaic.core.pipeline.graph.conversion_remainder import stand_ins
from mosaic.core.pipeline.tracks_index import read_tracks_index
from mosaic.runlog import JsonlRunLog, run_log_path
from tests.helpers import (
    calms21_dataset,
    exited_pid,
    make_dataset,
    write_trex_npz,
)

RAW_FILES = 10
CONVERTED = 6


class _Stop(Exception):
    """Raised by a cancel check to end a conversion after some files."""


def _convert_recipe(source_format: str) -> Recipe:
    return read_recipe(
        {
            "steps": [
                {
                    "id": "convert",
                    "type": "op",
                    "kind": "convert-tracks",
                    "params": {"source_format": source_format},
                }
            ]
        }
    )


def _trex_dataset(root: Path) -> Dataset:
    """``RAW_FILES`` TREx exports, each one sequence, scanned and not converted."""
    ds = make_dataset(root)
    raw = ds.base_dir / "raw"
    for number in range(RAW_FILES):
        write_trex_npz(raw / f"s{number:02d}_fish0.npz")
    _ = ds.index_tracks_raw([raw], patterns=["*.npz"], src_format="trex_npz")
    return ds


def _stop_after(done: int):
    calls = 0

    def check() -> None:
        nonlocal calls
        calls += 1
        if calls > done:
            raise _Stop

    return check


def _interrupted(root: Path) -> Dataset:
    """A TREx dataset whose conversion was cancelled after ``CONVERTED`` files."""
    ds = _trex_dataset(root)
    try:
        _ = ds.convert_all_tracks(
            source_format="trex_npz", cancel_check=_stop_after(CONVERTED)
        )
    except _Stop:
        pass
    assert len(read_tracks_index(ds)) == CONVERTED
    return ds


def _table_stamps(ds: Dataset) -> dict[str, int]:
    return {
        str(path): path.stat().st_mtime_ns
        for path in sorted((ds.get_root("tracks")).rglob("*.parquet"))
    }


def test_a_conversion_that_stopped_part_way_plans_the_remainder(
    tmp_path: Path,
) -> None:
    ds = _interrupted(tmp_path / "ds")

    plan = plan_pipeline(ds, _convert_recipe("trex_npz"))

    step = plan.step("convert")
    assert not step.is_complete
    assert len(step.coverage.covered) == CONVERTED
    assert len(step.coverage.missing) == RAW_FILES - CONVERTED
    assert isinstance(step.reason, CoverageShort)
    assert (step.reason.covered, step.reason.target) == (CONVERTED, RAW_FILES)
    assert plan.pending == (step,)


def test_a_resubmit_converts_only_the_remainder(tmp_path: Path) -> None:
    ds = _interrupted(tmp_path / "ds")
    recipe = _convert_recipe("trex_npz")
    before = _table_stamps(ds)

    run = run_pipeline(ds, recipe)

    assert [outcome.state for outcome in run.outcomes] == ["ran"]
    assert len(read_tracks_index(ds)) == RAW_FILES
    after = _table_stamps(ds)
    assert len(after) == RAW_FILES
    assert all(after[path] == stamp for path, stamp in before.items())
    assert plan_pipeline(ds, recipe).step("convert").is_complete


def test_a_finished_conversion_plans_complete(tmp_path: Path) -> None:
    ds = _trex_dataset(tmp_path / "ds")
    recipe = _convert_recipe("trex_npz")
    assert not plan_pipeline(ds, recipe).step("convert").is_complete

    _ = run_pipeline(ds, recipe)

    plan = plan_pipeline(ds, recipe)
    assert plan.step("convert").is_complete
    assert plan.step("convert").coverage.missing == frozenset()


def test_a_file_holding_several_sequences_is_pending_until_it_is_converted(
    tmp_path: Path,
) -> None:
    """A CalMS21 file names no sequence, so the plan counts the file."""
    ds = calms21_dataset(
        tmp_path / "ds", {"train": {"a": {"s1": 4, "s2": 3}}, "test": {"a": {"s3": 5}}}
    )
    recipe = _convert_recipe("calms21_npy")
    pending = plan_pipeline(ds, recipe).step("convert")
    assert not pending.is_complete
    assert len(pending.coverage.missing) == 2

    _ = run_pipeline(ds, recipe)

    assert plan_pipeline(ds, recipe).step("convert").is_complete


def test_a_stand_in_for_a_file_names_it_relative_to_the_dataset(
    tmp_path: Path,
) -> None:
    ds = calms21_dataset(tmp_path / "ds", {"train": {"a": {"s1": 4}}})

    wire = plan_wire(plan_pipeline(ds, _convert_recipe("calms21_npy")))

    [step] = wire.steps
    assert step.coverage.missing_sample == (("", "raw/train.npy"),)
    assert str(tmp_path) not in str(step.coverage.missing_sample)


def test_a_stand_in_outside_the_dataset_names_the_file(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds")

    named = stand_ins(
        ds, [tmp_path / "elsewhere" / "a.npy", tmp_path / "elsewhere" / "b.npy"]
    )
    shared = stand_ins(ds, [tmp_path / "x" / "a.npy", tmp_path / "y" / "a.npy"])

    assert named == {("", "a.npy"), ("", "b.npy")}
    assert shared == {
        ("", str(tmp_path / "x" / "a.npy")),
        ("", str(tmp_path / "y" / "a.npy")),
    }


def test_a_converted_file_leaves_the_target_and_its_entries_count(
    tmp_path: Path,
) -> None:
    """Three entries of two files read 3 of 3 once converted, not 5 of 5."""
    ds = calms21_dataset(
        tmp_path / "ds", {"train": {"a": {"s1": 4, "s2": 3}}, "test": {"a": {"s3": 5}}}
    )
    recipe = _convert_recipe("calms21_npy")
    before = plan_pipeline(ds, recipe).step("convert").coverage
    assert (len(before.covered), len(before.target)) == (0, 2)

    _ = run_pipeline(ds, recipe)

    after = plan_pipeline(ds, recipe).step("convert").coverage
    assert (len(after.covered), len(after.target)) == (3, 3)
    assert not any(entry[1].endswith(".npy") for entry in after.target)


def test_an_orphaned_request_does_not_block_the_resubmit(tmp_path: Path) -> None:
    """A driver killed mid-conversion leaves a request that reads orphaned."""
    ds = _interrupted(tmp_path / "ds")
    recipe = _convert_recipe("trex_npz")
    killed = submit_request(ds, recipe).request
    execution_id = killed.step_executions["convert"]
    with JsonlRunLog(run_log_path(ds.base_dir, execution_id), execution_id) as log:
        log.started(
            kind="op",
            target="trex_npz",
            host=socket.gethostname(),
            pid=exited_pid(),
        )

    assert request_summary(ds.base_dir, killed).verdict == "orphaned"
    resubmitted = submit_request(ds, recipe)

    assert resubmitted.duplicates == ()
    assert plan_pipeline(ds, recipe).step("convert").coverage.missing
