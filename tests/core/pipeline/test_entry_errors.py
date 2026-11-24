"""An entry error fails its entry alone, in serial, thread and process mode.

Three errors fail an entry: its input lacks a column the feature requires, an
arena-blind feature meets an input laid out in several arenas, and ``apply``
raises. Each records one ``entry_error`` with the error's type and message, names
the entry in ``Result.failed_entries`` and keeps the rows of the other entries. A
run whose every entry fails raises :class:`AllEntriesFailed`. A process worker
raises each error in its own process, so in process mode the error crosses the
process boundary.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar, Literal, NamedTuple

import pytest

from mosaic.core.arenas import ArenaLayout
from mosaic.core.dataset import Dataset
from mosaic.core.entry import parse_entry_key
from mosaic.core.pipeline.column_contract import Column, Requirement
from mosaic.core.pipeline.index import feature_index, feature_index_path
from mosaic.core.pipeline.parallelism import ParallelMode
from mosaic.core.pipeline.run import AllEntriesFailed, run_feature
from mosaic.core.pipeline.types import Feature, Result
from mosaic.runlog import read_run, run_log_dir
from tests.helpers import (
    ParallelFeature,
    add_tracks_table,
    logged_events,
    make_dataset,
    track_table,
)

VARIANT = "convert-x.0.1-aaaaaaaaaa"
_SEQUENCES = ("seq_a", "seq_b", "seq_c", "seq_d")
_MARKER = "marker"
"""The column :class:`NeedsTheMarker` requires, which the tables of kept entries hold."""

_TWO_ARENAS = ArenaLayout.grid(xs=(0, 50, 100), ys=(0, 100))


class NeedsTheMarker(ParallelFeature):
    """A process-capable feature that requires the column :data:`_MARKER`."""

    name = "prov-parallel-needs-marker"
    requires: ClassVar[tuple[Requirement, ...]] = (Column(_MARKER),)


class LosesSeqB(ParallelFeature):
    """A process-capable feature whose ``apply`` raises on ``seq_b``."""

    name = "prov-parallel-loses"
    fail_on = ("seq_b",)


class LosesEvery(LosesSeqB):
    """A :class:`LosesSeqB` that raises on every sequence."""

    fail_on = _SEQUENCES


type EntryError = Literal["column-check", "arena-refusal", "apply"]


class _Expected(NamedTuple):
    """What an ``entry_error`` of one kind records."""

    type: str
    message: str


_EXPECTED: dict[EntryError, _Expected] = {
    "column-check": _Expected("MissingInputColumns", f"lacks '{_MARKER}'"),
    "arena-refusal": _Expected("ArenaBlindFeature", "layout of 2 arenas"),
    "apply": _Expected("RuntimeError", "a lost entry"),
}

_ERRORS: tuple[EntryError, ...] = ("column-check", "arena-refusal", "apply")


class _Mode(NamedTuple):
    """How a run computes its entries."""

    name: str
    workers: int
    kind: ParallelMode


_MODES = (
    _Mode("serial", 1, "thread"),
    _Mode("thread", 3, "thread"),
    _Mode("process", 3, "process"),
)


def _dataset(base: Path, error: EntryError, lost: frozenset[str]) -> Dataset:
    """Four entries of two individuals each, of which those of *lost* fail with *error*.

    For the column check the other entries hold :data:`_MARKER`. For the arena
    refusal the entries of *lost* are laid out in two arenas, and their tables
    assign none.
    """
    ds = make_dataset(base)
    for sequence in _SEQUENCES:
        table = track_table("", sequence, n_ids=2)
        if error == "column-check" and sequence not in lost:
            table[_MARKER] = 1.0
        add_tracks_table(ds, VARIANT, table, std_format="mosaic_v1")
    if error == "arena-refusal":
        for sequence in sorted(lost):
            _ = ds.set_arena_layout(_TWO_ARENAS, group="", sequence=sequence)
    return ds


def _feature(error: EntryError, lost: frozenset[str]) -> Feature:
    if error == "column-check":
        return NeedsTheMarker()
    if error == "arena-refusal":
        from mosaic.behavior.feature_library.nearestneighbor import NearestNeighbor

        return NearestNeighbor()
    return LosesSeqB() if lost == frozenset({"seq_b"}) else LosesEvery()


def _run(
    ds: Dataset, error: EntryError, lost: frozenset[str], mode: _Mode, execution_id: str
) -> Result:
    return run_feature(
        ds,
        _feature(error, lost),
        parallel_workers=mode.workers,
        parallel_mode=mode.kind,
        execution_id=execution_id,
    )


def _check_errors(
    ds: Dataset,
    execution_id: str,
    error: EntryError,
    lost: frozenset[str],
    mode: _Mode,
) -> None:
    """Check one ``entry_error`` of *error* per entry of *lost*, and the mode run in.

    A feature that cannot run in processes would run in threads where processes
    are asked, so the last ``resources`` event must name the mode asked for.
    """
    errors = logged_events(ds.base_dir, execution_id, "entry_error")
    keys = (str(record["key"]) for record in errors)
    assert sorted(parse_entry_key(key)[1] for key in keys) == sorted(lost)
    expected = _EXPECTED[error]
    for record in errors:
        recorded = json.loads(str(record["error"]))
        assert recorded["type"] == expected.type
        assert expected.message in recorded["message"]
    (*_, resources) = logged_events(ds.base_dir, execution_id, "resources")
    assert resources["workers"] == mode.workers
    assert resources.get("parallel_mode", "") == (mode.kind if mode.workers > 1 else "")


@pytest.mark.parametrize("error", _ERRORS)
@pytest.mark.parametrize("mode", _MODES, ids=lambda mode: mode.name)
def test_an_entry_error_fails_its_entry_and_keeps_the_others(
    tmp_path: Path, mode: _Mode, error: EntryError
) -> None:
    lost = frozenset({"seq_b"})
    ds = _dataset(tmp_path / "ds", error, lost)
    execution_id = "TESTENTRYERROR"

    result = _run(ds, error, lost, mode, execution_id)

    assert [parse_entry_key(key)[1] for key in result.failed_entries] == ["seq_b"]
    _check_errors(ds, execution_id, error, lost, mode)
    assert result.run_id is not None
    storage = feature_index_path(ds, result.feature)
    rows = feature_index(storage).read(run_id=result.run_id)
    assert sorted(rows["sequence"]) == ["seq_a", "seq_c", "seq_d"]
    snapshot = read_run(run_log_dir(ds.base_dir), execution_id)
    assert snapshot is not None
    assert snapshot["status"] == "finished"
    assert snapshot["entries_failed"] == 1
    assert snapshot["entries_written"] == len(_SEQUENCES) - 1


@pytest.mark.parametrize("error", _ERRORS)
@pytest.mark.parametrize("mode", _MODES, ids=lambda mode: mode.name)
def test_an_error_in_every_entry_fails_the_run(
    tmp_path: Path, mode: _Mode, error: EntryError
) -> None:
    lost = frozenset(_SEQUENCES)
    ds = _dataset(tmp_path / "ds", error, lost)
    execution_id = "TESTEVERYENTRYERROR"

    with pytest.raises(AllEntriesFailed):
        _ = _run(ds, error, lost, mode, execution_id)

    _check_errors(ds, execution_id, error, lost, mode)
    snapshot = read_run(run_log_dir(ds.base_dir), execution_id)
    assert snapshot is not None
    assert snapshot["status"] == "failed"
    assert snapshot["entries_failed"] == len(_SEQUENCES)
    assert snapshot["entries_written"] == 0
