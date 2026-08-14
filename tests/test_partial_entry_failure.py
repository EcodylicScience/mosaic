"""A run that lost entities must not report itself finished.

A per-entity ``apply`` exception is caught, printed to stderr, and skipped, in
both the inline and the parallel-drain paths. Nothing propagates, so
``job_context`` takes its success branch and writes ``finished`` to the run-log,
``mosaic run --json`` emits a hard-coded ``"status": "finished"`` and exits 0, and
mosaic-queue maps that exit code to a ``finished`` ledger row.

The corruption is durable and silent in the specific way that matters: under the
queue, the child's stderr goes to DEVNULL, so the one line naming the failed entry
is destroyed before anyone can read it. What remains is a run reporting success
with N outputs missing, indistinguishable from a run whose scope was smaller. The
feature index does leave such a run unfinished -- ``mark_finished`` is gated on
every entry's parquet existing -- but that signal never reaches the ledger and
names no entity.

Both catch sites are pinned by one parametrized body: ``parallel_workers=1``
takes the inline path, ``2`` the drain path. A fix that repairs one and not the
other fails here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mosaic.core.helpers import make_entry_key
from mosaic.core.pipeline.run import AllEntriesFailed, run_feature
from mosaic.core.pipeline.types import Inputs, InputStream, Params, TrackInput
from mosaic.runlog import read_run, run_log_dir
from tests.helpers import MockDataset

_DOOMED = "s2"


def _make_parquet(path: Path, sequence: str, n_rows: int = 10) -> None:
    """A track table carrying its own sequence name.

    ``apply`` receives only a frame, so the name is how a feature can fail for one
    entity and succeed for another without depending on iteration order.
    """
    df = pd.DataFrame(
        {
            "frame": range(n_rows),
            "time": [f / 30.0 for f in range(n_rows)],
            "id": [0] * n_rows,
            "sequence": [sequence] * n_rows,
            "feat_a": np.arange(n_rows, dtype=float),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


def _setup_tracks(ds: MockDataset, sequences: list[str]) -> None:
    rows: list[dict[str, str]] = []
    for sequence in sequences:
        path = ds.get_root("tracks") / f"g1__{sequence}.parquet"
        _make_parquet(path, sequence)
        rows.append({"group": "g1", "sequence": sequence, "abs_path": str(path)})
    pd.DataFrame(rows).to_csv(ds.get_root("tracks") / "index.csv", index=False)


class _FailsOnOneSequence:
    """Raises for ``_DOOMED`` and succeeds for every other entity."""

    name = "test-fails-on-one"
    version = "0.1"
    parallelizable = True
    scope_dependent = False
    consumed_roots: tuple[str, ...] = ()

    class Inputs(Inputs[TrackInput]):
        pass

    class Params(Params):
        pass

    def __init__(self, doomed: tuple[str, ...] = (_DOOMED,)) -> None:
        self._inputs = self.Inputs(("tracks",))
        self._params = self.Params.from_overrides(None)
        self._doomed = doomed

    @property
    def inputs(self) -> _FailsOnOneSequence.Inputs:
        return self._inputs

    @property
    def params(self) -> _FailsOnOneSequence.Params:
        return self._params

    def load_state(
        self,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_lookups: dict[str, dict[tuple[str, str], Path]],
    ) -> bool:
        return True

    def fit(self, inputs: InputStream) -> None:
        pass

    def save_state(self, run_root: Path) -> None:
        pass

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if str(df["sequence"].iloc[0]) in self._doomed:
            msg = "apply failed on purpose"
            raise RuntimeError(msg)
        return pd.DataFrame({"frame": df["frame"], "value": df["feat_a"] * 2})


@pytest.mark.parametrize("parallel_workers", [1, 2], ids=["inline", "parallel"])
def test_one_failed_entity_is_not_a_finished_run(
    tmp_path: Path, parallel_workers: int
) -> None:
    ds = MockDataset(tmp_path)
    _setup_tracks(ds, ["s1", _DOOMED])
    execution_id = "TESTPARTIAL1"

    result = run_feature(
        ds,
        _FailsOnOneSequence(),
        parallel_workers=parallel_workers,
        execution_id=execution_id,
    )

    # The entity that succeeded is still on disk: a partial run keeps its work.
    assert result.run_id is not None
    run_root = (
        ds.get_root("features") / "test-fails-on-one__from__tracks" / result.run_id
    )
    assert (run_root / "g1__s1.parquet").exists()
    assert not (run_root / f"g1__{_DOOMED}.parquet").exists()

    # The result names what it lost, rather than looking like a clean run.
    assert result.failed_entries == (make_entry_key("g1", _DOOMED),)

    # And the run-log carries the count, so the queue can see it without stderr.
    snapshot = read_run(run_log_dir(ds.base_dir), execution_id)
    assert snapshot is not None
    assert snapshot["entries_failed"] == 1


def test_every_entity_failing_is_a_failed_run(tmp_path: Path) -> None:
    """Losing every entity is a failure, not a finished run with no outputs."""
    ds = MockDataset(tmp_path)
    _setup_tracks(ds, ["s1", _DOOMED])
    execution_id = "TESTPARTIAL2"

    with pytest.raises(AllEntriesFailed):
        _ = run_feature(
            ds,
            _FailsOnOneSequence(doomed=("s1", _DOOMED)),
            execution_id=execution_id,
        )

    snapshot = read_run(run_log_dir(ds.base_dir), execution_id)
    assert snapshot is not None
    assert snapshot["status"] == "failed"
    assert snapshot["entries_failed"] == 2


def test_an_empty_scope_is_not_an_all_entries_failure(tmp_path: Path) -> None:
    """No entities is not the same as every entity failing.

    The all-failed check has to be guarded on having had work to do, or a run over
    an empty scope -- a legitimate outcome the suite already relies on -- would
    start raising.
    """
    ds = MockDataset(tmp_path)
    _setup_tracks(ds, ["s1"])

    result = run_feature(ds, _FailsOnOneSequence(), sequences=["nosuchsequence"])

    assert result.failed_entries == ()
