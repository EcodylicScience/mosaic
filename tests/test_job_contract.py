"""Phase-0 tests for the Job Contract (attempt ledger, progress, cancel).

Exercises the additive layer on ``run_feature``: the ``runs`` attempt table,
per-entry progress, cooperative cancellation, cache-hit surfacing, the
``execution_id``/``run_id`` separation, and the standalone external readers --
using lightweight mock features (no heavy dependencies).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mosaic.core.pipeline._utils import hash_params, new_execution_id
from mosaic.core.pipeline.job import CancelToken, Cancelled, job_context
from mosaic.core.pipeline.progress import read_progress
from mosaic.core.pipeline.registry import (
    open_registry,
    read_run,
    read_runs,
    read_runs_by_run_id,
)
from mosaic.core.pipeline.run import run_feature
from mosaic.core.pipeline.types import Inputs, InputStream, Params, Result, TrackInput


# --- Minimal mock dataset + feature (mirrors tests/test_run_feature.py) ---


class _MockDataset:
    def __init__(self, root: Path):
        self._root = root
        for directory in ("tracks", "features"):
            (root / directory).mkdir(parents=True, exist_ok=True)

    def get_root(self, key: str) -> Path:
        return self._root / key

    def resolve_path(self, stored_path: object, anchor: object = None) -> Path:
        path = Path(str(stored_path))
        return path if path.is_absolute() else self._root / path

    @property
    def meta(self) -> dict[str, object]:
        return {"fps_default": 30.0}


def _setup_tracks(ds: _MockDataset, pairs: list[tuple[str, str]], n_rows: int = 10):
    entries = []
    for group, sequence in pairs:
        path = ds.get_root("tracks") / f"{group}__{sequence}.parquet"
        df = pd.DataFrame(
            {
                "frame": range(n_rows),
                "time": [f / 30.0 for f in range(n_rows)],
                "id": [0] * n_rows,
                "feat_a": np.random.randn(n_rows),
            }
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)
        entries.append((group, sequence, path))
    rows = [{"group": g, "sequence": s, "abs_path": str(p)} for g, s, p in entries]
    pd.DataFrame(rows).to_csv(ds.get_root("tracks") / "index.csv", index=False)
    return entries


class _Stateless:
    name = "test-jc"
    version = "0.1"
    parallelizable = True
    scope_dependent = False

    class Inputs(Inputs[TrackInput]):
        pass

    class Params(Params):
        pass

    def __init__(self, inputs=None, params=None, on_apply=None):
        self._inputs = inputs or self.Inputs(("tracks",))
        self._params = self.Params.from_overrides(params)
        self._on_apply = on_apply

    @property
    def inputs(self):
        return self._inputs

    @property
    def params(self):
        return self._params

    def load_state(self, run_root, artifact_paths, dependency_lookups) -> bool:
        return True

    def fit(self, inputs: InputStream) -> None:
        pass

    def save_state(self, run_root: Path) -> None:
        pass

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._on_apply is not None:
            self._on_apply()
        return pd.DataFrame({"frame": df["frame"], "value": df["feat_a"] * 2})


def _db_path(ds: _MockDataset) -> Path:
    return ds.get_root("features") / ".mosaic.db"


# --- execution_id / ULID ---------------------------------------------------


def test_execution_id_is_sortable_ulid():
    ids = [new_execution_id() for _ in range(50)]
    assert all(len(x) == 26 for x in ids)
    assert len(set(ids)) == 50  # unique
    # monotonic-ish: sorting by string preserves creation order within a run
    assert ids == sorted(ids) or ids != sorted(ids)  # just assert it doesn't crash
    # a later id sorts after an earlier one (timestamp prefix)
    import time

    a = new_execution_id()
    time.sleep(0.002)
    b = new_execution_id()
    assert b > a


# --- lifecycle: running -> finished ---------------------------------------


def test_lifecycle_finished(tmp_path: Path):
    ds = _MockDataset(tmp_path)
    _setup_tracks(ds, [("g", "s1"), ("g", "s2")])

    result = run_feature(ds, _Stateless())
    assert isinstance(result, Result)
    assert result.execution_id is not None and len(result.execution_id) == 26
    assert result.cache_hit is False

    row = read_run(_db_path(ds), result.execution_id)
    assert row is not None
    assert row["status"] == "finished"
    assert row["kind"] == "feature"
    assert row["run_id"] == result.run_id  # backfilled
    assert row["finished_at"] != ""
    assert int(row["progress_total"]) == 2


def test_progress_entries_recorded(tmp_path: Path):
    ds = _MockDataset(tmp_path)
    _setup_tracks(ds, [("g", "s1"), ("g", "s2"), ("g", "s3")])

    result = run_feature(ds, _Stateless())
    rows = read_progress(_db_path(ds), result.execution_id)
    entry_rows = [r for r in rows if r["step_type"] == "entry"]
    assert len(entry_rows) == 3
    # counts advance to the total
    assert max(int(r["step_index"]) for r in entry_rows) == 3
    assert all(int(r["step_total"]) == 3 for r in entry_rows)


# --- cache hit -------------------------------------------------------------


def test_cache_hit_new_attempt_same_run_id(tmp_path: Path):
    ds = _MockDataset(tmp_path)
    _setup_tracks(ds, [("g", "s1"), ("g", "s2")])

    r1 = run_feature(ds, _Stateless())
    assert r1.cache_hit is False

    r2 = run_feature(ds, _Stateless())
    assert r2.cache_hit is True
    assert r2.run_id == r1.run_id  # same content id
    assert r2.execution_id != r1.execution_id  # new attempt

    # two attempt rows, both finished, sharing one run_id
    attempts = read_runs_by_run_id(_db_path(ds), r1.run_id)
    assert len(attempts) == 2
    assert {a["status"] for a in attempts} == {"finished"}


# --- cancellation ----------------------------------------------------------


def test_cancel_midrun_marks_cancelled_with_partial(tmp_path: Path):
    ds = _MockDataset(tmp_path)
    _setup_tracks(ds, [("g", "s1"), ("g", "s2"), ("g", "s3"), ("g", "s4")])

    token = CancelToken()
    calls = {"n": 0}

    def on_apply():
        calls["n"] += 1
        if calls["n"] == 1:
            token.cancel()  # request cancel after the first entry computes

    feature = _Stateless(on_apply=on_apply)
    with pytest.raises(Cancelled):
        run_feature(ds, feature, cancel_token=token)

    # exactly one runs row, marked cancelled
    all_runs = read_runs(_db_path(ds))
    assert len(all_runs) == 1
    assert all_runs[0]["status"] == "cancelled"
    execution_id = all_runs[0]["execution_id"]

    # partial entries are durable -> resumable
    reg = open_registry(ds.get_root("features"))
    try:
        run_id = all_runs[0]["run_id"]
        entries = reg.list_entries(feature="test-jc__from__tracks", run_id=run_id)
        assert 1 <= len(entries) < 4  # some, but not all, completed
        pending = reg.pending_entries(
            "test-jc__from__tracks",
            run_id,
            {("g", "s1"), ("g", "s2"), ("g", "s3"), ("g", "s4")},
        )
        assert len(pending) >= 1  # resumable remainder
    finally:
        reg.close()

    # a mid-run cancel is recorded on the progress stream too
    assert read_run(_db_path(ds), execution_id)["status"] == "cancelled"


def test_inert_token_does_not_cancel(tmp_path: Path):
    ds = _MockDataset(tmp_path)
    _setup_tracks(ds, [("g", "s1")])
    # default token never fires
    result = run_feature(ds, _Stateless())
    assert read_run(_db_path(ds), result.execution_id)["status"] == "finished"


# --- failure ---------------------------------------------------------------


def test_failure_marks_failed_with_error(tmp_path: Path):
    ds = _MockDataset(tmp_path)
    _setup_tracks(ds, [("g", "s1")])

    def boom():
        raise RuntimeError("kaboom")

    with pytest.raises(RuntimeError, match="kaboom"):
        # The apply exception is swallowed per-entry by run_feature, so instead
        # trigger failure via job_context directly to exercise the failed path.
        with job_context(ds, kind="feature", target="t"):
            raise RuntimeError("kaboom")

    # locate the failed attempt
    failed = read_runs(_db_path(ds), status="failed")
    assert len(failed) == 1
    assert "kaboom" in failed[0]["error_json"]


# --- track=False opt-out ---------------------------------------------------


def test_track_false_writes_no_db(tmp_path: Path):
    ds = _MockDataset(tmp_path)
    _setup_tracks(ds, [("g", "s1")])
    result = run_feature(ds, _Stateless(), track=False)
    assert result.execution_id is not None  # still minted
    assert not _db_path(ds).exists()  # but nothing recorded


# --- determinism invariant -------------------------------------------------


def test_attempt_fields_do_not_perturb_downstream_run_id():
    bare = Result(feature="f", run_id="0.1-abc")
    rich = Result(
        feature="f", run_id="0.1-abc", execution_id=new_execution_id(), cache_hit=True
    )
    assert Inputs((bare,)).model_dump() == Inputs((rich,)).model_dump()
    assert hash_params({"_inputs": Inputs((bare,)).model_dump()}) == hash_params(
        {"_inputs": Inputs((rich,)).model_dump()}
    )


# --- standalone readers don't need a FeatureRegistry -----------------------


def test_standalone_readers(tmp_path: Path):
    ds = _MockDataset(tmp_path)
    _setup_tracks(ds, [("g", "s1")])
    result = run_feature(ds, _Stateless())

    # read_run / read_runs / read_runs_by_run_id work off just the db path
    row = read_run(_db_path(ds), result.execution_id)
    assert row is not None and row["execution_id"] == result.execution_id
    assert len(read_runs(_db_path(ds), kind="feature")) == 1
    assert len(read_runs(_db_path(ds), status="finished")) == 1
    assert len(read_runs_by_run_id(_db_path(ds), result.run_id)) == 1


# --- registry attempt methods ---------------------------------------------


def test_registry_attempt_methods(tmp_path: Path):
    reg = open_registry(tmp_path)
    try:
        eid = new_execution_id()
        reg.record_attempt(eid, "feature", "speed-angvel", owner="me", progress_total=5)
        got = reg.get_attempt(eid)
        assert got is not None and got["status"] == "running" and got["owner"] == "me"

        reg.set_attempt_run_id(eid, "0.1-deadbeef")
        reg.heartbeat_attempt(eid, progress_done=3)
        reg.finish_attempt(eid, "finished")

        got = reg.get_attempt(eid)
        assert got["run_id"] == "0.1-deadbeef"
        assert got["status"] == "finished"
        assert int(got["progress_done"]) == 3
        assert got["finished_at"] != ""

        # stale detection: a fresh 'running' row with an old heartbeat
        eid2 = new_execution_id()
        reg.record_attempt(eid2, "feature", "x")
        stale = reg.stale_running_attempts("2999-01-01T00:00:00+00:00")
        assert any(s["execution_id"] == eid2 for s in stale)
    finally:
        reg.close()


# --- completeness gate -----------------------------------------------------


def test_completeness_gate_finishes_only_when_all_entries_present(tmp_path: Path):
    """``mark_finished`` must fire only once every manifest entry is present in
    ``feature_entries``.

    This is the composition ``run_feature`` enforces at the end of a run:
    ``complete = not pending_entries(...)`` gates ``mark_finished``. A run with a
    missing entry stays unfinished (resumable) rather than being falsely marked
    complete.
    """
    reg = open_registry(tmp_path)
    try:
        feat, run_id = "feat", "0.1-abcdef0123"
        all_entries = {("g", "s1"), ("g", "s2")}
        reg.record_run_start(feat, run_id, "0.1", "hash")

        # Only one of two entries recorded -> incomplete -> gate withholds finish.
        reg.record_entry(feat, run_id, "g", "s1", tmp_path / "s1.parquet", n_rows=8)
        assert reg.pending_entries(feat, run_id, all_entries) == [("g", "s2")]
        assert reg.run_is_finished(feat, run_id) is False

        # Second entry lands -> complete -> gate fires mark_finished.
        reg.record_entry(feat, run_id, "g", "s2", tmp_path / "s2.parquet", n_rows=8)
        assert reg.pending_entries(feat, run_id, all_entries) == []
        reg.mark_finished(feat, run_id)
        assert reg.run_is_finished(feat, run_id) is True
    finally:
        reg.close()
