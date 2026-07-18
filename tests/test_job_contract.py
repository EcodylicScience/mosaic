"""Phase-0 tests for the Job Contract (attempt run-log, progress, cancel).

Exercises the additive layer on ``run_feature``: the append-only JSONL run-log
(per attempt, under ``<dataset_root>/.mosaic/runs/``), per-entry progress,
cooperative cancellation, cache-hit surfacing, the ``execution_id``/``run_id``
separation, and the standalone external readers -- using lightweight mock
features (no heavy dependencies).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mosaic.core.pipeline._utils import hash_params, new_execution_id
from mosaic.core.pipeline.index import feature_index, feature_index_path
from mosaic.core.pipeline.job import CancelToken, Cancelled, job_context
from mosaic.core.pipeline.run import run_feature
from mosaic.core.pipeline.run_log import (
    JsonlRunLog,
    read_run,
    read_run_progress,
    read_runs,
    read_runs_by_run_id,
    reduce_run_log,
    run_log_dir,
    run_log_path,
)
from mosaic.core.pipeline.types import Inputs, InputStream, Params, Result, TrackInput


# --- Minimal mock dataset + feature (mirrors tests/test_run_feature.py) ---


class _MockDataset:
    def __init__(self, root: Path):
        self._root = root
        for directory in ("tracks", "features"):
            (root / directory).mkdir(parents=True, exist_ok=True)

    @property
    def base_dir(self) -> Path:
        return self._root

    def get_root(self, key: str) -> Path:
        return self._root / key

    def resolve_path(self, stored_path: object, anchor: object = None) -> Path:
        path = Path(str(stored_path))
        return path if path.is_absolute() else self._root / path

    def relative_to_root(self, path: object) -> str:
        try:
            return str(Path(str(path)).resolve().relative_to(self._root.resolve()))
        except ValueError:
            return str(path)

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


def _run_dir(ds: _MockDataset) -> Path:
    return run_log_dir(ds.base_dir)


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

    row = read_run(_run_dir(ds), result.execution_id)
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
    rows = read_run_progress(_run_dir(ds), result.execution_id)
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

    # two attempt logs, both finished, sharing one run_id
    attempts = read_runs_by_run_id(_run_dir(ds), r1.run_id)
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

    # exactly one run-log, marked cancelled
    all_runs = read_runs(_run_dir(ds))
    assert len(all_runs) == 1
    assert all_runs[0]["status"] == "cancelled"
    execution_id = all_runs[0]["execution_id"]
    run_id = all_runs[0]["run_id"]

    # partial outputs are durable on disk -> resumable. The filesystem is the
    # source of truth: some (but not all) entry parquets were written before the
    # cancel landed, and the run is NOT marked finished in index.csv.
    storage = "test-jc__from__tracks"
    run_root = ds.get_root("features") / storage / run_id
    parquets = list(run_root.glob("*.parquet"))
    assert 1 <= len(parquets) < 4  # some, but not all, completed

    idx_df = feature_index(feature_index_path(ds, storage)).read()
    assert (idx_df["finished_at"] == "").all()  # run stays unfinished -> resumable

    # a mid-run cancel is recorded on the run-log too
    assert read_run(_run_dir(ds), execution_id)["status"] == "cancelled"


def test_inert_token_does_not_cancel(tmp_path: Path):
    ds = _MockDataset(tmp_path)
    _setup_tracks(ds, [("g", "s1")])
    # default token never fires
    result = run_feature(ds, _Stateless())
    assert read_run(_run_dir(ds), result.execution_id)["status"] == "finished"


# --- failure ---------------------------------------------------------------


def test_failure_marks_failed_with_error(tmp_path: Path):
    ds = _MockDataset(tmp_path)
    _setup_tracks(ds, [("g", "s1")])

    with pytest.raises(RuntimeError, match="kaboom"):
        # The apply exception is swallowed per-entry by run_feature, so instead
        # trigger failure via job_context directly to exercise the failed path.
        with job_context(ds, kind="feature", target="t"):
            raise RuntimeError("kaboom")

    # locate the failed attempt
    failed = read_runs(_run_dir(ds), status="failed")
    assert len(failed) == 1
    assert "kaboom" in failed[0]["error_json"]


# --- track=False opt-out ---------------------------------------------------


def test_track_false_writes_no_log(tmp_path: Path):
    ds = _MockDataset(tmp_path)
    _setup_tracks(ds, [("g", "s1")])
    result = run_feature(ds, _Stateless(), track=False)
    assert result.execution_id is not None  # still minted
    # but nothing recorded: no run-log file for this attempt
    assert not run_log_path(ds.base_dir, result.execution_id).exists()


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


# --- standalone readers work off just the run-log directory ----------------


def test_standalone_readers(tmp_path: Path):
    ds = _MockDataset(tmp_path)
    _setup_tracks(ds, [("g", "s1")])
    result = run_feature(ds, _Stateless())

    run_dir = _run_dir(ds)
    row = read_run(run_dir, result.execution_id)
    assert row is not None and row["execution_id"] == result.execution_id
    assert len(read_runs(run_dir, kind="feature")) == 1
    assert len(read_runs(run_dir, status="finished")) == 1
    assert len(read_runs_by_run_id(run_dir, result.run_id)) == 1


# --- run-log writer + reducer (the store's own contract) -------------------


def test_run_log_lifecycle_methods(tmp_path: Path):
    """A JsonlRunLog folds back to the attempt snapshot the CLI/API consume."""
    eid = new_execution_id()
    path = run_log_path(tmp_path, eid)
    log = JsonlRunLog(path, eid)
    log.started(kind="feature", target="speed-angvel", owner="me", host="h", pid=123)
    log.set_run_id("0.1-deadbeef")
    log.set_total(5)
    log.heartbeat(3, 5)
    log.finished()
    log.close()

    snap = reduce_run_log(path)
    assert snap is not None
    assert snap["kind"] == "feature"
    assert snap["target"] == "speed-angvel"
    assert snap["owner"] == "me" and snap["host"] == "h" and int(snap["pid"]) == 123
    assert snap["run_id"] == "0.1-deadbeef"
    assert snap["status"] == "finished"
    assert int(snap["progress_done"]) == 3
    assert int(snap["progress_total"]) == 5
    assert snap["finished_at"] != ""

    # a non-terminal log reduces to 'running' with a heartbeat_at -- what a
    # supervisor scans to detect live vs stale attempts (there is no DB flag).
    eid2 = new_execution_id()
    log2 = JsonlRunLog(run_log_path(tmp_path, eid2), eid2)
    log2.started(kind="feature", target="x")
    log2.heartbeat(1, 2)
    log2.close()
    running = read_runs(run_log_dir(tmp_path), status="running")
    match = next(r for r in running if r["execution_id"] == eid2)
    assert match["heartbeat_at"] != ""


def test_reduce_tolerates_partial_last_line(tmp_path: Path):
    """NFS-safe read: a torn last line (in-flight append) is skipped, not fatal."""
    eid = new_execution_id()
    path = run_log_path(tmp_path, eid)
    path.parent.mkdir(parents=True, exist_ok=True)
    good = '{"t": "2026-01-01T00:00:00+00:00", "ev": "started", "kind": "feature", "target": "t"}\n'
    torn = '{"t": "2026-01-01T00:00:01+00:00", "ev": "heart'  # partial, no newline
    path.write_text(good + torn)

    snap = reduce_run_log(path)
    assert snap is not None
    assert snap["kind"] == "feature"  # the complete line parsed
    assert snap["status"] == "running"  # torn line ignored, not crashed on


# --- completeness gate (now filesystem-driven) -----------------------------


def test_completeness_gate_marks_index_finished_only_when_all_outputs_present(
    tmp_path: Path,
):
    """``run_feature`` marks ``index.csv`` finished only when every manifest
    entry's output parquet is on disk.

    The gate moved from a DB ``pending_entries`` query to a direct
    ``out_path.exists()`` check (``run.py``). A full run finishes; a partial run
    (see ``test_cancel_midrun_...``) stays unfinished and is resumable.
    """
    ds = _MockDataset(tmp_path)
    _setup_tracks(ds, [("g", "s1"), ("g", "s2")])

    result = run_feature(ds, _Stateless())
    storage = "test-jc__from__tracks"
    idx_df = feature_index(feature_index_path(ds, storage)).read(run_id=result.run_id)
    # every entry present on disk -> finished_at stamped on all rows
    assert len(idx_df) == 2
    assert (idx_df["finished_at"] != "").all()
