"""Regression test: a multi-step Pipeline records an attempt row for EVERY step.

``Pipeline.run`` shares one ``FeatureRegistry`` across its steps. Each step's
per-job ``SQLiteProgressCallback`` opens (and closes) its own connection to the
same ``.mosaic.db``. On a WAL-capable filesystem all steps' ``runs`` attempt rows
persist; this test locks that in so a future change to the shared-registry path
can't silently drop later steps from the status bridge.

(Note: on a WAL-hostile filesystem such as exFAT, closing that second connection
makes the long-lived registry connection lose subsequent writes, so only the first
step's row survives -- an environmental limitation, not exercised here since pytest's
tmp_path is on a native filesystem.)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mosaic.behavior.feature_library import SpeedAngvel
from mosaic.core.dataset import Dataset, new_dataset_manifest
from mosaic.core.pipeline import FeatureStep, Pipeline
from mosaic.core.pipeline.index import feature_run_root
from mosaic.core.pipeline.registry import open_registry, read_runs


def _dataset_with_tracks(tmp_path: Path) -> Dataset:
    manifest = new_dataset_manifest("t", base_dir=tmp_path)
    ds = Dataset(manifest_path=manifest).load()
    tracks_root = ds.get_root("tracks")
    rows = []
    for group, sequence in [("g", "s1"), ("g", "s2")]:
        n = 12
        df = pd.DataFrame(
            {
                "frame": range(n),
                "time": [f / 30.0 for f in range(n)],
                "id": [0] * n,
                "X": np.linspace(0.0, 5.0, n),
                "Y": np.linspace(0.0, 2.0, n),
            }
        )
        path = tracks_root / f"{group}__{sequence}.parquet"
        df.to_parquet(path)
        rows.append({"group": group, "sequence": sequence, "abs_path": str(path)})
    pd.DataFrame(rows).to_csv(tracks_root / "index.csv", index=False)
    return ds


def test_pipeline_records_every_step(tmp_path: Path) -> None:
    ds = _dataset_with_tracks(tmp_path)

    # Two independent tracks-sourced steps (same feature, different params -> two
    # distinct run_ids). The pipeline runs them sequentially on one shared registry.
    pipe = Pipeline()
    pipe.add(FeatureStep("speed_a", SpeedAngvel, {"step_size": 1}))
    pipe.add(FeatureStep("speed_b", SpeedAngvel, {"step_size": 2}))
    pipe.run(ds)

    db = ds.get_root("features") / ".mosaic.db"
    runs = read_runs(db, kind="feature")

    # BOTH steps must have left an attempt row -- not just the first.
    assert len(runs) == 2, f"expected 2 attempt rows, got {[r['run_id'] for r in runs]}"
    assert {str(r["status"]) for r in runs} == {"finished"}
    assert len({str(r["run_id"]) for r in runs}) == 2


# ---------------------------------------------------------------------------
# Completeness-aware caching: a partial run must NOT read as cached.
# ---------------------------------------------------------------------------


def _resolve(pipe: Pipeline, ds: Dataset, name: str) -> dict:
    return next(r for r in pipe._resolve_step_cache(ds) if r["step"].name == name)


def _write_dummy_parquet(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"frame": [0], "X": [0.0], "Y": [0.0]}).to_parquet(path)


def test_partial_run_reads_not_cached_and_resumes(tmp_path: Path) -> None:
    """A run dir with only some target sequences is 'partial', not 'cached'."""
    ds = _dataset_with_tracks(tmp_path)  # target = {(g, s1), (g, s2)}
    pipe = Pipeline()
    pipe.add(FeatureStep("speed", SpeedAngvel, {"step_size": 1}))

    info = _resolve(pipe, ds, "speed")
    run_root = feature_run_root(ds, info["storage_name"], info["expected_run_id"])
    _write_dummy_parquet(run_root / "g__s1.parquet")  # 1 of 2 -> partial

    row = pipe.status(ds)
    row = row[row["step"] == "speed"].iloc[0]
    assert row["cached"] == "partial"
    assert row["n_seq"] == "1/2"

    # The resolve reports it not-cached (bool), so run() executes/resumes it.
    assert _resolve(pipe, ds, "speed")["cached"] is False
    pipe.run(ds)
    assert len(list(run_root.glob("*.parquet"))) == 2


def test_scope_relative_finished_is_not_trusted(tmp_path: Path) -> None:
    """finished_at set over a truncated scope must NOT make a step read cached.

    Reproduces the ffgm_main failure: a downstream step whose partial upstream
    truncated its manifest honestly marks itself finished for that smaller scope.
    """
    ds = _dataset_with_tracks(tmp_path)
    pipe = Pipeline()
    pipe.add(FeatureStep("speed", SpeedAngvel, {"step_size": 1}))

    info = _resolve(pipe, ds, "speed")
    storage, rid = info["storage_name"], info["expected_run_id"]
    run_root = feature_run_root(ds, storage, rid)
    entry = run_root / "g__s1.parquet"
    _write_dummy_parquet(entry)

    reg = open_registry(ds.get_root("features"))
    reg.record_run_start(storage, rid, "0.1", "hash")
    reg.record_entry(storage, rid, "g", "s1", entry, 1)
    reg.mark_finished(storage, rid)
    assert reg.run_is_finished(storage, rid) is True  # DB says done...
    reg.close()

    # ...but only 1 of 2 target sequences exists, so it is NOT cached.
    assert _resolve(pipe, ds, "speed")["cached"] is False
    pipe.run(ds)
    assert len(list(run_root.glob("*.parquet"))) == 2


def test_complete_run_stays_cached_including_legacy_no_db(tmp_path: Path) -> None:
    """A fully complete run reads cached; removing .mosaic.db doesn't change that."""
    ds = _dataset_with_tracks(tmp_path)
    pipe = Pipeline()
    pipe.add(FeatureStep("speed", SpeedAngvel, {"step_size": 1}))
    pipe.run(ds)

    assert _resolve(pipe, ds, "speed")["cached"] is True

    # Legacy fallback: the check is disk-based, so a full on-disk run with no DB
    # still caches (old runs must not be force-recomputed).
    (ds.get_root("features") / ".mosaic.db").unlink(missing_ok=True)
    assert _resolve(pipe, ds, "speed")["cached"] is True


class _ScopedSpeed(SpeedAngvel):
    """A scope_dependent variant to exercise the run_id parity fix."""

    name = "scoped-speed"
    scope_dependent = True


def test_scope_dependent_run_id_matches_on_disk(tmp_path: Path) -> None:
    """For scope_dependent features, status()'s predicted run_id must include the
    scope term so it matches the on-disk run_id run_feature wrote."""
    ds = _dataset_with_tracks(tmp_path)
    pipe = Pipeline()
    pipe.add(FeatureStep("scoped", _ScopedSpeed, {"step_size": 1}))
    pipe.run(ds)

    info = _resolve(pipe, ds, "scoped")
    assert info["cached"] is True  # predicted run_id (with _scope_entries) matched
    run_root = feature_run_root(ds, info["storage_name"], info["expected_run_id"])
    assert run_root.exists() and any(run_root.glob("*.parquet"))
