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
from mosaic.core.pipeline.registry import read_runs


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
