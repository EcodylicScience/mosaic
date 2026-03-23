# Task 9: Rewrite `run_feature()` for new protocol

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** All features now implement the new protocol. Replace the existing `run_feature()` implementation with the new pipeline flow.

**Phase:** C (Protocol Transition -- clean break, all Phase C tasks land together)

**Parent plan:** `docs/plans/2026-03-12-pipeline-unification-implementation.md`

**Depends on:** Tasks 5-8 (all features must be migrated first)

---

## Files

- Modify: `src/mosaic/core/pipeline/run.py`
- Create: `tests/test_run_feature.py`

---

## Terminology note

> Task 4 renamed `KeyData` -> `EntryData`, `load_key_data` -> `load_entry_data`,
> and `key` -> `entry_key` throughout the codebase. The code below reflects that.

## Step 1: Write failing test

```python
# tests/test_run_feature.py
"""Test run_feature with the new protocol."""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mosaic.core.pipeline.loading import EntryData
from mosaic.core.pipeline.types import Inputs, Params, TrackInput


class _StatelessFeature:
    """Minimal stateless feature for testing."""
    name = "stateless-test"
    version = "0.1"
    parallelizable = True
    scope_dependent = False

    class Inputs(Inputs[TrackInput]):
        pass

    class Params(Params):
        pass

    def __init__(self):
        self.inputs = self.Inputs(("tracks",))
        self._params = self.Params()

    @property
    def params(self):
        return self._params

    def load_state(self, run_root, artifact_paths):
        return True

    def fit(self, inputs):
        pass

    def save_state(self, run_root):
        pass

    def apply(self, entry_key, entry_data):
        return pd.DataFrame({
            "frame": entry_data.frames,
            "value": np.ones(len(entry_data.frames)),
        })


# Integration test requires a mock dataset setup
# (follow pattern from test_iteration.py)
```

## Step 2: Rewrite `run_feature()`

Delete the old `run_feature()` body and replace with the new pipeline flow:

```python
def run_feature(ds, feature, scope, run_root, run_id, ...):
    """Run a feature through the unified pipeline."""
    from .manifest import build_manifest, iter_manifest
    from .loading import EntryData

    # Storage name
    suffix = feature.inputs.storage_suffix()
    if suffix:
        storage_feature_name = f"{feature.name}__from__{suffix}"
    else:
        storage_feature_name = feature.name

    # 1. Resolve artifact paths from params
    artifact_paths = _resolve_artifact_paths(ds, feature.params)

    # 2. Call load_state
    state_ready = feature.load_state(run_root, artifact_paths)

    # 3. Fit if needed
    if not state_ready:
        manifest, _ = build_manifest(ds, feature.inputs, ...)
        fit_iter = iter_manifest(manifest, scope, progress_label=feature.name)
        feature.fit(fit_iter)
        feature.save_state(run_root)

    # 4. Apply phase
    manifest, _ = build_manifest(ds, feature.inputs, ...)
    out_rows = []
    for entry_key, entry_data in iter_manifest(manifest, scope, progress_label=feature.name):
        group, sequence = _resolve_entry(entry_key, scope)
        out_path = _build_output_path(run_root, group, sequence)

        # Skip existing if state was cached (not re-fitted)
        if state_ready and not overwrite and out_path.exists():
            continue

        df_out = feature.apply(entry_key, entry_data)
        n_rows = _write_apply_output(out_path, df_out)
        out_rows.append(_make_index_row(
            run_id, storage_feature_name, feature.version,
            group, sequence, out_path, n_rows,
        ))

    return out_rows
```

Also delete `process_transform_worker()` (line 73-93), the `__global__` marker parquet logic (lines 595-616), and all old-protocol dispatch code (`needs_fit()`, `partial_fit()`, `finalize_fit()`, `transform()` calls, `skip_transform_phase` handling).

---

## Key implementation detail -- `_resolve_artifact_paths`

```python
def _resolve_artifact_paths(ds, params: Params) -> dict[str, Path]:
    """Introspect params for ArtifactSpec fields and resolve their paths."""
    from .types import ArtifactSpec
    from .index import feature_run_root, latest_feature_run_root

    artifact_paths: dict[str, Path] = {}
    for field_name, field_info in type(params).model_fields.items():
        value = getattr(params, field_name, None)
        if not isinstance(value, ArtifactSpec):
            continue
        if not value.feature:
            continue

        feat_name = value.feature
        run_id = value.run_id
        if run_id is None:
            run_id, run_root = latest_feature_run_root(ds, feat_name)
        else:
            run_root = feature_run_root(ds, feat_name, run_id)

        files = sorted(run_root.glob(value.pattern))
        if files:
            artifact_paths[field_name] = files[0]

    return artifact_paths
```

---

## Step 3: Run tests

Run: `cd mosaic && uv run pytest tests/ -x -q`
Expected: All pass.

## Step 4: Commit

```
feat: rewrite run_feature for new Feature protocol
```
