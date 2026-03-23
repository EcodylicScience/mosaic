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

    def load_state(self, run_root, artifact_paths, dependency_indices):
        return True

    def fit(self, inputs):
        pass

    def save_state(self, run_root):
        pass

    def apply(self, df):
        return pd.DataFrame({
            "frame": df["frame"],
            "value": np.ones(len(df)),
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

    # 1. Resolve dependencies from params
    artifact_paths, dependency_indices = _resolve_dependencies(ds, feature.params)

    # 2. Call load_state
    state_ready = feature.load_state(run_root, artifact_paths, dependency_indices)

    # 3. Fit if needed
    #    fit() takes a Callable[[], Iterator[...]] (iterator factory),
    #    not a bare iterator. Wrap in a lambda so the feature can
    #    iterate multiple times (e.g. exact-allocation two-pass).
    if not state_ready:
        manifest, _ = build_manifest(ds, feature.inputs, ...)
        feature.fit(lambda: iter_manifest(manifest, scope, progress_label=feature.name))
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

        df_out = feature.apply(entry_data.df)
        n_rows = _write_apply_output(out_path, df_out)
        out_rows.append(_make_index_row(
            run_id, storage_feature_name, feature.version,
            group, sequence, out_path, n_rows,
        ))

    return out_rows
```

Also delete `process_transform_worker()` (line 73-93), the `__global__` marker parquet logic (lines 595-616), and all old-protocol dispatch code (`needs_fit()`, `partial_fit()`, `finalize_fit()`, `transform()` calls, `skip_transform_phase` handling).

---

## Key implementation detail -- `_resolve_dependencies`

Resolve all upstream dependencies declared in a feature's Params. Returns
two separate dicts:

- `artifact_paths: dict[str, Path]` -- file paths for `ArtifactSpec` and
  `LabelsSource` fields.
- `dependency_indices: dict[str, pd.DataFrame]` -- IndexCSV DataFrames for
  plain `Result` fields (e.g. `NNResult`, `BodyScaleResult`). Features use
  `load_result_for(index, group, sequence)` from `helpers.py` to look up
  per-sequence parquets from these indices at apply time.

Three cases:

1. **`ArtifactSpec` fields** (e.g. model files, scaler files) -- resolved to a
   specific file via `run_root.glob(pattern)`. Goes into `artifact_paths`.
2. **Plain `Result` fields** (e.g. `NNResult`, `BodyScaleResult`) -- resolved to
   the upstream feature's IndexCSV DataFrame. Goes into `dependency_indices`.
   Features store the index in `load_state` and use `load_result_for()` in
   `apply()` to read per-sequence upstream parquets on demand.
3. **`LabelsSource` fields** (e.g. `GroundTruthLabelsSource`, id-tag labels) --
   resolved to `<dataset_root>/labels/<kind>/`. Goes into `artifact_paths`.

`LabelsSource` is a new base class that `GroundTruthLabelsSource` subclasses.
Any Params field that is a `LabelsSource` gets its `kind` resolved to a
labels directory path.

```python
def _resolve_dependencies(
    ds, params: Params
) -> tuple[dict[str, Path], dict[str, pd.DataFrame]]:
    """Introspect params for dependency fields and resolve paths/indices."""
    from .types import ArtifactSpec, LabelsSource, Result
    from .index import feature_run_root, latest_feature_run_root
    from .index_csv import IndexCSV

    artifact_paths: dict[str, Path] = {}
    dependency_indices: dict[str, pd.DataFrame] = {}

    for field_name in type(params).model_fields:
        value = getattr(params, field_name, None)

        match value:
            case ArtifactSpec(feature=feature_name, run_id=run_id, pattern=pattern) if feature_name:
                if run_id is None:
                    run_id, run_root = latest_feature_run_root(ds, feature_name)
                else:
                    run_root = feature_run_root(ds, feature_name, run_id)
                files = sorted(run_root.glob(pattern))
                if files:
                    artifact_paths[field_name] = files[0]

            case Result(feature=feature_name, run_id=run_id) if feature_name:
                if run_id is None:
                    run_id, run_root = latest_feature_run_root(ds, feature_name)
                else:
                    run_root = feature_run_root(ds, feature_name, run_id)
                index = IndexCSV(run_root).read()
                dependency_indices[field_name] = index

            case LabelsSource(kind=kind) if kind:
                labels_root = Path(ds.get_root("labels")) / kind
                if labels_root.exists():
                    artifact_paths[field_name] = labels_root

    return artifact_paths, dependency_indices
```

---

## Step 3: Run tests

Run: `cd mosaic && uv run pytest tests/ -x -q`
Expected: All pass.

## Step 4: Commit

```
feat: rewrite run_feature for new Feature protocol
```
