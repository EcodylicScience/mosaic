# Pre-Implementation Assessment: Task 11 - Remove StreamingFeatureHelper and Old Iteration Functions

**Plan:** `docs/plans/pipeline/task-11-remove-streaming-helper.md`
**Date:** 2026-03-17
**Baseline:** 330 tests passing, 0 regressions

---

## 1. Plan Thoroughness Assessment

**Verdict: The plan has significant gaps. Several steps are already done (Task 10), several are unsafe, and the actual minimal scope is much smaller than described.**

### Steps already completed by Task 10

| Plan step | Status |
|-----------|--------|
| Step 7: Remove `PartialIndexRow`, `_build_index_row`, `get_additional_index_rows`, `set_run_root`, `skip_transform_phase`, `storage_feature_name`, `storage_use_input_suffix` from feature files | **Already done.** Task 10 removed all old protocol remnants from feature files. None of these symbols exist in any feature file. |
| Step 8: Update feature templates | **Already done.** Task 10 rewrote both `feature_template__global.py` and `feature_template__per_sequence.py` to the new 4-method protocol. |
| Step 9: Remove `bind_dataset`, `set_scope` from all feature files | **Already done in feature_library.** No feature_library file has these methods. However, the plan is wrong about removing them from visualization_library (see below). |

### Steps that are unsafe or incorrect

| Plan step | Problem |
|-----------|---------|
| Step 3: Remove `yield_input_data()` and `yield_feature_data()` from iteration.py | **Test impact not addressed.** `test_iteration.py` has 14 tests for `yield_feature_data` (TestYieldFeatureData: 10 tests) and `yield_input_data` (TestYieldInputData: 4 tests). These must be removed too. The plan says "Remove or update tests" in the Files section but gives no specifics. |
| Step 5: Remove re-exports from helpers.py that are no longer needed | **Vague.** Does not specify which re-exports. Needs analysis (see section 3 below). |
| Step 6: Update `behavior/feature_library/__init__.py` -- remove helpers from __all__ if it's now empty | **Wrong assumption.** helpers.py will NOT be empty after cleanup. It contains actively-used utility functions (ensure_columns, feature_columns, clean_animal_track, smooth_1d, unwrap_diff, wrap_angle, ego_rotate, nn_lookup_for, load_result_for). 16 feature files import from helpers. helpers MUST stay in __init__.py. |
| Step 9: Remove bind_dataset, set_scope from all feature files | **Unsafe for visualization_library.** Three visualization features actively use these methods: `egocentric_crop.py`, `viz_timeline.py`, `viz_global_colored.py`. They are NOT on the new protocol and have a separate execution path. Removing bind_dataset/set_scope from them would break visualization. |

### What the plan misses

1. **Test cleanup details.** Removing `yield_feature_data`, `yield_input_data`, `resolve_input_scope`, `resolve_tracks_entries`, `resolve_feature_entries`, and `_resolve_input_specs` from iteration.py means removing these test classes from test_iteration.py:
   - `TestYieldFeatureData` (10 tests)
   - `TestYieldInputData` (4 tests)
   - `TestResolveTracksEntries` (4 tests)
   - `TestResolveFeatureEntries` (8 tests)

   Total: ~26 tests to remove. Test count will drop from 330 to ~304.

2. **`resolve_tracks_entries` and `resolve_feature_entries` should also be removed.** They are only called by `_resolve_input_specs` and `resolve_input_scope`, both of which are dead. The plan mentions removing `yield_input_data` and `yield_feature_data` but not these helper functions.

3. **`ResolvedInput` in `_utils.py`.** Once the dead iteration functions are removed, `ResolvedInput` has zero remaining consumers (iteration.py was its only user). It should also be removed. The plan does not mention this.

4. **helpers.py `__all__` cleanup.** The plan does not specify what `__all__` should contain after cleanup. Currently it exports `StreamingFeatureHelper`, `PartialIndexRow`, `_build_index_row`, and 10 re-exports from loading.py. After removing the dead items, the remaining re-exports need audit too.

---

## 2. Code Cleanliness and DRY Assessment

### helpers.py re-export audit

helpers.py re-exports 11 symbols from `mosaic.core.pipeline.loading`:

| Re-export | Used by feature files? | Verdict |
|-----------|----------------------|---------|
| `_get_feature_run_root` | No | Remove from __all__, keep import only if used internally |
| `_load_array_from_spec` | No | Remove |
| `_load_artifact_matrix` | No | Remove |
| `_load_joblib_artifact` | No | Remove |
| `_load_parquet_dataframe` | No | Remove |
| `_normalize_identity_columns` | No | Remove |
| `_pose_column_pairs` | **Yes** (body_scale.py) | **Keep** |
| `build_nn_lookup` | No | Remove from __all__ |
| `nn_pair_mask` | No | Remove from __all__ |
| `resolve_sequence_identity` | No | Remove from __all__ |
| `load_entry_data` (as `_load_entry_data`) | No (only StreamingFeatureHelper used it) | Remove |

**Action:** Remove all re-export imports from helpers.py except `_pose_column_pairs`. Remove the entire first `from mosaic.core.pipeline.loading import (...)` block. Keep `_pose_column_pairs` as a targeted import. Remove the `load_entry_data` aliased import.

The `_EXTRA_META` set and `feature_columns`, `ensure_columns`, `nn_lookup_for`, `load_result_for`, `clean_animal_track`, `smooth_1d`, `unwrap_diff`, `wrap_angle`, `ego_rotate` are all defined locally in helpers.py and actively used. These stay.

### Updated __all__ for helpers.py

After cleanup, `__all__` should be:

```python
__all__ = [
    "_pose_column_pairs",
    "clean_animal_track",
    "ego_rotate",
    "ensure_columns",
    "feature_columns",
    "load_result_for",
    "nn_lookup_for",
    "smooth_1d",
    "unwrap_diff",
    "wrap_angle",
]
```

### iteration.py after cleanup

After removing dead functions, iteration.py retains:
- `_read_tracks_index()` (helper, used by yield_sequences)
- `_filter_index()` (helper, used by yield_sequences)
- `yield_sequences()` (ACTIVE: visualization_library/data_loading.py)
- `yield_sequences_with_overlap()` (supports yield_sequences)

These are coherent and well-tested (TestReadTracksIndex: 3 tests, TestYieldSequences: 8 tests, TestYieldSequencesWithOverlap: 6 tests).

The `ResolvedInput` import in iteration.py should be removed along with the dead functions. The `Scope` import stays (used by resolve_input_scope which is being deleted, but check if yield_sequences needs it -- it does not).

### No overlap with other planned tasks

Task 12 (update imports and final verification) depends on Task 11 completing first. No overlap concerns.

---

## 3. Maintainability Concerns

### Visualization library features remain on old protocol

The 3 visualization features (`egocentric_crop`, `viz_timeline`, `viz_global_colored`) plus `model_library/behavior_xgboost.py` still use `bind_dataset()` and `set_scope()`. These are NOT dispatched through `run_feature()` -- they have their own execution paths in the visualization library. Removing these methods would break visualization.

**Recommendation:** Do NOT touch bind_dataset/set_scope in visualization_library or model_library. These are a separate concern for a future visualization protocol migration task.

### ResolvedInput removal is safe

`ResolvedInput` (in `_utils.py`) is used only by:
1. `iteration.py` -- in `resolve_feature_entries`, `_resolve_input_specs`, `resolve_input_scope`, `yield_input_data` (all being deleted)
2. `tests/test_iteration.py` -- imported at line 10, used in `TestYieldInputData._make_inputs` (being deleted)

No other consumers exist. Safe to remove.

### Import chain after cleanup

After removing the dead loading.py re-exports from helpers.py, `body_scale.py` still imports `_pose_column_pairs` from helpers. This is the only re-export that survives. All other feature files import only locally-defined helpers functions. The import chain is clean.

---

## 4. Recommended Implementation Plan

### Step 1: Delete dead code from helpers.py

- Delete `StreamingFeatureHelper` class (lines 60-238)
- Delete `PartialIndexRow` dataclass (lines 241-252)
- Delete `_build_index_row()` function (lines 255-267)
- Remove all loading.py re-export imports EXCEPT `_pose_column_pairs`
- Remove `load_entry_data` aliased import
- Update `__all__` to only include actively-used symbols

### Step 2: Delete dead code from iteration.py

- Delete `yield_feature_data()` (lines 209-244)
- Delete `resolve_tracks_entries()` (lines 250-257)
- Delete `resolve_feature_entries()` (lines 260-296)
- Delete `_resolve_input_specs()` (lines 302-356)
- Delete `resolve_input_scope()` (lines 359-386)
- Delete `yield_input_data()` (lines 389-482)
- Remove `ResolvedInput` from imports (no longer needed)
- Remove unused imports (`sys`, `np`, `feature_index`, `feature_index_path`, `latest_feature_run_root`) that become dead after deletion

### Step 3: Delete `ResolvedInput` from `_utils.py`

- Remove `ResolvedInput` dataclass
- Remove from `_utils.py` exports if listed

### Step 4: Update test_iteration.py

- Remove `TestYieldFeatureData` class (lines 307-433)
- Remove `TestResolveTracksEntries` class (lines 439-455)
- Remove `TestResolveFeatureEntries` class (lines 461-561)
- Remove `TestYieldInputData` class (lines 567-633)
- Remove `ResolvedInput` import
- Remove `resolve_feature_entries`, `resolve_tracks_entries`, `yield_feature_data`, `yield_input_data` from imports
- Remove `_setup_feature` helper if no remaining tests use it
- Keep: `TestReadTracksIndex`, `TestYieldSequences`, `TestYieldSequencesWithOverlap`
- Keep: `_MockDataset`, `_make_parquet`, `_write_tracks_index` (used by remaining tests)

### Step 5: Verify no other consumers

- Grep for all removed symbols across the entire codebase
- Confirm no feature files, tests, or other modules import them

### Step 6: Do NOT touch

- `bind_dataset`/`set_scope` in visualization_library or model_library
- Feature templates (already rewritten in Task 10)
- `feature_library/__init__.py` helpers export (helpers is still actively used)

### Step 7: Verification

```bash
uv run ruff format src/mosaic/behavior/feature_library/helpers.py src/mosaic/core/pipeline/iteration.py src/mosaic/core/pipeline/_utils.py tests/test_iteration.py
uv run ruff check --fix --select I src/mosaic/behavior/feature_library/helpers.py src/mosaic/core/pipeline/iteration.py src/mosaic/core/pipeline/_utils.py tests/test_iteration.py
uv run basedpyright src/mosaic/core/pipeline/ src/mosaic/behavior/feature_library/
uv run pytest tests/ -x -q
```

Expected: ~304 tests passing (down from 330 due to removed dead-code tests).

### Step 8: Commit

```
refactor: remove StreamingFeatureHelper, dead iteration functions, and unused re-exports

Delete StreamingFeatureHelper, PartialIndexRow, _build_index_row from
helpers.py. Remove yield_feature_data, yield_input_data, resolve_input_scope,
and related dead functions from iteration.py. Remove ResolvedInput from
_utils.py. Clean up helpers.py re-exports. Remove 26 tests for deleted code.
```

---

## 5. Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Removing iteration functions breaks visualization_library | Low | `yield_sequences` is kept. Only dead functions removed. Grep verification in step 5. |
| Removing ResolvedInput breaks something | Low | Only used in iteration.py dead code and test_iteration.py dead tests. |
| helpers.py __all__ incomplete | Low | Explicit audit above. All actively-imported symbols listed. |
| _pose_column_pairs re-export missed | Medium | body_scale.py imports it from helpers. Must keep this one re-export. |
| Test count drop raises alarms | Low | Expected and documented. Dead tests for dead code. |

---

## 6. Summary

The plan as written has 3 steps already done (Task 10), 1 unsafe step (removing bind_dataset/set_scope from visualization_library), and significant gaps (ResolvedInput removal, test cleanup specifics, helpers __all__ audit, iteration.py helper function removal). The actual scope is:

1. **Delete from helpers.py**: StreamingFeatureHelper, PartialIndexRow, _build_index_row, unused loading.py re-exports (~200 lines)
2. **Delete from iteration.py**: 6 dead functions (~230 lines)
3. **Delete from _utils.py**: ResolvedInput dataclass (~10 lines)
4. **Delete from test_iteration.py**: 4 test classes, ~26 tests (~200 lines)
5. **Update __all__ and imports**: helpers.py, iteration.py

Net change: ~640 lines removed. No new code needed.
