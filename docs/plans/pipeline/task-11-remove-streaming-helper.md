# Task 11: Remove StreamingFeatureHelper and old iteration functions

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Delete the old data loading and iteration code that has been replaced by the manifest builder and new pipeline flow.

**Phase:** D (Cleanup)

**Parent plan:** `docs/plans/2026-03-12-pipeline-unification-implementation.md`

**Depends on:** Task 10

---

## Files

- Modify: `src/mosaic/behavior/feature_library/helpers.py`
- Modify: `src/mosaic/core/pipeline/iteration.py`
- Modify: All feature files that import from helpers
- Remove or update tests that test removed functions

---

## Steps

1. Delete `StreamingFeatureHelper` class from `helpers.py`
2. Delete `PartialIndexRow` and `_build_index_row` from `helpers.py` (or `loading.py` if moved there)
3. Remove `yield_input_data()` and `yield_feature_data()` from `iteration.py`
4. Keep `yield_sequences()` and `yield_sequences_with_overlap()` in `iteration.py` -- these may still be used by the manifest builder for track resolution
5. Remove re-exports from `helpers.py` that are no longer needed
6. Update `behavior/feature_library/__init__.py` -- remove `helpers` from `__all__` if it's now empty
7. Remove all references to `PartialIndexRow`, `_build_index_row`, `get_additional_index_rows`, `set_run_root`, `skip_transform_phase`, `storage_feature_name`, `storage_use_input_suffix` from feature files
8. Update feature templates (`feature_template__global.py`, `feature_template__per_sequence.py`)
9. Remove `bind_dataset`, `set_scope` from all feature files (pipeline no longer calls these)
10. Run tests: `cd mosaic && uv run pytest tests/ -x -q`
11. Run type checker: `cd mosaic && uv run basedpyright src/mosaic/core/pipeline/ src/mosaic/behavior/feature_library/`
12. Commit: `refactor: remove StreamingFeatureHelper, old iteration, and dead protocol methods`
