# Task 12: Update imports and final verification

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Update all imports to use canonical `core/pipeline/` locations. Remove re-exports from `spec.py` and `helpers.py`. Final verification with tests, type checker, and linter.

**Phase:** D (Cleanup)

**Parent plan:** `docs/plans/2026-03-12-pipeline-unification-implementation.md`

**Depends on:** Task 11

---

## Files

- Modify: All feature files importing from `spec.py` -- update to import from `core.pipeline.types`
- Modify: `src/mosaic/behavior/feature_library/spec.py` -- remove re-exports
- Modify: `src/mosaic/behavior/feature_library/__init__.py` -- update exports
- Modify: All test files

---

## Steps

1. Replace all `from mosaic.behavior.feature_library.spec import X` with `from mosaic.core.pipeline.types import X` across feature files (for moved symbols only)
2. Replace all `from mosaic.behavior.feature_library.helpers import X` with `from mosaic.core.pipeline.loading import X` across feature files (for moved functions)
3. Remove re-exports from `spec.py` -- it now contains only Columns/configs/registry
4. Remove re-exports from `helpers.py` -- it now contains only PartialIndexRow remnants (or is empty)
5. Update `__init__.py` in `behavior/feature_library/` to export from new locations
6. Update test imports

## Verification

```bash
cd mosaic && uv run pytest tests/ -x -q
cd mosaic && uv run basedpyright src/mosaic/
cd mosaic && uv run ruff check src/mosaic/
```

Expected: All tests pass, no new type errors in target modules, no new lint errors.

7. Commit: `refactor: update all imports to canonical core/pipeline locations`
