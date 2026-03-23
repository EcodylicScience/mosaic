# Task 10: Remove old protocol remnants from `types.py` and `run.py`

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove any remaining dead code from the old protocol that wasn't already cleaned up in Tasks 5-9.

**Phase:** D (Cleanup)

**Parent plan:** `docs/plans/2026-03-12-pipeline-unification-implementation.md`

**Depends on:** Task 9

---

## Files

- Modify: `src/mosaic/core/pipeline/types.py`
- Modify: `src/mosaic/core/pipeline/run.py`
- Modify: `src/mosaic/core/pipeline/__init__.py`

---

## Steps

1. Remove any remaining old-protocol types from `types.py` (e.g. old `OutputType` if unused, old imports)
2. Remove any leftover helper functions from `run.py` that served the old protocol
3. Remove `__global__` marker parquet logic if still present -- replace with a `__global__` index row pointing at `run_root` for global features
4. Update `__init__.py` exports
5. Run tests: `cd mosaic && uv run pytest tests/ -x -q`
6. Commit: `refactor: remove old protocol remnants`
