# Task 8: Migrate special-case features

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Migrate TemporalStacking, KPMS, and ModelPredict features to the new protocol. These have unusual data loading or naming patterns that require special handling.

**Phase:** C (Protocol Transition -- clean break, all Phase C tasks land together)

**Parent plan:** `docs/plans/2026-03-12-pipeline-unification-implementation.md`

**Depends on:** Tasks 5-7

---

## IMPORTANT: Sequential Migration

Same sequential migration discipline applies here -- pseudocode extraction first, then clean implementation.

---

## TemporalStackingFeature

This feature reimplements the pipeline's data loading from scratch (lines 250-460 of `temporal_stacking.py`). Under the new protocol, all custom loading code is deleted. It becomes a standard stateless feature:

```python
scope_dependent = False
parallelizable = True

def load_state(self, run_root, artifact_paths):
    return True

def fit(self, inputs):
    pass

def save_state(self, run_root):
    pass

def apply(self, key, kd):
    # kd.features is the merged input data from the manifest
    # Apply temporal stacking directly
    return self._stack(kd)
```

Delete: `_ensure_inputs_ready()`, `_build_sequence_mapping()`, `_load_sequence_matrix()`, `wants_full_input_data()`. Fix the latent bug at line 454 by deleting the code entirely.

---

## KpmsFit and KpmsApply

These are the most complex. They use external keypoint-MoSeq libraries and have custom data serialization. Migrate following the global pattern -- `apply()` replaces the streaming output in `fit()`.

KpmsApply's `skip_transform_phase = True` is removed. Instead, `apply()` does per-sequence prediction using the fitted KPMS model.

---

## ModelPredictFeature

Uses `params.output_feature_name` to override `name`. This stays as-is (feature sets `self.name` at construction time). Migration follows the per-sequence-with-fit pattern.

---

## Step 1: Migrate TemporalStacking

Delete ~250 lines of custom loading code. Convert to stateless pattern.

## Step 2: Migrate KpmsFit, KpmsApply

## Step 3: Migrate ModelPredictFeature

## Step 4: Run tests

Run: `cd mosaic && uv run pytest tests/ -x -q`
Expected: All pass

## Step 5: Commit

```
feat: migrate TemporalStacking, KPMS, ModelPredict to new protocol
```
