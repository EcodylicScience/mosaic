# Drop `loads_own_data()` Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove `loads_own_data()` from the Feature protocol entirely, replacing all pipeline checks with `inputs.is_empty`, and passing a lazy generator to `fit()` to avoid memory regression.

**Architecture:** `feature.inputs` already tells the pipeline everything `loads_own_data()` communicated. Empty inputs means no pipeline-provided data. Non-empty inputs means the pipeline provides data. The eager list collection in `fit()` becomes a lazy generator so features that bypass the iterator (GlobalTSNE, WardAssign, etc.) pay zero memory cost. This also fixes the incorrect frame-filter rejection for features with non-empty inputs.

**Tech Stack:** Python, dataclasses, Pydantic (RootModel), pytest

---

### Task 1: Add `is_empty` property to Inputs

**Files:**
- Modify: `src/mosaic/behavior/feature_library/spec.py:353-399`
- Test: `tests/test_inputs_subclass.py`

**Step 1: Write the failing test**

In `tests/test_inputs_subclass.py`, add:

```python
def test_is_empty() -> None:
    """Inputs.is_empty reflects whether the tuple is empty."""
    empty = Inputs(())
    assert empty.is_empty
    non_empty = Inputs(("tracks",))
    assert not non_empty.is_empty
```

Note: `Inputs(())` will fail validation because `_require` defaults to `"nonempty"`.
Use a subclass with `_require = "any"` to allow empty:

```python
def test_is_empty() -> None:
    """Inputs.is_empty reflects whether the tuple is empty."""
    from mosaic.behavior.feature_library.spec import InputRequire
    from typing import ClassVar

    class _AnyInputs(Inputs[TrackInput | Result]):
        _require: ClassVar[InputRequire] = "any"

    empty = _AnyInputs(())
    assert empty.is_empty
    non_empty = _AnyInputs(("tracks",))
    assert not non_empty.is_empty
    also_non_empty = Inputs(("tracks",))
    assert not also_non_empty.is_empty
```

**Step 2: Run test to verify it fails**

Run: `cd /home/paul/ecodylic/mosaic && uv run pytest tests/test_inputs_subclass.py::test_is_empty -v`
Expected: FAIL with `AttributeError: ... has no attribute 'is_empty'`

**Step 3: Implement `is_empty`**

In `spec.py`, add to `InputsLike` protocol (after line 365, the `is_multi` property):

```python
    @property
    def is_empty(self) -> bool: ...
```

In `spec.py`, add to `Inputs` class (after `is_multi` property, around line 446):

```python
    @property
    def is_empty(self) -> bool:
        return len(self.root) == 0
```

Also update the Inputs docstring (line 387-389):

```python
    Features that take no pipeline inputs:
        class Inputs(Inputs[Result]):
            _require: ClassVar[InputRequire] = "empty"
```

And the comment (line 397):

```python
    # "empty": must be empty (no pipeline inputs)
```

And the validation error message (line 419):

```python
                raise ValueError("This feature takes no pipeline inputs")
```

**Step 4: Run test to verify it passes**

Run: `cd /home/paul/ecodylic/mosaic && uv run pytest tests/test_inputs_subclass.py -v`
Expected: all tests PASS

**Step 5: Commit**

```bash
cd /home/paul/ecodylic/mosaic && git add src/mosaic/behavior/feature_library/spec.py tests/test_inputs_subclass.py
git commit -m "add Inputs.is_empty property"
```

---

### Task 2: Replace pipeline checks in run.py

**Files:**
- Modify: `src/mosaic/core/pipeline/run.py:197-205, 354-361, 375-415`

**Step 1: Replace entry routing (lines 197-205)**

Change:

```python
    elif feature.loads_own_data():
        if groups is not None or sequences is not None:
            raise ValueError(
                f"Feature '{feature.name}' has empty inputs and loads its "
                f"own data; groups/sequences filters cannot be applied."
            )
        entries_all = set()
    else:
        raise ValueError("Feature.inputs is empty")
```

To:

```python
    elif inputs.is_empty:
        if groups is not None or sequences is not None:
            raise ValueError(
                f"Feature '{feature.name}' has empty inputs; "
                f"groups/sequences filters cannot be applied."
            )
        entries_all = set()
    else:
        raise ValueError("Feature.inputs is empty and unrecognized")
```

**Step 2: Replace frame filter rejection (lines 354-361)**

Change:

```python
    # Validate time/frame filters against ds-loading features
    if _has_frame_filter and feature.loads_own_data():
        raise RuntimeError(
            f"[feature:{feature.name}] Time/frame filters are set but "
            f"this feature loads its own data and cannot apply them. "
            f"Remove the filters or use a feature that processes data "
            f"from the sequence iterator."
        )
```

To:

```python
    if _has_frame_filter and inputs.is_empty:
        raise RuntimeError(
            f"[feature:{feature.name}] Time/frame filters are set but "
            f"this feature has no inputs and cannot apply them."
        )
```

**Step 3: Replace fit skip and data loading (lines 375-423)**

Change:

```python
        # Check if fit phase can be skipped (for global features with existing outputs)
        loads_own = feature.loads_own_data()
        model_path = run_root / "model.joblib"
        # Also check for global-specific artifacts (e.g., global_opentsne_embedding.joblib)
        embedding_path = run_root / "global_opentsne_embedding.joblib"
        fit_complete = model_path.exists() or embedding_path.exists()

        skip_fit = not overwrite and loads_own and fit_complete
        if skip_fit:
            print(
                f"[feature:{feature.name}] fit phase skipped (overwrite=False, outputs exist)",
                file=sys.stderr,
            )
        elif feature.supports_partial_fit():
            for item in iter_inputs():
                df = item[2]
                try:
                    feature.partial_fit(df)
                except Exception as e:
                    print(
                        f"[feature:{feature.name}] partial_fit failed: {e}",
                        file=sys.stderr,
                    )
            try:
                feature.finalize_fit()
            except Exception:
                pass
        else:
            # Check if feature loads its own data (e.g., GlobalTSNE) - avoid pre-loading
            if loads_own:
                # Feature will load data itself; pass empty iterator to satisfy protocol
                all_dfs = []
            else:
                all_dfs = []
                for item in iter_inputs():
                    df = item[2]
                    all_dfs.append(df)
            # Always call fit, even if no streamed inputs were found.
            # Many "global/artifact" features load their own matrices from disk.
            try:
                feature.fit(all_dfs)
            except TypeError:
                # Backward-compat: some features define fit() with no args.
                try:
                    getattr(feature, "fit")()
                except Exception as e:
                    print(
                        f"[feature:{feature.name}] fit() failed: {e}", file=sys.stderr
                    )
```

To:

```python
        # Skip fit for empty-input features when outputs already exist
        model_path = run_root / "model.joblib"
        embedding_path = run_root / "global_opentsne_embedding.joblib"
        fit_complete = model_path.exists() or embedding_path.exists()

        skip_fit = not overwrite and inputs.is_empty and fit_complete
        if skip_fit:
            print(
                f"[feature:{feature.name}] fit phase skipped (overwrite=False, outputs exist)",
                file=sys.stderr,
            )
        elif feature.supports_partial_fit():
            for item in iter_inputs():
                df = item[2]
                try:
                    feature.partial_fit(df)
                except Exception as e:
                    print(
                        f"[feature:{feature.name}] partial_fit failed: {e}",
                        file=sys.stderr,
                    )
            try:
                feature.finalize_fit()
            except Exception:
                pass
        else:
            def _fit_iter() -> Iterator[pd.DataFrame]:
                for item in iter_inputs():
                    yield item[2]

            try:
                feature.fit([] if inputs.is_empty else _fit_iter())
            except TypeError:
                try:
                    getattr(feature, "fit")()
                except Exception as e:
                    print(
                        f"[feature:{feature.name}] fit() failed: {e}", file=sys.stderr
                    )
```

Note: `Iterator` is already imported from `collections.abc` in run.py. Verify this; if not, add the import.

**Step 4: Run tests**

Run: `cd /home/paul/ecodylic/mosaic && uv run pytest tests/ -x -q`
Expected: all tests PASS

**Step 5: Commit**

```bash
cd /home/paul/ecodylic/mosaic && git add src/mosaic/core/pipeline/run.py
git commit -m "replace loads_own_data() checks with inputs.is_empty, lazy generator for fit()"
```

---

### Task 3: Remove `loads_own_data()` from Feature protocol

**Files:**
- Modify: `src/mosaic/behavior/feature_library/spec.py:482`

**Step 1: Delete the protocol method**

Remove this line from the `Feature` protocol class:

```python
    def loads_own_data(self) -> bool: ...
```

**Step 2: Run type checker**

Run: `cd /home/paul/ecodylic/mosaic && uv run basedpyright src/mosaic/behavior/feature_library/spec.py`
Expected: no new errors

**Step 3: Commit**

```bash
cd /home/paul/ecodylic/mosaic && git add src/mosaic/behavior/feature_library/spec.py
git commit -m "remove loads_own_data() from Feature protocol"
```

---

### Task 4: Remove `loads_own_data()` from all features returning False

These 17 features all have a trivial `def loads_own_data(self) -> bool: return False` method. Delete the method from each file.

**Files (feature_library/):**
- `speed_angvel.py:83-84`
- `orientation_relative.py:124-125`
- `pair_position.py:102-103`
- `pairposedistancepca.py:90-91`
- `pair_egocentric.py:95-96`
- `pair_wavelet.py:150-151`
- `body_scale.py:70-71`
- `nearestneighbor.py:85-86`
- `nn_delta_response.py:111-112`
- `nn_delta_bins.py:156-157`
- `id_tag_columns.py:76-77`
- `ffgroups.py:149-150`
- `ffgroups_metrics.py:83-84`
- `approach_avoidance.py:129-130`
- `model_predict.py:111-112`
- `temporal_stacking.py:136-137`

**Files (visualization_library/):**
- `egocentric_crop.py:143-144`

**Step 1: Delete the method from all 17 files**

In each file, delete the `loads_own_data` method and its surrounding blank line. Example from `speed_angvel.py`:

Delete:
```python
    def loads_own_data(self) -> bool:
        return False
```

**Step 2: Run tests**

Run: `cd /home/paul/ecodylic/mosaic && uv run pytest tests/ -x -q`
Expected: all tests PASS

**Step 3: Commit**

```bash
cd /home/paul/ecodylic/mosaic && git add -u src/mosaic/behavior/
git commit -m "remove loads_own_data() from features returning False"
```

---

### Task 5: Remove `loads_own_data()` from features returning True

These 9 features have `loads_own_data() -> True`, some with multi-line NOTE comments about frame filter limitations. Delete the method and NOTE comments from each.

**Files (feature_library/):**
- `global_tsne.py:274-278` - method + 3-line NOTE
- `global_kmeans.py:179-183` - method + 3-line NOTE
- `global_ward.py:113-117` - method + 3-line NOTE
- `ward_assign.py:145-149` - method + 3-line NOTE
- `kpms_fit.py:438-439` - method only
- `kpms_apply.py:121-122` - method only
- `feature_template__global.py:122-126` - method + 3-line NOTE

**Files (visualization_library/):**
- `viz_timeline.py:186-187` - method only
- `viz_global_colored.py:125-126` - method + comment on same line

**Step 1: Delete the method from all 9 files**

For files with NOTE comments (global_tsne, global_kmeans, global_ward, ward_assign, feature_template__global), delete the method AND the NOTE block. Example from `global_tsne.py`:

Delete:
```python
    def loads_own_data(self) -> bool:
        # NOTE: time/frame scope filters (filter_start_frame, etc.) are not
        # applied when loading own data. run_feature() raises RuntimeError
        # if these filters are set. Future work: apply them during loading.
        return True
```

For `viz_global_colored.py`, delete:
```python
    def loads_own_data(self):
        return True  # Skip run_feature pre-loading; we load from artifacts
```

For `kpms_fit.py`, `kpms_apply.py`, `viz_timeline.py`, delete the bare method:
```python
    def loads_own_data(self) -> bool:
        return True
```

**Step 2: Run tests**

Run: `cd /home/paul/ecodylic/mosaic && uv run pytest tests/ -x -q`
Expected: all tests PASS

**Step 3: Commit**

```bash
cd /home/paul/ecodylic/mosaic && git add -u src/mosaic/behavior/
git commit -m "remove loads_own_data() from features returning True"
```

---

### Task 6: Verify and clean up

**Step 1: Verify no remaining references**

Run: `cd /home/paul/ecodylic/mosaic && grep -r "loads_own_data" src/ tests/`
Expected: zero matches

Run: `cd /home/paul/ecodylic/mosaic && grep -r "loads_own" src/ tests/`
Expected: zero matches

**Step 2: Run full test suite**

Run: `cd /home/paul/ecodylic/mosaic && uv run pytest tests/ -v`
Expected: all ~200 tests PASS

**Step 3: Run type checker**

Run: `cd /home/paul/ecodylic/mosaic && uv run basedpyright src/mosaic/core/pipeline/run.py src/mosaic/behavior/feature_library/spec.py`
Expected: no new errors from our changes

**Step 4: Update the issue doc**

In `docs/issues/2026-03-11-loads-own-data-misuse.md`, add a resolution section at the bottom:

```markdown
## Resolution

Resolved by dropping `loads_own_data()` from the Feature protocol entirely.
The pipeline now uses `inputs.is_empty` to determine whether a feature has
pipeline-provided data. `fit()` receives a lazy generator instead of a
materialized list, so features that bypass the pipeline iterator
(GlobalTSNE, WardAssign, etc.) pay zero memory cost. Frame/time filters
are no longer blocked for features with non-empty inputs.
```

**Step 5: Commit**

```bash
cd /home/paul/ecodylic/mosaic && git add docs/issues/2026-03-11-loads-own-data-misuse.md
git commit -m "update loads_own_data issue doc with resolution"
```
