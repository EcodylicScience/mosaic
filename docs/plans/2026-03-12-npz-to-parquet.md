# Migrate per-sequence npz to standard parquet -- Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace legacy per-sequence `.npz` outputs with standard per-frame parquet, drop `ArtifactSpec` subclasses for those outputs, drop `seq=` naming, update all consumers.

**Architecture:** Bottom-up: add `ResultColumn` to spec, then update producers (drop npz writes in WardAssign/GlobalKMeans, convert GlobalTSNE to parquet), then update consumers (VizGlobalColored, analysis.py). Each producer change is independent; consumer changes depend on all producers being done.

**Tech Stack:** Python, pydantic, pandas, numpy, parquet

**Design doc:** `docs/plans/2026-03-12-npz-to-parquet-design.md`

---

### Task 1: Add `ResultColumn` to spec.py, replace `FeatureLabelsSource`

**Files:**
- Modify: `src/mosaic/behavior/feature_library/spec.py`
- Test: `tests/test_feature_params.py`

`ResultColumn` is a `Result` subclass with a required `column: str` field. It replaces `FeatureLabelsSource` (which was an `ArtifactSpec[NpzLoadSpec]`).

**Step 1: Add `ResultColumn` class**

In `spec.py`, after the `NNResult` class (line 285), add:

```python
class ResultColumn(Result[str]):
    """Reference to a column in a feature's standard parquet output.

    Used for labels, continuous color values, or any per-frame column
    from another feature's output. Follows the same pattern as NNResult.

    Attributes:
        feature: Source feature name.
        column: Column name to extract from the parquet output.
        run_id: Specific run ID, or None for latest.
    """

    column: str
```

**Step 2: Replace `FeatureLabelsSource` with `ResultColumn` in `LabelsSourceSpec`**

Change `LabelsSourceSpec` (line 347) from:

```python
LabelsSourceSpec = FeatureLabelsSource | GroundTruthLabelsSource
```

to:

```python
LabelsSourceSpec = ResultColumn | GroundTruthLabelsSource
```

Keep `FeatureLabelsSource` in the file for now (it may have other importers); we remove it in a later task once all consumers are updated.

**Step 3: Add test for `ResultColumn`**

In `tests/test_feature_params.py`, add:

```python
def test_result_column_requires_column():
    from mosaic.behavior.feature_library.spec import ResultColumn

    rc = ResultColumn(feature="ward-assign", column="cluster")
    assert rc.feature == "ward-assign"
    assert rc.column == "cluster"
    assert rc.run_id is None


def test_result_column_with_run_id():
    from mosaic.behavior.feature_library.spec import ResultColumn

    rc = ResultColumn(feature="global-kmeans", column="cluster", run_id="0.1-abc")
    assert rc.run_id == "0.1-abc"
```

**Step 4: Run tests, commit**

```bash
uv run pytest tests/test_feature_params.py -v
git add src/mosaic/behavior/feature_library/spec.py tests/test_feature_params.py
git commit -m "add ResultColumn to spec.py, update LabelsSourceSpec"
```

---

### Task 2: Drop `seq=` regex from `_extract_key`

**Files:**
- Modify: `src/mosaic/behavior/feature_library/helpers.py`
- Modify: `src/mosaic/behavior/visualization_library/viz_global_colored.py`
- Test: `tests/test_load_spec.py`

**Step 1: Simplify `_extract_key` in helpers.py**

In `StreamingFeatureHelper._extract_key()` (line 681), replace the method body with:

```python
def _extract_key(self, path: Path) -> str:
    """Extract entry key from file path (the bare stem)."""
    return path.stem
```

**Step 2: Simplify `_extract_key` in viz_global_colored.py**

In `VizGlobalColored._extract_key()` (line 290), replace:

```python
def _extract_key(self, path: Path) -> str:
    """Extract entry key from file path (the bare stem)."""
    return path.stem
```

Remove the `regex` parameter. Update call sites:
- Line ~418: `key = self._extract_key(f, key_regex)` -> `key = self._extract_key(f)`
- Line ~450: `lab_key = self._extract_key(lf, label_regex)` -> `lab_key = self._extract_key(lf)`

Note: `coord_key_regex` and `label_key_regex` Params fields are removed later in Task 6 with the full VizGlobalColored Params rewrite.

**Step 3: Update test**

In `tests/test_load_spec.py`, `TestExtractKey::test_seq_pattern` currently passes a `seq=` filename and expects the sequence part extracted. Since the `seq=` files are being removed, update the test:

Rename `test_seq_pattern` to `test_returns_bare_stem` and update:

```python
def test_returns_bare_stem(self) -> None:
    from mosaic.behavior.feature_library.helpers import StreamingFeatureHelper

    helper = StreamingFeatureHelper(None, "test")
    path = Path("/tmp/calms21_task1_test__task1%2Ftest%2Fmouse075.parquet")
    assert helper._extract_key(path) == "calms21_task1_test__task1%2Ftest%2Fmouse075"
```

**Step 4: Run tests, commit**

```bash
uv run pytest tests/test_load_spec.py tests/test_feature_params.py -v
git add src/mosaic/behavior/feature_library/helpers.py \
        src/mosaic/behavior/visualization_library/viz_global_colored.py \
        tests/test_load_spec.py
git commit -m "drop seq= regex from _extract_key, return bare stem"
```

---

### Task 3: WardAssign -- drop npz write and `SeqLabelsArtifact`

**Files:**
- Modify: `src/mosaic/behavior/feature_library/ward_assign.py`

WardAssign already writes per-sequence parquet (lines 193-215). The npz write (lines 183-191) is redundant.

**Step 1: Delete npz write in `_write_sequence_outputs()`**

Remove lines 183-191 (the `np.savez_compressed(npz_path, ...)` block and the `npz_path` assignment).

**Step 2: Delete `SeqLabelsArtifact` class**

Remove the class definition (lines 73-78) and any imports of it within the file.

**Step 3: Check for external imports of `SeqLabelsArtifact`**

Search for imports of `ward_assign.SeqLabelsArtifact` across the codebase. Update or remove any found. Key places to check:
- `viz_global_colored.py`
- `analysis.py`
- Any test files

**Step 4: Run tests, commit**

```bash
uv run pytest tests/ -v
git add src/mosaic/behavior/feature_library/ward_assign.py
# Add any other files that needed import cleanup
git commit -m "ward_assign: drop npz write and SeqLabelsArtifact"
```

---

### Task 4: GlobalKMeans -- drop npz write and `SeqLabelsArtifact`

**Files:**
- Modify: `src/mosaic/behavior/feature_library/global_kmeans.py`

Same pattern as Task 3. GlobalKMeans writes both npz (lines 530-538) and parquet (lines 513-526) in `save_model()`.

**Step 1: Delete npz write in `save_model()`**

Remove lines 530-538 (the `np.savez_compressed(run_root / fname, ...)` block and the `fname` assignment).

**Step 2: Delete `SeqLabelsArtifact` class**

Remove the class definition (lines 111-115) and any imports.

**Step 3: Check for external imports**

Same as Task 3 -- search for `global_kmeans.SeqLabelsArtifact` across codebase.

**Step 4: Run tests, commit**

```bash
uv run pytest tests/ -v
git add src/mosaic/behavior/feature_library/global_kmeans.py
git commit -m "global_kmeans: drop npz write and SeqLabelsArtifact"
```

---

### Task 5: GlobalTSNE -- convert npz output to standard parquet

**Files:**
- Modify: `src/mosaic/behavior/feature_library/global_tsne.py`

This is the most involved change. GlobalTSNE currently writes `global_tsne_coords_seq={safe_seq}.npz` with key `Y` (N x 2 float32). Convert to standard parquet with columns `tsne_x`, `tsne_y`, `frame`, `sequence`, `group`.

**Step 1: Replace `_persist_mapped_coords()` to write parquet**

Current method (lines 569-578) writes npz. Replace with parquet write:

```python
def _persist_mapped_coords(
    self, safe_seq: str, Y: np.ndarray, frames: np.ndarray | None
) -> None:
    group, sequence = _resolve_sequence_identity(
        safe_seq, self._scope.entry_map
    )
    out_name = f"{entry_key(group, sequence)}.parquet"
    out_path = self._run_root / out_name

    n_rows = int(Y.shape[0])
    df = pd.DataFrame({
        "tsne_x": Y[:, 0].astype(np.float32),
        "tsne_y": Y[:, 1].astype(np.float32),
        "frame": frames if frames is not None else np.arange(n_rows, dtype=np.int64),
        "sequence": sequence,
        "group": group,
    })
    df.to_parquet(out_path, index=False)
    self._append_index_row(safe_seq, out_path, n_rows)
```

Note: `frames` is already available in `_map_sequences_streaming()` -- check that it's passed through. Currently `_persist_mapped_coords` is called at line 673; verify the `frames` array from `helper.load_key_data()` (line 649) is threaded to the write call.

**Step 2: Update `_map_sequences_streaming()` to pass frames**

At line 649, `helper.load_key_data()` returns `(X, frames)`. Verify that `frames` is passed to `_persist_mapped_coords()` at line 673. If not, add it. (Note: `load_key_data_with_identity()` is the 5-tuple variant used by WardAssign/GlobalKMeans; GlobalTSNE uses the simpler 2-tuple `load_key_data()`.)

**Step 3: Update `save_model()` coord persistence**

In `save_model()` (lines 446-450), there's a loop over `self._mapped_coords` that writes npz files. Replace with parquet writes using the same pattern as step 1. This path handles the case where coords are stored in memory rather than streamed.

**Step 4: Delete `SeqCoordsArtifact` class**

Remove the class definition (lines 152-162) and any imports. Search for external imports.

**Step 5: Update `_discover_existing_coord_rows()`**

This method (called by `get_additional_index_rows()` when `_additional_index_rows` is empty) globs for `global_tsne_coords_seq=*.npz`. Update to glob for `*.parquet` instead, and extract key from stem.

**Step 6: Run tests, commit**

```bash
uv run pytest tests/ -v
git add src/mosaic/behavior/feature_library/global_tsne.py
git commit -m "global_tsne: convert per-sequence output from npz to parquet"
```

---

### Task 6: VizGlobalColored -- `ResultColumn` params for x, y, and labels

**Files:**
- Modify: `src/mosaic/behavior/visualization_library/viz_global_colored.py`
- Modify: `src/mosaic/behavior/feature_library/spec.py` (remove `FeatureLabelsSource` if no other importers)

VizGlobalColored becomes a generic scatter plot. It keeps empty `Inputs` (same pattern as GlobalWard) and uses `ResultColumn` params for axes and labels.

**Step 1: Keep `Inputs` empty, update `Params`**

`Inputs` stays `_require = "empty"` (no pipeline inputs). Replace params:

- Remove `coords: ArtifactSpec` -- replaced by `x` and `y`
- Remove `coord_key_regex` and `label_key_regex`
- Add `x: ResultColumn` (required) -- scatter plot x-axis column
- Add `y: ResultColumn` (required) -- scatter plot y-axis column
- Replace `labels: FeatureLabelsSource | GroundTruthLabelsSource | None` with `labels: ResultColumn | GroundTruthLabelsSource | None`

Example usage for a t-SNE plot:
```python
Params(
    x=ResultColumn(feature="global-tsne", column="tsne_x"),
    y=ResultColumn(feature="global-tsne", column="tsne_y"),
    labels=ResultColumn(feature="ward-assign", column="cluster"),
)
```

Or a speed vs approach scatter:
```python
Params(
    x=ResultColumn(feature="speed-angvel", column="speed"),
    y=ResultColumn(feature="approach-avoidance", column="approach"),
)
```

**Step 2: Rewrite coordinate loading**

Replace the artifact-based coord loading (lines 408-430) with `ResultColumn`-based loading. For each `ResultColumn` (`x`, `y`, and optionally `labels`), resolve the feature run root and glob for `*.parquet` files per sequence. Load as DataFrames (via `pd.read_parquet`), extracting only the needed column plus `frame`/`sequence`/`group` for alignment.

When `x` and `y` reference the same feature+run_id, load each parquet file once and extract both columns. When they reference different features, load separately and align via `pd.merge` on `frame` (+ `sequence`/`group`), following the same pattern as `yield_input_data` in `iteration.py:489`.

`StreamingFeatureHelper.build_manifest_from_results()` can be reused to discover files per sequence. But do NOT use `load_key_data()` for the actual loading -- it does numpy hstack with min-trimming, which silently misaligns data when inputs have different row counts (e.g., from mixing frame-scoped and unscoped runs, or partial runs with different sequence coverage). Load parquet DataFrames directly and merge on frame columns.

**Step 3: Rewrite label loading for `ResultColumn`**

When `labels` is a `ResultColumn`, load the referenced feature's parquet output and extract the specified column. Align via `pd.merge` on `frame` (matching coord keys). `GroundTruthLabelsSource` path stays largely unchanged.

**Step 4: Remove artifact loading infrastructure**

Remove `_load_artifacts_glob()` usage for coords. Check if it's still needed for anything else in the class; if not, remove. Remove `_extract_key()` if no longer needed (stem-based key extraction may still be useful).

**Step 5: Remove `FeatureLabelsSource` from spec.py**

If no other files import `FeatureLabelsSource`, delete the class from spec.py. Search the codebase first.

**Step 6: Run tests, commit**

```bash
uv run pytest tests/ -v
git add src/mosaic/behavior/visualization_library/viz_global_colored.py \
        src/mosaic/behavior/feature_library/spec.py
git commit -m "viz_global_colored: ResultColumn params for x, y, labels"
```

---

### Task 7: analysis.py -- update npz consumers to read parquet

**Files:**
- Modify: `src/mosaic/core/analysis.py`

**Step 1: Update `_augment_with_saved_sequences()`**

This function (line 517) scans for `*_labels_seq=*.npz` and extracts safe_seq via `stem.split("labels_seq=", 1)[1]`. Since those files no longer exist, switch to scanning for `*.parquet` files in run directories. Extract key from stem directly (it's already in entry_key format).

**Step 2: Update `_load_cluster_labels()`**

This function (line 395) has separate NPZ and parquet branches. The NPZ branch loads `npz[column]` and optional `frames`. Since per-sequence npz files are gone, the NPZ branch can be simplified or removed if only the per-sequence label files used it. Check whether any global artifacts (cluster_centers.npz, artifact_labels.npz) are loaded through this path -- if so, keep the NPZ branch for those.

**Step 3: Run tests, commit**

```bash
uv run pytest tests/ -v
git add src/mosaic/core/analysis.py
git commit -m "analysis.py: read standard parquet instead of seq= npz files"
```

---

### Task 8: Final cleanup

**Files:**
- Modify: various

**Step 1: Search for any remaining `seq=` references**

```bash
grep -r "seq=" src/mosaic/ --include="*.py" | grep -v ".pyc"
```

Update or remove any remaining references.

**Step 2: Search for remaining `FeatureLabelsSource` imports**

```bash
grep -r "FeatureLabelsSource" src/ tests/ --include="*.py"
```

Remove the class from spec.py if fully unused. Update any remaining imports.

**Step 3: Search for remaining `SeqLabelsArtifact` / `SeqCoordsArtifact` references**

```bash
grep -rE "SeqLabelsArtifact|SeqCoordsArtifact" src/ tests/ --include="*.py"
```

**Step 4: Run full test suite, commit**

```bash
uv run pytest tests/ -v
git commit -m "final cleanup: remove remaining seq= and artifact references"
```
