# load_key_data unification Implementation Plan

> **Status: COMPLETE.** All tasks implemented. 221 tests pass (207 baseline + 14 new).

**Goal:** Replace the buggy min-trim + hstack alignment in `load_key_data()` with frame-level pd.merge, merge the duplicated `load_key_data` / `load_key_data_with_identity` into one method, add identity columns to GlobalTSNE output, and rename the misleading `safe_seq` variable to `entry_key` across all global features.

**Architecture:** A new internal method `_load_parquet_dataframe` loads raw DataFrames from parquet inputs. The unified `load_key_data` merges them on frame alignment columns (`frame`, `time`, `id`) and extracts identity via `_normalize_identity_columns`. It returns a `KeyData` dataclass containing the feature matrix, frames, and identity arrays. All callers are updated to destructure from `KeyData`. The `safe_seq` variable -- which is actually an entry key (`group__sequence`), not a safe sequence name -- is renamed to `entry_key` throughout. The `entry_key()` function in `core/helpers.py` was renamed to `make_entry_key()` to avoid shadowing.

**Tech Stack:** Python 3.13, pandas, numpy, pytest

**Issues addressed:** #1 (alignment bug), #2 (method duplication), #4 (GlobalTSNE missing identity), plus `safe_seq` naming cleanup

**Assessment:** `docs/assessments/2026-03-12-streaming-feature-helper-issues.md`

---

## Commits

| SHA | Tasks | Description |
|-----|-------|-------------|
| `9bea4ee` | 1 | Add `KeyData` dataclass and `_load_parquet_dataframe` helper |
| `f782df8` | 2 | Add failing tests for unified `load_key_data` |
| `15f4509` | 3 | Unify `load_key_data`: frame-level alignment via merge, identity extraction, `KeyData` return |
| `3583a67` | 4, 5, 6 | Update `iter_sequences`, GlobalTSNE, GlobalWard to use unified `load_key_data` with identity |
| `92dd652` | 7, 8, 9 | Update WardAssign, GlobalKMeans, feature template to use unified `load_key_data` |
| `f7fb481` | 10, 11 | Fix type annotation, rename `safe_seq` to `entry_key` across global features |
| `0481a8e` | -- | Fix stale docstring in `StreamingFeatureHelper` (found in code review) |
| `e971dc8` | -- | Fix feature column handling in `_merge_parquet_inputs` (post-implementation fix) |
| `ba3a273` | -- | Rename `entry_key` function to `make_entry_key`, use `entry_key` for variables |

## Post-implementation fixes

Two issues were discovered after the initial implementation was complete:

1. **`_merge_parquet_inputs` dropped duplicate feature columns** (`e971dc8`).
   When two inputs shared column names (e.g. both have `feat_0, feat_1, ...`),
   the plan's `right_keep` filter silently dropped the right input's features
   because `c not in merged.columns` excluded them. The old `np.hstack` kept
   all columns regardless of name. Fixed by suffixing duplicate column names
   (e.g. `feat_0__1`) so they survive the merge.

2. **`_keydata_from_merged` included non-numeric columns** (`e971dc8`). The
   plan's `meta_cols` exclusion set correctly identified metadata columns, but
   `feat_cols` was built from `df.columns` without filtering to numeric types.
   Non-numeric columns outside `meta_cols` would be coerced to NaN. Fixed by
   intersecting with `df.select_dtypes(include=[np.number])` to match the old
   `_load_array_from_spec` behavior.

Both fixes were covered by 5 additional tests added in the same commit:
`test_duplicate_column_names_across_inputs`, `test_aligned_values_correct`,
`test_meta_columns_excluded_from_features`, `test_non_numeric_columns_excluded`,
`test_misaligned_row_alignment_correct`.

---

## File inventory

All paths relative to repo root (`mosaic/`).

| File | Role |
|------|------|
| `src/mosaic/behavior/feature_library/helpers.py` | Core: `load_key_data`, `load_key_data_with_identity`, `_load_identity_from_spec`, `iter_sequences` |
| `src/mosaic/behavior/feature_library/global_tsne.py` | Caller: `fit()` line 301, `_map_sequences_streaming()` line 655, `_persist_mapped_coords` line 563, `save_model` line 430 |
| `src/mosaic/behavior/feature_library/global_ward.py` | Caller: `_load_artifact_matrix()` line 208 |
| `src/mosaic/behavior/feature_library/ward_assign.py` | Caller: `fit()` line 268 |
| `src/mosaic/behavior/feature_library/global_kmeans.py` | Caller: `fit()` line 343 |
| `src/mosaic/behavior/feature_library/feature_template__global.py` | Caller: `fit()` line 150 |
| `tests/test_load_spec.py` | Existing tests for `_load_array_from_spec`, `_load_identity_from_spec`, `_extract_key`, `build_manifest` |

---

## Task 1: Add `KeyData` dataclass and `_load_parquet_dataframe` helper -- DONE

> Commit: `9bea4ee`

Introduce the return type and the low-level DataFrame loader that replaces
the double-read in `_load_identity_from_spec`.

**Files:**
- Modify: `src/mosaic/behavior/feature_library/helpers.py`
- Test: `tests/test_load_spec.py`

### Step 1: Write failing tests for `_load_parquet_dataframe`

Add to `tests/test_load_spec.py`:

```python
class TestLoadParquetDataFrame:
    """Tests for _load_parquet_dataframe."""

    def test_basic_load(self, parquet_path: Path) -> None:
        from mosaic.behavior.feature_library.helpers import _load_parquet_dataframe
        from mosaic.behavior.feature_library.spec import ParquetLoadSpec

        spec = ParquetLoadSpec()
        df = _load_parquet_dataframe(parquet_path, spec)
        assert df is not None
        # Should contain feature columns + frame + id (all original columns)
        assert "feat_a" in df.columns
        assert "feat_b" in df.columns
        assert "frame" in df.columns

    def test_with_df_filter(self, parquet_path: Path) -> None:
        from mosaic.behavior.feature_library.helpers import _load_parquet_dataframe
        from mosaic.behavior.feature_library.spec import ParquetLoadSpec

        spec = ParquetLoadSpec()
        df = _load_parquet_dataframe(
            parquet_path, spec, df_filter=lambda d: d[d["frame"] >= 1]
        )
        assert df is not None
        assert len(df) == 2

    def test_empty_after_filter_returns_none(self, parquet_path: Path) -> None:
        from mosaic.behavior.feature_library.helpers import _load_parquet_dataframe
        from mosaic.behavior.feature_library.spec import ParquetLoadSpec

        spec = ParquetLoadSpec()
        df = _load_parquet_dataframe(
            parquet_path, spec, df_filter=lambda d: d[d["frame"] > 100]
        )
        assert df is None

    def test_non_parquet_returns_none(self, npz_path: Path) -> None:
        from mosaic.behavior.feature_library.helpers import _load_parquet_dataframe
        from mosaic.behavior.feature_library.spec import NpzLoadSpec

        spec = NpzLoadSpec(key="features")
        df = _load_parquet_dataframe(npz_path, spec)
        assert df is None
```

### Step 2: Run tests to verify they fail

```bash
uv run pytest tests/test_load_spec.py::TestLoadParquetDataFrame -v
```

Expected: FAIL -- `_load_parquet_dataframe` not defined.

### Step 3: Implement `KeyData` and `_load_parquet_dataframe`

In `helpers.py`, add after the `_normalize_identity_columns` function
(after line 183):

```python
@dataclass(frozen=True, slots=True)
class KeyData:
    """Result of loading and merging data for a single manifest key.

    Attributes
    ----------
    features : np.ndarray
        (N, D) feature matrix, float32.
    frames : np.ndarray
        (N,) frame indices, int64.
    id1 : np.ndarray | None
        (N,) first identity column, float64. None when global.
    id2 : np.ndarray | None
        (N,) second identity column, float64. None when global.
    entity_level : str
        One of "global", "individual", "pair".
    """

    features: np.ndarray
    frames: np.ndarray
    id1: np.ndarray | None
    id2: np.ndarray | None
    entity_level: str


def _load_parquet_dataframe(
    path: Path,
    load_spec: LoadSpec,
    df_filter: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
) -> pd.DataFrame | None:
    """Load a parquet file as a full DataFrame, applying filter if given.

    Returns None for non-parquet specs or if the result is empty after
    filtering.
    """
    if not isinstance(load_spec, ParquetLoadSpec):
        return None
    df = pd.read_parquet(path)
    if df_filter is not None:
        df = df_filter(df)
    if df.empty:
        return None
    return df
```

### Step 4: Run tests to verify they pass

```bash
uv run pytest tests/test_load_spec.py::TestLoadParquetDataFrame -v
```

Expected: PASS

### Step 5: Commit

```bash
git add tests/test_load_spec.py src/mosaic/behavior/feature_library/helpers.py
git commit -m "add KeyData dataclass and _load_parquet_dataframe helper"
```

---

## Task 2: Write failing tests for the unified `load_key_data` -- DONE

> Commit: `f782df8`

Test the new alignment behavior and identity extraction before
implementing.

**Files:**
- Test: `tests/test_load_spec.py`

### Step 1: Write failing tests

These tests exercise the core bug fix: frame-level alignment via merge
instead of min-trim + hstack.

```python
class TestLoadKeyDataUnified:
    """Tests for the unified load_key_data returning KeyData."""

    @pytest.fixture()
    def aligned_parquets(self, tmp_path: Path) -> tuple[Path, Path]:
        """Two parquet files with same frames but different feature columns."""
        p1 = tmp_path / "feat1.parquet"
        p2 = tmp_path / "feat2.parquet"
        df1 = pd.DataFrame({
            "frame": [0, 1, 2, 3],
            "feat_a": [1.0, 2.0, 3.0, 4.0],
        })
        df2 = pd.DataFrame({
            "frame": [0, 1, 2, 3],
            "feat_b": [10.0, 20.0, 30.0, 40.0],
        })
        df1.to_parquet(p1, index=False)
        df2.to_parquet(p2, index=False)
        return p1, p2

    @pytest.fixture()
    def misaligned_parquets(self, tmp_path: Path) -> tuple[Path, Path]:
        """Two parquets with overlapping but different frame ranges."""
        p1 = tmp_path / "feat1.parquet"
        p2 = tmp_path / "feat2.parquet"
        # feat1 has frames 0-3, feat2 has frames 2-5
        df1 = pd.DataFrame({
            "frame": [0, 1, 2, 3],
            "feat_a": [1.0, 2.0, 3.0, 4.0],
        })
        df2 = pd.DataFrame({
            "frame": [2, 3, 4, 5],
            "feat_b": [30.0, 40.0, 50.0, 60.0],
        })
        df1.to_parquet(p1, index=False)
        df2.to_parquet(p2, index=False)
        return p1, p2

    @pytest.fixture()
    def identity_parquets(self, tmp_path: Path) -> tuple[Path, Path]:
        """Two parquets with identity columns (pair feature)."""
        p1 = tmp_path / "feat1.parquet"
        p2 = tmp_path / "feat2.parquet"
        df1 = pd.DataFrame({
            "frame": [0, 0, 1, 1],
            "id1": [1, 2, 1, 2],
            "id2": [2, 1, 2, 1],
            "feat_a": [1.0, 2.0, 3.0, 4.0],
        })
        df2 = pd.DataFrame({
            "frame": [0, 0, 1, 1],
            "id1": [1, 2, 1, 2],
            "id2": [2, 1, 2, 1],
            "feat_b": [10.0, 20.0, 30.0, 40.0],
        })
        df1.to_parquet(p1, index=False)
        df2.to_parquet(p2, index=False)
        return p1, p2

    def test_aligned_inputs_merged(self, aligned_parquets: tuple[Path, Path]) -> None:
        from mosaic.behavior.feature_library.helpers import (
            KeyData,
            StreamingFeatureHelper,
        )
        from mosaic.behavior.feature_library.spec import ParquetLoadSpec

        p1, p2 = aligned_parquets
        helper = StreamingFeatureHelper(None, "test")
        spec = ParquetLoadSpec()
        result = helper.load_key_data([(p1, spec), (p2, spec)])
        assert isinstance(result, KeyData)
        assert result.features.shape == (4, 2)  # 4 frames x 2 feature cols
        np.testing.assert_array_equal(result.frames, [0, 1, 2, 3])

    def test_misaligned_inputs_inner_join(
        self, misaligned_parquets: tuple[Path, Path]
    ) -> None:
        """Overlapping frames only -- the core alignment fix."""
        from mosaic.behavior.feature_library.helpers import (
            KeyData,
            StreamingFeatureHelper,
        )
        from mosaic.behavior.feature_library.spec import ParquetLoadSpec

        p1, p2 = misaligned_parquets
        helper = StreamingFeatureHelper(None, "test")
        spec = ParquetLoadSpec()
        result = helper.load_key_data([(p1, spec), (p2, spec)])
        assert isinstance(result, KeyData)
        # Only frames 2 and 3 overlap
        assert result.features.shape == (2, 2)
        np.testing.assert_array_equal(result.frames, [2, 3])
        # feat_a values at frames 2,3 are 3.0, 4.0
        np.testing.assert_array_almost_equal(result.features[:, 0], [3.0, 4.0])
        # feat_b values at frames 2,3 are 30.0, 40.0
        np.testing.assert_array_almost_equal(result.features[:, 1], [30.0, 40.0])

    def test_identity_columns_extracted(
        self, identity_parquets: tuple[Path, Path]
    ) -> None:
        from mosaic.behavior.feature_library.helpers import (
            KeyData,
            StreamingFeatureHelper,
        )
        from mosaic.behavior.feature_library.spec import ParquetLoadSpec

        p1, p2 = identity_parquets
        helper = StreamingFeatureHelper(None, "test")
        spec = ParquetLoadSpec()
        result = helper.load_key_data([(p1, spec), (p2, spec)])
        assert isinstance(result, KeyData)
        assert result.entity_level == "pair"
        assert result.id1 is not None
        assert result.id2 is not None
        assert result.features.shape == (4, 2)  # 4 rows x 2 feature cols
        np.testing.assert_array_equal(result.id1, [1, 2, 1, 2])
        np.testing.assert_array_equal(result.id2, [2, 1, 2, 1])

    def test_single_input(self, parquet_path: Path) -> None:
        """Single input should work without merge."""
        from mosaic.behavior.feature_library.helpers import (
            KeyData,
            StreamingFeatureHelper,
        )
        from mosaic.behavior.feature_library.spec import ParquetLoadSpec

        helper = StreamingFeatureHelper(None, "test")
        spec = ParquetLoadSpec()
        result = helper.load_key_data([(parquet_path, spec)])
        assert isinstance(result, KeyData)
        assert result.features.shape == (3, 2)  # feat_a, feat_b
        np.testing.assert_array_equal(result.frames, [0, 1, 2])
        assert result.entity_level == "individual"  # parquet_path has "id" col

    def test_no_overlap_returns_none(self, tmp_path: Path) -> None:
        """Two inputs with no shared frames should return None."""
        from mosaic.behavior.feature_library.helpers import StreamingFeatureHelper
        from mosaic.behavior.feature_library.spec import ParquetLoadSpec

        p1 = tmp_path / "a.parquet"
        p2 = tmp_path / "b.parquet"
        pd.DataFrame({"frame": [0, 1], "f": [1.0, 2.0]}).to_parquet(p1)
        pd.DataFrame({"frame": [5, 6], "f": [3.0, 4.0]}).to_parquet(p2)

        helper = StreamingFeatureHelper(None, "test")
        spec = ParquetLoadSpec()
        result = helper.load_key_data([(p1, spec), (p2, spec)])
        assert result is None

    def test_global_entity_level_when_no_id(self, tmp_path: Path) -> None:
        from mosaic.behavior.feature_library.helpers import (
            KeyData,
            StreamingFeatureHelper,
        )
        from mosaic.behavior.feature_library.spec import ParquetLoadSpec

        p = tmp_path / "no_id.parquet"
        pd.DataFrame({"frame": [0, 1], "feat": [1.0, 2.0]}).to_parquet(p)

        helper = StreamingFeatureHelper(None, "test")
        result = helper.load_key_data([(p, ParquetLoadSpec())])
        assert isinstance(result, KeyData)
        assert result.entity_level == "global"
        assert result.id1 is None
        assert result.id2 is None

    def test_npz_input_fallback(self, npz_path: Path) -> None:
        """NPZ inputs cannot merge on frames -- fall back to hstack."""
        from mosaic.behavior.feature_library.helpers import (
            KeyData,
            StreamingFeatureHelper,
        )
        from mosaic.behavior.feature_library.spec import NpzLoadSpec

        helper = StreamingFeatureHelper(None, "test")
        spec = NpzLoadSpec(key="features")
        result = helper.load_key_data([(npz_path, spec)])
        assert isinstance(result, KeyData)
        assert result.features.shape == (3, 4)
        assert result.entity_level == "global"
```

### Step 2: Run tests to verify they fail

```bash
uv run pytest tests/test_load_spec.py::TestLoadKeyDataUnified -v
```

Expected: FAIL -- `load_key_data` returns a tuple, not `KeyData`.

### Step 3: Commit test file

```bash
git add tests/test_load_spec.py
git commit -m "add failing tests for unified load_key_data"
```

---

## Task 3: Implement the unified `load_key_data` -- DONE

> Commit: `15f4509`
>
> **Deviation:** The plan's `_merge_parquet_inputs` had two bugs that were
> caught and fixed post-implementation in `e971dc8`:
>
> 1. The `right_keep` filter dropped feature columns from the right input
>    when both inputs shared column names (e.g. `feat_0`). Fixed by suffixing
>    duplicate column names (`feat_0__1`) instead of dropping them.
>
> 2. `_keydata_from_merged` selected feature columns by excluding `meta_cols`
>    from `df.columns` without filtering to numeric types. Non-numeric columns
>    outside `meta_cols` would be coerced to NaN. Fixed by intersecting with
>    `df.select_dtypes(include=[np.number])`.
>
> Both fixes are covered by 5 additional tests added in the same commit.

Replace both `load_key_data` and `load_key_data_with_identity` with a
single method returning `KeyData`.

**Files:**
- Modify: `src/mosaic/behavior/feature_library/helpers.py:510-684`

### Step 1: Implement

Replace the `load_key_data` method (lines 510-572) with the new
implementation. Delete `load_key_data_with_identity` (lines 608-683) and
`_load_identity_from_spec` (lines 574-606).

The new `load_key_data`:

```python
def load_key_data(
    self,
    file_specs: list[tuple[Path, LoadSpec]],
    key: str | None = None,
) -> KeyData | None:
    """Load and merge data for a single manifest key.

    For parquet inputs, merges on shared frame/identity columns
    (inner join) so that misaligned inputs are correctly aligned.
    For non-parquet inputs (npz), falls back to min-trim + hstack.

    Returns None when no usable data is found or when parquet inputs
    have no overlapping frames.
    """
    import pyarrow as pa

    df_filter = self._make_df_filter(key) if key else None

    parquet_dfs: list[pd.DataFrame] = []
    array_mats: list[np.ndarray] = []
    array_frames: np.ndarray | None = None

    for path, load_spec in file_specs:
        df = _load_parquet_dataframe(path, load_spec, df_filter=df_filter)
        if df is not None:
            parquet_dfs.append(df)
        else:
            arr, frame_vals = _load_array_from_spec(
                path, load_spec, extract_frame_col="frame", df_filter=df_filter
            )
            if arr is not None and arr.size > 0:
                array_mats.append(arr)
                if array_frames is None and frame_vals is not None:
                    array_frames = frame_vals

    if not parquet_dfs and not array_mats:
        return None

    # Parquet path: merge on alignment columns
    if parquet_dfs:
        merged = self._merge_parquet_inputs(parquet_dfs)
        if merged is None or merged.empty:
            return None
        return self._keydata_from_merged(merged)

    # Non-parquet fallback: min-trim + hstack (npz inputs)
    return self._keydata_from_arrays(array_mats, array_frames)
```

Add these private methods to `StreamingFeatureHelper`:

```python
_ALIGN_COLS = ("frame", "time", "id", "id1", "id2")

def _merge_parquet_inputs(
    self, dfs: list[pd.DataFrame]
) -> pd.DataFrame | None:
    """Merge multiple parquet DataFrames on shared alignment columns."""
    if len(dfs) == 1:
        return dfs[0]

    merged = dfs[0]
    for i, df_next in enumerate(dfs[1:], 1):
        on_cols = [
            c for c in self._ALIGN_COLS
            if c in merged.columns and c in df_next.columns
        ]
        if not on_cols:
            on_cols = [
                c for c in ("frame", "time")
                if c in merged.columns and c in df_next.columns
            ]
        if not on_cols:
            # No shared alignment columns -- fall back to row-aligned concat
            min_len = min(len(merged), len(df_next))
            right = df_next.iloc[:min_len].reset_index(drop=True)
            # Suffix duplicate columns so they survive concat
            right_rename = {
                c: f"{c}__{i}"
                for c in right.columns if c in merged.columns
            }
            if right_rename:
                right = right.rename(columns=right_rename)
            merged = pd.concat(
                [merged.iloc[:min_len].reset_index(drop=True), right],
                axis=1,
            )
            continue
        # Suffix duplicate feature columns so they survive the merge
        # (old hstack kept all columns regardless of name overlap)
        rename_map = {
            c: f"{c}__{i}"
            for c in df_next.columns
            if c not in on_cols and c in merged.columns
        }
        if rename_map:
            df_next = df_next.rename(columns=rename_map)
        # Drop non-feature meta columns that already exist in merged
        meta_dupes = [
            c for c in df_next.columns
            if c not in on_cols
            and c in merged.columns
            and c not in rename_map.values()
        ]
        if meta_dupes:
            df_next = df_next.drop(columns=meta_dupes)
        merged = merged.merge(df_next, how="inner", on=on_cols)

    if merged.empty:
        return None
    return merged

def _keydata_from_merged(self, df: pd.DataFrame) -> KeyData:
    """Extract KeyData fields from a merged DataFrame."""
    # Extract identity
    id1_series, id2_series, entity_level = _normalize_identity_columns(df)

    # Extract frame column
    if "frame" in df.columns:
        frames = df["frame"].to_numpy(dtype=np.int64, copy=True)
    elif "time" in df.columns:
        frames = df["time"].to_numpy(dtype=np.int64, copy=True)
    else:
        frames = np.arange(len(df), dtype=np.int64)

    # Extract feature matrix: numeric non-metadata columns only
    meta_cols = {
        "frame", "time", "id", "id1", "id2",
        "id_a", "id_b", "id_A", "id_B",
        "group", "sequence", "entity_level",
    }
    numeric_cols = set(df.select_dtypes(include=[np.number]).columns)
    feat_cols = [
        c for c in df.columns
        if c not in meta_cols and c in numeric_cols
    ]

    features = df[feat_cols].to_numpy(dtype=np.float32, copy=True)

    id1 = (
        id1_series.to_numpy(dtype=np.float64, copy=True)
        if id1_series is not None else None
    )
    id2 = (
        id2_series.to_numpy(dtype=np.float64, copy=True)
        if id2_series is not None else None
    )

    return KeyData(
        features=features,
        frames=frames,
        id1=id1,
        id2=id2,
        entity_level=entity_level,
    )

@staticmethod
def _keydata_from_arrays(
    mats: list[np.ndarray],
    frames: np.ndarray | None,
) -> KeyData:
    """Build KeyData from raw numpy arrays (npz fallback)."""
    import pyarrow as pa

    T_min = min(m.shape[0] for m in mats)
    trimmed = [m[:T_min] for m in mats]
    features = np.hstack(trimmed)

    del mats, trimmed
    gc.collect()
    pa.default_memory_pool().release_unused()

    if frames is not None and len(frames) >= T_min:
        frames = frames[:T_min]
    else:
        frames = np.arange(T_min, dtype=np.int64)

    return KeyData(
        features=features,
        frames=frames,
        id1=None,
        id2=None,
        entity_level="global",
    )
```

### Step 2: Run tests

```bash
uv run pytest tests/test_load_spec.py -v
```

Expected: all `TestLoadKeyDataUnified` tests PASS, plus existing tests.
Some existing tests for the old return type may need updating -- see step 3.

### Step 3: Update existing tests that assumed tuple return

The existing `TestLoadIdentityFromSpec` tests call
`helper._load_identity_from_spec()` which is being removed. Update them to
test the unified method instead, or remove them if covered by
`TestLoadKeyDataUnified`.

### Step 4: Run full test suite

```bash
uv run pytest tests/ -x -q
```

Expected: all pass.

### Step 5: Commit

```bash
git add src/mosaic/behavior/feature_library/helpers.py tests/test_load_spec.py
git commit -m "unify load_key_data: frame-level alignment via merge, identity extraction, KeyData return"
```

---

## Task 4: Update `iter_sequences` to use `KeyData` -- DONE

> Commit: `3583a67` (combined with Tasks 5, 6)

**Files:**
- Modify: `src/mosaic/behavior/feature_library/helpers.py:460-508`

### Step 1: Update `iter_sequences`

Change the yield type from `tuple[str, np.ndarray, np.ndarray | None]` to
`tuple[str, KeyData]`:

```python
def iter_sequences(
    self,
    manifest: dict[str, list[tuple[Path, LoadSpec]]],
    progress_interval: int = 10,
) -> Iterator[tuple[str, KeyData]]:
    """Iterate through manifest, yielding (key, KeyData) one at a time."""
    import pyarrow as pa

    n_keys = len(manifest)
    for i, (key, file_specs) in enumerate(manifest.items()):
        kd = self.load_key_data(file_specs, key=key)
        if kd is None:
            continue

        yield key, kd

        del kd
        gc.collect()
        pa.default_memory_pool().release_unused()

        if (i + 1) % progress_interval == 0 or i == n_keys - 1:
            print(
                f"[{self.feature_name}] Processed {i + 1}/{n_keys} sequences",
                file=sys.stderr,
            )
```

Note: `iter_sequences` is not currently called by any global feature (they
all call `load_key_data` directly). No caller updates needed for this
method. Confirm with grep.

### Step 2: Run tests

```bash
uv run pytest tests/ -x -q
```

### Step 3: Commit

```bash
git add src/mosaic/behavior/feature_library/helpers.py
git commit -m "update iter_sequences to yield KeyData"
```

---

## Task 5: Update GlobalTSNE callers and add identity to output -- DONE

> Commit: `3583a67` (combined with Tasks 4, 6)

This is the largest caller update. Three call sites in GlobalTSNE, plus
the output-writing code needs identity columns.

**Files:**
- Modify: `src/mosaic/behavior/feature_library/global_tsne.py`

### Step 1: Update `_mapped_coords` type

The `_mapped_coords` dict currently stores `(Y, frames)`. It needs to
store identity too. Change to store `(Y, frames, id1, id2, entity_level)`:

```python
# In __init__, line 220:
self._mapped_coords: dict[str, tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None, str]] = {}
```

### Step 2: Update `_persist_mapped_coords` to write identity

Change signature and body (lines 563-583):

```python
def _persist_mapped_coords(
    self,
    entry_key: str,
    Y: np.ndarray,
    frames: np.ndarray | None,
    id1: np.ndarray | None = None,
    id2: np.ndarray | None = None,
    entity_level: str = "global",
) -> None:
    if self._run_root is None:
        return
    group, sequence = _resolve_sequence_identity(
        entry_key, self._scope.entry_map
    )
    out_name = f"{entry_key(group, sequence)}.parquet"
    out_path = self._run_root / out_name

    n_rows = int(Y.shape[0])
    data: dict[str, object] = {
        "tsne_x": Y[:, 0].astype(np.float32),
        "tsne_y": Y[:, 1].astype(np.float32),
        "frame": frames if frames is not None else np.arange(n_rows, dtype=np.int64),
        "sequence": sequence,
        "group": group,
    }
    if id1 is not None:
        data["id1"] = pd.array(id1, dtype="Int64")
    if id2 is not None:
        data["id2"] = pd.array(id2, dtype="Int64")
    if entity_level != "global":
        data["entity_level"] = np.full(n_rows, entity_level, dtype=object)

    df = pd.DataFrame(data)
    df.to_parquet(out_path, index=False)
    self._append_index_row(entry_key, out_path, n_rows)
```

### Step 3: Update `save_model` output loop (lines 430-444)

Same identity columns pattern:

```python
for entry_key, (Y, frames, id1, id2, entity_level) in mapped.items():
    group, sequence = _resolve_sequence_identity(
        entry_key, self._scope.entry_map
    )
    out_name = f"{entry_key(group, sequence)}.parquet"
    out_path = run_root / out_name
    n_rows = int(Y.shape[0])
    data: dict[str, object] = {
        "tsne_x": Y[:, 0].astype(np.float32),
        "tsne_y": Y[:, 1].astype(np.float32),
        "frame": frames if frames is not None else np.arange(n_rows, dtype=np.int64),
        "sequence": sequence,
        "group": group,
    }
    if id1 is not None:
        data["id1"] = pd.array(id1, dtype="Int64")
    if id2 is not None:
        data["id2"] = pd.array(id2, dtype="Int64")
    if entity_level != "global":
        data["entity_level"] = np.full(n_rows, entity_level, dtype=object)
    df = pd.DataFrame(data)
    df.to_parquet(out_path, index=False)
```

### Step 4: Update `fit()` pass 1 sampling (line 301)

```python
kd = helper.load_key_data(key_file_manifest[key], key=key)
if kd is None or kd.features.shape[0] == 0:
    continue
X = kd.features
```

### Step 5: Update `_map_sequences_streaming` (line 655)

```python
kd = helper.load_key_data(key_file_manifest[key], key=key)
if kd is None or kd.features.shape[0] == 0:
    continue
X = kd.features
frames = kd.frames
```

Then update the storage into `mapped` dict and `_persist_mapped_coords`
calls to pass identity:

```python
if self._run_root is not None:
    self._persist_mapped_coords(key, Y_seq, frames, kd.id1, kd.id2, kd.entity_level)
    del Y_seq
else:
    mapped[key] = (Y_seq, frames, kd.id1, kd.id2, kd.entity_level)
```

### Step 6: Update imports

Add `KeyData` to the imports from `.helpers` (if not already re-exported).
Remove `_load_array_from_spec` from imports if no longer used directly
(check -- it's used in `_load_templates` on line 519, so keep it).

### Step 7: Run tests

```bash
uv run pytest tests/ -x -q
```

### Step 8: Commit

```bash
git add src/mosaic/behavior/feature_library/global_tsne.py
git commit -m "GlobalTSNE: use unified load_key_data, write identity columns to output"
```

---

## Task 6: Update GlobalWard caller -- DONE

> Commit: `3583a67` (combined with Tasks 4, 5)

**Files:**
- Modify: `src/mosaic/behavior/feature_library/global_ward.py:206-220`

### Step 1: Update `_load_artifact_matrix` stacked-features path

```python
kd = helper.load_key_data(entries, key=key)
if kd is not None and kd.features.size > 0:
    blocks[key] = kd.features
```

### Step 2: Update imports

Add `KeyData` to imports from `.helpers` if needed (or just use the
`kd.features` attribute without importing the type).

### Step 3: Run tests

```bash
uv run pytest tests/ -x -q
```

### Step 4: Commit

```bash
git add src/mosaic/behavior/feature_library/global_ward.py
git commit -m "GlobalWard: use unified load_key_data"
```

---

## Task 7: Update WardAssign caller -- DONE

> Commit: `92dd652` (combined with Tasks 8, 9)

**Files:**
- Modify: `src/mosaic/behavior/feature_library/ward_assign.py:267-295`

### Step 1: Update `fit()` loading loop

```python
kd = helper.load_key_data(manifest[entry_key], key=entry_key)
if kd is None:
    continue
```

Then update the downstream usage:

```python
if self._scaler is not None:
    ...
    X_use = self._scaler.transform(kd.features)
else:
    X_use = kd.features

idxs = self._assign_nn.kneighbors(X_use, return_distance=False)
labels = self._cluster_ids[idxs.ravel()]
self._write_sequence_outputs(
    entry_key, labels, kd.frames,
    id1_vals=kd.id1, id2_vals=kd.id2, entity_level=kd.entity_level,
)
del X_use, kd, idxs, labels
```

### Step 2: Update imports

Remove `load_key_data_with_identity` usage (method no longer exists).

### Step 3: Run tests

```bash
uv run pytest tests/ -x -q
```

### Step 4: Commit

```bash
git add src/mosaic/behavior/feature_library/ward_assign.py
git commit -m "WardAssign: use unified load_key_data"
```

---

## Task 8: Update GlobalKMeans caller -- DONE

> Commit: `92dd652` (combined with Tasks 7, 9)

**Files:**
- Modify: `src/mosaic/behavior/feature_library/global_kmeans.py:342-403`

### Step 1: Update `fit()` assign loop

```python
kd = helper.load_key_data(manifest[key], key=key)
if kd is None:
    continue
D_total = kd.features.shape[1]
```

Then update all downstream references:

```python
X_use = scaler.transform(kd.features)
# ...
X_use = kd.features

labels = self._kmeans.predict(X_use)
self._assign_labels[key] = labels.astype(np.int32)
self._assign_frames[key] = kd.frames
if kd.id1 is not None:
    self._assign_id1[key] = np.asarray(kd.id1, dtype=np.float64).ravel()
if kd.id2 is not None:
    self._assign_id2[key] = np.asarray(kd.id2, dtype=np.float64).ravel()
self._assign_entity_level[key] = kd.entity_level

del X_use, kd, labels
```

### Step 2: Run tests

```bash
uv run pytest tests/ -x -q
```

### Step 3: Commit

```bash
git add src/mosaic/behavior/feature_library/global_kmeans.py
git commit -m "GlobalKMeans: use unified load_key_data"
```

---

## Task 9: Update feature template -- DONE

> Commit: `92dd652` (combined with Tasks 7, 8)

**Files:**
- Modify: `src/mosaic/behavior/feature_library/feature_template__global.py:149-155`

### Step 1: Update `fit()` loading

```python
kd = helper.load_key_data(manifest[key], key=key)
if kd is None or kd.features.shape[0] == 0:
    continue
```

Update the downstream usage of `X` and `frames` to use `kd.features` and
`kd.frames`. The template's `_process_sequence` call (line 159) takes
`(key, X, frames)` -- update to `(key, kd.features, kd.frames)` or
update the template method signature.

### Step 2: Run tests

```bash
uv run pytest tests/ -x -q
```

### Step 3: Commit

```bash
git add src/mosaic/behavior/feature_library/feature_template__global.py
git commit -m "feature template: use unified load_key_data"
```

---

## Task 10: Fix type annotation bug and cleanup -- DONE

> Commit: `f7fb481` (combined with Task 11)

**Files:**
- Modify: `src/mosaic/behavior/feature_library/global_tsne.py:585`
- Modify: `src/mosaic/behavior/feature_library/helpers.py` (remove dead code)

### Step 1: Fix `_discover_existing_coord_rows` return type

```python
# Old (line 585):
def _discover_existing_coord_rows(self) -> list[dict[str, object]]:
    ...
    rows: list[dict[str, object]] = []

# New:
def _discover_existing_coord_rows(self) -> list[PartialIndexRow]:
    ...
    rows: list[PartialIndexRow] = []
```

### Step 2: Remove dead code from helpers.py

If `_load_identity_from_spec` and `load_key_data_with_identity` were not
removed in Task 3, remove them now. Verify no remaining references:

```bash
uv run basedpyright src/mosaic/behavior/feature_library/helpers.py
```

### Step 3: Run full test suite

```bash
uv run pytest tests/ -x -q
```

### Step 4: Commit

```bash
git add src/mosaic/behavior/feature_library/global_tsne.py src/mosaic/behavior/feature_library/helpers.py
git commit -m "fix _discover_existing_coord_rows return type, remove dead code"
```

---

## Task 11: Rename `safe_seq` to `entry_key` across global features -- DONE

> Commit: `f7fb481` (combined with Task 10), then `ba3a273` (rename
> `ekey` -> `entry_key` and `entry_key()` function -> `make_entry_key()`)
>
> **Deviation:** The plan originally used `ekey` as the variable name.
> This violated the project's "no unnecessary abbreviations" rule. Renamed
> to `entry_key` instead. Since `entry_key` was already an imported function
> name, the function in `core/helpers.py` was renamed to `make_entry_key()`
> and all imports/call sites across the codebase (19 files) were updated.

The variable named `safe_seq` throughout global features is actually an
entry key (the `group__sequence` composite string produced by
`make_entry_key()`), not a "safe sequence name." This is confusing because it
conflates the entry key with a sequence identifier. Rename to `entry_key`
throughout.

Also rename the parameter in `_resolve_sequence_identity` and
`_write_sequence_outputs` / `_persist_mapped_coords` / `_append_index_row`.

**Files:**
- Modify: `src/mosaic/behavior/feature_library/helpers.py` -- `_resolve_sequence_identity` parameter
- Modify: `src/mosaic/behavior/feature_library/global_tsne.py` -- all `safe_seq` occurrences
- Modify: `src/mosaic/behavior/feature_library/ward_assign.py` -- all `safe_seq` occurrences
- Modify: `src/mosaic/behavior/feature_library/global_kmeans.py` -- all `safe_seq` occurrences
- Modify: `src/mosaic/behavior/feature_library/feature_template__global.py` -- all `safe_seq` occurrences
- Modify: `src/mosaic/behavior/feature_library/temporal_stacking.py` -- all `safe_seq` occurrences

### Step 1: Rename in helpers.py

```python
# _resolve_sequence_identity (line 695):
def _resolve_sequence_identity(
    entry_key: str,
    entry_map: dict[str, tuple[str, str]],
) -> tuple[str, str]:
    """Map an entry key to (group, sequence).

    Looks up in entry_map, falls back to ("", entry_key).
    """
    if entry_key in entry_map:
        return entry_map[entry_key]
    return "", entry_key
```

### Step 2: Rename in each global feature file

Mechanical rename: `safe_seq` -> `entry_key` in all variable names, parameter
names, and loop variables. Use find-and-replace within each file.

Files and approximate occurrence counts:
- `global_tsne.py`: ~8 occurrences (lines 430, 432, 549, 551, 564, 568, 583, 597)
- `ward_assign.py`: ~8 occurrences (lines 150, 159, 176, 197, 267, 270, 272, 289)
- `global_kmeans.py`: ~5 occurrences (lines 487, 489, 494, 495, 500, 502)
- `feature_template__global.py`: ~2 occurrences (lines 210, 214)
- `temporal_stacking.py`: ~8 occurrences (lines 157, 159, 160, 163, 164, 188, 192, 199, 345, 362, 371)

### Step 3: Run full test suite

```bash
uv run pytest tests/ -x -q
```

### Step 4: Run basedpyright on changed files

```bash
uv run basedpyright src/mosaic/behavior/feature_library/helpers.py src/mosaic/behavior/feature_library/global_tsne.py src/mosaic/behavior/feature_library/ward_assign.py src/mosaic/behavior/feature_library/global_kmeans.py
```

### Step 5: Commit

```bash
git add src/mosaic/behavior/feature_library/
git commit -m "rename safe_seq to entry_key across global features"
```

---

## Verification checklist -- DONE

After all tasks:

1. `uv run pytest tests/ -x -q` -- 221 tests pass (207 baseline + 14 new)
2. `uv run basedpyright src/mosaic/behavior/feature_library/helpers.py` -- no new errors
3. No remaining references to `load_key_data_with_identity` or `_load_identity_from_spec`:
   ```bash
   grep -r "load_key_data_with_identity\|_load_identity_from_spec" src/
   ```
4. No remaining `safe_seq` or `ekey` variables:
   ```bash
   grep -rn "safe_seq\|ekey" src/mosaic/behavior/feature_library/
   ```
5. GlobalTSNE output parquets include `id1`/`id2`/`entity_level` columns
   when input is a pair feature
6. Misaligned inputs (different frame ranges) are correctly inner-joined
   on frame columns
7. Multi-input with duplicate column names preserves all features (verified
   empirically -- old hstack and new merge produce identical matrices)
