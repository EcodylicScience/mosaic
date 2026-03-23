# Task 6: Migrate stateless per-sequence features

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Convert all features that have `needs_fit() = False` and `parallelizable = True` to the new protocol.

**Phase:** C (Protocol Transition -- clean break, all Phase C tasks land together)

**Parent plan:** `docs/plans/2026-03-12-pipeline-unification-implementation.md`

**Depends on:** Task 5

---

## Features to migrate (10)

- `speed_angvel.py` (SpeedAngvel)
- `body_scale.py` (BodyScaleFeature)
- `orientation_relative.py` (OrientationRelativeFeature)
- `pair_position.py` (PairPositionFeatures)
- `pair_egocentric.py` (PairEgocentricFeatures)
- `nearestneighbor.py` (NearestNeighbor)
- `nn_delta_response.py` (NearestNeighborDelta)
- `nn_delta_bins.py` (NearestNeighborDeltaBins)
- `id_tag_columns.py` (IdTagColumns)
- `approach_avoidance.py` (ApproachAvoidance)

---

## Terminology note

> Task 4 renamed `KeyData` -> `EntryData`, `load_key_data` -> `load_entry_data`,
> and `key` -> `entry_key` throughout the codebase. The code below reflects that.

## Migration pattern for each

1. Remove: `output_type`, `storage_feature_name`, `storage_use_input_suffix`, `skip_existing_outputs`
2. Add: `scope_dependent = False`
3. Remove methods: `needs_fit()`, `supports_partial_fit()`, `fit()`, `partial_fit()`, `finalize_fit()`, `save_model()`, `load_model()`, `bind_dataset()`, `set_scope()`
4. Add methods:
   - `load_state(self, run_root, artifact_paths) -> True` (always skip fit)
   - `fit(self, inputs)` -- empty (never called)
   - `save_state(self, run_root)` -- empty
   - `apply(self, entry_key, entry_data) -> pd.DataFrame` -- adapted from `transform()`
5. Rename `transform(df) -> apply(entry_key, entry_data)`:
   - The `entry_key` parameter is the entry key string
   - The `entry_data: EntryData` carries `features`, `frames`, `id1`, `id2`, `entity_level`
   - Features that operate on DataFrames need to reconstruct one from EntryData, or refactor to work with arrays directly
   - The return type is always `pd.DataFrame` (no more payload types)

---

## Example: SpeedAngvel migration

Before:
```python
def transform(self, df: pd.DataFrame) -> pd.DataFrame:
    return self._compute_speed_angvel(df)
```

After:
```python
def load_state(self, run_root, artifact_paths):
    return True

def fit(self, inputs):
    pass

def save_state(self, run_root):
    pass

def apply(self, entry_key, entry_data):
    # SpeedAngvel works on raw track DataFrames with pose columns
    # Reconstruct minimal df from EntryData
    # OR: if the feature works with arrays, operate on entry_data.features
    # For now, reconstruct df from the original parquet path
    # (the pipeline passes the track DataFrame)
    ...
```

---

## Important design decision: EntryData vs DataFrame in `apply()`

The new protocol passes `EntryData` (numpy arrays) to `apply()`. But many per-sequence features currently receive a full DataFrame with named columns (frame, time, id, poseX0, poseY0, ...). They use column names for computation.

Two options:
1. Features that need column names reconstruct a DataFrame from EntryData
2. `apply()` receives the full merged DataFrame alongside EntryData

Option 2 is simpler and avoids information loss. Add the merged DataFrame to EntryData:

```python
@dataclass
class EntryData:
    features: np.ndarray
    frames: np.ndarray
    id1: np.ndarray | None
    id2: np.ndarray | None
    entity_level: str
    df: pd.DataFrame | None = None  # full merged DataFrame when available
```

Features can use `entry_data.df` when they need column names (most per-sequence features), or `entry_data.features` when they work with numeric arrays (global features).

---

## Step 1: Migrate first feature (SpeedAngvel) as a proof of concept

Apply the migration pattern to `speed_angvel.py`. Remove the old methods and add the new ones.

## Step 2: Run tests

Run: `cd mosaic && uv run pytest tests/ -x -q`
Expected: All pass

## Step 3: Migrate remaining 9 stateless features

Apply the same pattern to each feature. Each feature's `apply()` method adapts its existing `transform()` logic.

## Step 4: Run tests after all 10 migrations

Run: `cd mosaic && uv run pytest tests/ -x -q`
Expected: All pass

## Step 5: Commit

```
feat: migrate 10 stateless per-sequence features to new protocol
```
