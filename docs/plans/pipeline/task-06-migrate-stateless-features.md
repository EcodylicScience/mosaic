# Task 6: Migrate stateless per-sequence features

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Convert all 10 per-sequence features to the new protocol.

**Phase:** C (Protocol Transition -- clean break, all Phase C tasks land together)

**Parent plan:** `docs/plans/2026-03-12-pipeline-unification-implementation.md`

**Assessment:** `docs/plans/pipeline/task-06-assessment.md`

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

## Migration pattern

For each feature:

1. Remove: `output_type`, `storage_feature_name`, `storage_use_input_suffix`, `skip_existing_outputs`
2. Add: `scope_dependent = False`
3. Remove methods: `needs_fit()`, `supports_partial_fit()`, `fit()`, `partial_fit()`, `finalize_fit()`, `save_model()`, `load_model()`, `bind_dataset()`, `set_scope()`
4. Remove dead state: `self._ds`, `self._scope`
5. Add stub methods:
   - `load_state(self, run_root, dependency_paths) -> True` (always skip fit)
   - `fit(self, inputs)` -- empty (never called)
   - `save_state(self, run_root)` -- empty
6. Rename `transform(self, df)` to `apply(self, df: pd.DataFrame) -> pd.DataFrame`
   - The signature is identical -- `apply()` receives a DataFrame and returns a DataFrame
   - The pipeline unpacks `EntryData` and passes `entry_data.df` to the feature
   - The feature never sees `entry_key` or `entity_level`
7. Clean up deprecated typing imports (`List` -> `list`, `Tuple` -> `tuple`, `Optional` -> `| None`, `Any` -> concrete type, dead `Iterable` imports)

### Special cases

**OrientationRelativeFeature** has a `bind_dataset()` that loads body-scale data. Replace with a typed `BodyScaleResult` Params field. `load_state()` reads body-scale outputs from `dependency_paths["scale"]` and builds `_scale_lookup`. See Step 1b.

**IdTagColumns** has a `bind_dataset()` that loads id-tag labels. Replace with a `LabelsSource` Params field. `load_state()` reads labels from `dependency_paths["labels"]` and builds `_labels`. See Step 1b.

Both contracts are resolved by `_resolve_dependency_paths` (Task 9). Until Task 9 lands, `load_state()` implementations are inert -- same as all other migrated features.

---

## Step 0: Simplify EntryData to DataFrame-only

Replace the 5 numpy-oriented fields with `(df, entity_level)`:

```python
# Before (loading.py)
@dataclass(frozen=True, slots=True)
class EntryData:
    features: np.ndarray
    frames: np.ndarray
    id1: np.ndarray | None
    id2: np.ndarray | None
    entity_level: str

# After
@dataclass(frozen=True, slots=True)
class EntryData:
    df: pd.DataFrame
    entity_level: str
```

Simplify `_entrydata_from_merged()`: derive `entity_level` via `_normalize_identity_columns(df)`, return `EntryData(df=df, entity_level=entity_level)`. Remove all `to_numpy(copy=True)` extraction.

Update `_TestFeature.apply()` in `tests/test_inputs_subclass.py`: change signature to `apply(self, df)`, use `len(df)`.

Run tests to verify baseline.

---

## Step 1: Protocol and type additions

### 1a: Simplify `apply()` signature in `types.py`

```python
# Before
def apply(self, entry_key: str, entry_data: EntryData) -> pd.DataFrame: ...

# After
def apply(self, df: pd.DataFrame) -> pd.DataFrame: ...
```

### 1b: Add dependency types to `types.py`

```python
class BodyScaleResult(Result[Literal["body-scale"]]):
    """Result narrowed to the body-scale feature."""
    feature: Literal["body-scale"] = "body-scale"

class LabelsSource(DictModel):
    """Base class for dataset label dependencies."""
    kind: str
```

`GroundTruthLabelsSource` should subclass `LabelsSource` (existing class, update inheritance).

Run tests to verify.

---

## Step 2: Extract shared helpers to `helpers.py`

### `clean_animal_track()`

Duplicated across PairPositionFeatures, PairEgocentricFeatures, ApproachAvoidance.

```python
def clean_animal_track(
    g: pd.DataFrame,
    data_cols: list[str],
    order_col: str,
    config: InterpolationConfig,
) -> pd.DataFrame:
    """Sort, interpolate, fill, and drop rows with excessive missing data."""
```

### Math helpers

| Helper | Currently in |
|--------|-------------|
| `smooth_1d()` | pair_position, pair_egocentric |
| `unwrap_diff()` | pair_position, pair_egocentric |
| `wrap_angle()` | nearestneighbor, nn_delta_response |
| `ego_rotate()` | nearestneighbor, nn_delta_response |

Extract all to `helpers.py`. Update imports in source features. Run tests to verify -- the helpers are already called, just moved.

---

## Step 3: Migrate SpeedAngvel (proof of concept)

Apply the migration pattern. This is the simplest feature -- no dependencies, no special cases.

```python
# Before
def transform(self, df: pd.DataFrame) -> pd.DataFrame:
    ...

# After
def apply(self, df: pd.DataFrame) -> pd.DataFrame:
    ...  # same body
```

Run tests to verify.

---

## Step 4: Migrate remaining 9 features

Apply the migration pattern to each:

1. **BodyScaleFeature** -- straightforward
2. **OrientationRelativeFeature** -- replace `scale_feature`/`scale_run_id` with `BodyScaleResult` Params field, move `_load_scales()` logic into `load_state()`
3. **PairPositionFeatures** -- replace `_clean_one_animal()` with `clean_animal_track()` import
4. **PairEgocentricFeatures** -- replace `_clean_one_animal()`, `_smooth_1d()`, `_unwrap_diff()` with imports
5. **NearestNeighbor** -- replace `_wrap_angle()`, `_ego_rotate()` with imports
6. **NearestNeighborDelta** -- replace `_wrap_angle()`, `_ego_rotate()` with imports
7. **NearestNeighborDeltaBins** -- straightforward
8. **IdTagColumns** -- add `LabelsSource` Params field, move label loading into `load_state()`
9. **ApproachAvoidance** -- replace `_clean_one_animal()` with `clean_animal_track()` import

Run tests after all 9 to verify.

---

## Files to modify

| File | Changes |
|------|---------|
| `src/mosaic/core/pipeline/loading.py` | Replace `EntryData` fields with `(df, entity_level)`; simplify `_entrydata_from_merged()` |
| `src/mosaic/core/pipeline/types.py` | Add `BodyScaleResult`, `LabelsSource`; simplify `apply()` signature |
| `tests/test_inputs_subclass.py` | Update `_TestFeature.apply(self, df)` signature |
| `src/mosaic/behavior/feature_library/helpers.py` | Extract `clean_animal_track`, `smooth_1d`, `unwrap_diff`, `wrap_angle`, `ego_rotate` |
| `src/mosaic/behavior/feature_library/speed_angvel.py` | Full migration |
| `src/mosaic/behavior/feature_library/body_scale.py` | Full migration |
| `src/mosaic/behavior/feature_library/orientation_relative.py` | Full migration + `BodyScaleResult` Params field |
| `src/mosaic/behavior/feature_library/pair_position.py` | Full migration + use shared helpers |
| `src/mosaic/behavior/feature_library/pair_egocentric.py` | Full migration + use shared helpers |
| `src/mosaic/behavior/feature_library/nearestneighbor.py` | Full migration + use shared helpers |
| `src/mosaic/behavior/feature_library/nn_delta_response.py` | Full migration + use shared helpers |
| `src/mosaic/behavior/feature_library/nn_delta_bins.py` | Full migration |
| `src/mosaic/behavior/feature_library/id_tag_columns.py` | Full migration + `LabelsSource` Params field |
| `src/mosaic/behavior/feature_library/approach_avoidance.py` | Full migration + use shared helpers |

---

## Commit

```
feat: migrate 10 stateless per-sequence features to new protocol
```
