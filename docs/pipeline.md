# Pipeline Migration

Summary of the pipeline migration. Replaces the original feature execution infrastructure with a composable pipeline. Work spanned multiple branches; the `pipeline` branch contains the bulk of it (193 commits, 115 files, +24k/-15k lines).

## Motivation

The original architecture served the project well during early development but accumulated structural constraints as the feature library grew:

1. **Output handling.** GlobalTSNE, WardAssign, and GlobalKMeans managed their own output writing during `fit()`/`save_model()`, separate from the pipeline's standard output path. This made it harder to reason about where outputs lived and to add cross-cutting concerns (overlap trimming, index updates).

2. **Two data-loading paths.** `iteration.py:yield_input_data()` and `StreamingFeatureHelper.load_key_data()` evolved independently to serve different feature types. Having two paths for the same job meant changes had to be synchronized across both.

3. **Large Feature protocol surface.** The protocol accumulated 12+ methods (`fit`, `transform`, `needs_fit`, `partial_fit`, `finalize_fit`, `set_run_root`, `bind_dataset`, `loads_own_data`, ...) as features with different lifecycle needs were added. This made implementing new features harder than necessary.

4. **Module placement.** Protocol types (`Feature`, `Params`, `Inputs`, `Result`, `LoadSpec`) lived in `behavior/feature_library/` but were used by the pipeline in `core/`. Moving them to `core/pipeline/` establishes a clear dependency direction.

## What changed

### Feature protocol: 4 methods + 4 attributes

```
load_state(run_root, artifact_paths, dependency_lookups) -> bool
fit(inputs: InputStream) -> None
apply(df: DataFrame) -> DataFrame
save_state(run_root) -> None
```

Class attributes: `name`, `version`, `parallelizable`, `scope_dependent`.

The pipeline owns all data loading, output writing, and index management. Features own computation only.

### Pipeline package: `core/pipeline/`

Extracted from `dataset.py` and `behavior/feature_library/`:

| Module | Responsibility |
|--------|----------------|
| `types/` | Feature protocol, Params, Inputs, Result, ArtifactSpec, InputStream, DependencyLookup |
| `run.py` | `run_feature()` orchestration, dependency resolution, parallel apply |
| `manifest.py` | Unified manifest builder + per-sequence iterator |
| `loading.py` | Sequence identity resolution, NN lookup construction |
| `index_csv.py` | Generic typed IndexCSV with dataclass rows |
| `writers.py` | Parquet output writing with overlap trimming |
| `_loaders.py` | LoadSpec dispatcher (NPZ, Parquet, Joblib) |
| `_utils.py` | Scope dataclass, param hashing |

### Unified manifest and iterator

Single code path replaces both `yield_input_data()` and `StreamingFeatureHelper`. Handles tracks, single-feature, and multi-feature inputs through one manifest builder with inner-join semantics across all inputs.

### Type system

- **Typed artifacts.** `ParquetArtifact`, `JoblibArtifact`, `NpzArtifact` with `from_path()` methods. Defined at module level per the Feature Artifact Pattern.
- **Typed index rows.** `FeatureIndexRow`, `FramesIndexRow` replace raw dicts.
- **`GlobalModelParams[M]`.** Generic base for features that fit on templates or load a pre-fitted model. Mutual exclusivity validation (`templates XOR model`).
- **Upstream dependency resolution** into `dict[tuple[str, str], Path]` (`DependencyLookup` type alias) for both `Result` and `LabelsSource` fields. Features receive ready-to-use `(group, sequence) -> Path` lookups.
- **`InputStream`.** Wraps the input factory and exposes `n_entries` so features can make exact allocation decisions (e.g. train/test split counts) without extra data passes.

### Per-sequence output standardization

GlobalTSNE, GlobalKMeans, and GlobalWard migrated from NPZ to standard per-frame Parquet output. `apply()` returns a DataFrame the pipeline writes.

### Composable global feature pipeline

GlobalTSNE decomposed into independent pipeline stages:

```
per-sequence features (wavelets, egocentric, etc.)
  -> ExtractTemplates (subsampling, scope-dependent)
    -> GlobalScaler (fit StandardScaler on templates, scale per-sequence)
      -> [ExtractTemplates] (optional second extraction on scaled data)
        -> GlobalTSNE (fit embedding, map per-sequence)
        -> GlobalKMeans / GlobalWard (fit clustering, assign per-sequence)
```

Each stage is a separate feature with its own run_id, cache, and artifact. `scope_dependent` features -- `ExtractTemplates`, `ExtractLabeledTemplates`, `PairPoseDistancePCA`, and `KpmsFeature` -- include the manifest entries in their run_id hash so that different scopes produce separate runs. All other features are scope-independent.

### XGBoost as a feature

`BehaviorXGBoostModel` (model_library) migrated to two feature classes:

```
scaled features
  -> ExtractLabeledTemplates (label alignment, train/test split, subsampling)
    -> XgboostFeature (train classifier, per-sequence inference)
```

**ExtractLabeledTemplates:**
- Streams upstream features, aligns ground truth labels from NPZ files via `dependency_lookups["labels"]`
- Exact train/test split assignment using `InputStream.n_entries` (guarantees at least 1 test sequence when `test_fraction > 0`)
- Per-(split, class) reservoir sampling with lazy class discovery
- `apply()` adds `label` and `split` columns to per-sequence outputs, enabling downstream filtering for held-out evaluation
- Farthest-first selection option for diverse template coverage

**XgboostFeature:**
- Multiclass (single `multi:softprob` model) or one-vs-rest (N binary models)
- Balanced class weights via `sample_weight` / `scale_pos_weight`
- Optional multiclass-aware undersampling + SMOTE
- Decision thresholds (global, per-class, or None) with `default_class` fallback
- Test split evaluation with `reports.json` + `summary.csv`

**Scope reduction vs BehaviorXGBoostModel:** The new features do not handle data loading, label loading, train/test splitting, or scaling internally -- those are handled by upstream pipeline stages. External memory mode and parameter presets are also dropped (bounded templates and direct hyperparameter passing replace them).

### Visualization features replaced by `load_values()`

`VizTimeline` and `VizGlobalColored` removed. Their role is now served by `load_values()` -- a free function that loads and aligns columns from tracks, features, and labels into a single DataFrame for notebook-level analysis.

### KPMS unification

`KpmsFit` + `KpmsApply` merged into single `KpmsFeature` with persistent subprocess server for the external keypoint-MoSeq process.

### Other changes

- `StreamingFeatureHelper` removed (replaced by unified manifest)
- `loads_own_data()` removed from protocol (replaced by `Inputs._require`)
- `group_safe`/`sequence_safe` columns removed from indexes
- `load_key_data()` replaced by manifest-based data loading

## What this enables

- **Composable pipelines.** Chain features with typed artifact references. Each stage cached independently.
- **`load_values()` for ad-hoc analysis.** Load any combination of feature columns, track columns, and ground truth labels into one DataFrame.
- **Train/test split as per-frame metadata.** `ExtractLabeledTemplates.apply()` writes `split` and `label` columns. Filter to test-only data for held-out evaluation in notebooks.
- **Exact split allocation.** `InputStream.n_entries` enables features to compute deterministic split assignments without extra data passes.
- **Pre-resolved dependencies.** `dependency_lookups` gives features ready-to-use `(group, sequence) -> Path` lookups.
- **Parallel apply.** `parallelizable=True` features run per-sequence apply in thread or process pools (existing capability, simplified in the rewrite).

## Open issues

### Feature output column declaration

`feature_columns()` uses a hardcoded `_EXTRA_META` set to distinguish metadata from feature output. Adding `label` and `split` required updating this set manually. A more robust solution would be explicit `output_columns` declarations per feature, validated at registration time. Tracked in `docs/issues/2026-03-19-feature-output-column-declaration.md`.

### `model_library/` not yet removed

`BehaviorXGBoostModel` still exists in `model_library/`. To be removed after `XgboostFeature` is validated on production datasets.

### Bout-aware sampling

`ExtractLabeledTemplates` uses proportional random sampling, which over-represents long stationary periods relative to short behavioral events. A bout-aware strategy would distribute sampling across behavioral episodes. Planned future enhancement (documented in code comment).

## User-facing API changes

See `notebooks/migration-guide.md` for the full migration guide. Key changes:

- `Inputs((result,))` replaces `input_kind`/`input_feature`/`input_run_id` kwargs
- `run_feature()` returns `Result` dataclass, not raw `run_id` string
- Typed artifact references: `Feature.Artifact().from_result(result)`
- Frame/time filtering moved to `run_feature()` parameters
- Pair filtering moved to feature `Params`
