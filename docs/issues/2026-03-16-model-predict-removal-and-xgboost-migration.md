# ModelPredictFeature removal and XGBoost-as-feature migration

## Background

`ModelPredictFeature` is a generic wrapper in `feature_library/model_predict.py` that dynamically imports an external model class, loads a trained model bundle from disk, and delegates per-sequence prediction to `model.predict_sequence(df, meta)`.

The only model that implements this interface is `BehaviorXGBoostModel` in `model_library/behavior_xgboost.py`. No other model class exists in the codebase.

## Why ModelPredictFeature was removed

Under the old protocol, the pipeline distinguished between "features" (data transforms) and "models" (trained classifiers). Features lived in `feature_library/`, models in `model_library/`. `ModelPredictFeature` bridged the two: it was a feature that loaded a model and called its `predict_sequence()` method.

The new protocol unifies this. Every computation that takes input data and produces output data is a feature with `fit()` and `apply()`. A trained classifier is just a feature whose `fit()` trains the classifier and whose `apply()` runs inference. There is no separate "model" concept at the pipeline level.

With this unification, `ModelPredictFeature` becomes pure indirection:

- It dynamically imports a class and calls two methods (`load_trained_model`, `predict_sequence`). A proper feature class does the same thing statically with `load_state` and `apply`.
- It adds a layer of generic parameter passing (`model_class`, `model_params`, `model_run_id`) that obscures the actual configuration. A proper feature class has typed `Params`.
- It cannot validate input columns because it does not know what the model expects. A proper feature class stores `feature_columns` in its model bundle and validates in `apply()`.

## What ModelPredictFeature did

```
__init__(inputs, params)
    params: model_class, model_params, model_run_id, model_name, output_feature_name

bind_dataset(ds)
    1. Import model_class dynamically
    2. Instantiate with model_params
    3. Resolve model path via model_run_root(ds, model_name, run_id)
    4. Call model.load_trained_model(run_root)

transform(df)
    1. Extract sequence/group metadata from df
    2. Call model.predict_sequence(df, meta)
    3. Attach sequence/group/model_run_id columns to result
```

Stateless (noop `fit`), scope-independent, parallelizable.

## Target pipeline

```
features (speed-angvel, etc.)
  -> GlobalScaler
    -> ExtractLabeledTemplates (subsampling, splits, labels)
      -> XgboostFeature (train + per-sequence inference)
```

Each step does exactly one thing. No step knows about the internals of another. Data flows through standard artifacts (parquet templates, joblib model bundles). Scaling, sampling, labeling, and classification are fully decoupled.

## XGBoost as a feature

`BehaviorXGBoostModel` should be migrated to a proper feature class in `feature_library/`. This is not a thin wrapper -- it is a substantial rewrite because the current class reimplements its own data loading, train/test splitting, and evaluation pipeline.

### Current BehaviorXGBoostModel structure

- `__init__`: raw dict params with manual defaults
- `bind_dataset`: stores dataset reference
- `train_model(ds, config, run_root)`: the full training pipeline
  - Resolves input features from dataset (own data loading)
  - Loads labels via `load_labels_for_feature_frames`
  - Splits train/test by sequence
  - Applies undersampling/SMOTE rebalancing
  - Fits StandardScaler on train split
  - Trains one-vs-rest XGBClassifier per class
  - Evaluates on test split
  - Saves bundle: models, scaler, classes, feature_columns, label_map, config, metrics
- `load_trained_model(run_root)`: loads the bundle
- `predict_sequence(df, meta)`: per-sequence inference
  - `_prepare_features_from_df`: column alignment with zero-fill/truncation
  - Applies scaler
  - Runs each per-class model
  - Stacks probabilities, applies threshold, assigns labels

### Migrated XgboostFeature structure

```
feature_library/xgboost.py

class XgboostFeature:
    name = "xgboost"
    scope_dependent = False
    parallelizable = True

    class Inputs(Inputs[Result]):
        _require = "nonempty"

    class Params(GlobalModelParams[XgboostModelArtifact]):
        model: XgboostModelArtifact | None = ...
        # templates inherited from GlobalModelParams (mutually exclusive with model)
        # XGBoost hyperparams (from xgb_params_preset + overrides)
        decision_threshold: float = 0.5
        # Optional SMOTE rebalancing during fit
        use_smote: bool = False
        undersample_ratio: float = 3.0

    load_state(run_root, artifact_paths, dependency_indices):
        # Branch 1: cached model in run_root
        # Branch 2: pre-fitted model from artifact_paths["model"]
        # Branch 3: labeled templates from artifact_paths["templates"]
        #   -> return False (fit required)
        # Bundle contains: models, scaler, classes, feature_columns, label_map

    fit(inputs):
        # Templates artifact has feature columns + label column + split column
        # Train on "train" split rows, evaluate on "test" split rows (if present)
        # Optional SMOTE on train split
        # Train one-vs-rest XGBClassifier per class
        # (no scaler -- GlobalScaler is upstream of ExtractLabeledTemplates)

    apply(df):
        # Validate columns via ensure_columns(df, self._feature_columns)
        # Run inference, assign labels (no scaler -- data arrives pre-scaled)
        # Return DataFrame with frame, per-class probabilities, predicted label

    save_state(run_root):
        # Save model bundle via joblib
        # If test split was present: save evaluation metrics (summary.csv, reports.json)
```

### What changes

| Aspect | Current | Migrated |
|--------|---------|----------|
| Data loading | Own `_resolve_inputs` + index parsing | Labeled templates artifact |
| Labels | `load_labels_for_feature_frames` on dataset | Embedded in templates artifact (label column) |
| Train/test split | Manual sequence-level split in XGBoost | Split column in templates artifact, produced by ExtractLabeledTemplates |
| Scaling | StandardScaler inside XGBoost | GlobalScaler upstream of ExtractLabeledTemplates |
| Rebalancing | undersample + SMOTE in XGBoost | Coarse balance in ExtractLabeledTemplates, optional SMOTE in XGBoost fit |
| Column validation | Loose (zero-fill, truncate) | Strict `ensure_columns` from bundle |
| Model bundle | `models.joblib` with scaler in model_library path | `xgboost_model.joblib` (no scaler) in feature run_root |
| Evaluation metrics | Always saved during training | Conditional: saved only when test split present |
| External memory | `use_external_memory` for huge datasets | Dropped (bounded templates make this unnecessary) |
| Config | Raw dict | Typed Pydantic `Params` |

### Resolved design decisions

1. **Label source.** Labels come embedded in the labeled templates artifact. No separate `LabelsSource` dependency needed for XGBoost. Label awareness lives in `ExtractLabeledTemplates`, which handles alignment of sparse event-based labels (NPZ format) to feature frames.

2. **Train/test split.** Handled by `ExtractLabeledTemplates`, which produces a "split" column in the templates artifact. Splits are sequence-level, controlled via contribution weights (each split gets `contribution / sum(contributions)` of sequences) and a random seed. XGBoost just reads the split column. Evaluation metrics are a conditional artifact -- only produced when at least two splits are present.

3. **Rebalancing.** Two levels:
   - `ExtractLabeledTemplates` handles coarse balance via stratified subsampling. Two strategies: (a) proportional per-bout contribution, where each bout of consecutive labels contributes proportionally; (b) per-bout equal division, where requested samples per category are divided equally across bouts, with remainder redistributed to longer bouts. Params include requested samples per class or "balanced" mode.
   - XGBoost handles fine-grained rebalancing: optional SMOTE and `scale_pos_weight` per one-vs-rest classifier. This is model-specific training logic.

4. **Evaluation output.** Conditional on test split presence. `save_state()` writes summary.csv and reports.json alongside the model bundle when test-split evaluation was performed.

5. **External memory mode.** Dropped. Proper subsampling in `ExtractLabeledTemplates` produces bounded template sets, making external memory unnecessary.

### New dependency: ExtractLabeledTemplates

A new feature that produces labeled templates artifacts. Separate design doc.

Key responsibilities:
- Load upstream feature data + labels (via `LabelsSource` dependency)
- Stratified or per-bout subsampling with configurable samples per class
- Sequence-level train/test splits via contribution weights
- Output: templates parquet with feature columns, label column, split column

## model_library becomes obsolete

Once XGBoost is a feature, `model_library/` has no remaining contents:

- `behavior_xgboost.py` -> `feature_library/xgboost.py`
- `helpers.py` (XGB_PARAM_PRESETS, undersample_then_smote, to_jsonable) -> `feature_library/` or shared utils
- `__init__.py` -> deleted

The distinction "features extract, models predict" collapses into "features transform data". A classifier is a feature whose `apply()` produces label columns instead of numeric columns. The pipeline treats both identically.

Related pipeline code to clean up:
- `core/pipeline/models.py` (`model_run_root`, `model_index_path`) -- no longer needed once XGBoost uses feature run roots
- `ModelPredictFeature` references in `feature_library/__init__.py`

## Scope

- ModelPredictFeature deletion: immediate (task-08 step 3)
- ExtractLabeledTemplates: new feature, separate design doc and task
- XGBoost feature migration: separate task, depends on ExtractLabeledTemplates
- model_library removal: after XGBoost migration
