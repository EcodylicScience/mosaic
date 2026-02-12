# behavior

Tools for building end-to-end behavioral analysis pipelines: index and standardize tracks/labels, compute reusable features, bundle inputs into inputsets, fit/predict models, and visualize/cluster results.

## Installation

`pyproject.toml` is the canonical Python dependency source for this package.

```bash
conda create -n behavior python=3.11 -y
conda activate behavior
conda install -c conda-forge ffmpeg -y
pip install -e ".[all]"
```

Notes:
- `ffmpeg` provides `ffprobe`, which is used by dataset media indexing.
- If you want a minimal install, use `pip install -e .` instead of `.[all]`.

## High-level workflow

Data flows through a few core concepts (everything is versioned by `run_id`):

```
raw tracks/labels
   │
   ├─ index & convert ➜ standardized tracks + labels (dataset roots: tracks/, labels/)
   │
   ├─ feature runs ➜ features/<feature>/<run_id>/*.parquet
   │         (e.g., pose distance PCA, egocentric kinematics, wavelets, temporal stacks)
   │
   ├─ inputsets ➜ JSON pointers to aligned feature runs (for multi-input models/embeddings)
   │
   ├─ embeddings/cluster ➜ global-tsne, kmeans/ward outputs under features/
   │
   └─ models ➜ models/<model>/<run_id>/ (models.joblib, config.json, metrics)
              └─ predictions via ModelPredictFeature ➜ features/<model>-pred__from__...
```

### Core building blocks
- **Dataset manifest** (`behavior.dataset.Dataset`): manages roots (`tracks_raw`, `tracks`, `labels`, `features`, `models`, `media`), indexing, and path remapping.
- **Features** (`behavior.features.*`): transform per-sequence data into numeric matrices (e.g., `PairPoseDistancePCA`, `PairEgocentricFeatures`, `PairWavelet`, `TemporalStackingFeature`, `GlobalTSNE`, clustering features).
- **Inputsets**: saved bundles of feature specs for multi-input consumers (embeddings or models).
- **Models** (`behavior.models_behavior.BehaviorXGBoostModel`, etc.): train/eval on features or inputsets; `ModelPredictFeature` applies trained models back to sequences.
- **Labels**: standardized arrays (e.g., behavior class IDs) aligned to sequences/frames for training and evaluation.

## Repository layout
- `src/behavior/` — library code (dataset management, features, models, analysis, helpers)
- `feature_library/` — prebuilt feature configs
- `notebooks/` — worked examples; **start here:** `notebooks/calms21-template.ipynb`
- `dev-prev/` — scratch notebooks and earlier experiments
- `outputs/` — example or cached outputs (optional)

## Quick start (see the template notebook for details)
1. Create/load a dataset manifest pointing to your data roots.
2. Index/convert raw tracks and labels.
3. Run core features (pose distance, egocentric, wavelets).
4. Build an inputset if combining features.
5. (Optional) Global embedding + clustering (t-SNE, KMeans, Ward).
6. Temporal stack + train a model (feature or inputset).
7. Predict with `ModelPredictFeature`; visualize/overlay as needed.

The notebook `notebooks/calms21-template.ipynb` walks through a complete CalMS21 example with inline guidance, run_id tracking, and smoke-test slices you can remove for full runs.

## Notes on run_ids and reproducibility
- Every feature/model run is tagged with a hash-based `run_id`; outputs live under `features/<name>/<run_id>` or `models/<name>/<run_id>`.
- Keep track of printed run_ids between steps (e.g., temporal stack ➜ model train ➜ predict).
- Feature params are captured in `params.json`; model configs/metrics in `config.json`, `summary.csv`, `models.joblib`.

## Documentation & support
- Example pipeline: `notebooks/calms21-template.ipynb` (best reference for now).
- Feature and model docstrings live alongside code in `src/behavior/`.
- If you add new features/models, keep column names and input signatures stored in run artifacts to simplify downstream prediction.
