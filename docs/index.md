# Mosaic

A Python toolkit for end-to-end animal behavior analysis: video processing, pose estimation, track standardization, behavioral feature extraction, model training, and visualization.

## Package structure

```
src/mosaic/
├── core/
│   ├── dataset.py       # Dataset orchestrator
│   ├── schema.py        # Track schema validation
│   ├── analysis.py      # Clustering metrics
│   └── helpers.py       # Label loading, safe name encoding
├── behavior/
│   ├── feature_library/ # 30+ registered feature implementations
│   ├── model_library/   # ML models (XGBoost behavior classifier)
│   ├── label_library/   # Label converters (BORIS, CalMS21)
│   └── visualization_library/  # Overlay, playback, egocentric crops, timeline plots
├── tracking/
│   └── pose_training/   # YOLO pose, POLO point-detection, localizer heatmap training
│       └── converters/  # CVAT, Lightning Pose, COCO format converters
└── media/
    ├── video_io.py      # Video I/O with raw H.264 fallback
    ├── extraction.py    # Frame extraction (uniform / k-means)
    └── sampling.py      # Frame selection algorithms
```

## High-level workflow

Data flows through the `Dataset` orchestrator (everything is versioned by `run_id`):

```
video files
   │
   ├─ index_media()          → media/index.csv  (ffprobe metadata)
   ├─ extract_frames()       → media/frames/    (uniform or k-means sampled PNGs)
   │
raw tracks/labels
   │
   ├─ index_tracks_raw()     → tracks_raw/index.csv
   ├─ convert_all_tracks()   → tracks/<group>__<seq>.parquet  (standardized)
   ├─ convert_all_labels()   → labels/<kind>/<group>__<seq>.npz
   │
   ├─ run_feature()          → features/<name>/<run_id>/*.parquet
   │     per-frame: speed-angvel, pair-egocentric, nearest-neighbor, ...
   │     spectral:  pair-wavelet
   │     reduction: pairposedistancepca (PCA)
   │     context:   temporal-stacking
   │     global:    global-tsne, global-kmeans, global-ward
   │     external:  kpms-fit / kpms-apply (keypoint-moseq)
   │
   ├─ train_model()          → models/<name>/<run_id>/
   │
   └─ run_feature(xgboost)   → apply trained model back as a feature
```

## Core concepts

### Dataset

[`mosaic.core.Dataset`](api/core/dataset.md) is the central orchestrator. It manages named roots (`media`, `tracks_raw`, `tracks`, `labels`, `features`, `models`) and provides methods for every pipeline stage. Manifests are YAML or JSON.

```python
from mosaic.core.dataset import Dataset

ds = Dataset(manifest_path="path/to/dataset.yaml")
ds.load()
```

### Features

Plugin-based via `@register_feature`. Each feature implements a standard protocol (`name`, `version`, `params`, `fit()`, `apply()`). Features are organized by output type:

| Category | Examples |
|----------|----------|
| Per-frame kinematic | speed-angvel, body-scale, orientation-relative |
| Per-frame spatial | pair-egocentric, pair-position, approach-avoidance |
| Per-frame social | nearestneighbor, ffgroups, ffgroups-metrics |
| Spectral | pair-wavelet |
| Reduction | pairposedistancepca |
| Context | temporal-stacking |
| Global embed/cluster | global-tsne, global-kmeans, global-ward |
| External | kpms-fit, kpms-apply (keypoint-moseq) |
| Model prediction | xgboost-feature |

See the [Feature Library API](api/behavior/feature-library.md) for details.

### Run IDs and reproducibility

Every feature/model run is tagged with a hash-based `run_id` (`<version>-<SHA1(params)>`). Outputs live under `features/<name>/<run_id>/` or `models/<name>/<run_id>/`. Feature params are captured in `params.json`.

## Installation

```bash
conda create -n mosaic python=3.12 -y
conda activate mosaic
conda install -c conda-forge ffmpeg -y
pip install -e ".[all]"
```

Optional extras (install only what you need):

| Extra | Install | What it adds |
|-------|---------|-------------|
| `wavelets` | `pip install -e ".[wavelets]"` | PyWavelets for wavelet features |
| `pose` | `pip install -e ".[pose]"` | Ultralytics YOLO pose training & inference |
| `polo` | `pip install -e ".[polo]"` | POLO point-detection models |
| `localizer` | `pip install -e ".[localizer]"` | PyTorch localizer heatmap training |
| `all` | `pip install -e ".[all]"` | wavelets |
