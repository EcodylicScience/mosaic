# Mosaic

A Python toolkit for end-to-end animal behavior analysis: video processing,
pose estimation, track standardization, behavioral feature extraction, model
training, and visualization.

For an introduction aimed at researchers, see the [project README on
GitHub](https://github.com/EcodylicScience/mosaic#readme). This site is the
developer-facing reference: how the pipeline is structured, how to add
features, and the API.

## Package structure

```
src/mosaic/
├── core/
│   ├── dataset.py       # Dataset orchestrator
│   ├── pipeline/        # Feature execution engine + typed protocol
│   ├── schema.py        # Track schema validation
│   ├── analysis.py      # Clustering metrics
│   ├── helpers.py       # Label loading, safe name encoding
│   └── track_library/   # Track format converters (CalMS21, MABe22, TREx)
├── behavior/
│   ├── feature_library/ # ~30 registered feature implementations
│   ├── model_library/   # Legacy — being phased out (see repo summary)
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
   └─ run_feature()          → features/<name>/<run_id>/*.parquet
         per-frame:    speed-angvel, pair-egocentric, nearest-neighbor, ...
         spectral:     pair-wavelet
         reduction:    pair-posedistance-pca
         context:      temporal-stack
         templates:    extract-templates, extract-labeled-templates
         global:       global-scaler, global-tsne, global-kmeans, global-ward
         trainable:    xgboost, arhmm, feral, kpms, lightning-action,
                       global-identity-model
         visualization: egocentric-crop, viz-timeline, viz-global-colored,
                        interaction-crop-pipeline
```

## Core concepts

### Dataset

[`mosaic.core.Dataset`](api/core/dataset.md) is the central orchestrator. It manages named roots (`media`, `tracks_raw`, `tracks`, `labels`, `features`, `models`) and provides methods for every pipeline stage. Manifests are YAML or JSON.

```python
from mosaic.core.dataset import Dataset

ds = Dataset(manifest_path="path/to/dataset.yaml")
ds.load()
```

### Features and global features

Everything in mosaic is a feature, registered via `@register_feature`. Each feature implements a 4-method protocol (`load_state`, `fit`, `apply`, `save_state`) plus a few attributes (`name`, `version`, `parallelizable`, `scope_dependent`).

Two flavors:

- **Per-frame / per-sequence features** — stateless transforms of tracks or upstream feature output.
- **Global features** — trainable, fit-then-apply features that learn from a collection of sequences (or labeled examples) and then map per-sequence. This is where mosaic's "models" live.

| Category | Examples |
|----------|----------|
| Per-frame kinematic | speed-angvel, body-scale, orientation-rel |
| Per-frame spatial | pair-egocentric, pair-position, approach-avoidance |
| Per-frame social | nearest-neighbor, ffgroups, ffgroups-metrics, nn-delta-response, nn-delta-bins |
| Track preprocessing | trajectory-smooth, movement-smooth, movement-filter-interpolate, pair-interaction-filter, id-tag-columns |
| Spectral | pair-wavelet |
| Reduction | pair-posedistance-pca |
| Context | temporal-stack |
| Templates | extract-templates, extract-labeled-templates |
| **Global (trainable)** | global-scaler, global-tsne, global-kmeans, global-ward, xgboost, arhmm, feral, kpms, lightning-action, global-identity-model |
| Visualization | egocentric-crop, viz-global-colored, viz-timeline, interaction-crop-pipeline |

See the [Feature Library API](api/behavior/feature-library.md) for details.

### Run IDs and reproducibility

Every feature run is tagged with a hash-based `run_id` (`<version>-<SHA1(params)>`). Outputs live under `features/<name>/<run_id>/`. Feature params are captured in `params.json`. Identical params + identical inputs always produce the same `run_id`, so re-running is a no-op and parameter sweeps stay organized automatically.

## Installation

```bash
conda create -n mosaic python=3.12 -y
conda activate mosaic
conda install -c conda-forge ffmpeg -y
pip install -e ".[recommended]"
```

The `recommended` bundle covers the typical research workflow (wavelets +
YOLO pose + PyTorch localizer). See the [project
README](https://github.com/EcodylicScience/mosaic#installation) for the full
extras matrix and notes on the mutually exclusive `pose` / `polo` ultralytics
pins.
