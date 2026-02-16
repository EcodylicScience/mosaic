# mosaic

A Python toolkit for end-to-end animal behavior analysis: video processing, pose estimation, track standardization, behavioral feature extraction, model training, and visualization.

## Installation

`pyproject.toml` is the canonical dependency source.

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
| `remote` | `pip install -e ".[remote]"` | SSH/Jupyter remote execution support |
| `all` | `pip install -e ".[all]"` | wavelets + remote |

Notes:
- `ffmpeg` provides `ffprobe`, used by media indexing and raw H.264 support.
- Pose/POLO/localizer extras are separate because they pull in large GPU dependencies.

## Package structure

```
src/mosaic/
├── core/
│   ├── dataset.py       # Dataset orchestrator (~6k lines)
│   ├── schema.py        # Track schema validation
│   ├── analysis.py      # Clustering metrics
│   └── helpers.py       # Label loading, safe name encoding
├── behavior/
│   ├── feature_library/ # 25+ registered feature implementations
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
   ├─ index_media()          ➜ media/index.csv  (ffprobe metadata)
   ├─ extract_frames()       ➜ media/frames/    (uniform or k-means sampled PNGs)
   │
raw tracks/labels
   │
   ├─ index_tracks_raw()     ➜ tracks_raw/index.csv
   ├─ convert_all_tracks()   ➜ tracks/<group>__<seq>.parquet  (standardized)
   ├─ convert_all_labels()   ➜ labels/<kind>/<group>__<seq>.npz
   │
   ├─ run_feature()          ➜ features/<name>/<run_id>/*.parquet
   │     per-frame: speed-angvel, pair-egocentric, nearest-neighbor, ...
   │     spectral:  pair-wavelet
   │     reduction: pairposedistancepca (PCA)
   │     context:   temporal-stacking
   │     global:    global-tsne, global-kmeans, global-ward
   │     external:  kpms-fit / kpms-apply (keypoint-moseq)
   │
   ├─ run_model()            ➜ models/<name>/<run_id>/
   │     BehaviorXGBoostModel (one-vs-rest XGBoost classifier)
   │
   └─ run_feature(model-predict) ➜ apply trained model back as a feature
```

## Core concepts

### Dataset

`mosaic.core.Dataset` is the central orchestrator. It manages named roots (`media`, `tracks_raw`, `tracks`, `labels`, `features`, `models`) and provides methods for every pipeline stage. Manifests are YAML or JSON.

```python
from mosaic.core.dataset import Dataset

ds = Dataset(manifest_path="path/to/dataset.yaml")
ds.load()
```

### Features

Plugin-based via `@register_feature`. Each feature implements a standard protocol (`name`, `version`, `params`, `transform()`, optional `fit()`). Features are organized by output type:

| Category | Examples |
|----------|----------|
| Per-frame kinematic | speed-angvel, body-scale, orientation-relative |
| Per-frame spatial | pair-egocentric, pair-position, approach-avoidance |
| Per-frame social | nearestneighbor, ffgroups, ffgroups-metrics, nn-delta-response, nn-delta-bins |
| Spectral | pair-wavelet |
| Reduction | pairposedistancepca |
| Context | temporal-stacking |
| Global embed/cluster | global-tsne, global-kmeans, global-ward, ward-assign |
| External | kpms-fit, kpms-apply (keypoint-moseq via subprocess) |
| Model prediction | model-predict |
| Visualization | egocentric-crop, viz-global-colored, viz-timeline |

### Labels

Standardized NPZ format aligned to sequences/frames. Converters for CalMS21, BORIS aggregated CSV/TSV, and BORIS Pandas pickle. Plugin architecture via `@register_label_converter`.

### Models

`BehaviorXGBoostModel`: one-vs-rest XGBoost classifier with SMOTE balancing, PCA reduction, and full metrics. Supports single-feature or multi-input (inputset) training.

### Media & frame extraction

`Dataset.index_media()` scans directories and collects ffprobe metadata. `Dataset.extract_frames()` samples representative frames per video using uniform spacing or k-means diversity selection. The `video_io` module handles raw H.264 elementary streams (common from Raspberry Pi cameras) where OpenCV metadata is unreliable.

### Pose training

`mosaic.tracking.pose_training` provides an end-to-end pipeline for training custom pose estimation models:
- **Converters**: CVAT XML, Lightning Pose, COCO keypoints/points/localizer formats
- **Training**: YOLO pose, POLO point-detection, PyTorch localizer heatmap models
- **Inference**: evaluate trained models on video, export to DataFrames
- **Prep**: dataset splitting, label filtering/simplification, data.yaml generation

### Visualization

`mosaic.behavior.visualization_library` provides modular components for reviewing results:
- Track + label overlay on video frames
- Interactive video playback
- Egocentric crop generation
- Global embedding colored scatter plots
- Timeline plots

### Remote execution

For GPU-intensive operations (features, model training), the dataset can sync to a remote machine via SSH/rsync and execute jobs through SSH or Jupyter kernels:

```python
remote_cfg = {
    "ssh_host": "user@gpu-server",
    "local_root": "/local/path",
    "remote_root": "/remote/path",
}
ds.sync_to_remote(remote_cfg)
ds.run_feature_remote("mosaic.behavior.feature_library.global_tsne.GlobalTSNE", params, remote_cfg)
ds.sync_from_remote(remote_cfg)
```

## Run IDs and reproducibility

Every feature/model run is tagged with a hash-based `run_id` (`<version>-<SHA1(params)>`). Outputs live under `features/<name>/<run_id>/` or `models/<name>/<run_id>/`. Feature params are captured in `params.json`; model configs/metrics in `config.json`, `summary.csv`, `models.joblib`.

## Repository layout

- `src/mosaic/` — library code
- `notebooks/` — worked examples; start with `notebooks/calms21-template.ipynb`
- `sync-to-remote.ipynb` — rsync helper for pushing the package to a remote server
