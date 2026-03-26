# Getting Started

This guide walks through a typical mosaic workflow from raw data to feature extraction.

## Installation

```bash
conda create -n mosaic python=3.12 -y
conda activate mosaic
conda install -c conda-forge ffmpeg -y
pip install -e ".[all]"
```

`ffmpeg` provides `ffprobe`, used by media indexing and raw H.264 support.

## Create a dataset

A mosaic dataset is a directory with a YAML or JSON manifest pointing to
named roots for media, tracks, labels, features, and models.

```python
from mosaic.core.dataset import Dataset

ds = Dataset(manifest_path="my_project/dataset.yaml")
ds.set_root("media_raw", "my_project/videos")
ds.set_root("tracks_raw", "my_project/tracking_output")
ds.save()
```

## Index media

Scan directories for video files and collect metadata via ffprobe:

```python
ds.index_media(
    search_dirs=["my_project/videos"],
    extensions=(".mp4", ".avi", ".h264"),
)
```

This writes `media/index.csv` with columns for group, sequence, path,
width, height, fps, and codec.

## Extract frames

Sample representative frames for annotation or visualization:

```python
# Uniform sampling: evenly spaced frames
run_id = ds.extract_frames(n_frames=20, method="uniform")

# K-means sampling: visually diverse frames
run_id = ds.extract_frames(n_frames=20, method="kmeans")
```

## Index and convert tracks

```python
# Index raw tracking files
ds.index_tracks_raw(
    search_dirs=["my_project/tracking_output"],
    src_format="calms21_npy",
)

# Convert to standardized parquet format
ds.convert_all_tracks()
```

## Run features

Features are composable pipeline stages. Each produces per-sequence
parquet files versioned by `run_id`:

```python
from mosaic.behavior.feature_library import (
    SpeedAngvel, PairEgocentric, Inputs,
)

# Basic kinematic features (reads from tracks by default)
speed = SpeedAngvel()
speed_result = ds.run_feature(speed)

# Pair-egocentric features
ego = PairEgocentric()
ego_result = ds.run_feature(ego)
```

### Chain features together

```python
from mosaic.behavior.feature_library import (
    ExtractTemplates, GlobalScaler, GlobalTSNE,
)

# 1. Extract templates from upstream features
templates = ExtractTemplates(
    Inputs((speed_result, ego_result)),
    params={"n_templates": 2000},
)
templates_result = ds.run_feature(templates)

# 2. Fit scaler, then t-SNE
scaler = GlobalScaler(
    Inputs((speed_result, ego_result)),
    params={"templates": ExtractTemplates.TemplatesArtifact().from_result(templates_result)},
)
scaler_result = ds.run_feature(scaler)
```

## Next steps

- See the [Pipeline Architecture](pipeline.md) guide for the full composable pipeline design
- See the [Migration Guide](migration-guide.md) if upgrading from the older API
- Browse the [API Reference](api/core/dataset.md) for full method documentation
