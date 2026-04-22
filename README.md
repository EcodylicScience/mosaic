# mosaic

A Python toolkit for animal behavior analysis: track standardization, behavioral
feature extraction, unsupervised embedding and clustering, supervised classifier
training, and annotated video output.

**Documentation:** <https://ecodylicscience.github.io/mosaic/>

## Overview

Given pose tracks (per-frame keypoints with identities), mosaic produces:

- standardized parquet track tables from CalMS21, MABe22, TREx, or user-defined
  formats;
- per-frame and per-sequence behavioral features — kinematic, social, spectral,
  and reduction;
- unsupervised embeddings and clusters (t-SNE, k-means, Ward, ARHMM,
  [keypoint-MoSeq](https://keypoint-moseq.readthedocs.io/));
- supervised classifiers (XGBoost, Lightning-Action) trained from
  [BORIS](https://www.boris.unito.it) or other manual labels;
- visual identification models from egocentric crops;
- annotated overlay videos, embedding scatters, and behavior timelines.

If pose tracks are not yet available, the package also provides:

- frame sampling for annotation (uniform or k-means diversity);
- pose-model training from CVAT, COCO, or Lightning Pose annotations using
  YOLO pose, POLO point detection, or a PyTorch heatmap localizer.

## Installation

`pyproject.toml` is the canonical dependency source.

```bash
conda create -n mosaic python=3.12 -y
conda activate mosaic
conda install -c conda-forge ffmpeg -y
pip install -e ".[recommended]"
```

`ffmpeg` provides `ffprobe`, used by media indexing and raw H.264 support.

The `recommended` extra bundles wavelets, YOLO pose training/inference, and the
PyTorch localizer. For lighter or alternative installs, select extras
individually:

| Extra              | Adds                                                                                |
| ------------------ | ----------------------------------------------------------------------------------- |
| `recommended`      | `wavelets` + `pose` + `localizer`                                                   |
| `wavelets`         | PyWavelets for spectral features                                                    |
| `pose`             | Ultralytics YOLO pose training and inference                                        |
| `polo`             | POLO point detection (mutually exclusive with `pose`; different ultralytics fork)   |
| `localizer`        | PyTorch heatmap localizer training                                                  |
| `lightning-action` | Lightning-Action temporal action classifier                                         |
| `gpu`              | faiss for GPU-accelerated kNN in `global-tsne` (use `faiss-gpu` on Linux + CUDA)    |

## Quick start

The [CalMS21 template notebook](notebooks/calms21-template.ipynb) is the
canonical end-to-end example. It walks through:

1. building a `Dataset` from a manifest;
2. computing `pair-egocentric` and `pair-posedistance-pca` features;
3. wavelet expansion, global scaling, and t-SNE embedding;
4. k-means and Ward clustering with cluster-to-label agreement metrics;
5. supervised classification via `extract-labeled-templates` and XGBoost,
   with optional temporal-context stacking;
6. visualization of predictions on the embedding.

Additional notebooks for the MABe22 mouse-triplet and beetle-ant datasets are
available in [`notebooks/`](notebooks/).

## Concepts

Every transformation in mosaic is registered as a **feature** and executed
through a single `Dataset` orchestrator. Each feature implements a four-method
protocol (`load_state`, `fit`, `apply`, `save_state`) and declares a name,
version, and parallelizability.

Features are either:

- **per-frame / per-sequence** — stateless transforms of tracks or upstream
  feature output (e.g. `speed-angvel`, `pair-egocentric`, `nearest-neighbor`,
  `pair-wavelet`, `temporal-stack`, `body-scale`);
- **global** — fit-then-apply transforms trained on a collection of sequences
  (e.g. `global-scaler`, `global-tsne`, `global-kmeans`, `global-ward`,
  `xgboost`, `arhmm`, `kpms`, `lightning-action`, `global-identity-model`).

Visualization (`egocentric-crop`, `viz-timeline`, `viz-global-colored`,
`interaction-crop-pipeline`) is exposed as features and shares the same caching
and reproducibility machinery.

The full registry is documented in the
[feature library reference](docs/api/behavior/feature-library.md).

## Pipeline

`Dataset` manages named roots and produces deterministic, versioned outputs.
Each feature run is tagged with a `run_id` of the form
`<version>-<SHA1(params)>`; identical inputs and parameters resolve to the same
`run_id`, so re-runs are no-ops and parameter sweeps stay organized.

```
video files
   ├─ index_media()          → media/index.csv          (ffprobe metadata)
   └─ extract_frames()       → media/frames/            (uniform or k-means PNGs)

raw tracks/labels
   ├─ index_tracks_raw()     → tracks_raw/index.csv
   ├─ convert_all_tracks()   → tracks/<group>__<seq>.parquet
   └─ convert_all_labels()   → labels/<kind>/<group>__<seq>.npz

run_feature(...)             → features/<name>/<run_id>/*.parquet
```

## Repository layout

```
src/mosaic/
├── core/        # Dataset orchestrator, pipeline engine, schema, helpers
├── behavior/    # feature_library, label_library, visualization_library
├── tracking/    # pose-model training/inference and annotation converters
└── media/       # video I/O, frame extraction, sampling
```

## Status

Mosaic is in early development (0.x). Public APIs, feature names, and on-disk
layouts may change between releases.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Please open an issue before making
large changes.

## License

GNU Affero General Public License v3 or later (AGPLv3+). See
[LICENSE](LICENSE).
