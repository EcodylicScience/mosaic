# mosaic

**A Python toolkit for end-to-end animal behavior analysis.**

Mosaic is a toolkit to go from raw video and tracking output to interpretable behavioral
features, unsupervised embeddings, and trained behavior classifiers — in a
single, versioned pipeline. It can be used on mice, fish, bees, and other
animals across both lab and field setups.

If you already have tracks (poses, identities, trajectories), mosaic
standardizes them, extracts behavioral features (kinematic, social, spectral),
clusters them, lets you train classifiers from human labels, and renders
annotated overlay videos. If you don't have tracks yet, it can also help you
extract frames for annotation, train custom pose models, and run tracking.

---

## Example tasks with mosaic

- **Sample frames for annotation** — uniform or k-means diversity-maximizing
  selection across a video collection.
- **Train a custom pose model** from CVAT, COCO, or Lightning Pose annotations
  (YOLO pose, POLO point-detection, or PyTorch heatmap localizer).
- **Standardize tracks** from CalMS21, MABe22, TREx, or your own format into a
  single parquet schema.
- **Extract behavioral features** — speed/angular velocity, body scale,
  egocentric pose distances, nearest-neighbor relationships, social grouping
  (FFGroups), wavelet decompositions, PCA, and more.
- **Smooth and filter** raw tracks (`trajectory-smooth`,
  `movement-filter-interpolate`) before downstream computation.
- **Discover behaviors unsupervised** — t-SNE / UMAP-style global embeddings,
  k-means and Ward clustering, ARHMM, or
  [keypoint-MoSeq](https://keypoint-moseq.readthedocs.io/) via a managed
  subprocess environment.
- **Train supervised behavior classifiers** — XGBoost from
  [BORIS](https://www.boris.coding.systems/) or CalMS21 labels, with optional
  temporal context stacking. Lightning-Action temporal classifier also
  supported.
- **Train a visual identification model** (T-Rex-compatible CNN) from
  egocentric crops of known individuals.
- **Render annotated overlay videos**, interactive playback, egocentric crop
  videos, embedding-colored scatter plots, and behavior timelines.

## Features and "models"

Everything in mosaic is a **feature**. There are two flavors:

- **Per-frame / per-sequence features** — stateless transforms of tracks or
  upstream feature output. Examples: `speed-angvel`, `pair-egocentric`,
  `nearest-neighbor`, `pair-wavelet`, `temporal-stack`, `body-scale`.
- **Global features** — trainable, fit-then-apply features that learn from a
  collection of sequences and then map per-sequence. This is where mosaic's
  "models" live: `global-scaler`, `global-tsne`, `global-kmeans`,
  `global-ward`, `xgboost`, `arhmm`, `feral`, `kpms`, `lightning-action`,
  `global-identity-model`.

Visualization (overlay videos, timeline plots, embedding scatters) is also
exposed as features so it benefits from the same caching and reproducibility
machinery.

The full registry — currently ~30 features across kinematic, spatial, social,
spectral, embedding, clustering, supervised, and external (KPMS,
Lightning-Action) categories — is documented in the
[feature library reference](docs/api/behavior/feature-library.md).

## Examples

The canonical worked example is the
[CalMS21 template notebook](notebooks/calms21-template.ipynb), which walks
end-to-end through:

1. Building a `Dataset` from a manifest.
2. Computing pair-egocentric and pair-pose-distance-PCA features.
3. Wavelet expansion + global scaling + t-SNE embedding.
4. K-means / Ward clustering with cluster-to-label agreement metrics.
5. Supervised classification: `ExtractLabeledTemplates` + XGBoost training,
   with temporal context stacking.
6. Visualizing predictions on the embedding.

More examples (mice, bees, custom pose-model training) are in the works and
will live alongside this one.

## Documentation

The full mkdocs site (Getting Started, Pipeline Guide, Pipeline Architecture,
API Reference) lives under [docs/](docs/) and renders to a published site at
**https://ecodylicscience.github.io/mosaic/** (once published — see
[docs/](docs/) for the source).

Highlights:
- [Getting Started](docs/getting-started.md) — install, create a dataset, run
  your first features.
- [Pipeline Guide](docs/guide-pipeline.md) — declarative multi-step pipelines
  with caching.
- [Pipeline Architecture](docs/pipeline.md) — the feature protocol and how
  global features compose.
- [Migration Guide](docs/migration-guide.md) — for upgrading from the older
  API.

## Installation

`pyproject.toml` is the canonical dependency source.

```bash
conda create -n mosaic python=3.12 -y
conda activate mosaic
conda install -c conda-forge ffmpeg -y
pip install -e ".[recommended]"
```

`ffmpeg` provides `ffprobe`, used by media indexing and raw H.264 support.

The `recommended` bundle covers the typical research workflow (wavelets, YOLO
pose training/inference, PyTorch localizer). For lighter or different installs,
pick extras à la carte:

| Extra | What it adds |
|-------|--------------|
| `recommended` | wavelets + pose + localizer (recommended starting point) |
| `wavelets` | PyWavelets for spectral features |
| `pose` | Ultralytics YOLO pose training & inference |
| `polo` | POLO point-detection (mutually exclusive with `pose` — different ultralytics fork) |
| `localizer` | PyTorch localizer heatmap training |
| `lightning-action` | Lightning-Action temporal action classifier |
| `gpu` | faiss for GPU-accelerated kNN in global-tsne (use `faiss-gpu` on Linux+CUDA) |

## Pipeline at a glance

Data flows through a single `Dataset` orchestrator. Every stage's output is
versioned by `run_id` and cached.

```
video files
   ├─ index_media()          ➜ media/index.csv  (ffprobe metadata)
   └─ extract_frames()       ➜ media/frames/    (uniform or k-means PNGs)

raw tracks/labels
   ├─ index_tracks_raw()     ➜ tracks_raw/index.csv
   ├─ convert_all_tracks()   ➜ tracks/<group>__<seq>.parquet  (standardized)
   └─ convert_all_labels()   ➜ labels/<kind>/<group>__<seq>.npz

run_feature(...)            ➜ features/<name>/<run_id>/*.parquet
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

## Repository layout

The package lives under `src/mosaic/` and is organized into `core/`,
`behavior/` (features, labels, models, visualization), `tracking/`
(pose-model training/inference + converters), and `media/` (video I/O, frame
extraction). For the full developer-facing map see
[mosaic_repo_summary.md](mosaic_repo_summary.md).

## Status & contributions

Mosaic is actively developed by Ecodylic Science. It is in early use within
our group and not yet stable for outside users — the API may change. Internal
collaborators and curious colleagues, see [CONTRIBUTING.md](CONTRIBUTING.md).
