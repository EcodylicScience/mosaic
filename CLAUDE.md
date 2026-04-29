# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) and other AI coding
agents when working with code in this repository.

## Project Overview

`mosaic-behavior` (imported as `mosaic`) is a Python toolkit for animal
behavior analysis. Given pose tracks (per-frame keypoints with identities), it
produces standardized parquet track tables, behavioral features (kinematic,
social, spectral, reduction), unsupervised embeddings and clusters
(t-SNE / k-means / Ward / ARHMM / keypoint-MoSeq), supervised classifiers
(XGBoost, Lightning-Action), visual identification models from egocentric
crops, and annotated overlay videos / timelines.

If pose tracks are not yet available, mosaic also covers the upstream pipeline:
frame sampling for annotation, and pose-model training from CVAT / COCO /
Lightning Pose annotations using YOLO pose, POLO point detection, or a PyTorch
heatmap localizer.

Public docs: <https://ecodylicscience.github.io/mosaic/>.

## Development Commands

### Environment setup

```bash
conda create -n mosaic python=3.12 -y
conda activate mosaic
conda install -c conda-forge ffmpeg -y
pip install -e ".[recommended]"
```

`ffmpeg` (with `ffprobe`) must be on `PATH` — it is used by media indexing and
raw H.264 frame counting.

Python `>=3.12` is required (`pyproject.toml`).

### Optional extras

`[recommended]` bundles `wavelets` + `pose` + `localizer`. For the full extras
table, see [README.md](README.md). Notable points:

- `pose` and `polo` cannot be installed in the same environment — both ship
  under the `ultralytics` distribution name.
- `lightning-action` and `gpu` are intentionally excluded from `recommended`.
- `gpu` installs `faiss-cpu` by default; on Linux + CUDA, install `faiss-gpu`
  manually for GPU-accelerated kNN in `global-tsne`.

### Smoke import

```bash
python -c "from mosaic.core.dataset import Dataset; print('OK')"
```

(Same check as [CONTRIBUTING.md](CONTRIBUTING.md).)

### Testing

Pytest is configured in `pyproject.toml`. Slow tests are deselected by default.

```bash
pytest                                      # all tests except those marked slow
pytest -m slow                              # slow tests only
pytest -m "slow or not slow"                # everything
pytest tests/test_run_feature.py            # one file
pytest tests/test_run_feature.py::test_x    # one test
pytest -k "feature_params"                  # name pattern
pytest -v                                   # verbose
```

### Linting and formatting

```bash
ruff check --fix     # lint + autofix
ruff format          # format
```

Ruff is configured in `pyproject.toml`. There is no separate `ruff.toml`.

### Type checking

mosaic uses **basedpyright** in strict mode (target Python 3.13):

```bash
basedpyright            # full project
basedpyright src/mosaic/core/dataset.py    # one file
```

`src/mosaic/behavior/feature_library/external/` has its own execution
environment (it runs keypoint-MoSeq in an isolated venv) and is excluded from
the uv workspace — see `[tool.basedpyright]` and `[tool.uv.workspace]` in
`pyproject.toml`.

### Documentation

Built with MkDocs Material + mkdocstrings (Google-style docstrings):

```bash
mkdocs serve     # live-reload at http://127.0.0.1:8000/
mkdocs build     # static site into ./site/
```

API pages under `docs/api/` are auto-generated, so updating a docstring
updates the published docs.

### No CI / pre-commit

There is currently no GitHub Actions workflow and no `.pre-commit-config.yaml`.
Before reporting work done, manually run:

```bash
ruff check
basedpyright
pytest
```

## High-Level Architecture

### `Dataset` orchestrator

[`src/mosaic/core/dataset.py`](src/mosaic/core/dataset.py) defines `Dataset`,
the central object users interact with. A `Dataset` manages a fixed set of
named roots:

- `media/`        — video files + `index.csv` (ffprobe metadata)
- `tracks_raw/`   — raw input tracks/labels + `index.csv`
- `tracks/`       — standardized `<group>__<seq>.parquet`
- `labels/<kind>/` — converted manual labels (`.npz`)
- `features/<name>/<run_id>/` — per-feature outputs
- `models/<name>/<run_id>/`   — trained model artifacts
- `inputsets/`    — input-set definitions for grouped runs

`dataset_type` is `"discrete"` (default) or `"continuous"` (with
`segment_duration` and `time_column`).

### Plugin registries (everything is a plugin)

mosaic uses decorator-based registries; new functionality almost always means
"register a new plugin," not "edit a hot path."

| Decorator                  | Registry            | Lives in                          |
| -------------------------- | ------------------- | --------------------------------- |
| `@register_feature`        | `FEATURES`          | `behavior/feature_library/`       |
| `register_track_converter` | `TRACK_CONVERTERS`  | `core/track_library/`             |
| `@register_label_converter`| `LABEL_CONVERTERS`  | `behavior/label_library/`         |

### Feature protocol

Every feature implements four methods — `load_state`, `fit`, `apply`,
`save_state` — and declares a `name`, `version`, and parallelizability flags.
Two flavors:

- **per-frame / per-sequence** — stateless transforms (e.g. `speed-angvel`,
  `pair-egocentric`, `nearest-neighbor`, `pair-wavelet`, `temporal-stack`,
  `body-scale`).
- **global** (fit-then-apply) — trained on a collection of sequences, then
  applied (e.g. `global-scaler`, `global-tsne`, `global-kmeans`,
  `global-ward`, `xgboost`, `arhmm`, `kpms`, `lightning-action`,
  `global-identity-model`).

Visualization features (`egocentric-crop`, `viz-timeline`,
`viz-global-colored`, `interaction-crop-pipeline`) use the same protocol and
caching machinery.

### Pipeline package

[`src/mosaic/core/pipeline/`](src/mosaic/core/pipeline/) owns data loading,
output writing, dependency resolution, and indexing. **Features own
computation only.** The public typed surface lives in `pipeline/types/`:
`Params`, `Inputs`, `Result`, `ArtifactSpec`, `OutputType`, `InputStream`,
`DependencyLookup`, `FeatureLabelsSource`, `GroundTruthLabelsSource`.

### `run_id` reproducibility

Each feature run is tagged with `run_id = "<version>-<SHA1(params)>"`.
Identical inputs + params → identical `run_id` → no recompute. Parameter
sweeps stay organized under `features/<name>/<run_id>/`. Never bypass
`run_feature()` to write feature outputs directly — it would desync indexes
and break reproducibility.

## Module Organization

```
src/mosaic/
├── core/
│   ├── dataset.py              # Dataset orchestrator
│   ├── pipeline/               # feature execution engine, typed protocol
│   │   ├── types/              # Params, Inputs, Result, ArtifactSpec, ...
│   │   ├── run.py              # run_feature() orchestration
│   │   ├── manifest.py         # unified manifest + per-sequence iterator
│   │   ├── loading.py          # sequence identity / NN-lookup construction
│   │   ├── index_csv.py        # generic typed IndexCSV
│   │   ├── writers.py          # parquet output writing, overlap trimming
│   │   └── _loaders.py         # NPZ / Parquet / Joblib dispatcher
│   ├── schema.py               # track-schema validation (e.g. trex_v1)
│   ├── analysis.py             # clustering metrics
│   ├── helpers.py              # label loading, safe-name encoding, time/frame filtering
│   └── track_library/          # track converters (CalMS21, MABe22, TREx)
├── behavior/
│   ├── feature_library/        # ~30 per-frame + global features (plugin)
│   │   ├── movement/           # optional movement-library integration
│   │   └── external/           # keypoint-moseq subprocess runner (own venv)
│   ├── label_library/          # label converters (BORIS, CalMS21)
│   ├── model_library/          # legacy models (being phased out)
│   └── visualization_library/  # overlay, playback, egocentric crops, timelines
├── tracking/
│   └── pose_training/          # YOLO pose, POLO point, localizer training
│       ├── converters/         # CVAT XML, Lightning Pose, COCO, ...
│       └── augmentation.py     # YOLO + localizer augmentation presets
└── media/
    ├── video_io.py             # video I/O, raw H.264 fallback, ffmpeg hw accel
    ├── extraction.py           # uniform / k-means frame extraction
    └── sampling.py             # frame selection algorithms
```

## Data Flow Pipeline

```
video files
   ├─ index_media()          → media/index.csv          (ffprobe metadata)
   └─ extract_frames()       → media/frames/            (uniform or k-means PNGs)

raw tracks/labels
   ├─ index_tracks_raw()     → tracks_raw/index.csv
   ├─ convert_all_tracks()   → tracks/<group>__<seq>.parquet
   └─ convert_all_labels()   → labels/<kind>/<group>__<seq>.npz

run_feature(...)             → features/<name>/<run_id>/*.parquet
                                                 └── run_id = <version>-<SHA1(params)>
```

Models follow the same shape: `models/<name>/<run_id>/`.

## Important Conventions

### Track schema

Standardized tracks are validated by `core/schema.py`. The `trex_v1` schema,
for example, requires columns: `frame, time, id, group, sequence, X, Y, ANGLE,
SPEED` plus `poseX*` / `poseY*` for keypoints. New track converters must emit
schema-valid parquet.

### Params are Pydantic

Per-feature `Params` are Pydantic models — never pass raw `dict[str, Any]`
across feature boundaries. The `run_id` is a SHA1 of the serialized params, so
every field affects reproducibility. Add new params as typed fields with
defaults; don't reuse one field for two meanings across versions — bump
`version` instead.

### Determinism

Identical params + inputs must always produce the same `run_id` and outputs.
Don't introduce nondeterministic iteration order, unseeded random state, or
filesystem-order-dependent behavior in feature code.

### Type checking

basedpyright is in strict mode. Prefer dataclasses / Pydantic models over
`dict[str, Any]`. New code is expected to type-check cleanly.

### Docstrings

Google-style. mkdocstrings auto-renders public API into `docs/api/`, so a good
docstring is the documentation.

### `feature_library/external/` is sandboxed

keypoint-MoSeq lives in its own venv and is invoked via subprocess. It has a
separate basedpyright execution environment and is excluded from the uv
workspace (`[tool.uv.workspace]` in `pyproject.toml`). Don't import it
directly from the main mosaic package.

## Working with Notebooks

Reference end-to-end examples (not test fixtures):

- [`notebooks/calms21-template.ipynb`](notebooks/calms21-template.ipynb) —
  canonical end-to-end (manifest → features → wavelet/scaler/t-SNE →
  clustering → XGBoost classifier → visualization).
- [`notebooks/mabe22-mouse-triplets.ipynb`](notebooks/mabe22-mouse-triplets.ipynb)
- [`notebooks/mabe22-beetle-ant.ipynb`](notebooks/mabe22-beetle-ant.ipynb)

Notebooks may use sample data not present in the repo; check the first cell
for path expectations before running.

## Common Pitfalls

1. **`pose` vs `polo` install conflict.** Both extras install something named
   `ultralytics` (upstream pin vs. the [mooch443/POLO](https://github.com/mooch443/POLO)
   git pin), so pip resolves only one. POLO is a *full fork* of ultralytics —
   it retains all upstream tasks (detect/segment/classify/pose/track) and
   *adds* the `locate` (point-detection) task. The trade-off is update
   cadence: `[pose]` tracks upstream releases; `[polo]` is pinned to a fork
   that updates less often. Prefer `[pose]` unless you need point detection.
2. **Raw `.h264` files (Raspberry Pi)** have no container. OpenCV's
   `CAP_PROP_FRAME_COUNT` and `CAP_PROP_FPS` return garbage; random seeking
   silently corrupts the decoder. Always go through
   [`media/video_io.py`](src/mosaic/media/video_io.py) (`_has_container`,
   `_ffprobe_fps`, `_count_frames_by_decoding`) — don't open raw streams with
   `cv2.VideoCapture` directly.
3. **`ffmpeg` / `ffprobe` must be on `PATH`** — install via
   `conda install -c conda-forge ffmpeg`. Many failures in `index_media()`
   trace back to a missing `ffprobe`.
4. **Don't bypass `run_feature()`** to write feature outputs directly. The
   pipeline owns indexing, `run_id` registration, and output layout. Side-loaded
   files break reproducibility and downstream features.
5. **Schema-valid tracks only.** Track converters that emit non-schema columns
   will fail validation downstream. Test new converters against
   `core/schema.py` before relying on them.
6. **`recommended` is curated.** It deliberately omits `polo`,
   `lightning-action`, and `gpu`. Don't quietly fold them in.
7. **0.x APIs may move.** Per [CONTRIBUTING.md](CONTRIBUTING.md), breaking
   changes still warrant explicit discussion in an issue first.

## Pointers to Deeper Docs

- [`docs/getting-started.md`](docs/getting-started.md) — installation and first run.
- [`docs/guide-pipeline.md`](docs/guide-pipeline.md) — pipeline guide.
- [`docs/api/`](docs/api/) — auto-generated API reference (core, pipeline,
  behavior, media, tracking).
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — PR workflow and CLA.
