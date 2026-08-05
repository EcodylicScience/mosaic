# mosaic

A Python toolkit for animal behavior analysis: track standardization, behavioral
feature extraction, unsupervised embedding and clustering, supervised classifier
training, and annotated video output.

**Documentation:** <https://ecodylicscience.github.io/mosaic/>

## Overview

Given pose tracks (per-frame keypoints with identities), mosaic produces:

- standardized parquet track tables from CalMS21, MABe22, TREx, SLEAP,
  Lightning Pose, DeepLabCut, or user-defined formats;
- per-frame and per-sequence behavioral features — kinematic, social, spectral,
  and reduction;
- unsupervised embeddings and clusters (t-SNE, k-means, Ward, ARHMM,
  [keypoint-MoSeq](https://keypoint-moseq.readthedocs.io/) — non-commercial
  license, see [Licensing](docs/licensing.md));
- supervised classifiers (XGBoost, Lightning-Action, FERAL) trained from manual
  labels;
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

Frame decoding runs in-process via the `av` wheel (installed with
`mosaic-media[io]`), so no `ffmpeg` binary is required to read video. System
`ffprobe` is still used for media indexing and probing (`mosaic_media.probe_media`),
and system `ffmpeg` >= 5.1 is required for the transcode path (`mosaic media
transcode`). Installing `ffmpeg` via conda covers both.

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
| `identity`         | Image-backbone identity models (trained classifier, frozen timm backbones, DINOv2 + temporal); `torch` + `timm` |
| `lightning-action` | Lightning-Action temporal action classifier                                         |
| `gpu`              | faiss for GPU-accelerated kNN in `global-tsne` (use `faiss-gpu` on Linux + CUDA)    |
| `imgstore`         | Native imgstore (Motif / Loopbio) video support (directory-based stores as media)   |
| `sleap`            | `h5py`, to read the SLEAP analysis `.h5` its converter consumes (SLEAP itself is an external binary) |
| `feral`            | FERAL V-JEPA behavior classifier (`FeralFeature`, training + inference)              |

`pose`, `polo`, and `recommended` install Ultralytics, which is AGPL-3.0. That
matters more than it looks: AGPL section 13 extends the duty to offer
Corresponding Source to anyone who interacts with the program over a network, so
a commercial deployment is reached without ever redistributing a copy.
Ultralytics sells an Enterprise license for use that cannot meet those terms,
but it covers Ultralytics' own distribution only — `polo` is a third-party fork,
so it is AGPL-only. Mosaic is AGPL-3.0-or-later itself, so nothing here is
incompatible; the question is whether *your* use of the combined work can meet
the obligations. See [Licensing](docs/licensing.md).

There is deliberately no `kpms` extra. keypoint-MoSeq cannot share an
environment with mosaic, so the `kpms` feature drives it in a separate one that
you build yourself — and it is licensed for non-commercial research and academic
use only. See [Licensing](docs/licensing.md) for the terms and
[`external/README.md`](src/mosaic/behavior/feature_library/external/README.md)
for the setup.

### Platform support

mosaic runs natively on **macOS** and **Linux**. On **Windows**, the core
analysis pipeline runs natively, but several features depend on components with
no native-Windows build and need **WSL2** (or Linux):

| Capability                                                        | Native Windows          | WSL2 / Linux | macOS |
| ----------------------------------------------------------------- | ----------------------- | ------------ | ----- |
| Core analysis (indexing, tracks, features, clustering, ARHMM, XGBoost, visualization) | Yes | Yes | Yes |
| keypoint-MoSeq (`kpms`) -- JAX + Unix sockets                     | No                      | Yes          | Yes   |
| FERAL (`feral`) -- `decord`                                       | No                      | Yes          | Yes   |
| GPU kNN (`gpu`, `faiss-gpu`)                                      | No (`faiss-cpu` works)  | Yes          | n/a   |
| imgstore read/write (`imgstore`)                                  | Partial                 | Yes          | Yes   |
| TREx tracking and pose-model training                             | Partial                 | Yes          | Yes   |

Native-Windows support for the core is new; for any **No** / **Partial**
capability, or if anything misbehaves natively, use **WSL2** (`wsl --install` in
an admin PowerShell), then follow the Linux setup above inside Ubuntu. Keep the
repository on the WSL filesystem (for example `~/mosaic`) rather than under
`/mnt/c`, so index locking and I/O behave as on Linux.

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
  `xgboost`, `arhmm`, `kpms`\*, `lightning-action`, `feral`,
  `global-identity-model`, `global-identity-embedding`,
  `global-identity-dinov2-temporal`).

\* `kpms` drives keypoint-MoSeq, which is licensed for non-commercial research
and academic use only. See [Licensing](docs/licensing.md); `arhmm` is the
unrestricted alternative.

`global-identity-embedding` loads whatever image backbone you name, and mosaic
distributes no weights — each carries its own license. Its default is
permissive; `BVRA/MegaDescriptor-L-384` is the strongest option for academic
wildlife re-identification and is non-commercial. See
[Licensing](docs/licensing.md).

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
   ├─ scan_media()                     → media_raw/index.csv  (probed via mosaic_media/ffprobe)
   └─ tracking.extract_frames(ds, …)   → media/frames/     (uniform or k-means PNGs)

raw tracks/labels
   ├─ scan_tracks()          → tracks_raw/index.csv
   ├─ convert_all_tracks()   → tracks/<variant>/<group>__<seq>.parquet
   └─ convert_all_labels()   → labels/<kind>/<group>__<seq>.npz

run_feature(...)             → features/<name>/<run_id>/*.parquet
```

`group` in the `<group>__<seq>` name is an optional namespace (it may be empty),
not the canonical grouping — flexible, redefinable grouping of sequences is done
with tags, and a feature can be run over an arbitrary subset of sequences via
`run_feature(ds, feature, entries=[(group, sequence), ...])`.

## Repository layout

```
src/mosaic/
├── core/        # Dataset orchestrator, pipeline engine, schema, media I/O (core/media/)
├── behavior/    # feature_library, label_library, visualization_library
└── tracking/    # pose-model training/inference and annotation converters
```

## Status

Mosaic is in early development (0.x). Public APIs, feature names, and on-disk
layouts may change between releases.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Please open an issue before making
large changes.

Working with Claude Code or another AI coding agent? See [CLAUDE.md](CLAUDE.md)
for repo orientation, development commands, and architectural conventions.

## License

GNU Affero General Public License v3 or later (AGPLv3+). See
[LICENSE](LICENSE), and [NOTICE](NOTICE) for the third-party attributions that
must be preserved with it.

Mosaic bundles no third-party source and no model weights, but it drives tools
whose terms differ from its own — keypoint-MoSeq prohibits commercial use
outright, TRex requires a paid license for company use, and Ultralytics is
AGPL-3.0. [Licensing](docs/licensing.md) states which components carry
restrictions and what mosaic does about them.

