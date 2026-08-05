# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) and other AI coding
agents when working with code in this repository.

## Project Overview

`mosaic-behavior` (imported as `mosaic`) is a Python toolkit for animal
behavior analysis. Given pose tracks (per-frame keypoints with identities), it
produces standardized parquet track tables, behavioral features (kinematic,
social, spectral, reduction), unsupervised embeddings and clusters
(t-SNE / k-means / Ward / ARHMM / keypoint-MoSeq), supervised classifiers
(XGBoost, Lightning-Action, FERAL), visual identification models from egocentric
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

`mosaic-media` is a sibling package by the same authors, and it resolves from
PyPI like any other dependency — no second clone is needed. To work against its
unreleased `main`, install it editable *over* the released wheel:
`pip install -e "../mosaic-media[io,cli]"`. Do not reintroduce it as a
`[tool.uv.sources]` path: that table is a uv extension pip does not read, so a
path source there makes `pip install -e .` fail outright on a machine without
the sibling. See [CONTRIBUTING.md](CONTRIBUTING.md).

### Optional extras

`[recommended]` bundles `wavelets` + `pose` + `localizer`. For the full extras
table, see [README.md](README.md). Notable points:

- `pose` and `polo` cannot be installed in the same environment — both ship
  under the `ultralytics` distribution name.
- `lightning-action` and `gpu` are intentionally excluded from `recommended`.
- `gpu` installs `faiss-cpu` by default; on Linux + CUDA, install `faiss-gpu`
  manually for GPU-accelerated kNN in `global-tsne`.
- `imgstore` adds native support for imgstore (Motif / Loopbio) recordings — a
  store *directory* is indexed and read as a normal media entry (see "imgstore
  support" below). Not bundled in `recommended`.
- `feral` installs the FERAL V-JEPA behavior classifier (`FeralFeature`, train +
  infer) from a git pin (`Skovorp/feral@main`). It runs in-process (not
  sandboxed like keypoint-MoSeq) and is deliberately excluded from `recommended`.
- `identity` installs `torch` + `timm` for all three image-backbone identity
  models — `global-identity-model` (trains a classification head),
  `global-identity-embedding` (frozen backbone, prototype k-NN, trains nothing)
  and `global-identity-dinov2-temporal` (frozen DINOv2 plus a trained temporal
  head over clips). The first two take any timm architecture tag or Hugging Face
  hub id. Also excluded from `recommended`. Mosaic ships no weights — see
  [docs/licensing.md](docs/licensing.md).

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

### CI

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs three jobs on push
and pull request. There is no `.pre-commit-config.yaml`.

- **`test`** — `uv run --no-sync pytest -q`. It inherits `addopts = "-m 'not slow'"`,
  so **slow-marked tests never run in CI**. A change that only breaks a slow test
  goes green; run them locally.
- **`identity`** — four named test files under a torch-bearing environment, so
  `pytest.importorskip("torch")` cannot silently skip them.
- **`lint-changed`** — `uvx ruff check` **and** `uvx ruff format --check` over
  every changed `.py` file. Formatting is a merge gate, not a suggestion.

Not gated: project-wide `ruff check` / `ruff format --check` and `basedpyright`,
all three of which carry a pre-existing backlog. So before reporting work done,
run what CI will not:

```bash
ruff check && ruff format --check   # gated for changed files only
basedpyright                        # not gated at all
pytest -m "slow or not slow"        # CI skips the slow ones
```

## High-Level Architecture

### `Dataset` orchestrator

[`src/mosaic/core/dataset.py`](src/mosaic/core/dataset.py) defines `Dataset`,
the central object users interact with. A `Dataset` manages a fixed set of
named roots:

- `media/`        — video files + `index.csv` (ffprobe metadata)
- `tracks_raw/`   — user-uploaded raw tracks/labels + `index.csv`
- `_tracking/<tool>/` — run-addressed *raw* output of integrated trackers
                  (`trex` / `sleap` / `litpose`) and of model inference
                  (`infer-pose` / `infer-points` / `infer-localizer`), before
                  conversion. Kept out of `tracks_raw/` so that root holds only
                  user-uploaded content, and **excluded by name from every scan
                  that walks the dataset for user content** — the exclusion is a
                  path-component check — `is_under_tracking_root` in
                  `core/pipeline/tracking_roots.py`, called from `iter_track_files`
                  and from `_probe_dir_rows` (the prober behind both `scan_media`
                  and `index_media`) — because `exclude_patterns` matches
                  basenames and cannot express a directory. Locations come from
                  the registry in
                  [`core/pipeline/tracking_roots.py`](src/mosaic/core/pipeline/tracking_roots.py),
                  never spelled inline; `mosaic sweep-tracking` reclaims what is
                  finished and past its retention window
- `tracks/<variant>/` — standardized `<group>__<seq>.parquet`, one directory
                  per tracks recipe, + a single typed `index.csv`
- `labels/<kind>/` — converted manual labels (`.npz`)
- `features/<name>/<run_id>/` — per-feature outputs
- `models/<name>/<run_id>/`   — trained model artifacts
- `inputsets/`    — input-set definitions for grouped runs

### The dataset manifest

`dataset.yaml` is what makes a directory a mosaic dataset. Its format lives in
[`core/manifest.py`](src/mosaic/core/manifest.py) as pydantic models that take no
import from `Dataset`; `Dataset` holds one and exposes its fields as properties.

- `manifest_version: 2` is current. An older manifest is migrated **in memory**
  on read and stays as it is on disk until something saves, so a read-only mount
  works. A *newer* one raises rather than being read under the wrong rules.
- **Unknown top-level keys are preserved**, which is what made retiring
  `format`, `index_format`, `dataset_type`, `segment_duration` and `time_column`
  cost nothing: they are no longer modeled, and a file that holds them keeps
  them through a load-and-save round trip.
- `save()` is atomic. `Dataset.mutate_manifest()` is the read-modify-write seam
  for a writer that may be racing another one.
- **Roots live inside the dataset; sources deliberately do not.** See below.

### Scan sources: where a dataset draws its raw files from

`sources:` declares, per raw root, the directories and files a scan reads. A
source may point anywhere -- that is the mechanism that replaced an external
`media_raw`, with the files recorded by absolute `abs_path` into an index that
stays inside. A source directory is never created and never walked at load time.

Two modes. A **directory** source globs (`extensions` / `patterns`,
`recursive`); a **files** source claims exactly the paths it lists and nothing
beside them, which is what an import selecting some of a folder's contents
needs -- no glob expresses an arbitrary subset. Each source carries its whole
recipe, so one dataset can hold TREx output beside CalMS21 arrays.

**A scan replaces what it claims and preserves everything else**
([`core/pipeline/scan_claim.py`](src/mosaic/core/pipeline/scan_claim.py)). A row
under no scanned source survives: one written by an assignment scope, or one
pointing at a file outside the dataset. A file removed from a claimed directory
does leave. Two sources of one kind may not claim the same file, so the claimed
sets partition; two file sources sharing a directory with disjoint lists are
legal and expected. `--prune-unsourced` opts into dropping unclaimed rows.

A scan never overwrites an identity a caller **assigned** through a
`MediaIndexScope`; it refreshes the measured cells and keeps the identity ones.
Without that, declaring `media_raw` as a source would silently repartition every
project the control plane manages.

**Kinds are bare; roots carry `_raw`.** A source *kind* is `media` / `tracks` /
`labels` (`SourceKind`), the root it feeds is `media_raw` / `tracks_raw` /
`labels_raw`, and `_RAW_ROOT_FOR_KIND` joins them. So the scan verbs are
`scan_media` / `scan_tracks` / `scan_labels` — named for the kind, matching
`mosaic scan --kind`.

`_raw` is a **disambiguator, not a decoration**: it marks a name only where two
indexes would otherwise answer to one, because "picking the wrong one is a silent
wrong answer rather than an import error"
([`tracks_raw_index.py`](src/mosaic/core/pipeline/tracks_raw_index.py)). Hence
`read_tracks_raw_index` beside a real `read_tracks_index`, but no
`labels_raw_index` module at all — nothing collides with it. Nothing scans the
converted `tracks/`, so no scan verb needs the suffix either.

The media accessors (`index_media`, `write_media_index`, `read_media_index`)
carry no `_raw` for a different reason: `media_raw` is the one source root
`backfill_roots` cannot fill — an empty `media_raw/` on a dataset whose videos
sit in `media/` would make its media vanish — so it may legitimately be absent,
and those methods resolve through `resolve_media_root()` rather than pinning a
root a `_raw` name would promise. They read and write the **originals** index;
the derivative index (`media/index.csv`, one row per transcode) is reached
through `media_routing_context`.

### Dataset notes and tags

`notes:` is free text. `tags:` are typed attributes carrying the same
`type` / `type_constraints` / `value` shape as mosaic-api's sequence and
individual tags, validated by the shared grammar in
[`core/typed_attribute.py`](src/mosaic/core/typed_attribute.py) (`label`, `text`,
`int`, `float`, `bool`, `categorical`). Definition and value collapse into one
entry because a dataset-level tag has exactly one holder, so there is nothing
for a constraint change to narrow against. A `None` value means declared but not
yet set.

These describe the **dataset**. The per-sequence tags that categorize sequences
for analysis are a different thing, owned by mosaic-api.

### Plugin registries (everything is a plugin)

mosaic uses decorator-based registries; new functionality almost always means
"register a new plugin," not "edit a hot path."

| Decorator                  | Registry            | Lives in                          |
| -------------------------- | ------------------- | --------------------------------- |
| `@register_feature`        | `FEATURES`          | `behavior/feature_library/`       |
| `register_track_converter` | `TRACK_CONVERTERS`  | `core/track_converter.py` (impls in `core/track_library/`) |
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
  `global-ward`, `xgboost`, `arhmm`, `kpms`, `lightning-action`, `feral`,
  `global-identity-model`). `kpms` is license-restricted — see
  "`feature_library/external/` is sandboxed" below.

Visualization features (`egocentric-crop`, `viz-timeline`,
`viz-global-colored`, `interaction-crop-pipeline`) use the same protocol and
caching machinery.

`feral` (the FERAL V-JEPA behavior classifier) is a global fit-then-apply
feature but runs **in-process** — it imports the installed `feral` package
directly, unlike the sandboxed keypoint-MoSeq runner in
`feature_library/external/`. Each feature also declares a `category` used for
grouping/display; beyond per-frame / global / visualization, the taxonomy
includes `summary` (per-sequence aggregations, e.g. `frame-aggregate`) and
`tag` (e.g. `id-tag-columns`).

### Pipeline package

[`src/mosaic/core/pipeline/`](src/mosaic/core/pipeline/) owns data loading,
output writing, dependency resolution, and indexing. **Features own
computation only.** The public typed surface lives in `pipeline/types/`:
`Params`, `Inputs`, `Result`, `ArtifactSpec`, `OutputType`, `InputStream`,
`DependencyLookup`, `FeatureLabelsSource`, `GroundTruthLabelsSource`.

### `run_id` reproducibility

Each feature run is tagged with `run_id = "<version>-<hash>"`, where `<hash>`
is a 10-char SHA1 over the feature's params, inputs, and frame range (computed
from `Params.identity_dump()`, so `HASH_EXCLUDE`-tagged throughput knobs are
omitted — see "Params are Pydantic"). Identical inputs + params → identical
`run_id` → no recompute. Parameter sweeps stay organized under
`features/<name>/<run_id>/`. Never bypass `run_feature()` to write feature
outputs directly — it would desync indexes and break reproducibility.

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
│   ├── media/                  # foundational media I/O (read/decode/encode frames)
│   │   ├── video_io.py         # media I/O facade: libav reader/writer + dispatchers (mosaic-media)
│   │   ├── imgstore_io.py      # imgstore (Motif / Loopbio) dispatch + capture adapter
│   │   ├── imgstore_native.py  # native imgstore decode: mp4 via reader, raw via numpy
│   │   └── facts_columns.py    # MediaFacts / verdict <-> media-index row mapping
│   ├── schema.py               # track-schema validation (e.g. trex_v1)
│   ├── analysis.py             # clustering metrics
│   ├── helpers.py              # label loading, safe-name encoding, time/frame filtering
│   └── track_library/          # track converters (CalMS21, MABe22, TREx, SLEAP, DeepLabCut)
├── behavior/
│   ├── feature_library/        # ~35+ per-frame + global features (plugin)
│   │   ├── movement/           # optional movement-library integration
│   │   └── external/           # keypoint-moseq subprocess runner (own venv)
│   ├── label_library/          # label converters (BORIS, CalMS21, MABe22)
│   ├── model_library/          # identity networks (timm classifier / embedding, DINOv2 temporal)
│   └── visualization_library/  # overlay, playback, egocentric crops, timelines
└── tracking/
    ├── frame_extraction/       # uniform / k-means frame sampling → PNGs for annotation
    ├── pose_training/          # YOLO pose, POLO point, localizer training
    │   ├── converters/         # CVAT XML, Lightning Pose, COCO, ...
    │   └── augmentation.py     # YOLO + localizer augmentation presets
    ├── common/                 # everything a tracker run does around the tool
    │   ├── toolenv.py          # the MOSAIC_<TOOL>_CONDA_ENV / _BIN location ladder
    │   ├── mint.py             # root, run_id, tracks variant, run_params.json
    │   ├── scope.py            # media scope -> work items (video/camera collapse)
    │   ├── entry.py            # claim, marker reuse, cascade clearing, adoption
    │   ├── bridge.py           # converted frame -> tracks/<variant>/*.parquet
    │   ├── index.py            # TrackerRunRowBase + the typed run index
    │   ├── params.py           # TrackerOpParams (scope + HASH_EXCLUDE knobs)
    │   └── driver.py           # run_tracker(): the per-entry loop
    ├── trex/                   # TREx: two gated phases (convert -> track), own conda env
    ├── sleap/                  # SLEAP: one gated phase + an ungated atomic analysis export
    └── litpose/                # Lightning Pose: one gated phase, reuses the deeplabcut converter
```

**Layering.** `core` is the foundation: data model, schema, the pipeline engine,
and low-level **media I/O** (`core/media/` — read/decode/encode frames). `behavior`
and `tracking` are domain packages: they import `core` (including `core.media`)
but **never each other**, and they exchange data only through on-disk artifacts
(parquet tracks, feature/model files, index CSVs).

`core` reaches *upward* into `behavior` in exactly two places, both **deferred
into a call** so the two packages are not a cycle, and both for the same reason:
a registry fills only as a side effect of importing the modules that populate
it, and `core` is where the lookup happens. `core/__init__.py`'s module
`__getattr__` re-exports `register_feature`, and
`label_converter.ensure_label_converters_registered()` imports
`behavior.label_library` when `LABEL_CONVERTERS` is still empty (guarded on
emptiness, so a caller with its own converters pays nothing). `track_library`
needs neither — it lives in `core` and `core/__init__.py` imports it directly.
Anything else pointing from `core` at `behavior` or `tracking` is a layering
break.

`core.media` takes no import
from `behavior` or `tracking`. It is not a dependency-free leaf: it reads verdict
thresholds from the root-level `media_probe_config`, and `reprobe.py`
additionally reaches `core.helpers`, `core.stored_paths` and
`core.pipeline.media_index`. Frame *sampling/extraction*
(`tracking/frame_extraction/`, exposed as `mosaic.tracking.extract_frames(ds, …)`)
is a tracking-domain concern — it reads `media/frames` via `ds.get_root("frames")`
(downward) and is **not** a `Dataset` method; `core` has no frame-extraction code.

## Data Flow Pipeline

```
dataset.yaml  (mosaic init)
   └─ sources:                         → what every scan below reads

video files
   ├─ scan_media()  / index_media()    → media_raw/index.csv  (ffprobe metadata)
   └─ tracking.extract_frames(ds, …)   → media/frames/     (uniform or k-means PNGs)

raw tracks/labels
   ├─ scan_tracks() / scan_labels()   → <root>_raw/index.csv
   ├─ convert_all_tracks()   → tracks/<variant>/<group>__<seq>.parquet
   └─ convert_all_labels()   → labels/<kind>/<group>__<seq>.npz

run_trex / run_sleap / run_litpose / infer-*
   ├─ (working)              → _tracking/<tool>/<run_id>/<group>__<seq>/
   └─ (bridged)              → tracks/<variant>/<group>__<seq>.parquet
        ↑ reclaimed by `mosaic sweep-tracking`; a correction is promoted back
          out with `promote_correction` → tracks_raw/<entry>/corrected.rev<N>

run_feature(...)             → features/<name>/<run_id>/*.parquet
                                                 └── run_id = <version>-<SHA1 of params+inputs+frame range>
```

Models follow the same shape: `models/<name>/<run_id>/`.

## Important Conventions

### The tracks index

`tracks/index.csv` is a typed [`IndexCSV`](src/mosaic/core/pipeline/tracks_index.py)
(`TracksIndexRow`), written atomically under a per-file lock. Do not write it by
hand — `write_tracks_row()` is the only writer, and `read_tracks_index()` the only
reader.

Beyond `abs_path`/`group`/`sequence` each row carries `run_id` (the *tracks
variant* — which recipe produced the table, from
[`tracks_identity.py`](src/mosaic/core/pipeline/tracks_identity.py)), `producer`
(`convert-<fmt>` | `trex` | `infer-<kind>`, exactly `parse_op_run_id(run_id).kind`),
`producer_run_id` (the op run behind it, empty for a conversion, which has none),
and `consumed_source_roots` (dataset root *keys*, comma-joined and sorted).

Three invariants worth knowing:

- **One row per `(run_id, group, sequence)`** — the triple its sibling indexes
  use. An entry may carry several variants, because each writes into its own
  `tracks/<variant>/` directory.
- **Absent is empty.** A missing index reads as an empty frame carrying the full
  schema; nothing raises. Code that must tell a human to convert first checks for
  zero rows (see `mosaic sequences`).
- **Adopt on write, tolerate on read.** An older on-disk schema is widened in
  memory inside the write lock; readers project without touching disk, so a
  read-only mount works and looking at a legacy dataset does not rewrite it.

Writing a second row and *resolving* one are different questions.
`select_variant_rows()` answers the second, and is the only place that does: an
unlabelled row (`run_id` empty — every row written before variants existed) loses
to a labelled one for the same entry, and two genuinely different recipes for one
entry **raise** rather than guess. Different entries carrying different
variants — some converted, some tracked — stays legal, which is why `None` never
meant "the latest run".

Pass `tracks_run_id=` to answer a refusal: on `run_feature`,
`Dataset.run_feature`, `build_manifest`, `load_values`, and `--tracks-run-id` on
`mosaic run`; `run_id=` on `load_tracks` and `drop_entries`. `""` names the
unlabelled tables explicitly.

The resolved variants enter the feature `run_id` (the `_tracks` term) and
**never** the storage directory name, so `features/<name>__from__tracks/` stays
one directory and one index however many tracks recipes a dataset holds. The term
is omitted when the index names none, which is why a dataset converted before
this scheme keeps the identifiers it already has.

### Entry names are one path component

A `group` or `sequence` may not contain `/`, `\\` or NUL. mosaic itself survives a
slash — `to_safe_name` percent-encodes it — but an entry name doubles as a
directory name in mosaic-api, where it does not. Validated at the three write
boundaries (`EntryHints`, `write_tracks_row`, `build_tracks_raw_row`) and at no
read path, so an index that already holds one keeps resolving. Join levels with
`__`, which `parse_hierarchy` reads by default.

### Op and variant run identifiers

Tracking ops and tracks variants are named `<kind>.<version>-<10-hex-digest>`
(e.g. `convert-calms21_npy.0.2-6bb5efbf05`). The version is a *visible segment*,
not a hash term, so bumping it does not re-derive anything.
`extract-frames` is carved out and frozen — mosaic-api embeds its identifier in
annotation paths.

### Track schema

Standardized tracks are validated by `core/schema.py`. The `trex_v1` schema, for
example, *requires* columns `frame, time, id, group, sequence` plus at least one
`poseX*` and one `poseY*` keypoint column, and *recommends* (warn-only)
`X#wcentroid, Y#wcentroid, SPEED, ANGLE`. The validator never rejects unknown
columns, so additive columns (e.g. an optional `camera` axis for multi-camera
recordings) are back-compatible. New track converters must emit schema-valid
parquet.

### `group` is an optional namespace, not the grouping

`group` is a required column but may be empty (`""`). Together with `sequence` it
forms the composite identity / filename key (`<group>__<seq>`, or just `<seq>`
when empty) — kept for back-compat and to disambiguate non-unique sequence names.
It is **not** the canonical way to categorize sequences for analysis. Flexible,
redefinable grouping lives in **tags** (owned by mosaic-api). To run a feature
over an arbitrary, tag-resolved subset, pass explicit pairs:
`run_feature(ds, feature, entries=[(group, sequence), ...])` — unambiguous even
when sequence names repeat across groups (`groups=`/`sequences=` combine as a
cross-product and can't express an arbitrary set). `group` retains a *structural*
role only as a temporal-contiguity key for the (future) `continuous` dataset type:
overlap/windowed features pull prev/next neighbors only within the same group
(`core/pipeline/manifest.py`, `core/pipeline/iteration.py`) — preserve that when
softening `group` elsewhere.

### imgstore support

imgstore (Motif / Loopbio) recordings are *directories* (a `metadata.yaml` plus
chunk files), not single video files. mosaic treats a store as a normal media
entry — `scan_media()` (and the one-off `index_media()`) discovers stores
natively (one entry per store, `media_type="imgstore"`, internal chunks
excluded), and reading dispatches transparently:

- All frame-consuming features route through `MultiVideoReader`, which opens a
  store via `ImgStoreCapture` (a seek/read capture adapter) — so `egocentric-crop`,
  `interaction-crop-pipeline`, playback, and tracking's `extract_frames` work
  unchanged.
- Tracking inference uses `open_frame_reader()`: `ImgStoreFrameReader` for stores,
  else the in-process libav `VideoReader` for plain files — both in
  [`core/media/imgstore_io.py`](src/mosaic/core/media/imgstore_io.py) /
  [`core/media/video_io.py`](src/mosaic/core/media/video_io.py).
- Native decode: video-codec chunks (`.mp4` / h264) decode through the same
  `VideoReader`, raw-pixel chunks (`npy` / `npz`) load via numpy, and Bayer/YUV
  chunks convert through a local color map
  ([`core/media/imgstore_native.py`](src/mosaic/core/media/imgstore_native.py)),
  so the `imgstore` package leaves the read path.
- Frame addressing: the track-table `frame` column is the 0-based contiguous
  video frame index, which maps to imgstore's `frame_index` (**not** the
  camera-provided `frame_number`). Detection (`is_imgstore`) is import-free;
  reading a store no longer needs the `imgstore` package (native decode), so
  `[imgstore]` is only for writing stores, as the test fixtures do.

### Params are Pydantic

Per-feature `Params` are Pydantic models — never pass raw `dict[str, Any]`
across feature boundaries. The `run_id` hash covers the serialized params (plus
inputs and frame range), so every params field affects reproducibility
**except** those tagged `Annotated[T, HASH_EXCLUDE]` — throughput-only knobs
(e.g. `FeralFeature.infer_batch_size`) that `Params.identity_dump()` strips from
the hash so retuning them never busts the cache, even though they still appear
in `params.json` and propagate to parallel workers. Add new params as typed
fields with defaults; don't reuse one field for two meanings across versions —
bump `version` instead.

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

**The subprocess boundary is a licensing requirement, not a packaging
convenience.** keypoint-MoSeq is licensed by Harvard OTD for non-commercial
research and academic use only; mosaic is AGPL-3.0-or-later, and AGPL section 7
forbids adding a "non-commercial only" restriction to a covered work. Importing
it would make a combined work that could not be distributed at all. So:
keypoint-moseq must never be folded into mosaic's own dependencies, and never
bundled into a packaged installer.

The environment is not built by anything in the repo — the user builds it, and
[`external/README.md`](src/mosaic/behavior/feature_library/external/README.md)
is the documented bootstrap, with the license terms attached to that step.
`KpmsFeature._start_server` refuses to spawn until
`MOSAIC_KPMS_LICENSE_ACCEPTED=1`; `MOSAIC_KPMS_PYTHON` locates an interpreter,
matching the `MOSAIC_TREX_BIN` / `MOSAIC_SLEAP_BIN` convention. The check sits
at the spawn and deliberately not in `__init__`, because constructing a feature
(as `mosaic reconcile` does for every run it re-addresses) is not use of
keypoint-MoSeq. [`docs/licensing.md`](docs/licensing.md) is the user-facing
page, and covers the other third-party terms too — notably TRex, which requires
a paid license for company use.

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
2. **Raw `.h264` files (Raspberry Pi)** have no container, so header metadata is
   unreliable and seeking the bare stream corrupts the decoder. Read them through
   [`core/media/video_io.py`](src/mosaic/core/media/video_io.py): the packet-scan
   probe (`probe_media`) measures the true frame count and fps, and the in-process
   `VideoReader` decodes sequentially with those facts injected. Don't open a raw
   stream directly.
3. **`ffmpeg` / `ffprobe` must be on `PATH`** — install via
   `conda install -c conda-forge ffmpeg`. Many failures in `scan_media()` /
   `index_media()` trace back to a missing `ffprobe`.
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
- [`docs/adding-a-tracker.md`](docs/adding-a-tracker.md) — wiring a new
  external tracker in. `tracking/common/` owns the run loop; a tracker supplies
  its argv, its settings, its phases and its converter, plus one `TrackingRoot`
  row. `tests/test_tracker_conformance.py` is parametrized over every tracker
  root, so a half-implemented one fails by name.
- [`docs/api/`](docs/api/) — auto-generated API reference (core, pipeline,
  behavior, media, tracking).
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — PR workflow and CLA.
