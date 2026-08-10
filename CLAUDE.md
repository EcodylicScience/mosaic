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
  under the `ultralytics` distribution name. `pose` also carries `lap` and an
  `ultralytics>=8.4.63` floor, which is what `mosaic track ultralytics` needs:
  `lap` is the tracker's linear-assignment solver and is in no ultralytics
  extra, so undeclared it gets pip-installed mid-run, and the four newer
  tracker backends only exist from 8.4.63. `polo` carries `lap` too.
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

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs four jobs on push
and pull request. There is no `.pre-commit-config.yaml`.

- **`test`** — `uv run --no-sync pytest -q`. It inherits `addopts = "-m 'not slow'"`,
  so **slow-marked tests never run in CI**. A change that only breaks a slow test
  goes green; run them locally.
- **`identity`** — five named test files under a torch-bearing environment, so
  `pytest.importorskip("torch")` cannot silently skip them.
- **`tracking`** — the ultralytics preflight and marker suites under a `pose`
  environment, for the same reason: without `ultralytics` and `lap` installed they
  would skip green and prove nothing.
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

- `media_raw/`    — the originals index (`index.csv`, ffprobe metadata). The one
                  source root `backfill_roots` cannot fill, so it may legitimately
                  be absent; accessors resolve through `resolve_media_root()`
- `media/`        — transcode derivatives + their own `index.csv`, one row per
                  derivative, reached through `media_routing_context`
- `media/frames/` — extracted PNGs for annotation (root key `frames`)
- `tracks_raw/`   — user-uploaded raw tracks + `index.csv`
- `labels_raw/`   — user-uploaded raw labels + `index.csv`
- `_tracking/<tool>/` — run-addressed *raw* output of integrated trackers
                  (`trex` / `sleap` / `litpose` / `ultralytics`) and of model inference
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
| `@register_label_converter`| `LABEL_CONVERTERS`  | `core/label_converter.py` (impls in `behavior/label_library/`) |
| `@register_op`             | `OPS`               | `core/pipeline/ops.py`            |

### Feature protocol

Every feature implements four methods — `load_state`, `fit`, `apply`,
`save_state` — and declares a `name`, `version`, and parallelizability flags.
Two flavors:

- **per-frame / per-sequence** — stateless transforms (e.g. `speed-angvel`,
  `pair-egocentric`, `nearest-neighbor`, `pair-wavelet`, `temporal-stack`,
  `body-scale`, `heading`, `scale-to-cm`). The last two hold what converters
  used to do inline: `heading` derives `ANGLE` from keypoints under a *chosen*
  method, and `scale-to-cm` converts pixels to centimetres using the per-video
  `cm_per_pixel` on the media index. Both put the choice in a run identifier,
  which is the whole reason they are features rather than columns.
- **global** (fit-then-apply) — trained on a collection of sequences, then
  applied (e.g. `global-scaler`, `global-tsne`, `global-kmeans`,
  `global-ward`, `xgboost`, `arhmm`, `kpms`, `lightning-action`, `feral`,
  `global-identity-model`). `kpms` is license-restricted — see
  "`feature_library/external/` is sandboxed" below.

The crop features (`egocentric-crop`, `interaction-crop-pipeline`) use the same
protocol and caching machinery. They live under `visualization_library/` for
historical reasons but are categorized `media`, not visualization: they write
image and video artifacts that other features read, and egocentric crops are the
input all three identity models take.

`feral` (the FERAL V-JEPA behavior classifier) is a global fit-then-apply
feature but runs **in-process** — it imports the installed `feral` package
directly, unlike the sandboxed keypoint-MoSeq runner in
`feature_library/external/`. Each feature also declares a `category` used for
grouping/display; beyond per-frame and global, the taxonomy includes `summary`
(per-sequence aggregations, e.g. `frame-aggregate`), `tag` (e.g.
`id-tag-columns`) and `media` (writes crops or clips another feature reads, e.g.
`egocentric-crop`). There is deliberately no visualization category: rendering
lives in `visualization_library/` as plain functions rather than as features.

### Pipeline package

[`src/mosaic/core/pipeline/`](src/mosaic/core/pipeline/) owns data loading,
output writing, dependency resolution, and indexing. **Features own
computation only.** The public typed surface lives in `pipeline/types/`:
`Params`, `Inputs`, `Result`, `ArtifactSpec`, `InputStream`,
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
├── cli/                        # every `mosaic` verb
├── runlog.py                   # append-only JSONL run-log (a deliberate leaf)
├── media_probe_config.py       # verdict thresholds, read by core/media
├── core/
│   ├── dataset.py              # Dataset orchestrator
│   ├── manifest.py             # dataset.yaml models (NOT pipeline/manifest.py)
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
│   └── track_library/          # track converters (CalMS21, TREx, SLEAP, DeepLabCut, Ultralytics)
├── behavior/
│   ├── feature_library/        # ~35+ per-frame + global features (plugin)
│   │   ├── movement/           # optional movement-library integration
│   │   └── external/           # keypoint-moseq subprocess runner (own venv)
│   ├── label_library/          # label converters (BORIS, CalMS21)
│   ├── model_library/          # identity networks (timm classifier / embedding, DINOv2 temporal)
│   └── visualization_library/  # overlay, playback, egocentric crops, timelines
└── tracking/
    ├── ops/                    # @register_op layer behind `mosaic run --kind`
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
    ├── litpose/                # Lightning Pose: one gated phase, reuses the deeplabcut converter
    └── ultralytics_track/      # Ultralytics MOT: one gated phase, in process, no second env
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

Every `index.csv` has a zero-byte `index.csv.lock` beside it — `index_lock`'s
sidecar, created on the first locked write and **never removed**. It is not
data, nothing reads it, and deleting it while a writer holds it reintroduces the
lost update the lock prevents. Anything that enumerates a root should expect it.

## Important Conventions

### The tracks index

`tracks/index.csv` is a typed [`IndexCSV`](src/mosaic/core/pipeline/tracks_index.py)
(`TracksIndexRow`), written atomically under a per-file lock. Do not write it by
hand — `write_tracks_row()` is the only *row-append* writer, and
`read_tracks_index()` the only reader. Two other modules rewrite the same file
through sibling `IndexCSV` methods: `delete_set` (deletion) and
`reconcile_variants` (re-addressing).

Beyond `abs_path`/`group`/`sequence` each row carries `run_id` (the *tracks
variant* — which recipe produced the table, from
[`tracks_identity.py`](src/mosaic/core/pipeline/tracks_identity.py)), `producer`
(`convert-<fmt>` | `trex` | `infer-<kind>`, exactly `parse_op_run_id(run_id).kind`),
`producer_run_id` (the op run behind it, empty for a conversion, which has none),
`consumed_source_roots` (dataset root *keys*, comma-joined and sorted), and
`n_keypoints` (how many `poseX*`/`poseY*` pairs the table holds, **measured from
the parquet** at write time rather than passed in, so no call site can record a
false zero). Since keypoints are optional, "does this entry have any" would
otherwise need a parquet open per entry; a blank cell means *unknown*, not zero,
exactly as it does for `n_rows`.

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
directory name in mosaic-api, where it does not. Validated at the six write
boundaries (`EntryHints`, `Dataset.set_display_name`, `build_tracks_raw_row`,
`promote_correction`, `write_tracks_row`, `write_labels_row`) and at no
read path, so an index that already holds one keeps resolving. Join levels with
`__`, which `parse_hierarchy` reads by default.

### Op and variant run identifiers

Tracking ops and tracks variants are named `<kind>.<version>-<10-hex-digest>`
(e.g. `convert-calms21_npy.0.2-6bb5efbf05`). The version is a *visible segment*,
not a hash term, so bumping it does not re-derive anything.
`extract-frames` is carved out and frozen — mosaic-api embeds its identifier in
annotation paths.

### Track schema

Standardized tracks are validated by `core/schema.py`. Three schemas are
registered, and a converter declares which it emits through
`TrackConverter.output_schema` — a tracker through `TrackingRoot.output_schema`.
**That declaration is the only place the name is written.** It used to be spelled
in five (each converter, `Dataset` via a manifest setting, the tracker bridge,
the inference path), so one table was validated twice under two independently
chosen names and the index row recorded only the second.

- **`mosaic_v1`** — the tracker-neutral standard. Requires
  `frame, time, id, group, sequence, X, Y`. Every spatial column is **video
  pixels**, and `X`/`Y` are the individual's **body centre**. **Keypoints are
  optional**: a centroid-only tracker (TREx without posture, a box model, an
  export that carries a centroid and nothing else) emits no `poseX*`/`poseY*` at
  all. Requiring a pair made those trackers fabricate one — a verbatim copy of
  `X`/`Y` under a name promising a detected landmark, which is the same
  plausible-but-not-measured column the forbidden set exists to refuse. The
  features that are *defined* on keypoints (`heading`, `body-scale`,
  `orientation-rel`, `kpms`, the `pair-*` family) refuse a table without them,
  naming themselves; the ones that only prefer them (`egocentric-crop`,
  `interaction-crop-pipeline`, the overlay) fall back to the body centre.
- **`trex_v2`** — `mosaic_v1` plus what TREx genuinely measures (`SPEED`,
  `ANGLE`, `X#wcentroid`, the midline family), also in pixels. TREx's own bare
  `X`/`Y` are the *head* and are preserved as `X#head`/`Y#head`.
- **`trex_v1`** — the legacy schema, kept registered permanently because a real
  archived dataset is in it. Its spatial columns are **centimetres** and its `X`
  is a head position.

Two rules the validator enforces beyond presence:

- **A named, closed `forbidden` set.** `mosaic_v1` forbids `VX, VY, AX, AY,
  SPEED*, ANGLE, ANGULAR_*, X#wcentroid, Y#wcentroid` — quantities a *feature*
  derives, not ones a tracker measures. Violating it raises **regardless of
  `strict`**, because every tracker write path validates with `strict=False` and
  a wrong table must not reach disk. A tracker that genuinely measures one
  declares a schema that `allows` it, as `trex_v2` does.
- **An unregistered schema name raises.** It used to return an empty report,
  validating nothing *and* silently disarming `strict=True`.

The validator still **never rejects a column merely for being unknown**, so
additive columns (e.g. an optional `camera` axis for multi-camera recordings)
remain back-compatible — only the named set is refused. New track converters
must emit schema-valid parquet.

### Tracks are pixels; a physical unit is a feature

Every spatial column in `tracks/` is in video pixels, and `X`/`Y` mean the body
centre on every tracker. Neither was true before: TREx reports centimetres scaled
by `cm_per_pixel` and puts the *head* in `X`, while every other converter wrote
pixels and a keypoint mean. A feature reading `X` across trackers was comparing
a head to a centroid in two unit systems, and nothing on disk recorded either
fact.

- The TREx converter divides the centimetre columns back out, reading the factor
  **from the file** (TREx writes `cm_per_pixel` into every export) rather than
  from mosaic's own parameter — the parameter records what mosaic *passed*, and
  TREx substitutes `meta_real_width / video_width` when it is unset.
- Centimetres are obtained downstream by the `scale-to-cm` feature, from a
  per-video `cm_per_pixel` on the media index. That column is **text**, so empty
  can mean *uncalibrated* rather than `0.0`, and a rescan never clears it.
- `mosaic upgrade-tracks` rescales centimetre-era tables whose raw export has
  been swept, and **refuses** a table that does not record its factor.

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

Notebooks may use sample data not present in the repo; check the first cell
for path expectations before running.

### Invariants a change must not quietly undo

Each of these replaced a silent wrong answer, and each has a test named for it.

- **One parquet writer.** `write_parquet_atomic` (and `atomic_savez`) are the only
  sanctioned ways to write an output. A direct `df.to_parquet(final_path)` leaves a
  torn file where a whole one belongs, and every reuse gate in the tracking layer
  tests for presence. A test fails if a new call site addresses a final path.
- **Tracks are pixels, and `X` is the body centre.** Both hold on every tracker,
  and neither did before: TREx reports centimetres and puts the *head* in `X`. A
  physical unit is obtained by the `scale-to-cm` feature, never stored in the
  table. A converter that writes a scaled column, or puts a landmark other than
  the body centre in `X`, reintroduces a difference that reads as a plausible
  number and is recorded nowhere.
- **A tracker reports; a feature derives.** `mosaic_v1` *forbids* `VX`, `VY`,
  `SPEED`, `ANGLE` and the rest, so a converter cannot compute one and present it
  as a measurement. Heading is the sharpest case: the principal-component fit the
  converters used has an arbitrary sign, and its flips read downstream as real
  turns. Anything wanting one runs `heading` and chooses the method, which then
  enters the run identifier.
- **Unreadable is not empty.** `readable_tracks_table` returns `None` for a table
  that cannot be read and `BridgeCounts(0, 0)` for one that reads and holds no rows.
  The second is a legitimate result -- a video with no detected individuals -- and
  must stay reusable.
- **A run reports what it lost.** A per-entity failure is recorded as an
  `entry_error` run-log event and surfaced as `Result.failed_entries` plus
  `"status": "partial"`; losing every entity raises. Do not add `partial` to
  `runlog.TERMINAL_STATUSES` -- mosaic-api's sweeper reaps that set.
- **Inputs align at one entity level.** `alignment_verdict` decides; the merge
  raises from it. Joining individual-level to pair-level output is a cartesian
  product, not an alignment. `loading.CROSS_JOIN_FEATURES` is the closed escape.
- **An entry is claimed before it is touched.** `open_entry` takes the claim with an
  exclusive create; release and refresh are ownership-checked. One-shot ops claim
  their run root and raise on contention, because two nondeterministic trainers in
  one run root interleave artifacts.
- **One rule resolves an unpinned `Result`.** `track_universe.current_run_id`:
  leaf-of-chain when the runs have edges among themselves, recorded time when they
  are siblings. Do not reintroduce a second rule.
- **`consumed_tracks_composition` is compared, not just recorded.** It is what
  notices a re-conversion from changed sources; the tracks variant identity is
  params-only and does not move.

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

8. **Exactly one distribution may provide `cv2`.** `albumentations` requires
   `opencv-python-headless` and mosaic + `ultralytics` require `opencv-python`; pip
   installs both without complaint because they are different distributions, then
   they overwrite each other's files and merge two ffmpeg builds into one
   `cv2/.dylibs`. The suite then dies with `Trace/BPT trap: 5` somewhere different
   every run, and whichever wheel wins may be the headless one, which has no
   `imshow` -- silently breaking playback. `tests/conftest.py` refuses to start when
   both are present. Keep `opencv-python`.

## Pointers to Deeper Docs

- [`docs/getting-started.md`](docs/getting-started.md) — installation and first run.
- [`docs/guide-pipeline.md`](docs/guide-pipeline.md) — pipeline guide.
- [`docs/adding-a-tracker.md`](docs/adding-a-tracker.md) — wiring a new
  external tracker in. `tracking/common/` owns the run loop; a tracker supplies
  its argv, its settings, its phases and its converter, plus one `TrackingRoot`
  row. `tests/test_tracker_conformance.py` is parametrized over every tracker
  root, so a half-implemented one fails by name.
- [`docs/adding-a-converter.md`](docs/adding-a-converter.md) — writing a track
  converter for files you already have: the file-to-sequence declarations, params
  versus hints, `output_schema`, and where a converter or a custom schema has to
  be registered to reach the CLI.
- [`docs/api/`](docs/api/) — auto-generated API reference (core, pipeline,
  behavior, media, tracking).
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — PR workflow and CLA.
