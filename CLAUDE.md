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
conda install -c conda-forge ffmpeg av py-opencv -y
pip install -e ".[all]"
```

`ffmpeg` (with `ffprobe`) must be on `PATH` — it is used by media indexing and
raw H.264 frame counting.

**`av` and `py-opencv` come from conda so that the environment holds one ffmpeg.**
The PyPI wheels for both vendor a complete ffmpeg build of their own, and two of
them in one process crash it nondeterministically — see pitfall 8. The conda-forge
builds link the `ffmpeg` installed on the line above instead, so there is one
`libavdevice` and one `libSvtAv1Enc`. Nothing here is pinned: `av` satisfies
`av>=18,<19` and `py-opencv` registers both `opencv-python` and
`opencv-python-headless`, so the `pip install` that follows finds every requirement
already met and installs neither wheel. Order matters — conda first, pip second.

A `uv venv` cannot take conda packages, so a checkout that wants this needs a conda
environment. `conda create -p ./.venv python=3.12` puts one at the path
`[tool.basedpyright]` and the editor integration already expect.

Python `>=3.12` is required (`pyproject.toml`).

`mosaic-media` is a sibling package by the same authors, and it resolves from
PyPI like any other dependency — no second clone is needed. To work against its
unreleased `main`, install it editable *over* the released wheel:
`pip install -e "../mosaic-media[io,cli]"`. Do not reintroduce it as a
`[tool.uv.sources]` path: that table is a uv extension pip does not read, so a
path source there makes `pip install -e .` fail outright on a machine without
the sibling. See [CONTRIBUTING.md](CONTRIBUTING.md).

### Optional extras

**A bare `pip install -e .` is a complete analysis install.** Converters,
features, wavelets, clustering, classifiers, overlays and crops all work with no
extra at all, because PyWavelets, h5py and PyTables are base dependencies -- each
gates reading a file the user already has, and the three are ~30 MB together.
`[all]` adds the deep-learning surface. For the full table, see
[docs/installation.md](docs/installation.md). Notable points:

- **`deep-learning` (`torch` + `timm`) is the one gate on PyTorch.** It powers
  the heatmap localizer and all three identity models, and replaced the separate
  `localizer` and `identity` extras, which installed near-identical environments
  once h5py moved to the base. On Linux torch pulls the whole `nvidia-cu12`
  stack -- about 4 GB of wheels -- which is the entire reason the default install
  does not carry it.
- `pose` and `polo` cannot be installed in the same environment — both ship
  under the `ultralytics` distribution name. Both self-reference
  `deep-learning`, so choosing the fork does not cost the identity models. What
  they serve is pose and point **model training and inference**, the paths that
  still import Ultralytics in mosaic's own process. `mosaic track ultralytics`
  needs neither extra: it runs in
  `src/mosaic/tracking/external/ultralytics-env/`, whose own `pyproject.toml`
  declares `lap` (the tracker's linear-assignment solver, in no ultralytics
  extra, so undeclared it gets pip-installed mid-run) and the
  `ultralytics>=8.4.63` floor the four newer tracker backends arrived in. Both
  extras still carry a copy of each.
- **`all` is self-referential** (`["mosaic-behavior[pose,faiss]"]`), so a bundle
  cannot drift from its parts. It excludes `polo` (mutually exclusive with
  `pose`), `yolo-augment` (changes what a training run does),
  `lightning-action` and `movement` (heavy, single-purpose), and `feral` (wants
  its own environment).
- `yolo-augment` installs `albumentations`, which Ultralytics picks up on its own
  and uses to add Blur / MedianBlur / ToGray / CLAHE at p=0.01 to YOLO and POLO
  training. **It is opt-in because it changes what a run does** and nothing
  records which way a run went — not for packaging reasons. It does require
  `opencv-python-headless`, but the documented conda environment satisfies that
  with no wheel; see pitfall 8, which now names the two ffmpeg hazards apart.
- `lightning-action` is capped at `<1.1`: 1.1.0 requires `nvidia-dali-cuda110`
  unconditionally, and PyPI serves it as an sdist only, so without the cap the
  extra fails to install anywhere without CUDA.
- `movement` declares the movement-library integration behind
  `movement-smooth` and `movement-filter-interpolate`. Before it existed those
  were two registered features with no declared dependency at all.
- `faiss` (formerly `gpu`, which promised a GPU and installed `faiss-cpu`) adds
  the `"faiss"` kNN backend for `global-tsne`; the default backend is `"annoy"`
  and needs nothing. On Linux + CUDA, install `faiss-gpu` manually.
- **`imgstore` is not an extra.** Reading a store is native — the package is
  needed only to *write* the fixture stores the suite builds — so it lives in
  the `test` dependency group, which every CI job installs.
- `recommended`, `identity`, `localizer` and `gpu` survive as self-referential
  aliases through 0.12 and are removed in 0.13. They exist only because pip
  *warns* about an unknown extra and carries on: a saved `.[recommended]` would
  otherwise produce a working install with no torch in it and no error to say
  so. `wavelets`, `sleap` and `hdf5` are not aliased — the base provides them,
  so the warning is harmless.
- `feral` installs the FERAL V-JEPA behavior classifier (`FeralFeature`, train +
  infer) from PyPI, as `feral>=1.0,<2`. It runs in-process (not sandboxed like
  keypoint-MoSeq) and **wants an environment of its own, for a different reason
  from everything above**: FERAL pins its dependencies exactly while every
  mosaic requirement is a floor, so pip resolves the pair happily and downgrades
  each one. No install layout fixes that. The upper bound is load-bearing:
  mosaic imports thirteen symbols from FERAL's submodules, ten of them outside
  that package's `__all__`, and the `feral` CI job exists to make a release that
  moves one fail here.
- Mosaic ships no weights — see [NOTICE](NOTICE).

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
pytest -m identity                          # what CI's identity job runs
pytest -m tracker                           # what CI's tracking job runs
pytest -m feral                             # what CI's feral job runs
pytest tests/test_run_feature.py            # one file
pytest tests/test_run_feature.py::test_x    # one test
pytest -k "feature_params"                  # name pattern
pytest -v                                   # verbose
```

Five markers are declared, all in `[tool.pytest.ini_options]`: `slow`, `media`
(needs `ffmpeg` **and** `ffprobe` on PATH), `tracker`, `identity` and `feral`. The
last three are how CI selects its extra jobs, so a new test file in any of those
areas is covered the day it lands rather than when someone remembers to edit the
workflow.

`-m` on the command line **replaces** the `-m "not slow"` in `addopts` rather than
intersecting with it. That is what makes `pytest -m slow` work, and it also means
`pytest -m "not media"` quietly re-enables the slow tests.
`tests/test_pytest_config.py` pins the default so that stays deliberate.

Three tests keep the suite honest about its own environment.
`tests/test_optional_dependency_coverage.py` reads the suite's guards out of its
AST — both `importorskip` calls and the literal `find_spec` probes that back a
two-directional `skipif`, as `feral`'s do — and fails when one names a module no
CI job installs, **or when one guards a module that is a base dependency**, since
a guard that can never fire masks a broken install.
`tests/test_optional_dependency_messages.py` checks the other direction: every
extra named in a `pip install "mosaic-behavior[...]"` hint or passed to
`optional_dependency.require` must be declared, and every self-referential extra
must resolve — a dangling one would make pip warn and install the base, which is
a silently torch-less environment. And `tests/test_pytest_config.py` asserts the
configuration above.

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

**`docs/reference/` is generated wholesale** by `scripts/gen_docs_reference.py`
from the live registries and the Typer app — features, ops, the CLI, and the track
formats. Never edit those files. Run the script with `--write` after changing a
registry, and commit the result; the docs workflow runs it with `--check` and fails
when a committed page no longer matches the code.

Every other page under `docs/` is hand-written prose, and there is no rendered
Python API reference. `docs/api/` held one: fourteen mkdocstrings stubs reaching 87
of 303 public modules, with no gate to keep them honest and nothing that swept them.
A public object is documented by its docstring, read in the source or in an editor.

The site deploys from `.github/workflows/docs.yml` on every push to `main`, and
builds with `--strict` on every pull request. Do not run `mkdocs gh-deploy`: Pages is
fed from Actions, and a local deploy would publish whatever untracked internal docs
happen to be in the working tree.

### CI

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs five jobs on push
and pull request. There is no `.pre-commit-config.yaml`. **Every job installs
the `test` dependency group**, because `tests/conftest.py` demands `imgstore` of
any run with `CI` set and a job without it fails at collection.

- **`test`** — `uv run --no-sync pytest -q`. It inherits `addopts = "-m 'not slow'"`,
  so **slow-marked tests never run in CI**. A change that only breaks a slow test
  goes green; run them locally.
- **`identity`** — the identity-marked suites under a `deep-learning`
  environment, so `pytest.importorskip("torch")` cannot silently skip them.
- **`tracking`** — the `tracker`-marked suites, deliberately with **no** `pose`
  extra, against a real Ultralytics environment built the way a user builds it:
  `uv sync --python 3.12` in `src/mosaic/tracking/external/ultralytics-env/`,
  located by `MOSAIC_ULTRALYTICS_BIN`. Installing `[pose]` here would put
  Ultralytics one careless import away from mosaic's own process and buy
  nothing. Without the built environment the preflight comparison against the
  shipped tracker tables would skip green and prove nothing, so
  `MOSAIC_CI_TRACKING=1` promotes an environment that does not resolve from
  "skip" to "broken environment".
- **`feral`** — the FERAL-marked suites under a `feral` environment. Mosaic
  imports ten symbols FERAL's `__all__` does not carry, so this is what notices a
  release that moves one. Its own runner because FERAL's exact dependency pins
  would otherwise re-resolve the environment the other jobs share.
- **`lint`** — `uvx ruff check` **and** `uvx ruff format --check` over `src` and
  `tests`, on push as well as on pull request. Formatting is a merge gate, not a
  suggestion. It was once scoped to the files a pull request touched, which meant
  it never ran at all on a repository that pushes to `main`.

Not gated: `basedpyright`, which carries a pre-existing backlog of ~11k
strict-mode errors. So before reporting work done, run what CI will not:

```bash
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

### Continuous groups

`continuous_groups:` names the groups whose sequences are **time divisions of one
recording** rather than independent trials. A 6-hour session is either one discrete
sequence covering all 6 hours, or a continuous group whose sequences are its
half-hour divisions -- never a group of independent half-hours.

The declaration asserts two things nothing else records, and mosaic acts on both:

- Its sequences are adjacent in time, so a feature may read across their boundary
  (`overlap_frames`), with neighbours ordered by their recorded frame extent rather
  than by name.
- Its `frame` column is one axis spanning the group, so its media resolves as **one
  shared timeline** -- every sequence resolves the group's whole ordered clip set,
  ordered by `(where the sequence starts, video_order)`. `video_order`'s counter
  restarts per `(group, sequence, camera)`, so it cannot order a group by itself.

**Declaration and measurement are both required.** No measurement can establish that
two sequences divide one recording rather than being two recordings numbered
consecutively; no declaration can be trusted about an axis that is there to be read.
So `overlap_frames > 0` checks the declaration *and* verifies the recorded extents
are disjoint and increasing, naming both sequences and both ranges when they are
not. A continuous group's `group` may not be empty -- the one place "group is an
optional namespace" does not hold.

Nothing in mosaic produces a continuous dataset yet: every mechanism that meets a
split recording collapses it into one sequence. Making one means converting with
frames numbered across the whole recording (sum the frame counts of the earlier
files as an offset), then declaring the group with
`ds.set_continuous_groups([...])`. A dataset converted before the extent was
recorded reads blank and is refused; `ds.measure_frame_extents()` fills it in.

### Plugin registries (everything is a plugin)

mosaic uses decorator-based registries; new functionality almost always means
"register a new plugin," not "edit a hot path."

| Registrar                  | Registry            | Lives in                          |
| -------------------------- | ------------------- | --------------------------------- |
| `@register_feature`        | `FEATURES`          | `behavior/feature_library/`       |
| `register_track_converter` | `TRACK_CONVERTERS`  | `core/track_converter.py` (impls in `core/track_library/`) |
| `register_label_converter` | `LABEL_CONVERTERS`  | `core/label_converter.py` (impls in `behavior/label_library/`) |
| `@register_op`             | `OPS`               | `core/pipeline/ops.py`            |

**`register_label_converter` is called, never decorated.**
`behavior/label_library/__init__.py` imports each converter module and then calls it
on the class, rebinding the result — so importing a converter module does not
register it, and the registry holds exactly what that file names. A label converter
class carrying a decorator instead will not appear.

`register_track_converter` accepts both spellings and is mostly used as a decorator
(`track_library/deeplabcut.py:213`); `track_library/calms21.py` uses both, calling it
for one class and decorating the other.

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

The `media` features (`overlay`, `egocentric-crop`, `interaction-crop-pipeline`)
use the same protocol and caching machinery. They live under
`visualization_library/` but are categorized `media`, not visualization: they
write image and video artifacts something else reads. Egocentric crops are the
input all three identity models take, and `overlay` renders the annotated video a
biologist looks at — a feature so that a graph can end on the deliverable rather
than one step short of it, and so that the video is addressed by `run_id` like
everything else.

`feral` (the FERAL V-JEPA behavior classifier) is a global fit-then-apply
feature but runs **in-process** — it imports the installed `feral` package
directly, unlike the sandboxed keypoint-MoSeq runner in
`feature_library/external/`. Each feature also declares a `category` used for
grouping/display; beyond per-frame and global, the taxonomy includes `summary`
(per-sequence aggregations, e.g. `frame-aggregate`), `tag` (e.g.
`id-tag-columns`) and `media` (writes an image or video artifact something else
reads, e.g. `egocentric-crop`, `overlay`). There is deliberately no visualization
category: a rendered artifact is `media` like any other, and a second,
non-artifact status mechanism beside the first is what that avoids.

### Pipeline package

[`src/mosaic/core/pipeline/`](src/mosaic/core/pipeline/) owns data loading,
output writing, dependency resolution, and indexing. **Features own
computation only.** The public typed surface lives in `pipeline/types/`:
`Params`, `Inputs`, `Result`, `ArtifactSpec`, `InputStream`,
`DependencyLookup`, `FeatureLabelsSource`, `GroundTruthLabelsSource`.

### The inventory: what a dataset holds

[`core/pipeline/inventory/`](src/mosaic/core/pipeline/inventory/) answers "what
has been computed here, with what params, over which entries" -- the question
`mosaic sequences` (a tracks listing), `mosaic runs`/`status` (job-log surfaces
reporting *attempts*) and `mosaic features list` (the registry) each do not.
`inventory(ds)` in the library, `mosaic inventory --json` at the CLI.

- **Coverage is which keys exist, never a flag**, and the key type differs by
  kind: `(group, sequence)` for a feature run or tracks variant,
  `(group, sequence, camera)` for a frame run (the cameras of one recording
  share an entry), the run id for a trained model, and a media row's
  `video_uuid` for a transcode. **Transcode has no run-addressed directory at
  all**, so a single `coverage(storage, run_id)` signature makes an
  already-clean corpus read as permanently incomplete forever.
- **Status is derived, never stored**: `absent` / `partial` / `complete` /
  `complete-but-drifted` / `inconsistent`, decided in one function. Files ahead
  of index rows is damage only on a *finished* run -- outputs are written before
  their rows.
- **Truth is on disk.** Every view is a cache, thrown away rather than
  reconciled; there is deliberately no `.mosaic/inventory.json` and no
  filesystem watcher. Stale is safe, so `InventoryCache.revalidate()` stats the
  index files rather than subscribing to anything.
- **`core` does not import `tracking`**, so tracker runs, frame runs and trained
  models arrive through `register_inventory_contributor`, the same import-side-
  effect seam `register_reconcilable_index` uses. A kind nobody registered is
  reported in `unavailable_kinds`, never as zero artifacts.
- **`read_run_params` is the only reader of a feature run's `params.json`**, and
  distinguishes *absent* from *unreadable* -- the sidecar write is best-effort,
  so a run root with none is a real state. It is tolerant by design: a block it
  cannot read is dropped and the rest is returned.

### The graph: a pipeline as a file

[`core/pipeline/graph/`](src/mosaic/core/pipeline/graph/) is the pipeline as a
**document**: a JSON recipe of steps and the references between them, validated
against the real registries, resolved against a dataset into a plan, and run.
Deliberately separate from the live-object `Pipeline`, which holds feature
*classes* and a `CallbackStep` wrapping a live callable and so has no wire form.
`mosaic pipeline validate|plan|show|run --recipe @file.json` at the CLI.

- **The recipe is portable; the request beside it holds a submission's choices.**
  A recipe never carries a resolved `run_id` (those are dataset state) and never
  carries an entry list (those are about one dataset), so the same file runs over
  several. `Request` holds the narrowing, `bind` (an out-of-graph artifact pinned
  by the submission), `allow_partial`, `max_concurrent_steps`, the `step_id →
  execution_id` map and the resolved `step_versions`. `submit_request` writes
  both files — a `<digest>.json` copy under `<dataset>/.mosaic/pipelines/` and
  the request under `.../requests/<rid>.json` — and assigns every step its
  attempt id **before anything runs**, which is what makes the document complete
  at submit rather than filled in as work lands.
- **A step is addressed by name, not spelled out.**
  `mosaic run --manifest <path> --graph-request <rid> --step <id> --execution-id
  <eid>` is strictly more expressive than the spelled-out form, because several
  arguments that reach a feature's identity have no flag at all — the entry
  narrowing, the frame filters, the overlap width — and a step re-planning itself
  reads all of them out of the recipe. **The request path is derived from
  `--manifest`'s parent and there is deliberately no second path flag**: a path
  mosaic-queue does not know about is one `translate_manifest_path` cannot
  rewrite for a substrate that mounts the dataset elsewhere. `--overwrite` stays
  an argv flag, being a property of an attempt rather than of the recipe, and is
  now *refused* with `--kind` rather than accepted and dropped.
- **One step body, two drivers.** `execute_step` is what a queued job runs and
  what `run_pipeline` loops over, so the preflight, the parent pinning and the
  failure record cannot drift by being edited on one path only.
- **A step pins its parents from their run-logs, and only the feature ones.**
  Resolving an input by feature *name* falls through to
  `track_universe.current_run_id`, whose sibling rule is wall clock — so two
  requests running one feature with different params on one dataset would
  cross-bind. **Every ancestor is pinned, not only the immediate parents**,
  because identity chains. An op parent is not pinned (its identity is a function
  of the params the plan handed it, so the two cannot differ); what it *is*
  checked for is the tracks variant it produced, via `variant_for_producer_run`.
  An ancestor with no run-log is a cache hit from an earlier request, not a
  fault.
- **A refusal is a reserved exit code, never a new terminal status.**
  `REFUSED_EXIT_CODE = 65`, the run-log status stays `failed`, and the reason
  travels in `error_json` as one of a closed `RefusalReason` set. Do **not** add
  a member to `runlog.TERMINAL_STATUSES` — three repositories read it and
  mosaic-api's sweeper reaps it, which is why `partial` was kept out too.
- **`allow_partial` answers exactly one refusal.** A shortfall is a question
  about *how much*; a digest mismatch, a moved version, a disagreeing variant and
  an upstream that finished having written nothing are not, and no flag unlocks
  them.
- **Waiting and quarantined are different, and collapsing them is expensive.** An
  entry inside its backoff needs a few more seconds and simply narrows this
  attempt; an entry past `QUARANTINE_AFTER` will not succeed and is what
  `allow_partial` decides about. Treating a wait as a verdict would let one
  gesture permanently drop an entry from a fit.
- **The failure record is the one durable non-derived state**, under
  `<dataset>/.mosaic/claims/`: attempts keyed `(storage_name, run_id, group,
  sequence)` **accumulate across resubmits** (a counter reset by the cheap
  recovery would bound nothing), while the *exclusion decision* is request-scoped
  desired state, so one request's scientific call cannot bind another's. The
  lease is the existing `try_create_inflight` over the run root, reused rather
  than reinvented — which is what already makes two concurrent training
  dispatches safe. G4 forbids stored *status*, not this.
- **A request is one-shot.** `request_rollup` reads the steps' run-logs only —
  no registry, no planning — and closes on `finished` / `failed` / `cancelled`. A
  branch that failed while a sibling is still running is **not** terminal yet.
- **One place per fact.** Cross-step references sit at the exact site they
  substitute, so there is no `edges` array to drift from the bodies; `edges()` is
  a derived read-only view. The one explicit list is `after`, which is
  ordering-only and corresponds to nothing in any payload.
- **`plan_pipeline(ds, recipe)` resolves every step in one topological walk and
  submits nothing.** Step A's identity is a function of its params, B's of its
  params plus A's identity, C's of B's. It closes because every term is in the
  recipe or on disk beforehand: a feature-to-feature edge reads nothing, a tracks
  variant is *minted* from the recipe's settings rather than read back from
  tables an op has not written, and a `scope_dependent` step's entry set comes
  from `intended_scope`.
- **A resolved `run_id` is never load-bearing at execution.** It drives the
  preview, the estimate, validation and the decision to enqueue; it never skips a
  step and never enters a downstream job's payload. Every step resolves its own
  identity at its own start. Do not "optimize" this away by trusting a submitted
  identifier.
- **One answer to what a step will be called.** `resolve_feature_identity`
  ([run.py](src/mosaic/core/pipeline/run.py)) is that answer, and both the graph
  planner and the live `Pipeline` call it.
- **Only `resolve.py` may import `FEATURES`,** and only inside its functions.
  Parsing a recipe, ordering it, listing parents, deciding a lane and rendering a
  status view must not pay the multi-second feature-library import, because the
  gate runs far more often than a submit does.
  `tests/test_graph_imports.py` holds the line.
- **`can_connect` / `can_join` answer with no dataset at all**, from declarations
  (`declaration_catalog()`), so a canvas refuses a wire as it is drawn. The
  sharpest refusal is `can_join`'s: a multi-input join of mismatched entity
  granularity is a silent per-frame cartesian product.
- **`reject_unless_valid` runs before the dataset is touched** and reports every
  problem rather than the first. `overwrite` is refused in `params` on presence,
  and `extract-frames` is excluded by *ownership* -- mosaic-api embeds its frozen
  identifier in annotation image paths.
- **An op step's scope comes from the plan, not the recipe**, through
  `Op.scoped_params`. `TranscodeOp` overrides it, being the one op whose params
  refuse an unscoped run and whose identity moves with what it covers.
- **A `scope_dependent` step is asked for all of its scope, never the
  remainder** -- its identity *is* its scope, so a fit over what is left under
  the name of a fit over everything is exactly what the scheme prevents. A
  scope-free step gets the remainder, as it should.

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
│   │   ├── index_csv.py        # generic typed IndexCSV + index_records
│   │   ├── inventory/          # what a dataset holds: coverage, status, params.json
│   │   ├── graph/              # a pipeline as a file: recipe, plan, submit, run a step
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
│   └── visualization_library/  # overlay renderer + the media features (overlay, crops)
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
    ├── ultralytics_track/      # Ultralytics MOT: one gated phase, own env, reached as a subprocess
    └── external/               # the Ultralytics environment definition + the programs that run in it
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

`inventory(ds)` reads across all of the above and reports what is there; it
writes nothing. `mosaic inventory --json` is the same answer at the CLI.

A recipe drives any run of the above as one graph. `plan_pipeline(ds, recipe)`
says what each step would be called and what is already done, `run_pipeline`
executes it here, and `mosaic pipeline validate|plan|show|submit|run|status` is
the same at the CLI. The recipe is copied to
`<dataset>/.mosaic/pipelines/<digest>.json` on first use, so the dataset records
which pipelines were applied to it, and each submission lands beside it:

```
mosaic pipeline submit           → .mosaic/pipelines/<digest>.json      (the recipe)
                                 → .mosaic/pipelines/requests/<rid>.json (the submission)
mosaic run --graph-request <rid> --step <id>
                                 → .mosaic/runs/<execution_id>.jsonl    (the attempt)
                                 → features/<name>/<run_id>/            (the work)
   ↑ failures counted under .mosaic/claims/, which is the only durable state
     that cannot be re-derived from the artifacts
```

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
(e.g. `convert-calms21_npy.0.3-6bb5efbf05`). The version is a *visible segment*,
not a hash term, so bumping it does not re-derive anything.
`extract-frames` is carved out and frozen — mosaic-api embeds its identifier in
annotation paths.

### Track schema

Standardized tracks are validated by `core/schema.py`. Four schemas are
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
- **`mosaic_cm_v1`** — the same contract as `trex_v2`, in **centimetres**, with
  `X`/`Y` still the body centre. It exists because the unit is sometimes not
  recoverable: TREx scaled its output long before it recorded `cm_per_pixel`
  (2025-02-18, TREx 2.0.0), and nothing can divide back out a factor nobody
  wrote down. Deliberately **its own schema family**, extending nothing — the
  columns mean the same things as `mosaic_v1`'s and not the same numbers, so a
  scope resolving both is refused. `STANDARD_COLUMNS` is shared between the two
  families precisely because the second cannot inherit it.
- **`trex_v1`** — the legacy schema, kept registered permanently because a real
  archived dataset is in it. Its spatial columns are **centimetres** and its `X`
  is a head position. Not the schema for new centimetre data — it also requires
  keypoints and does not require `X`/`Y`; `mosaic_cm_v1` is.

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
  reading a store no longer needs the `imgstore` package (native decode). The
  package is needed only to *write* stores, as the test fixtures do, so it sits
  in the `test` dependency group rather than in an extra.

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

Google-style. A public object's docstring is the only place it is documented —
there is no rendered API reference to carry the explanation instead.

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
keypoint-MoSeq. [NOTICE](NOTICE) records this arrangement and the other
third-party terms — notably TRex, which requires a paid license for company use.
A reader-facing licensing page is being written and is not yet published, so
nothing under `docs/` carries these terms today.

### `tracking/external/` is where Ultralytics runs

Same shape, different license. Ultralytics is AGPL-3.0, and a program that
imports it is one work with it, so `mosaic track ultralytics` drives it as a
separate program: `tracking/external/ultralytics-env/` declares an environment
the user builds with `uv sync --python 3.12`, `tracking/external/runner/` holds
the two files that run inside it, and what crosses between them is a JSON request
file, a JSON response file and progress lines on stdout.
[`ultralytics_track/run.py`](src/mosaic/tracking/ultralytics_track/run.py) is
mosaic's side: `ULTRALYTICS_ENV` on the same five-step location ladder every other
external tool uses, plus `probe_ultralytics`, `ultralytics_tracker_defaults` and
`run_ultralytics_tool`. One subprocess per entry.

`tests/test_ultralytics_separation.py` holds both directions: no mosaic module
outside the runner and two named `pose_training` modules may import Ultralytics,
and the runner may not import mosaic. **The separation covers tracking only.**
`train-pose`, `train-points`, `infer-pose` and `infer-points` still import
Ultralytics in mosaic's process, which is what `pose` and `polo` are for, so a
claim that mosaic installs no AGPL dependency is false today.

Two consequences a change should not undo. The tool is handed a video *path*, so
an imgstore recording needs `mosaic run --kind export-store` first, exactly as the
other three subprocess trackers do (`common/tool_input.py` is that boundary and
raises naming the command); tracking a store natively is a capability this cost.
And the environment is excluded from the uv workspace and has its own basedpyright
execution environment pinned at `python3.12`. Built with any other interpreter,
its packages land where nothing looks and the runner type-checks against no
Ultralytics at all.

## Working with Notebooks

Reference end-to-end examples (not test fixtures):

- [`notebooks/calms21-template.ipynb`](notebooks/calms21-template.ipynb) —
  canonical end-to-end (manifest → features → wavelet/scaler/t-SNE →
  clustering → XGBoost classifier → visualization).
- [`notebooks/collective-motion-shiners.ipynb`](notebooks/collective-motion-shiners.ipynb) —
  a track converter written inside the notebook, then the collective-motion
  features across four group sizes.
- [`notebooks/collective-motion-zebrafish.ipynb`](notebooks/collective-motion-zebrafish.ipynb) —
  a converter for a tracker with **no pose keypoints** (Ctrax/JAABA `trx`), then
  collective motion and the `nearest-neighbor` → `nn-delta-response` →
  `nn-delta-bins` social-force chain.

Notebooks may use sample data not present in the repo; check the first cell
for path expectations before running.

### Invariants a change must not quietly undo

Each of these replaced a silent wrong answer, and each has a test named for it.

- **One parquet writer.** `write_parquet_atomic` (and `atomic_savez`) are the only
  sanctioned ways to write an output. A direct `df.to_parquet(final_path)` leaves a
  torn file where a whole one belongs, and every reuse gate in the tracking layer
  tests for presence. A test fails if a new call site addresses a final path.
- **One tracks file per sequence, and the frame axis is global over the enclosing
  unit.** The first half is structural: `tracks_table_path` addresses one parquet per
  `(variant, group, sequence)` and the index holds one row per
  `(run_id, group, sequence)`. The second says what `frame` counts *from*, and it
  applies at two levels. In a **discrete** dataset the enclosing unit is the
  sequence: a multi-clip sequence numbers frames across its ordered clips, which is
  what `ConcatenatedTimeline` and `MultiVideoReader` already build and what
  `joins_sources=True` makes a tracker deliver. In a **continuous** group -- one
  declared in `continuous_groups`, whose sequences are time divisions of one
  recording -- the enclosing unit is the group: frames are numbered across the whole
  recording, and its media resolves as one shared timeline for the same reason.
  Never a group of independent divisions each restarting at zero; that makes one
  frame number name a different moment in each, which is exactly what
  `overlap_frames` refuses. Note the invariant is *not* enforced on the write paths
  today -- `merges_per_sequence` is the individual axis, not the time axis, and only
  TREx joins an entry's clips -- so a converter fed per-clip files can still produce
  colliding frames inside one sequence.
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
- **A run also reports what it holds, and the run-log is the only place it can.**
  `entries_written`, `cache_hit` and `tracks_variant` fold onto `RunLogSnapshot`
  beside `entries_failed`, because a queue spawns `mosaic run` with stdout *and*
  stderr on `DEVNULL` and so can never see a stdout payload or a returned
  `Result`. Three points a change must not undo:
  - **`entries_written` counts what the scope holds, cache hits included** -- so a
    resumed run and a fresh one report the same number, which is what lets one
    value be read as coverage without knowing which kind of run wrote it. It is
    last-write-wins, where `entries_failed` accumulates. For a *tracker* it is
    attempted-minus-lost, **not** the index-row count: a failed bridge still
    writes a row, because the tool output is durable and a re-run adopts it and
    redoes only the conversion.
  - **`tracks_variant` is what a run read**, never what an op produced. Those are
    different relations and one key cannot hold both.
  - **Zero / `False` / `""` mean *not reported*.** A job that never asked the
    question writes nothing, the same convention as the tracks index's blank
    `n_keypoints` cell meaning *unknown* rather than zero.
  Adding an event kind is safe for an older reader by construction --
  `reduce_run_log` is an if/elif fold, so an unrecognised `ev` advances liveness
  and changes nothing else. Emit **before** the terminal event and before the
  context closes: `JsonlRunLog._emit` returns silently on a closed file, so a late
  write is dropped without an error, and anything after `finished` never reaches
  the ledger.
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
  params-only and does not move. `consumed_media_composition` on
  `TrackerRunRowBase` is the same rule one level up: a tracker's identity is its
  settings with no term for the media it read, so without it a re-transcode
  leaves every tracker run reading as current over different pixels.
- **An index lock that cannot be taken is not the same as one that is held.**
  `index_lock` classifies the errno: a filesystem that refuses locking raises
  `IndexLockUnsupported` immediately rather than spinning the full timeout and
  then blaming a writer that does not exist. It never degrades to an unlocked
  write, and an unfamiliar errno keeps polling rather than guessing.

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
6. **`all` is what belongs in one environment, not everything.** It resolves to
   `pose` + `faiss`. `polo` is excluded because it cannot coexist with `pose`,
   `yolo-augment` because it changes what a training run does, `movement` and
   `lightning-action` because they are heavy and single-purpose, and `feral`
   because it re-resolves the environment. Each exclusion has its own reason;
   don't quietly fold any of them in, and don't restate the reasons as one.
7. **0.x APIs may move.** Per [CONTRIBUTING.md](CONTRIBUTING.md), breaking
   changes still warrant explicit discussion in an issue first.

8. **One ffmpeg build per environment.** Two independent copies of ffmpeg in one
   process crash it: the suite dies with `Trace/BPT trap: 5` at a different point
   every run, and on macOS the Objective-C runtime warns on every import that
   `AVFFrameReceiver` is implemented twice. **These are two different hazards
   with different blast radii, and running them together is what once put
   `albumentations` in quarantine for a reason that does not apply:**

   - **Two wheels providing `cv2` — plain pip only.** `albumentations`,
     `lightning-action` and `movement` (via `pyvideoreader`) require
     `opencv-python-headless` while mosaic and `ultralytics` require
     `opencv-python`. pip installs both without complaint -- they are different
     distributions -- and they then overwrite each other's files and merge two
     ffmpeg builds into one `cv2/.dylibs`. **The documented conda environment is
     immune**: one conda-forge `py-opencv` registers *both* pip distribution
     names for its single build, so those extras resolve with no wheel installed
     at all. `tests/conftest.py` refuses to start when it finds two *builds*, and
     tells conda's one-build-two-names case apart by `INSTALLER`.
   - **`av` and any `cv2` wheel each vendoring their own ffmpeg.** Both bundle a
     complete build, so two collide even when only one provides `cv2`
     (`av/.dylibs/libSvtAv1Enc.4.1.0` against `cv2/.dylibs/libSvtAv1Enc.3.0.2`).
     **Unaffected by which `cv2` flavor you pick** -- the wheels carry the same
     payload -- so no dependency edit fixes it. It does not arise on Windows,
     where OpenCV's ffmpeg is a separate lazily-loaded DLL.

   **Conda is one route to one ffmpeg, not the only one.** conda-forge's `av` and
   `py-opencv` link the `ffmpeg` installed beside them rather than vendoring
   their own, which is why "Environment setup" installs them before pip runs. But
   `apt install python3-av python3-opencv` and `pip install av --no-binary av`
   reach the same invariant, and mosaic's **Linux CI installs `av` and
   `opencv-python` as plain wheels against an apt `ffmpeg`, on every job, and has
   never crashed**. Treat conda as required on a macOS workstation and as one
   option elsewhere.

## Pointers to Deeper Docs

- [`docs/installation.md`](docs/installation.md) — the environment, the extras, and
  the four tools mosaic drives but does not install.
- [`docs/dataset.md`](docs/dataset.md) — what a dataset is: the manifest, the named
  roots, the indexes, and how a derived directory gets its name.
- [`docs/guides/`](docs/guides/) — tracking, analysis and pipelines, one page per
  thing a reader is trying to do. `guides/pipelines/chain-steps.md` replaced the
  old `guide-pipeline.md`.
- [`docs/guides/tracking/write-a-converter.md`](docs/guides/tracking/write-a-converter.md)
  — writing a track converter for files you already have: the file-to-sequence
  declarations, params versus hints, `output_schema`, and where a converter or a
  custom schema has to be registered to reach the CLI.
- [`docs/concepts/`](docs/concepts/) — why the structure is the way it is: tracks
  and units, feature composition, reproducibility, pipelines as documents.
- Wiring a **new external tracker** in is not a published page. `tracking/common/`
  owns the run loop; a tracker supplies its argv, its settings, its phases and its
  converter, plus one `TrackingRoot` row. `tests/test_tracker_conformance.py` is
  parametrized over every tracker root, so a half-implemented one fails by name,
  and it reads `docs/drafts/adding-a-tracker.md` when that draft is present.
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — PR workflow and CLA.
