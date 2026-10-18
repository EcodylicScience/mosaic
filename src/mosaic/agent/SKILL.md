---
name: mosaic
description: >-
  Use when code imports mosaic, when running the mosaic CLI, when editing a
  dataset.yaml, or when working with tracks, features, pipelines, trackers or pose
  models. Gives the workflow, where answers for the installed version live, and the
  dataset rules.
---

# Using mosaic in a project

You are working in the user's project, which uses mosaic (the `mosaic-behavior`
distribution, imported as `mosaic`) as a library and a CLI. This guide ships with the
installed mosaic and matches its version. The block after it names this installation's
`mosaic` executable, its Python interpreter and its documentation.

- **Do not modify mosaic's installed source**, including an editable install of a
  clone. Extend mosaic from the user's project through its plugin seams: a feature, a
  track converter, a label converter.
- **Report a defect upstream** at https://github.com/EcodylicScience/mosaic/issues with
  the command, the full error and the `mosaic-behavior` version, which
  `pip show mosaic-behavior` prints and `mosaic.version.installed_version()` returns. An empty
  string there means mosaic is running from a source tree that is not installed, and
  the report then needs that tree's commit. Work around it in the user's code
  meanwhile.

## Mosaic in brief

Mosaic runs trackers (TRex, SLEAP, Lightning Pose, Ultralytics) or imports tracks other
tools made, standardizes every table to one schema, and computes **features** from it:
kinematic, social, collective-motion and spectral measures, embeddings and clusters,
supervised behavior classifiers, identity models and annotated videos. Before tracks
exist, it samples frames for annotation and trains pose models.

## The workflow end to end

```bash
mosaic init study --name "Experiment 1"
mosaic sources add -m study/dataset.yaml --kind media --path /data/day1 --extensions .mp4
mosaic scan -m study/dataset.yaml
mosaic track trex -m study/dataset.yaml --set track_max_individuals=4
mosaic run -m study/dataset.yaml --feature speed-angvel
mosaic inventory -m study/dataset.yaml
```

For tracks made elsewhere, skip `mosaic track`. Declare them as a `--kind tracks` source
with `--patterns` and `--src-format`, scan, then run `mosaic convert-tracks -m
study/dataset.yaml`. The same work from Python:

```python
from mosaic.behavior.feature_library import SpeedAngvel
from mosaic.core.dataset import new_dataset_manifest, open_dataset
from mosaic.core.manifest import MediaScanSource, TracksScanSource
from mosaic.core.pipeline.inventory import inventory
from mosaic.tracking import run_trex
from mosaic.tracking.trex import TrexParams

ds = open_dataset(new_dataset_manifest("Experiment 1", "study"))
ds.add_scan_source(MediaScanSource(id="day1", path="/data/day1", extensions=(".mp4",)))
ds.scan_media()
run_trex(ds, TrexParams(track_max_individuals=4))
# For imported tracks, in place of run_trex:
# ds.add_scan_source(TracksScanSource(id="day1-tracks", path="/data/day1",
#                                     patterns=("*.npz",), src_format="trex_npz"))
# ds.scan_tracks(); ds.convert_all_tracks()
result = ds.run_feature(SpeedAngvel())
inventory(ds)
```

Reopen a dataset with `open_dataset("study/dataset.yaml")`. The bare `Dataset(path)`
constructor reads nothing, and every accessor on it fails.

## Finding answers for this version

Prefer these sources, in this order, over memory and over the website. The website
follows mosaic's main branch, which may differ from this installation.

1. **The local docs**, when the block below names a directory. A pip install carries
   no `docs/`, and the block then names the website instead. The directory, when it
   exists, also says which file each site URL in this guide maps to.
2. **`mosaic --help`** and `mosaic <verb> --help`.
3. **The registries:** `mosaic features list`, `mosaic features describe <slug>` (a
   table of its params; `--detail schema` adds the JSON Schema), `mosaic pipeline
   catalog <name>` (the same for any op), `mosaic tracking list`, `mosaic tracking
   describe <kind> --json`, and `mosaic pipeline catalog --json`, which also says why
   a feature is unavailable here and what to install.
4. **The generated reference** (https://ecodylicscience.github.io/mosaic/reference/):
   features with their class paths, ops, media steps, the CLI and the track formats.
5. **`help(obj)`** or the docstring in the source. No Python API reference is published.

When `mosaic` is not on `PATH`, run `<python> -m mosaic` with the interpreter below.

## Python entry points

- `mosaic.core.dataset`: `open_dataset`, `new_dataset_manifest`, and `Dataset`, whose
  methods scan (`scan_media`, `scan_tracks`, `scan_labels`), convert
  (`convert_all_tracks`, `convert_all_labels`), run (`run_feature`) and read
  (`load_tracks`).
- `mosaic.core.manifest`: `MediaScanSource`, `TracksScanSource`, `LabelsScanSource`.
- `mosaic.core.scope`: `Scope`, which narrows a feature run or a pipeline, to
  entries (`Scope(entries=[(group, sequence)])`) or to whole groups
  (`Scope(groups=["train"])`).
- `mosaic.behavior.feature_library`: the feature classes, such as `SpeedAngvel`,
  `HeadingFeature` and `ScaleToCm`. `Inputs` and `Result` from
  `mosaic.core.pipeline.types` chain one feature onto another.
- `mosaic.tracking`: `run_trex`, `run_sleap`, `run_litpose`, `run_ultralytics` and
  `extract_frames`. Each tracker's parameter model is in `mosaic.tracking.<tool>`.
- `mosaic.core.pipeline.inventory`: `inventory`. `mosaic.core.pipeline.graph`:
  `load_recipe`, `plan_pipeline`, `run_pipeline`.
- `mosaic.core.pipeline`: `load_values` reads columns of runs and labels into one
  frame, aligned row by row. Name a column with `ResultColumn` and labels with
  `GroundTruthLabelsSource`, both from `mosaic.core.pipeline.types`.
- `mosaic.core.evaluation`: `align_predictions_to_labels` pairs a run's rows with
  their labels, `classification_scores` scores predicted classes,
  `cluster_label_scores` a clustering, and `select_scored_rows` chooses the
  perspective of pair rows to score.

## Installation

Commands and environments are on https://ecodylicscience.github.io/mosaic/installation/.

- Mosaic installs from PyPI as `mosaic-behavior`, imported as `mosaic`. Never run
  `pip install mosaic`, which is an unrelated project. `ffmpeg`, `av` and `py-opencv`
  come from conda-forge before `pip install "mosaic-behavior[all]"`, which keeps one ffmpeg
  build in the environment. On macOS use conda and do not add pip wheels of `av` or
  OpenCV.
- An extra named in an older install command may be retired. pip only warns and
  installs the base without it. Check the retired names on the installation page.
- Install the `feral` extra in a separate environment, since its exact pins downgrade
  mosaic's dependencies.
- TRex, SLEAP, Lightning Pose, Ultralytics, POLO, keypoint-MoSeq and VAME each run in an
  environment the user builds, and mosaic finds each through the variables the
  installation page lists. The Ultralytics, POLO, keypoint-MoSeq and VAME environments
  are built from files in the mosaic repository that a pip install does not carry, so
  they need a clone. Do not install Ultralytics into mosaic's environment.
- Licenses differ. keypoint-MoSeq is for non-commercial research and academic use
  only, and mosaic refuses to start it until the user sets the acceptance variable.
  Setting it is the user's assertion. Ask, and do not set it yourself. TRex needs a
  paid license for company use. Ultralytics is AGPL-3.0.

## Dataset files

A dataset is a directory with `dataset.yaml` and named roots
(https://ecodylicscience.github.io/mosaic/dataset/).

- The user's files go in `media_raw/`, `tracks_raw/` or `labels_raw/`, or stay where
  they are as a declared source. Mosaic writes everything else.
- Do not write into `tracks/`, `labels/`, `features/`, `models/`, `media/` or
  `_tracking/`. Produce outputs through `run_feature`, `mosaic run` and the ops, which
  keep the indexes and run ids consistent.
- Do not hand-edit an `index.csv`, delete a `*.lock` file, or edit `.mosaic/`. Keep
  `.mosaic/claims/`, the only record of which entries are known to fail.
- Change `dataset.yaml` with `mosaic sources`, `mosaic tags`, `mosaic notes` and
  `mosaic libraries`, or the matching `Dataset` methods. A top-level key added by hand
  survives. A newer `manifest_version` raises `ManifestVersionError`. Upgrade mosaic
  then, and leave the version as it is.
- Derived and temporary folders cost only time to delete. Reclaim space with `mosaic
  sweep-tracking`, `prune-media` and `prune-joined`. `mosaic reindex` drops index rows
  whose files are gone. All four are dry runs until given `--apply`. For superseded
  feature runs, plan with `mosaic pipeline plan --cleanup superseded`, which lists
  them, and submit with `--cleanup superseded --expect-plan <fingerprint>`.

## Bringing data in

- An **entry** is a `(group, sequence)` pair. Neither name may contain `/`, `\` or NUL.
  Join hierarchy levels with `__`, which `mosaic.core.helpers.parse_hierarchy` splits.
- `group` may be empty. It is a namespace, and it does not define analysis groups. To
  run over an arbitrary set, pass `Scope(entries=[(group, sequence), ...])` or
  `--entries group:sequence`. `groups` and `sequences` combine as a cross product.
- A scan replaces the rows its sources claim and keeps every other row. A files source
  (`mosaic sources add --file ...`) claims only the paths it lists.
- A recording split into clips is one entry, with frames numbered across its clips.
  Declare a continuous group only for parts of one recording kept as separate
  sequences, following
  https://ecodylicscience.github.io/mosaic/dataset/#declare-a-continuous-recording.
- Labels: https://ecodylicscience.github.io/mosaic/guides/analysis/import-labels/.

## Track tables

The rules are in https://ecodylicscience.github.io/mosaic/concepts/tracks/.

- Every spatial column is in **video pixels**, and `X`/`Y` are the **body center**, for
  every tracker. For centimeters, record the scale with `ds.set_media_calibration(...)`
  and run the `scale-to-cm` feature. Do not rescale a tracks table yourself.
- Velocity, speed and heading angle come from features. The `mosaic_v1` schema refuses
  them in a tracks table. Run `speed-angvel`, or `heading` with its method chosen.
- Keypoints are optional. Features defined on keypoints refuse a table without them.
- `frame` counts media frames from 0 across an entry's clips (imgstore: `frame_index`).
- A format mosaic does not read needs a converter in the user's project:
  https://ecodylicscience.github.io/mosaic/guides/tracking/write-a-converter/.
- Read raw `.h264` streams (Raspberry Pi) only through mosaic, which measures their
  frame count. Other readers misjudge a bare stream.
- Run `mosaic upgrade-tracks` on TRex tables an older mosaic wrote in centimeters.
- TRex tables an older mosaic tracked are timed at TRex's whole-number frame rate, not
  the media's measured one. Run `mosaic track trex` again with the same `--set`
  values. It tracks from the cached conversion and publishes a retimed table under a
  new variant.

## Running features

- A run is named `<version>-<10 hex>`, a hash of its parameters, inputs and frame range.
  The same call again computes nothing. `--overwrite` (`overwrite=True`) recomputes. To
  change a result, change its parameters. Editing output files breaks the cache.
- A tracks variant, op run or model is named `<kind>.<version>-<10 hex>`, such as
  `convert-trex_npz.0.3-6bb5efbf05`. That string is what `tracks_run_id=`, `media=`,
  `model=` and `data=` take.
- An entry with two tracks variants raises an error naming them. Pass `tracks_run_id=`
  to `run_feature` (`--tracks-run-id` on `mosaic run`), or `run_id=` to `load_tracks`
  and `drop_entries`. `""` names tables converted before mosaic recorded variants.
- A run that loses some entries finishes as `partial` and names them in
  `result.failed_entries`. Check it before any global fit, since a fit over fewer
  entries is a different result. Losing every entry raises.
- A global fit reads every numeric payload column of its input unless the `columns`
  param of `extract-templates`, `extract-labeled-templates`, `arhmm` or
  `temporal-stack` names the measurements to fit.
- A template step (`extract-templates`, `extract-labeled-templates`) writes only its
  sample, and each entry's table is empty. A consumer reads the template step's inputs
  and takes the sample through `templates`. A labeled run records each entry's `split`
  in `template_provenance.parquet`.
- Inputs combine at one entity level. Mosaic refuses to merge per-individual output
  with per-pair output.
- A label of `-1` means unlabeled: no annotation covers the row. Templates leave it
  out and scoring drops and counts it. Never read it as a class. A pair row takes the
  label of its own direction, so a CalMS21 label marks only the resident-focal row.
- A custom feature must be deterministic, with typed Pydantic parameters:
  https://ecodylicscience.github.io/mosaic/guides/analysis/write-your-own-feature/.

## Trackers, pose models and annotations

Start at https://ecodylicscience.github.io/mosaic/guides/tracking/run-a-tracker/ and
https://ecodylicscience.github.io/mosaic/guides/tracking/train-a-pose-model/.

- Tracker settings are `--set key=value`, and `mosaic tracking describe <kind> --json`
  lists them.
- TRex, SLEAP and Lightning Pose read one file. A multi-clip entry or an imgstore
  needs `--kind export-joined` or `--kind export-store` first, and a run that needs one
  stops and names the command. After an upgrade changes the join recipe, re-join and
  then run `mosaic prune-joined --apply`.
- A media variant (a cropped, trimmed or re-encoded copy, from the `preprocess` op) is
  tracked by passing its run id as `media` (`--set media=<run_id>`). Its tracks are
  published in the source video's pixels and frames. Do not add crop offsets. A frame
  window cannot be combined with `media`.
- Lightning Pose cannot decode AV1. Give it a variant made with `"codec": "h264"`.
- Frame extraction is `mosaic run --kind extract-frames`, or the function
  `mosaic.tracking.extract_frames(ds, ...)`, which is not a `Dataset` method.
- Annotation revisions (`labels_raw/keypoints/<set>/rev<N>/`) are immutable. A changed
  save becomes a new revision, and an edited revision is refused.
- Refer to a model by its run id. To use a model trained in another dataset, link that
  dataset with `mosaic libraries add`. Do not pass a path into its `models/`.
  `mosaic models provenance <run_id>` shows what a model was trained on.
- A hand-corrected track set becomes source data through `promote_correction`, as
  https://ecodylicscience.github.io/mosaic/guides/tracking/import/ describes.

## Pipelines and inventory

- A recipe is a JSON file of steps
  (https://ecodylicscience.github.io/mosaic/guides/pipelines/chain-steps/). Run
  `mosaic pipeline validate`, then `plan`, then `run`. From Python,
  `run_pipeline(ds, load_recipe("recipe.json"))`.
- Exit code 65 is a refusal, with a named reason (`"reason"` under `--json`). Python
  raises `Refusal`, or `StepRefused` for a pipeline step. Fix the cause. Only
  `coverage_shortfall` is answered with `--allow-partial` (`allow_partial=True`), and
  only once the user accepts results over fewer entries. Details:
  https://ecodylicscience.github.io/mosaic/concepts/pipelines/.
- `mosaic inventory` reports what exists, with coverage and a status such as
  `partial` or `complete-but-drifted`. `mosaic runs` and `mosaic status` report
  attempts. A feature run is measured over the entries it was asked for, and
  `complete` means it has all of them. A classifier applied through a step scope to
  the test groups is complete when it has the test groups. A run invoked without a
  scope was asked for every entry, and reads `partial` once the dataset gains a
  sequence.
- Conversion is a recipe step: `convert-tracks` and `convert-labels` with
  `source_format`. Scan the sources first. A feature reads the converted tracks
  through `"tracks": {"step": <id>}` and the converted labels through
  `"labels": {"step": <id>}`.
- A train/test split is one request: step scopes narrow single steps
  (`--step-scope 'fit={"groups": ["train"]}'`). A step reading another through
  `inputs` or `tracks` must be narrowed to entries the other covers; a params
  reference such as `templates` or `label_map` need not be.

## Reading results with labels

- A frame or row no annotation covers is unlabeled, `-1`, and not class 0. Scores and
  templates leave it out and count it. Labels attach to rows: a pair row takes the
  label of its own direction, so a CalMS21 label labels the resident-focal row only.
- Score a classifier or a clustering with an `evaluate-labels` step, narrowed to the
  held-out groups. Set `perspective` on pair output (`0`, `"pooled"` or
  `"mean_probabilities"`), and `average_over` to leave a catch-all class such as
  `other` out of the macro F1. It writes `metrics.json` in its run directory
  (https://ecodylicscience.github.io/mosaic/guides/analysis/train-a-classifier/).
- For a clustering, `fit-label-map` on the training groups fits the cluster to
  behavior map that `evaluate-labels` applies to the test groups through
  `label_map`. Without one, its accuracy is the purity of the scored rows.
- In Python, `load_values(ds, [ResultColumn(...), GroundTruthLabelsSource()])` gives
  each row its label, and `labels_run_id=` picks a labels variant when the dataset
  holds two.

## Mosaic's MCP tools

When your client offers mosaic's MCP tools (`mosaic agent serve`), use them for the
work they cover (https://ecodylicscience.github.io/mosaic/agents/).

- Find step types with `describe_step_types` and a query, then call it with
  `names=[...]` for the params of the ones you choose.
- Validate a recipe with `validate_recipe`, then plan it with `plan_recipe`. Show the
  user the plan and its `cautions` (a plan over the whole dataset, a step asked for
  more than 100 entries) and wait for their approval, then pass its fingerprint to
  `submit_pipeline`.
- `entries` takes `[group, sequence]` pairs or `{"groups": [...]}`, and
  `step_entries` narrows single steps by step id. Pass the same ones to
  `plan_recipe` and `submit_pipeline`, or the fingerprint will not match.
- A run's report, such as `evaluate-labels`' scores, is the resource
  `mosaic://dataset/{id}/artifacts/feature/{name}/{run_id}/report`.
- A read returns at most 40,000 characters. A `result_too_large` error names the
  arguments that narrow the call. Narrow a preview with `columns` patterns
  (`W_*_f0`) and `max_columns`. `preview_output` also previews a labels variant
  (`labels-variant/<kind>/<run_id>`) with its events per class.
- This guide is also the `mosaic://guide` resource. Read it once.
- Set `confirm` only after the user agrees in the conversation to the conditions
  that the refusal lists.
- After `confirmation_unanswered` or `confirmation_declined`, the call did not run.
  Stop and report the plan and the conditions. Do not call it again, or run the same
  work through the `mosaic` CLI, unless the user asks. An operator who runs you
  unattended may approve conditions in advance with a hook
  (`mosaic agent confirm-hook`). A confirmation it does not approve comes back
  declined or unanswered.
- Poll `get_request_status` after the `poll_after_seconds` it reports. For a
  request that reads `orphaned`, propose
  `mosaic pipeline run --manifest <manifest> --request <id>`, which resumes it.
- A step whose `liveness` reads `unknown` runs on another machine that cannot be
  checked. Ask the user whether its process still runs. Once they confirm it has
  exited, propose `mosaic cancel --manifest <manifest> --request <id>
  --assume-exited`, then submit the recipe again.
- The tools delete nothing except superseded feature runs, through `cleanup` on
  `submit_pipeline`, which the operator must allow and the user confirms. They do not
  overwrite, sweep, prune, reindex or change the manifest. Propose the `mosaic`
  command and let the user run it.

## Pitfalls

- **`IndexLockUnsupported`**: the filesystem refuses file locks. Keep the dataset on a
  disk that locks and declare the share as a source.
- **Several machines on one dataset**: a lock excludes other machines only when the
  filesystem enforces it across them. A synced folder or an NFS volume mounted with
  `nolock` does not. Write such a dataset from one machine at a time.
- **Mosaic is 0.x.** Interfaces, feature names and on-disk layouts may change. Record
  the `mosaic-behavior` version with every result, read the changelog
  (https://github.com/EcodylicScience/mosaic/blob/main/CHANGELOG.md) after an update,
  and run `mosaic reconcile` if it says an identifier moved.
- **Two ffmpeg builds** in one environment crash a process at a different point each
  run (`Trace/BPT trap: 5`). On macOS every import also warns that `AVFFrameReceiver`
  is implemented twice, with or without a crash. Either symptom means the environment
  needs rebuilding as the installation page says.
- **`scan_media` failures** usually mean `ffprobe` is not on `PATH`.
- **Ultralytics and POLO** need two separate environments, and the point ops refuse an
  upstream Ultralytics build.

## Examples

The repository's `notebooks/` on GitHub are end-to-end examples that download their
data (https://ecodylicscience.github.io/mosaic/examples/). A pip install does not carry
them, so download one from the repository and run the copy in the user's project.
