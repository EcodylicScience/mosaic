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
  `load_run_matrix` reads one finished run's columns as an array.
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
- `splits` in `dataset.yaml` gives a group the role `train`, `test` or `sealed`. Read it
  with `describe_dataset` or `mosaic splits list`. Only the user sets it, with `mosaic
  splits`. Never write it, even when it blocks a plan, because it is the record of
  which data no fit may see.
- Derived and temporary folders cost only time to delete. Reclaim space with `mosaic
  sweep-tracking`, `prune-media` and `prune-joined`. `mosaic reindex` drops index rows
  whose files are gone. All four are dry runs until given `--apply`. For superseded
  feature runs, plan with `mosaic pipeline plan --cleanup superseded`, which lists
  them, and submit with `--cleanup superseded --expect-plan <fingerprint>`. The plan
  also lists the saved recipes whose requests wrote runs. Give a digest to `--retire`
  on the plan and again on the submission to reclaim the runs that recipe wrote in
  storages the new recipe no longer names.
- Give a recipe a `name` and keep it across the recipe's versions. A cleanup can then
  remove the runs of its earlier versions, and it keeps every run that a request of
  another recipe recorded, unless you retire that recipe. An unnamed recipe is a version
  only of itself.

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
  and run the `scale-to-cm` feature. For body lengths, run `body-length` and then
  `scale-to-bl` referencing it. Do not rescale a tracks table yourself. The overlay
  and the crop features refuse a table recorded in centimeters (`trex_v1`,
  `mosaic_cm_v1`). `mosaic upgrade-tracks` converts a `trex_v1` table to pixels.
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
  The same call again computes nothing. `--overwrite` (`overwrite=True`) recomputes each
  entry's output but loads a stored fit and never fits again. To change a result,
  including a model, change a parameter. Editing output files breaks the cache. To check
  that a stored result is the one its recipe produces, see `mosaic pipeline verify`
  below.
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
- `arhmm` reads each individual, or each ordered pair and perspective of a pair table,
  as its own sequence, and writes `id1`, `id2`, `perspective`, `group` and `sequence`
  beside its states, so `fit-label-map` can score them (`prediction_column: syllable`).
  `lightning-action` trains and predicts on each `(id1, id2, perspective)` as a
  sequence of its own, from the entries of its `templates` run's train split. Its scope
  is not part of its run id: the `templates` run names the training set, so a wider
  request is the same run, which applies the stored fit to the new entries. The first
  request must hold the training entries and is refused otherwise.
  `sequence-classifier` (`[deep-learning]`, GPU worker) is the same kind of step with a
  registered network as `head` (`tcn`, `gru` or `mlp`, options in `head_config`). It takes
  `cv_folds` and `cv_seed` as `xgboost` does, with the same folds. A head registered in
  Python with `register_head` is visible to that process only, not to the CLI or a queue
  worker. `probability-ensemble` averages or blends the `prob_<class>` columns of
  classifier runs: `weights` is `uniform`, a list, or `fit`, which chooses them on the
  members' out-of-fold probabilities and refuses a member fitted without `cv_folds`. It
  writes the members' fold table when they all have the same one, so an average of
  cross-validated runs can be a member of a blend. A list or `uniform` weights fit nothing,
  so those members need no fold table, and mixed members are averaged without one.
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
  needs an `export-joined` or `export-store` run first. Before a tracker, an `infer-*`
  op, `extract-frames` or `preprocess`, run `mosaic prerequisites` with the `--kind`,
  `--params` and scope flags of its `mosaic run` command. For `mosaic track <kind>`,
  pass `--kind <kind>` and give its `--set` settings as one `--params` object, such as
  `--params '{"media": "<run_id>"}'`. Of a tracker's settings, the verb reads only
  `media`. It prints the runs that the op needs, in order, each with its command, then
  each refused entry with its cause, and then the command of the op. It does not run
  them, and `--json` prints the same as one object. A run that lacks a join or an
  export stops and names the command. After an upgrade changes the join recipe,
  re-join and then run `mosaic prune-joined --apply`.
- A media variant (a cropped, trimmed or re-encoded copy, from the `preprocess` op) is
  tracked by passing its run id as `media` (`--set media=<run_id>`). Its tracks are
  published in the source video's pixels and frames. Do not add crop offsets. A frame
  window cannot be combined with `media`.
- Lightning Pose cannot decode AV1. Give it a variant made with `"codec": "h264"`.
  `mosaic prerequisites --kind litpose` plans that variant and names it in `media`.
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
- A step given a fitted `model` (`arhmm`, `kpms`, `vame`, the identity models,
  `lightning-action`) applies it to each entry alone, so its scope is not in its run id
  and a wider request is the same run. `evaluate-labels` takes several inputs over
  disjoint entries and scores them as one, and the plan refuses inputs asked for the
  same entry. `extract-templates`, `extract-labeled-templates`, `fit-label-map` and a
  `pair-posedistance-pca` without a `model` still hash the entries they cover, so scope
  them to the training groups once.
- `mosaic pipeline verify --manifest <manifest> --recipe <recipe>` (with the
  `--step-scope` you ran) fits each fitting step of a finished recipe again in a scratch
  directory and compares it with the stored run. It exits 1 on a difference and when it
  compared nothing, and it writes nothing to the dataset. Features that train a network
  match exactly on CPU at one thread count and otherwise within `--atol`. Propose it to
  the user to check a result reproduces, and run it only when they ask.
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
  reference such as `templates`, `label_map` or `model` need not be. To project every
  group onto a `pair-posedistance-pca` basis fitted on the training groups, add a second
  step of that feature with `"model": {"step": "<fit step>", "pattern":
  "model.joblib"}` and the fit's other params.
- When the dataset gives groups roles (`splits`), the plan refuses a step that fits on
  the entries it covers, such as `extract-templates`, `extract-labeled-templates`,
  `fit-label-map`, `arhmm`, `kpms`, `vame` or the identity models (`fits` in
  `describe_step_types`), when they include a `test` or `sealed` group. Scope it to the
  `train` groups, and apply what it fitted to the others through the params reference
  as above. A scope-free fitter such as `xgboost` is held to the same rule by its
  templates step. A plan over the whole dataset names the groups it covers and their
  roles in its cautions: narrow it before you submit.
- A `sealed` group is scored once, by the user, and its labels stay outside the dataset.
  Data added after your fits is scored with `mosaic pipeline score --manifest <manifest>
  --recipe <recipe> --heldout <file> --answers <answers.json> --out <dir>`, which works
  on a copy of the dataset and exits 1 with `RUN IDS CHANGED` when adding the data would
  fit a step of your recipe again. Propose it, and run it only when the user asks. Give
  `--step-scope` as you ran the recipe, the raw format's source options (`--src-format`,
  `--multi-sequences-per-file`, `--group-from`, `--patterns`, `--labels-path`) and the
  evaluation protocol (`--perspective`, `--average-over`); see `mosaic pipeline score
  --help`. To find out beforehand, call `plan_recipe` with
  `assume_added: [group]`: its cautions name the steps that would refit and the steps
  that would only re-score. A `test` group that is already in the dataset is scored
  with an `evaluate-labels` step narrowed to it, once. An `evaluate-labels` step that
  reads a sealed group asks the user to confirm (`score_sealed`).

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
  `label_map`. Without one, its accuracy is the purity of the scored rows, and the
  report's `label_map_source` reads `scored_rows`: report it as purity, not as a
  held-out score. `weighting: "class_balanced"` keeps a common behavior from taking
  every cluster, which leaves a rare one at F1 0.
- To compare classifier settings without spending the test groups, set `cv_folds` on
  `xgboost` and `test_fraction: 0` on the labeled templates, and scope both and
  `evaluate-labels` to the training groups: that score is the pooled out-of-fold one.
  For the final score widen the same `xgboost` step to the test groups, which is the
  same run, and keep the labeled templates on the training groups. Widening a
  `sequence-classifier`, `lightning-action` or fitted-weight `probability-ensemble` step
  is the same run too: their `templates` run (or the members' folds) names the training
  set, so the wider request fits nothing again and computes only the new entries. Ask
  the first request over a scope that holds the training entries. A fit does not move
  when a training entry is converted again; only that entry's output is recomputed with
  the stored model, so refit on purpose by changing a parameter. `fit_scope.json` lists
  the entries of the templates run, not of the request. The score is not nested, and a
  fold is whole sequences unless `cv_unit` is `block`
  (https://ecodylicscience.github.io/mosaic/guides/analysis/train-a-classifier/).
- With one training sequence there is nothing to hold out, and `cv_folds` refuses it. Set
  `cv_unit: "block"` on `xgboost` or `sequence-classifier`: each sequence is cut into
  `cv_blocks_per_sequence` blocks of frames, a fold holds blocks out, and `cv_folds.parquet`
  has a row per block with `start` and `stop`. Set `cv_gap_frames` to the longest window
  above the classifier. The plan refuses a gap shorter than the `temporal_reach` the steps
  above declare, and cautions on a step that declares none while the gap is 0 or short of
  what is known. `xgboost` needs templates from `extract-labeled-templates` 0.7, which
  moves the run id of every classifier built on them. `lightning-action` has no `cv_unit`.
- Choose a decision threshold on the out-of-fold probabilities, never on the test groups:
  `tune-threshold` over a classifier fitted with `cv_folds`, then a one-member
  `probability-ensemble` over the same classifier with `decision_threshold: {"step":
  "<tune>"}`. The report `tune_report.json` gives `plain` (optimistic) and `nested` scores,
  and `warnings` when a cut is the lowest or highest `grid` value, which is the edge of what
  was tried and not a tuned value. The default class takes threshold 1.0, and a class left
  out of `classes` has 0
  (https://ecodylicscience.github.io/mosaic/guides/analysis/train-a-classifier/).
- Each output row of `evaluate-labels` carries `tp_<class>`, `fp_<class>` and
  `fn_<class>`. `mosaic.core.evaluation` pools them (`pool_entry_counts`), bootstraps
  over sequences from them (`bootstrap_sequences`, with a required `random_state`),
  and combines the reports of separate runs (`pool_reports`, `aggregate_reports`).
  In a recipe, give one `evaluate-labels` step several inputs over disjoint entries,
  such as a cross-validated run over the training groups and an applied one over the
  test groups, and it reports their union with the pooled counts of one run.
- In Python, `load_values(ds, [ResultColumn(...), GroundTruthLabelsSource()])` gives
  each row its label, and `labels_run_id=` picks a labels variant when the dataset
  holds two.
- To fit or score on a run's columns in your own code, do not read the parquet files.
  `load_run_matrix(ds, result, columns="prob_*", perspective=0, scope=...)` returns
  `.index` (group, sequence, frame, and id or id1/id2/perspective), `.values` (float32,
  one row per row of the index) and `.columns`, in order of group and sequence. It
  refuses a run that is not finished, an entry of the scope the run lacks, and a read
  too large for the memory, and the message says how to narrow it. Pass
  `allow_partial=True` only once the user accepts results over fewer entries, and
  report `.entries` as what was read: `.finished` is then False for a run that lost
  entries, and `.missing` lists only the scope's entries that have no file. It
  carries no labels, so take those from `align_predictions_to_labels`.

## Mosaic's MCP tools

When your client offers mosaic's MCP tools (`mosaic agent serve`), use them for the
work they cover (https://ecodylicscience.github.io/mosaic/agents/).

- Find step types with `describe_step_types` and a query, then call it with
  `names=[...]` for the params of the ones you choose.
- A dataset that is not ready says so in `describe_dataset`'s `next_steps`. Where the
  server allows it, `manage_dataset` creates a dataset (`create`, a `path` inside the
  server's dataset roots), declares raw files (`add_source` with `kind`, `path` inside
  its source roots, and for tracks `patterns` and `src_format`), indexes them
  (`scan`) and sets tags (`set_tags`). Creating and changing sources ask the user.
  Then plan a recipe that begins with `convert-tracks` (and `convert-labels`). A
  refusal with `roots_not_configured` means the user runs `mosaic init`,
  `mosaic sources add` and `mosaic scan` instead. Never try to change `splits`.
- Validate a recipe with `validate_recipe`, then plan it with `plan_recipe`. Show the
  user the plan and its `cautions` (a plan over the whole dataset, a step asked for
  more than 100 entries) and wait for their approval, then pass its fingerprint to
  `submit_pipeline`.
- A plan of about 40 steps or more is over the read limit. Plan it with `limit` (about
  25) and then `cursor`, the previous page's `next_cursor`, until `next_cursor` is
  empty, and show the user every page. Every page carries the whole plan's fingerprint;
  `problems`, `cautions` and the other plan-level blocks are on the first page only.
- `entries` takes `[group, sequence]` pairs, `{"entries": [...]}` or
  `{"groups": [...], "sequences": [...]}`, and `step_entries` narrows single steps by
  step id in the same forms. A step scope is intersected with `entries`, so leave
  `entries` unset when steps are scoped to different groups. Pass the same ones to
  `plan_recipe` and `submit_pipeline`, or the fingerprint will not match.
- Write a step repeated over seeds, behaviors or folds once with `repeat`:
  `{"id": "xgb_s{seed}", "repeat": {"seed": [1, 2, 3]}, ..., "params": {"random_state":
  "{seed}"}}` is three steps, and `{"step": "xgb_s{seed}", "repeat": {"seed": [1, 2, 3]}}`
  in `inputs` is three inputs. Save a long recipe with `patch_recipe` (no `ops`), then pass
  `{"digest": "<digest>"}` as `recipe` to `validate_recipe` (with `dataset`),
  `plan_recipe` and `submit_pipeline`, and change it with `patch_recipe` ops
  (`add_step`, `remove_step`, `set_param`, `replace_step`), each returning a new digest.
- To reclaim the runs of an earlier recipe that wrote to storages this one no longer
  names, call `plan_recipe` with `cleanup: "superseded"` and read `retirable`: name,
  digest, request ids and run count; `retirable` lists the eight newest, and
  `list_activity` shows the `recipe_digest` of any other request. Plan again with
  `retire: [digest]`, show the user `cleanup.retired_runs` and every entry of a step's
  `superseded` whose `retired` is set, and pass the same `retire` and the new
  fingerprint to `submit_pipeline`. A retirement lasts for that call, so repeat it each
  time.
- A run's report, such as `evaluate-labels`' scores, is the resource
  `mosaic://dataset/{id}/artifacts/feature/{name}/{run_id}/report`.
- A read returns at most 40,000 characters. A `result_too_large` error names the
  arguments that narrow the call. Narrow a preview with `columns` patterns
  (`W_*_f0`) and `max_columns`. `preview_output` also previews a labels variant
  (`labels-variant/<kind>/<run_id>`) with its events per class.
- This guide is also the `mosaic://guide` resource. Read it once.
- Set `confirm` only after the user agrees in the conversation to the conditions
  that the refusal lists.
- After `confirmation_unanswered`, `confirmation_declined` or
  `confirmation_unavailable` (a server over HTTP that does not take `confirm` from you),
  the call did not run.
  Stop and report the plan and the conditions. Do not call it again, or run the same
  work through the `mosaic` CLI, unless the user asks. An operator who runs you
  unattended may approve conditions in advance with a hook
  (`mosaic agent confirm-hook`). A confirmation it does not approve comes back
  declined or unanswered.
- Wait with `get_request_status(wait_seconds=45)`, which returns when a step or the
  request ends; `brief: true` lists only the steps still running or not finished
  cleanly. Do not watch `driver_pid` (`kill -0`) or `tail` the driver log for
  progress: the log holds output for explaining a failure. When no step has started
  after a few waits, read `driver_log` once, because a driver that failed to start
  leaves the request `running`. For a request that reads `orphaned`, propose
  `mosaic pipeline run --manifest <manifest> --request <id>`, which resumes it.
- A failed step ends only the branch below it. The steps below it are recorded
  failed with error type `ParentFailed`, whose `parent` names the failed step. The
  other steps still run. Answer the failed step. An ended request is not run again.
  Submit the recipe again once the failed step is answered. Its finished steps are
  cache hits.
- A step whose `liveness` reads `unknown` runs on another machine that cannot be
  checked. Ask the user whether its process still runs. Once they confirm it has
  exited, propose `mosaic cancel --manifest <manifest> --request <id>
  --assume-exited`, then submit the recipe again.
- Parallel workers: a `parallelizable` step can compute several entries at once.
  `plan_recipe` reports the operator's limits, the host's processors and memory, and
  each step's `workers`. `submit_pipeline` takes `workers` and `parallel_mode`, never
  above those limits, and a request over one is refused with `parallel_limit`. Start
  with `thread`, which speeds up numpy and pandas work. Use `process` only for a step
  that stays slow in threads because it runs Python code that holds the interpreter
  lock, after checking that the host's available memory holds one interpreter and
  several copies of an entry's table per worker.
- Watch memory while a step runs: `get_request_status` reports each step's
  `peak_rss_bytes` and `min_available_bytes`. When a step ends with `WorkerProcessDied`
  or a memory-floor failure, or its peak nears the host's memory, halve `workers` or
  use `thread`, then submit the recipe again. Written entries are kept, because the
  worker count is not part of a run's identity. Do not retry at the same setting.
- The tools delete nothing except superseded feature runs, and with `retire` the
  runs of recipes retired in that call, through `cleanup` on `submit_pipeline`, which
  the operator must allow and the user confirms. They do not
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
