# Train a behavior classifier

Supervised classification predicts an annotated behavior on frames nobody annotated.
It needs labels: see [Import behavior annotations](import-labels.md) first.

Four classifiers ship. All are features, so all cache and take a `run_id`.

| Feature | Reads | Install |
| --- | --- | --- |
| `xgboost` | per-frame feature columns | base install |
| `sequence-classifier` | per-frame feature columns, modeled as sequences by a registered network | `[deep-learning]` |
| `lightning-action` | per-frame feature columns, modeled as sequences | `[lightning-action]` |
| `feral` | video directly, through a V-JEPA backbone | `[feral]`, in its own environment |

For mouse social behavior without labels of your own, the pretrained `hidra` feature
runs HiDRA's published heads ([Classify mouse behavior with HiDRA](hidra.md)).

`probability-ensemble` is a step beside them. It averages or blends the probabilities of
several classifier runs ([Average and blend classifiers](#average-and-blend-classifiers)).

## The XGBoost route

Two steps: draw a labeled, class-balanced sample with a held-out test split, then fit.

```python
from mosaic.behavior.feature_library import (
    ExtractLabeledTemplates, GroundTruthLabelsSource, XgboostFeature,
)
from mosaic.core.pipeline.types import Inputs

labeled = ExtractLabeledTemplates(
    Inputs((features_result,)),
    params={
        "labels": GroundTruthLabelsSource(),
        "n_per_class": 500,
        "test_fraction": 0.2,
    },
)
labeled_result = ds.run_feature(labeled)

xgb = XgboostFeature(
    Inputs((features_result,)),
    params={
        "templates": ExtractLabeledTemplates.LabeledTemplatesArtifact().from_result(labeled_result),
        "strategy": "multiclass",
        "default_class": 3,
        "n_estimators": 400,
        "max_depth": 6,
    },
)
xgb_result = ds.run_feature(xgb)
```

`n_per_class` caps how many frames each behavior contributes, so a rare behavior is not
drowned by a common one. `test_fraction` holds sequences back, not frames — a split by
frame leaks, because neighboring frames of one bout are nearly identical.
`default_class` names the majority or "other" class.

Templates are drawn only from labeled rows. A row no annotation covers is unlabeled
(`-1`) and is never sampled. On a pair feature, each row takes the label of its own
direction. A CalMS21 label describes the resident's behavior toward the intruder, so
CalMS21 pair features train on the resident-focal rows, and the intruder-focal rows
are unlabeled. See [Labeled, background and unlabeled
frames](import-labels.md#labeled-background-and-unlabeled-frames).

`strategy="multiclass"` fits one model over all behaviors. `"one_vs_rest"` fits one
model per behavior, which is what you want when behaviors can co-occur.

`n_per_class` also takes a mapping from class to count, where `null` takes every
labeled frame of that class: `{"0": 6000, "1": null, "2": 6000}`. A class the mapping
leaves out contributes nothing. With `pool.allocation` set to `"exact"`, each sequence
draws its share of a class's count in proportion to the frames it holds. `class_weight`
is `"balanced"` by default (each frame weighs inversely to its class's frequency), `null`
for equal weights, or a mapping from class to weight for the classes to weigh up or
down. `n_jobs` limits the threads of a fit and does not enter the run id. `device` is `"cpu"`
by default, and `"cuda"` fits on a GPU where the machine has one. It enters the run id,
and the fitted model applies on any machine. The template
draw and the fit each refuse a sample that would not fit in the available memory, and
the message says what to lower.

## Train and score in one request

A recipe can run the whole route at once, from the raw files to a score on held-out
groups. Step scopes make the split: each one narrows a single step to part of what the
request covers, so the fitting steps see the training groups and the scoring step the
test group.

```json
{
  "schema_version": 1,
  "name": "supervised",
  "steps": [
    {"id": "tracks", "type": "op", "kind": "convert-tracks",
     "params": {"source_format": "calms21_npy"}},
    {"id": "labels", "type": "op", "kind": "convert-labels",
     "params": {"source_format": "calms21_npy"}},
    {"id": "egocentric", "type": "feature", "feature": "pair-egocentric",
     "inputs": ["tracks"], "tracks": {"step": "tracks"}},
    {"id": "templates", "type": "feature", "feature": "extract-templates",
     "inputs": [{"step": "egocentric"}], "params": {"n_templates": 2000}},
    {"id": "scaler", "type": "feature", "feature": "global-scaler",
     "inputs": [{"step": "egocentric"}], "params": {"templates": {"step": "templates"}}},
    {"id": "labeled", "type": "feature", "feature": "extract-labeled-templates",
     "inputs": [{"step": "scaler"}],
     "params": {"labels": {"step": "labels"}, "n_per_class": 500}},
    {"id": "classifier", "type": "feature", "feature": "xgboost",
     "inputs": [{"step": "scaler"}],
     "params": {"templates": {"step": "labeled"}, "default_class": 3}},
    {"id": "evaluate", "type": "feature", "feature": "evaluate-labels",
     "inputs": [{"step": "classifier"}],
     "params": {"labels": {"step": "labels"}, "perspective": 0,
                "average_over": [0, 1, 2]}}
  ]
}
```

```bash
mosaic pipeline run -m dataset.yaml --recipe @supervised.json \
  --step-scope 'templates={"groups": ["train"]}' \
  --step-scope 'labeled={"groups": ["train"]}' \
  --step-scope 'classifier={"groups": ["test"]}' \
  --step-scope 'evaluate={"groups": ["test"]}'
```

- `{"step": "labels"}` in a `labels` field reads the labels that step converts, so the
  scores are against that conversion even when the dataset holds another.
- CalMS21 labels belong to a pair, so the features are pair features, and
  `pair-egocentric` writes a row for each direction of a pair. `"perspective": 0` in
  `evaluate-labels` scores the row of each pair that a label marks. A feature of single
  individuals, such as `speed-angvel`, has no `perspective` column and takes no such
  setting.
- A step that reads another through `inputs` or `tracks` must be narrowed to entries
  the other is asked for, and the plan names the step that is not. A params reference
  carries no such rule: the classifier reads templates drawn from the training groups
  and is applied to the test group.
- Scan the dataset's sources before planning (`mosaic scan`): a conversion step whose
  format has no scanned file is refused.

`evaluate-labels` writes `metrics.json` in its run directory: accuracy, a macro F1 over
`average_over` (here every class but `other`), precision, recall and F1 per class, and
a confusion matrix, which is also `confusion.parquet`. Each of its output rows is one
entry's own score, with its true positives, false positives and false negatives per class
(`tp_<class>`, `fp_<class>`, `fn_<class>`). A row no label covers is left out and counted
in `n_unlabeled`.
`perspective` chooses which rows of a pair output are scored: one perspective, both
(`"pooled"`), or the mean of a pair's two probability vectors
(`"mean_probabilities"`). It refuses pair rows of both perspectives unless it is set.

The counts add across entries, so `mosaic.core.evaluation` can score any set of them as
one and put an interval on the score. `pool_entry_counts` takes the output table and
returns the macro F1 of all its rows together, which is the report's number and not the
mean of the entries' scores. `bootstrap_sequences` resamples the entries, with the
`random_state` you pass, and returns percentile intervals for the accuracy, the macro F1
and each class's F1. A report from a run over other entries, such as one fold of a
cross-validation, joins the rest through `pool_reports`, and `aggregate_reports` gives
the mean and standard deviation of repeated runs.

## Choose a setting by cross-validation

Scoring every setting you try on the test groups spends the test groups. `cv_folds` on
`xgboost` scores the training sequences instead, each by a model that never saw it.
Changed steps of the recipe above:

```json
{"id": "labeled", "type": "feature", "feature": "extract-labeled-templates",
 "inputs": [{"step": "scaler"}],
 "params": {"labels": {"step": "labels"}, "n_per_class": 500, "test_fraction": 0}},
{"id": "classifier", "type": "feature", "feature": "xgboost",
 "inputs": [{"step": "scaler"}],
 "params": {"templates": {"step": "labeled"}, "default_class": 3,
            "cv_folds": 5, "cv_seed": 0}}
```

Scope the classifier and its evaluation to the training groups, with the template
steps:

```bash
mosaic pipeline run -m dataset.yaml --recipe @cv.json \
  --step-scope 'templates={"groups": ["train"]}' \
  --step-scope 'labeled={"groups": ["train"]}' \
  --step-scope 'classifier={"groups": ["train"]}' \
  --step-scope 'evaluate={"groups": ["train"]}'
```

`evaluate-labels` over the training groups is now the pooled out-of-fold score, with the
same `perspective` and `average_over` as the test score.

- The training sequences of the labeled templates are dealt into K folds by a hash of
  `cv_seed`, group and sequence. The folds differ by at most one sequence, and they are
  the same for every setting whose templates cover the same sequences: `random_state`,
  `n_per_class` and the model's parameters do not move them. Vary `random_state` for
  repeats and `cv_seed` for other folds.
- `xgboost` fits K models, each on the training rows outside one fold, then the full
  model on all training rows. A training sequence is scored by the model of its fold.
  Any other entry, such as a test group, is scored by the full model, the one a run
  without `cv_folds` fits. The step takes up to K + 1 times as long as one fit.
- Set `test_fraction` to 0 on the labeled templates. Any other value moves a share of the
  training sequences to the template test split, out of every fold's training rows.
- The fold of each sequence and its template rows per class (`rows_<class>`) are in
  `cv_folds.parquet` in the run directory. Read it when a class scores badly. A class
  held by few sequences can be missing from a fold's training rows, and its probability
  is 0 for that fold's sequences. The fit warns when a class has fewer sequences than
  folds, or when a fold's training rows hold under half of a class's rows, and it
  refuses a fold whose training rows hold fewer than two classes, or with `use_smote`
  fewer than six rows of a class that SMOTE oversamples.
- `fit_scope.json` of the classifier lists the training sequences, as it does for any
  model fitted on them. Each of them was still scored by a model that did not see it, and
  `cv_folds.parquet` says which.

When you have chosen a setting, widen the same classifier step to the training and test
groups, and score the test groups once. Leave `templates` and `labeled` on the training
groups, so that no label of a test sequence enters the sample:

```bash
mosaic pipeline run -m dataset.yaml --recipe @cv.json \
  --step-scope 'templates={"groups": ["train"]}' \
  --step-scope 'labeled={"groups": ["train"]}' \
  --step-scope 'classifier={"groups": ["train", "test"]}' \
  --step-scope 'evaluate={"groups": ["test"]}'
```

The scope is not part of the classifier's run id, so this is the same run. The model is a
cache hit, and only the test groups are computed. A classifier step scoped to the
training and test groups from the start would score the test groups for every setting
you try. Keep one classifier step: two steps with the same parameters and different
scopes are one run, and can fit at the same time.

A fit that does not hash its scope also does not notice that a training entry changed. If
you convert a training entry again, the model stays the run it was, and only that entry's
output is computed again with it. To fit again on the new data, change a parameter, such
as `cv_seed`. `--overwrite` does not do it: it recomputes the entries' output and loads
the stored model.

The out-of-fold probabilities are inputs a second stage can train on. Stack the
`prob_*` columns with `temporal-stack`, draw templates from the stacked table, and fit a
second `xgboost` with the same `cv_seed`. Its folds are the first stage's. Name the
columns exactly, as `"columns": ["prob_0", "prob_1", "prob_2", "prob_3"]`, because a
prefix is refused and `temporal-stack` stacks every payload column otherwise, the
`predicted_label` among them.

What the score does not cover:

- It is not nested. The full model and the settings you compare were chosen with every
  training sequence in view, and a second stage is evaluated on the folds its inputs
  were made on.
- Label-free fits upstream (the scaler, k-means, t-SNE, PCA) see the features of the
  held-out sequences. The labeled sample is drawn over all training sequences, so a
  held-out sequence's class totals set how many rows of the other folds are kept. No
  label of a held-out sequence enters the model that scores it.
- A fold is a set of sequences, unless `cv_unit` is `block` (below). If several
  sequences come from one animal or one recording, a fold that splits them leaks across
  them, and mosaic does not know. Make that unit one sequence before you draw the
  templates.

## Cross-validate one training sequence by blocks

Folds of whole sequences need more training sequences than folds. A dataset with one
training sequence has no score of that kind, and `cv_folds` refuses it. Set `cv_unit` to
`block` on `xgboost` or `sequence-classifier` to hold out stretches of frames instead:

```json
{"id": "classifier", "type": "feature", "feature": "xgboost",
 "inputs": [{"step": "scaler"}],
 "params": {"templates": {"step": "labeled"}, "default_class": 3,
            "cv_folds": 5, "cv_unit": "block",
            "cv_blocks_per_sequence": 10, "cv_gap_frames": 60}}
```

- Each training sequence is cut into `cv_blocks_per_sequence` contiguous blocks (unset,
  `cv_folds`) that tile the frames from its first to its last labeled frame, and the
  blocks are dealt to the folds by a hash of `cv_seed`, group, sequence and block. The
  model of a fold trains on the frames outside its blocks and outside the `cv_gap_frames`
  frames on each side of them, and scores the frames inside its blocks. The frames outside
  every block, which no label covers, and the entries outside the training set are scored
  by the model fitted on all of it. `cv_folds.parquet` has one row per block, with its
  `start` and `stop` frames, and not one per entry.
- The numbers above suit sequences of minutes. On a few seconds of tracks, lower
  `cv_blocks_per_sequence`, `cv_gap_frames` and `cv_folds` until each fold's training
  frames hold every class: the fit refuses a fold that leaves no training rows of a
  class.
- Neighboring frames are not independent, so a block score is more optimistic than a
  sequence score wherever sequences are the unit you generalize over. The frames next to a
  held-out block resemble it, and a feature that reads a window of frames carries the
  block's frames into the rows beside it. `cv_gap_frames` removes that margin from
  training. It should be at least as wide as the longest window above the classifier.
  The plan checks it: each step can declare `temporal_reach`, the frames on each side that
  one of its output rows depends on (`mosaic pipeline catalog <name>` lists it), and the plan
  sums the declared reach along the steps above the classifier and refuses a gap that is
  shorter. A step that declares none, or whose reach no setting bounds (an interpolation
  over gaps of any length), makes the plan a caution instead. A `cv_gap_frames` above 0
  that covers the reach that is known is taken to be your answer to a step that says
  nothing, and the caution goes. The classifier's own reach is not counted: a network
  trains on pieces cut out of the runs, so it never reads a held-out frame.
- `xgboost` places each template row in a block by its `frame`, which
  `extract-labeled-templates` writes from version 0.7. It also records each entry's
  labeled frames in the provenance. Templates from an older run are refused under
  `cv_unit: block`. Drawing them again moves the run id of every classifier built on
  them, `cv_unit: sequence` ones included.
- `lightning-action` has no `cv_folds` yet, and so no `cv_unit`. To cross-validate a
  single sequence, use `sequence-classifier` or `xgboost`.

## Choose decision thresholds out of fold

The class with the highest probability is rarely the best call for a rare behavior.
`decision_threshold` takes a cut per class, and a cut chosen by looking at the
test groups spends them. `tune-threshold` chooses it on the out-of-fold probabilities of a
classifier fitted with `cv_folds`, against the labels, and saves it for a step to apply:

```json
{"id": "tune", "type": "feature", "feature": "tune-threshold",
 "inputs": [{"step": "classifier"}],
 "params": {"labels": {"step": "labels"}, "default_class": 3,
            "metric": "macro_f1", "average_over": [0, 1, 2]}},
{"id": "apply", "type": "feature", "feature": "probability-ensemble",
 "inputs": [{"step": "classifier"}],
 "params": {"default_class": 3, "decision_threshold": {"step": "tune"}}}
```

- A one-member `probability-ensemble` is the step that applies the thresholds. Its
  `decision_threshold` takes `{"step": "tune"}` beside a number or a mapping of class to
  cut. The classifier cannot name its own tuning, because the tuning needs the
  classifier's run: that is a cycle. For the same reason only `probability-ensemble` takes
  a tune step: the `decision_threshold` of `xgboost`, `sequence-classifier` and
  `lightning-action` stays a number or a mapping, and an `xgboost` stage stacked over
  probabilities takes the tuned cuts through a one-member ensemble as well. The ensemble
  must blend the run that was tuned and nothing else, and is refused for any other input,
  because the thresholds were scored on that run's output.
- `default_class` takes the rows no class claims and has no cut of its own: the step saves
  it with 1.0. A class left out of `classes` keeps a cut of 0, which it always clears. A
  threshold for one class in a binary problem is therefore `{"0": 1.0, "1": t}`: a lone
  `{"1": t}` with `t` under 0.5 changes nothing.
- `metric` is `macro_f1` over `average_over` or `pooled_f1` of the one class in `classes`.
  The step tries each cut of `grid` (fine below 0.1) and takes the lowest of equal scores.
  With several classes it improves one at a time, for up to five passes.
- `tune_report.json` in the run directory gives the estimate on the rows the cuts were
  chosen on (`plain`, optimistic by the freedom of the cuts) and, with `nested`, the
  estimate in which each fold is scored with cuts chosen on the other folds. Believe the
  nested one. Neither corrects for the hyperparameters, which were chosen on the same
  rows. A labeled row with no probability is scored as `default_class`, as the step that
  applies the cuts would assign it.
- The report's `warnings` and the log name a cut that is the lowest or the highest value of
  `grid`: the best cut may lie beyond it, and that value is the edge of what was tried and
  not a tuned value. Widen `grid` and run again.
- With block folds the step maps each row to the fold of its block and leaves out a labeled
  row in no block. The step hashes no scope, so widening the request to the test groups is
  the same run, which applies the stored cuts to the new entries. Ask the first request
  over a scope that holds the classifier's training entries.

## Give each frame its context

A single frame rarely identifies a behavior. `temporal-stack` widens each row into a
window before the classifier sees it:

```python
from mosaic.behavior.feature_library import TemporalStackingFeature

stacked_result = ds.run_feature(TemporalStackingFeature(
    Inputs((scaler_result,)),
    params={"half": 2, "skip": 1, "use_temporal_stack": True, "sigma_stack": 2, "fps": 30.0},
))
```

`half=2, skip=1` gives each frame the two before and the two after. Feed
`stacked_result` to `ExtractLabeledTemplates` and to the classifier in place of the raw
features.

`lightning-action` and `sequence-classifier` model the temporal structure themselves,
so they do not need this step. See [Model the sequence directly](#model-the-sequence-directly)
and [Train a sequence network](#train-a-sequence-network).

## Model the sequence directly

`lightning-action` trains a temporal network on frames in order. It reads one individual,
or one ordered pair, at a time: a pair table is split by `id1`, `id2` and `perspective`,
and each series is trained on and predicted as a sequence of its own. The recipe is the
XGBoost recipe with the classifier swapped and the scaler dropped, because the frames are
standardized by statistics that are stored with the model.

```json
{
  "schema_version": 1,
  "name": "lightning-action-sequence",
  "steps": [
    {"id": "tracks", "type": "op", "kind": "convert-tracks",
     "params": {"source_format": "calms21_npy"}},
    {"id": "labels", "type": "op", "kind": "convert-labels",
     "params": {"source_format": "calms21_npy"}},
    {"id": "egocentric", "type": "feature", "feature": "pair-egocentric",
     "inputs": ["tracks"], "tracks": {"step": "tracks"}},
    {"id": "labeled", "type": "feature", "feature": "extract-labeled-templates",
     "inputs": [{"step": "egocentric"}],
     "params": {"labels": {"step": "labels"}, "n_per_class": 500, "test_fraction": 0}},
    {"id": "classifier", "type": "feature", "feature": "lightning-action",
     "inputs": [{"step": "egocentric"}],
     "params": {"templates": {"step": "labeled"}, "labels": {"step": "labels"},
                "default_class": 3, "head": "dtcn", "sequence_length": 500,
                "num_epochs": 40}},
    {"id": "evaluate", "type": "feature", "feature": "evaluate-labels",
     "inputs": [{"step": "classifier"}],
     "params": {"labels": {"step": "labels"}, "perspective": 0,
                "average_over": [0, 1, 2]}}
  ]
}
```

```bash
mosaic pipeline run -m dataset.yaml --recipe @sequence.json \
  --step-scope 'labeled={"groups": ["train"]}' \
  --step-scope 'classifier={"groups": ["train", "test"]}' \
  --step-scope 'evaluate={"groups": ["test"]}'
```

What differs from `xgboost`:

- **The templates name the training entries, and the frames come from the step's
  inputs.** The model trains on the entries that `labeled` assigned to `train`, with the
  labels from `labels`. An entry that `labeled` held out, or never saw, does not train.
  The sample `labeled` drew is not read, so `n_per_class` shapes nothing here. `weight_classes`
  (on by default) weights the loss by class frequency in its place. The step may read other
  inputs than `labeled` did.
- **The classifier's scope is not part of its run id.** The `templates` run names the
  training entries, so `lightning-action`, like `xgboost`, keeps its run id when you widen
  the scope: the stored fit is applied to the new entries and nothing is fitted again.
  The first request must hold every training entry that has labels, and the step refuses
  a scope that leaves one out. To predict with a model fitted elsewhere, name it in a
  second step with `"model": {"step": "classifier"}` and no `templates` or `labels`.
  `fit_scope.json` lists the entries of the templates run, which are the ones that
  trained. A fit does not move when you convert a training entry again: that entry's
  output is computed again with the stored model. To fit again, change a parameter.
- **A run of consecutive labeled frames is the unit of training.** A frame no label covers
  ends a run, and so does a gap in the frame numbers. The network reads windows of
  `sequence_length` frames from the first frame of a run, so a run shorter than that does
  not train, and neither do the frames past its last whole window. `sequence_length` must
  exceed twice the context the encoder reads on each side of a frame, which `num_layers`
  and `num_lags` set. The fit names both numbers when it does not. The package validates on
  a tenth of its windows and warns when it has fewer than ten.
- **On CalMS21 pair features** the resident-focal series is labeled and trains. The
  intruder-focal series is unlabeled, so it writes no training run, and it is still
  predicted.
- **Every row is predicted.** Each series is predicted as a whole, so no row at the end of
  an entry falls to `default_class`. That class stands only for a row that no class clears
  `decision_threshold` for, and for a row with a missing `id1`, `id2` or `perspective`.
- **It trains on a GPU** when the machine has one, and the step is offered to GPU
  workers. `device` set to `cpu` trains on the CPU.

## Train a sequence network

`sequence-classifier` trains a neural network on frames in order and predicts each series
whole. A head from a registry is the network: `tcn` (dilated convolutions, centered on
each frame), `gru` (a bidirectional GRU) or `mlp` (each frame by itself). It needs the
`[deep-learning]` extra and is offered to a GPU worker.

Its defaults are those of the networks in the CalMS21 Task 1 agent runs, a published
benchmark, where a bidirectional GRU and a dilated TCN on 37 per-frame columns scored
0.859 and 0.855 on its test split, macro F1 over attack, investigation and mount, the
score of XGBoost on 981 columns. Read that as documented behavior of one dataset, scored
on a split these defaults saw, and not as a result for yours: cross-validate before you
trust it. The recipe is the `lightning-action` one with the classifier swapped:

```json
{
  "schema_version": 1,
  "name": "sequence-network",
  "steps": [
    {"id": "tracks", "type": "op", "kind": "convert-tracks",
     "params": {"source_format": "calms21_npy"}},
    {"id": "labels", "type": "op", "kind": "convert-labels",
     "params": {"source_format": "calms21_npy"}},
    {"id": "egocentric", "type": "feature", "feature": "pair-egocentric",
     "inputs": ["tracks"], "tracks": {"step": "tracks"}},
    {"id": "labeled", "type": "feature", "feature": "extract-labeled-templates",
     "inputs": [{"step": "egocentric"}],
     "params": {"labels": {"step": "labels"}, "n_per_class": 500, "test_fraction": 0}},
    {"id": "classifier", "type": "feature", "feature": "sequence-classifier",
     "inputs": [{"step": "egocentric"}],
     "params": {"templates": {"step": "labeled"}, "labels": {"step": "labels"},
                "default_class": 3, "head": "gru", "cv_folds": 5, "random_state": 42}},
    {"id": "evaluate", "type": "feature", "feature": "evaluate-labels",
     "inputs": [{"step": "classifier"}],
     "params": {"labels": {"step": "labels"}, "perspective": 0,
                "average_over": [0, 1, 2]}}
  ]
}
```

Choose a setting on the training groups first:

```bash
mosaic pipeline run -m dataset.yaml --recipe @network.json \
  --step-scope 'labeled={"groups": ["train"]}' \
  --step-scope 'classifier={"groups": ["train"]}' \
  --step-scope 'evaluate={"groups": ["train"]}'
```

Then widen the classifier to the test groups and score them once:

```bash
mosaic pipeline run -m dataset.yaml --recipe @network.json \
  --step-scope 'labeled={"groups": ["train"]}' \
  --step-scope 'classifier={"groups": ["train", "test"]}' \
  --step-scope 'evaluate={"groups": ["test"]}'
```

- **The second command fits nothing again.** The `templates` run names the training
  entries, so the classifier's scope is not part of its run id, and the wider step is the
  same run, as it is for `xgboost` in the cross-validation section above. Only the test
  groups are computed, with the final model, and the training entries keep their
  out-of-fold output. Ask the first request over a scope that holds the training entries,
  and the step refuses one that does not. A fitted-weight `probability-ensemble` behaves
  the same way, because its members' fold table names the entries its weights were chosen
  on.
- **The templates name the training entries, and the frames come from the step's
  inputs.** As with `lightning-action`, the sample that `labeled` drew is not read, an
  entry it held out never trains, and the step refuses a scope that leaves out a training
  entry that has labels. With `test_fraction` above 0, the sequences the sample sets
  aside are neither trained on nor dealt into a fold, and the full model scores them.
  Leave it at 0 to cross-validate every training sequence.
- **Heads and their options.** `head` is `tcn`, `gru` or `mlp`, and `head_config` holds
  the options of that head: `hidden`, `blocks`, `kernel_size` and `dropout` for `tcn`,
  `hidden`, `layers`, `bidirectional` and `dropout` for `gru`, and `hidden`, `layers` and
  `dropout` for `mlp`. An option the head does not have is refused. A `tcn` reads
  `(kernel_size - 1) * (2 ** blocks - 1) + 1` frames around each frame, 127 by default.
- **Name the columns.** `columns` lists the payload columns the network reads. A narrow set
  can beat a wide one (on CalMS21, 37 columns beat 562 to 999 by 0.04 of cross-validated
  macro F1), so compare sets by cross-validation. Unset, every payload column of the inputs
  is read.
- **A run of consecutive labeled frames is the unit of training.** A frame no label covers
  ends a run, and so does a gap in the frame numbers. A batch is `batch_size` random crops
  of up to `crop_length` frames of whole runs, drawn in proportion to the run's length,
  and a run shorter than `min_run` does not train. The loop is AdamW under a one-cycle
  schedule (`learning_rate`, `weight_decay`) for `steps` batches, with square-root class
  weights (`class_weight`), and the weights of the last step are kept. There is no
  validation split.
- **A series is predicted whole.** The two perspectives of a pair table are two series, a
  series is cut where its frames are not consecutive, and a row of no series takes
  `default_class`. Frames are standardized by the training frames' mean and standard
  deviation, which the model stores.
- **`cv_folds` works as for `xgboost`.** The training entries are dealt into folds by the
  same hash of `cv_seed`, group and sequence, so two classifiers on one templates run
  share their folds. One network is fitted without each fold, and the full model last. A
  training entry is scored by the network of its fold, and any other entry by the full
  model. `cv_folds.parquet` lists the fold of every entry and its labeled frames per class.
  With `cv_unit: block`, as for `xgboost` above, it holds one row per block with its
  `start` and `stop`, and the network of a fold trains on the labeled runs of every entry
  less its blocks and `cv_gap_frames` frames around them.
- **Device and repeatability.** `device` is `cuda` or `cpu` and is part of the run id,
  because a GPU and a CPU need not give one model. A model fitted on a GPU applies on a
  CPU (`apply_device`, not part of the run id). A CPU fit repeats at one thread count
  (`n_threads`, 1 by default), and a GPU fit repeats on the same machine. The torch and
  CUDA versions are recorded in the model's bundle.
- **A head of your own.** Register one in Python with `register_head(HeadSpec(...))`,
  giving its name, the model of its options and a function that builds a torch module
  from the options, the number of input columns and the number of classes. Raise
  `head_revision` when its code changes, so that the change is a new run. A head registered
  in Python exists only in that process. The command line and a queue's workers start their
  own, and cannot name it.

## Average and blend classifiers

`probability-ensemble` combines the `prob_<class>` columns of two or more classifier runs
and writes `prob_<class>` and `predicted_label` as a classifier does. `weights` is
`uniform` (an average, such as the seeds of one model), a list with one weight per input,
or `fit`, which chooses the weights that maximize the pooled macro F1 over `score_classes`
on the out-of-fold probabilities of the training entries. It searches a grid of
`grid_step` over the weights and takes the first maximum, and it refuses a member fitted
without `cv_folds`, so that no in-sample probability picks a weight.

Averages of cross-validated runs are still out of fold, and the step writes their fold
table, so a blend can take an average as a member. Two seeds of each of two networks, the
averages of each pair, and a blend of the averages:

```json
{
  "schema_version": 1,
  "name": "blend",
  "steps": [
    {"id": "tracks", "type": "op", "kind": "convert-tracks",
     "params": {"source_format": "calms21_npy"}},
    {"id": "labels", "type": "op", "kind": "convert-labels",
     "params": {"source_format": "calms21_npy"}},
    {"id": "egocentric", "type": "feature", "feature": "pair-egocentric",
     "inputs": ["tracks"], "tracks": {"step": "tracks"}},
    {"id": "labeled", "type": "feature", "feature": "extract-labeled-templates",
     "inputs": [{"step": "egocentric"}],
     "params": {"labels": {"step": "labels"}, "n_per_class": 500, "test_fraction": 0}},
    {"id": "gru_1", "type": "feature", "feature": "sequence-classifier",
     "inputs": [{"step": "egocentric"}],
     "params": {"templates": {"step": "labeled"}, "labels": {"step": "labels"},
                "default_class": 3, "head": "gru", "cv_folds": 5, "random_state": 1}},
    {"id": "gru_2", "type": "feature", "feature": "sequence-classifier",
     "inputs": [{"step": "egocentric"}],
     "params": {"templates": {"step": "labeled"}, "labels": {"step": "labels"},
                "default_class": 3, "head": "gru", "cv_folds": 5, "random_state": 2}},
    {"id": "tcn_1", "type": "feature", "feature": "sequence-classifier",
     "inputs": [{"step": "egocentric"}],
     "params": {"templates": {"step": "labeled"}, "labels": {"step": "labels"},
                "default_class": 3, "head": "tcn", "cv_folds": 5, "random_state": 1}},
    {"id": "tcn_2", "type": "feature", "feature": "sequence-classifier",
     "inputs": [{"step": "egocentric"}],
     "params": {"templates": {"step": "labeled"}, "labels": {"step": "labels"},
                "default_class": 3, "head": "tcn", "cv_folds": 5, "random_state": 2}},
    {"id": "gru", "type": "feature", "feature": "probability-ensemble",
     "inputs": [{"step": "gru_1"}, {"step": "gru_2"}], "params": {"default_class": 3}},
    {"id": "tcn", "type": "feature", "feature": "probability-ensemble",
     "inputs": [{"step": "tcn_1"}, {"step": "tcn_2"}], "params": {"default_class": 3}},
    {"id": "blend", "type": "feature", "feature": "probability-ensemble",
     "inputs": [{"step": "gru"}, {"step": "tcn"}],
     "params": {"default_class": 3, "weights": "fit", "labels": {"step": "labels"},
                "score_classes": [0, 1, 2]}},
    {"id": "evaluate", "type": "feature", "feature": "evaluate-labels",
     "inputs": [{"step": "blend"}],
     "params": {"labels": {"step": "labels"}, "perspective": 0,
                "average_over": [0, 1, 2]}}
  ]
}
```

Screen on the training groups:

```bash
mosaic pipeline run -m dataset.yaml --recipe @blend.json \
  --step-scope 'labeled={"groups": ["train"]}' \
  --step-scope 'gru_1={"groups": ["train"]}' \
  --step-scope 'gru_2={"groups": ["train"]}' \
  --step-scope 'tcn_1={"groups": ["train"]}' \
  --step-scope 'tcn_2={"groups": ["train"]}' \
  --step-scope 'gru={"groups": ["train"]}' \
  --step-scope 'tcn={"groups": ["train"]}' \
  --step-scope 'blend={"groups": ["train"]}' \
  --step-scope 'evaluate={"groups": ["train"]}'
```

Then widen the networks, the averages and the blend to the training and test groups, and
score the test groups:

```bash
mosaic pipeline run -m dataset.yaml --recipe @blend.json \
  --step-scope 'labeled={"groups": ["train"]}' \
  --step-scope 'gru_1={"groups": ["train", "test"]}' \
  --step-scope 'gru_2={"groups": ["train", "test"]}' \
  --step-scope 'tcn_1={"groups": ["train", "test"]}' \
  --step-scope 'tcn_2={"groups": ["train", "test"]}' \
  --step-scope 'gru={"groups": ["train", "test"]}' \
  --step-scope 'tcn={"groups": ["train", "test"]}' \
  --step-scope 'blend={"groups": ["train", "test"]}' \
  --step-scope 'evaluate={"groups": ["test"]}'
```

The second request fits nothing again: no step of the blend hashes its scope, so each is
the run of the first request and only the test groups are computed. The networks' folds
name the entries that the blend's weights were chosen on.

- A member of a fitted blend may be any classifier run that writes `cv_folds.parquet`,
  `xgboost` included: add `{"step": "classifier"}` to the blend's `inputs` beside the
  averages. The members must write the same classes and deal the same folds, so give them
  one templates run and one `cv_seed`. The step compares the folds, `group`, `sequence`
  and `fold`, with the `start` and `stop` of a block fold, and not the counts, because
  `xgboost` counts sampled template rows where `sequence-classifier` counts labeled
  frames.
- The blend's score over the training entries is optimistic by the freedom of its weights,
  one fewer than the number of members. `ensemble_report.json` in the run directory gives
  the weights, the objective, the objective of each member alone and of the uniform
  average. The held-out score is the test one.
- The grid has a limit of 20,000 weight vectors. Average the seeds of each model first, as
  above, and blend the averages. Six members at the default step of 0.05 would be
  well over the limit.
- `weights` as a list or `uniform` needs no labels and no folds, and the run does not
  depend on its scope. Members may then differ in whether they were cross-validated, such
  as a network fitted with `cv_folds` and one applied from a stored model. The blend is
  averaged all the same, and writes no fold table, because it is not out of fold wherever
  its members differ.
- To score runs that cover different entries as one, give `evaluate-labels` all of them
  as `inputs`, such as a cross-validated run over the training groups and an applied one
  over the test groups. It reports their union, with the pooled counts of a single run
  over all of them. An entry that two inputs hold is refused, because it would count
  twice, and the plan refuses it first.

## Read the predictions

The classifier writes its predictions into `features/xgboost__.../<run_id>/`, carrying
the input's identity columns through unchanged. On pair-level input that includes
`perspective`, so the predictions join one-to-one back against the features they were
trained on — the embedding, the clustering, anything else at that level. See [What
identifies a pair row](../../concepts/features.md#what-identifies-a-pair-row).

To fit or score in your own code, read the run as an array. `load_run_matrix` fills
one array with the columns you name, instead of building a frame of every column:

```python
from mosaic.core.pipeline import load_run_matrix

rows = load_run_matrix(ds, xgb_result, columns="prob_*", perspective=0)
rows.index     # group, sequence, frame and id (id1, id2 and perspective on pair output)
rows.values    # float32, one row for each row of rows.index
rows.columns   # ("prob_0", "prob_1", ...)
```

`columns` takes a name, a glob or a list. `scope=` narrows the entries, `perspective=`
keeps one ordering of a pair, and `dtype=` replaces `float32`. Rows come in order of
group and sequence. The call refuses a run that is not finished, a scope that names an
entry the run lacks, and a read that does not fit in the available memory, and the
message says how to narrow the read. `allow_partial=True` reads what the run holds:
`rows.entries` is what was read, `rows.finished` is `False` for a run that lost an entry
or is still going, and `rows.missing` lists the entries of the scope that have no file.
For the label of each row, use
[`align_predictions_to_labels`](discover-behaviors.md#score-the-result).

Draw them onto the video to see what the model learned:

```python
from mosaic.behavior.visualization_library import Overlay

ds.run_feature(Overlay(Inputs((xgb_result,)), params={"color_by": "predicted_label"}))
```

## FERAL

`feral` skips the feature chain and classifies from video through a V-JEPA backbone.
It runs in mosaic's own process but wants its own environment, because it pins exact
dependency versions — see [installation](../../installation.md#feral).

## Worked example

[`calms21-template.ipynb`][template] runs this whole path on CalMS21: features →
wavelets → scaler → temporal stack → labeled templates → XGBoost, then draws the
predictions back onto the t-SNE embedding.

[template]: https://github.com/EcodylicScience/mosaic/blob/main/notebooks/calms21-template.ipynb
