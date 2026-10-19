# Train a behavior classifier

Supervised classification predicts an annotated behavior on frames nobody annotated.
It needs labels: see [Import behavior annotations](import-labels.md) first.

Three classifiers ship. All are features, so all cache and take a `run_id`.

| Feature | Reads | Install |
| --- | --- | --- |
| `xgboost` | per-frame feature columns | base install |
| `lightning-action` | per-frame feature columns, modeled as sequences | `[lightning-action]` |
| `feral` | video directly, through a V-JEPA backbone | `[feral]`, in its own environment |

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
down. `n_jobs` limits the threads of a fit and does not enter the run id. The template
draw and the fit each refuse, before they start, a sample that would not fit in the
available memory, and the message says what to lower.

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
    {"id": "speed", "type": "feature", "feature": "speed-angvel",
     "inputs": ["tracks"], "tracks": {"step": "tracks"}},
    {"id": "templates", "type": "feature", "feature": "extract-templates",
     "inputs": [{"step": "speed"}], "params": {"n_templates": 2000}},
    {"id": "scaler", "type": "feature", "feature": "global-scaler",
     "inputs": [{"step": "speed"}], "params": {"templates": {"step": "templates"}}},
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
  refuses a fold whose training rows hold fewer than two classes.

When you have chosen a setting, widen the same classifier step to the training and test
groups, and score the test groups once:

```bash
  --step-scope 'classifier={"groups": ["train", "test"]}' \
  --step-scope 'evaluate={"groups": ["test"]}'
```

The scope is not part of the classifier's run id, so this is the same run. The model is a
cache hit, and only the test groups are computed. A classifier step scoped to the
training and test groups from the start would score the test groups for every setting
you try. Keep one classifier step: two steps with the same parameters and different
scopes are one run, and can fit at the same time.

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
- A fold is a set of sequences. If several sequences come from one animal or one
  recording, a fold that splits them leaks across them, and mosaic does not know. Make
  that unit one sequence before you draw the templates.

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

`lightning-action` models the temporal structure itself, so it does not need this step.
It reads an individual-level input and refuses a pair table. Train `xgboost` on pair
features.

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
