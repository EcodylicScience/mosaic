# Score and evaluate a classifier

Score a classifier on held-out groups, score predictions made elsewhere, and read the
predictions. Start from [Train a behavior classifier](train-a-classifier.md).

## XGBoost parameters

`n_per_class` caps how many frames each behavior contributes, so a rare behavior is not
drowned by a common one. `test_fraction` reserves whole sequences for testing. A split by
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
down. `n_jobs` limits the threads of a fit and does not enter the run id. `device` is
`"cpu"` by default, and `"cuda"` fits on a GPU where the machine has one. `"cuda"` also
moves the step to a GPU lane: submitting it raises the `gpu_lane` confirmation, and the
step takes a card of its own. `device` enters the run id, and the fitted model applies
on any machine. The plan cautions about a `"cpu"` fit over more than 50 entries on a
machine that has PyTorch and a GPU. The caution changes nothing: the default stays
`"cpu"` and the step keeps its lane. The template draw and the fit each refuse a sample
that would not fit in the available memory, and the message says what to lower.

## Train and score in one request

The `supervised` recipe of [Train a behavior classifier](train-a-classifier.md) runs the
whole route at once, from the raw files to a score on held-out groups. Step scopes make
the split: each one narrows a single step to part of what the request covers, so the
fitting steps see the training groups and the scoring step the test group.

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
`average_over` (here every class but `other`), `kappa` (Cohen's kappa of the scored
rows, `null` where it is undefined), precision, recall and F1 per class, and a confusion
matrix, which is also `confusion.parquet`. With `bootstrap` above 0, the report also
holds percentile intervals of the accuracy, the macro F1, kappa and each class's F1 from
that many resamples of the entries, drawn with `bootstrap_seed` and as wide as
`bootstrap_confidence`. Each of its output rows is one entry's own score, with its true
positives, false positives and false negatives per class (`tp_<class>`, `fp_<class>`,
`fn_<class>`). A row no label covers is left out and counted in `n_unlabeled`.
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

## Remap and mask labels

`remap` and `mask` change the classes a step sees, without converting the labels again.
`remap` maps a class id of the label file to the id that every step reads, and merges
classes when two ids map to one. `mask` lists class ids whose frames become unlabeled,
and steps neither train on nor score them. Both go on `labels`:

```json
{"labels": {"step": "labels", "remap": {"2": 0}, "mask": [3]}}
```

- **Controls apply to every entry a step reads.** They rewrite the labels as read and
  leave the file as it is. All swaps happen at once, so `{"1": 2, "2": 1}` exchanges two
  classes. A frame the file marks as background is remapped like any other. An id in
  both fields is refused, and so is a negative id.
- **Give each step that reads labels the same controls.** These are
  `extract-labeled-templates`, `sequence-classifier`, `lightning-action`,
  `tune-threshold`, `probability-ensemble` with `weights` set to `"fit"`,
  `evaluate-labels`, `fit-label-map` and `label-statistics`. `xgboost` reads labels
  through its templates and takes none. A classifier trains on the classes its templates
  and labels leave, and a step that scores it must score the same ones.
- **A truth class the classifier cannot predict is refused.** Where controls are set, on
  the step or on any step upstream of its input, `tune-threshold`,
  `probability-ensemble` with `weights` set to `"fit"` and `evaluate-labels` in
  `classifier` mode refuse labeled frames of a class for which the input has no
  `prob_<class>` column. The message names the classes and says to give the step the
  controls of the classifier's labels. A chain without controls is not checked, and a
  class missing from the classifier is scored as missed. That covers a class omitted by
  `n_per_class` and a class with only runs shorter than `min_run`.
- **A `labels` reference takes the controls.** `{"step": "labels", "remap": {...},
  "mask": [...]}` reads the labels that step converts, with the controls. Only a
  `labels` field accepts them. An `inputs`, `tracks` or `templates` reference with
  `remap` or `mask` is refused.
- **A setting changes the run id.** A run that reads labels records `remap` and `mask`
  in its `params.json`, and a run with neither keeps the id it had.
- `evaluate-labels` also takes `label_map`, which maps its predictions and not its
  truth. A `fit-label-map` step takes the same `remap`, and its map then targets the
  remapped classes.

## Score predictions made outside mosaic

A model that ran elsewhere, or a second annotator, can be scored with the metrics of
this page. Its predictions enter as the label kind `predictions`, and the
`label-predictions` feature reads them onto the rows of a tracks table or of a pair
feature. `evaluate-labels`, `tune-threshold` and `probability-ensemble` take that step as
they take a classifier.

- **The file** is a `.parquet` file, or a `.csv` file with a header row, with one
  prediction per row. `frame` is the frame of the entry's tracks table. `sequence` (and
  `group`, when the raw source does not give it) names the entry, so one file may hold
  several. `id` marks an individual row, `id1` and `id2` a pair row with `id1` the focal
  individual, and a row without ids applies to the scene. `predicted_label` is the class,
  and `prob_<class>` columns give its probabilities. Without `predicted_label`, a row
  takes the class of its highest probability, the lowest id on a tie. Names in place of
  ids need `class_ids`, a map from each name to its id.
- **The scan source** has format `predictions_table`, and its `patterns` name `*.parquet`
  beside `*.csv`, because the default patterns skip parquet files. Leave out
  `multi_sequences_per_file` for a file with a `sequence` column.

  ```bash
  mosaic sources add -m dataset.yaml --kind labels \
      --path labels_raw/predictions/own-model --patterns '*.parquet,*.csv' \
      --src-format predictions_table
  mosaic scan -m dataset.yaml --kind labels
  ```

- **The recipe** converts the labels and the predictions in two steps, reads the
  predictions onto the tracks, and scores them. `files` selects the files of one set of
  predictions when a source holds several, and a conversion with other params is another
  variant. One entry may come from one file only: two files for the same entry are
  refused before anything is written.

  ```json
  {
    "schema_version": 1,
    "name": "external-predictions",
    "steps": [
      {"id": "tracks", "type": "op", "kind": "convert-tracks",
       "params": {"source_format": "calms21_npy"}},
      {"id": "labels", "type": "op", "kind": "convert-labels",
       "params": {"source_format": "calms21_npy"}},
      {"id": "predictions", "type": "op", "kind": "convert-labels",
       "params": {"kind": "predictions", "source_format": "predictions_table",
                  "params": {"files": "labels_raw/predictions/own-model/*"}}},
      {"id": "own", "type": "feature", "feature": "label-predictions",
       "inputs": ["tracks"], "tracks": {"step": "tracks"},
       "params": {"predictions": {"step": "predictions"}}},
      {"id": "score", "type": "feature", "feature": "evaluate-labels",
       "inputs": [{"step": "own"}],
       "params": {"labels": {"step": "labels"}, "bootstrap": 2000}}
    ]
  }
  ```

  An entry without a file in the variant has its rows written without a prediction, and
  `score` counts them in `n_missing_prediction`. `missing_predictions: "error"` fails the
  entry instead. A row that two events of the file match is refused, since its prediction
  would depend on their order. `mosaic pipeline score` takes a recipe with one
  `convert-labels` step, so it does not score this one, which converts twice. To score
  the test groups, narrow the `score` step to them with `--step-scope`, as above.

- **Folds.** A `fold` column marks the rows that the model of fold `k` predicted out of
  fold, with `-1` or an empty cell for a row predicted in sample. The `label-predictions`
  run then writes `cv_folds.parquet`, which `tune-threshold` and `probability-ensemble`
  read as they read a classifier's. A `block` column beside `fold` names the
  cross-validation block of each row and gives block folds. Write every frame of each
  block, and the `block` column: without it, neighboring blocks of one fold merge, and a
  blend of this run with a block-validated classifier refuses the pair, because their
  folds differ.
- **Agreement between annotators.** Write the second annotator's classes as a
  `predictions_table` file with `predicted_label` and no probabilities, and score it
  against the first annotator's labels. Rows the first annotator left unlabeled are
  counted in `n_unlabeled`, and rows the second left unlabeled in
  `n_missing_prediction`. The scores cover the rows both labeled. Accuracy and kappa stay
  the same when the two annotators swap, per-class precision and recall exchange, and
  per-class F1 stays. For CalMS21, write the pair as `id1` 0 and `id2` 1: each frame is
  then scored once, on the resident's row.

## Read the predictions

The classifier writes its predictions into `features/xgboost__.../<run_id>/`, carrying
the input's identity columns through unchanged. On pair-level input that includes
`perspective`, so the predictions join one-to-one back against the features they were
trained on: the embedding, the clustering, anything else at that level. See [What
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
from mosaic.core.pipeline.types import Inputs

ds.run_feature(Overlay(Inputs((xgb_result,)), params={"color_by": "predicted_label"}))
```
