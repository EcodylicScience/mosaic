# Thresholds, smoothing and blends

Thresholds chosen out of fold, smoothing over frames, and averages and blends of several
classifier runs turn probabilities into calls.
Start from [Train a behavior classifier](train-a-classifier.md).

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

## Smooth probabilities and merge short bouts

A frame-by-frame classifier flickers inside a bout. `smooth-probabilities` takes one
classifier run (or one `probability-ensemble`), smooths each `prob_<class>` over
consecutive frames, divides each row by its sum and chooses `predicted_label` again from
the result:

- `method` is `gaussian` (`sigma` frames, 2.0 by default), `median` (an odd `window` of
  at least 3 frames, 5 by default) or `none`. `sigma` is refused without `gaussian`, and
  `window` without `median`.
- `min_bout_frames` merges bouts: a run of one predicted class shorter than it takes the
  class of the neighbor whose class has the higher mean probability over the short run,
  the shortest first. The default of 1 merges nothing. A merged frame can differ from
  the argmax of its probabilities.
- `default_class` is required, as on `probability-ensemble`, and `decision_threshold`
  takes a number or a mapping of class to cut.
- The step reads one series at a time, cut at a frame gap, at a row without a
  probability and where the scoring fold changes. A series sampled at a frame step
  other than 1, such as every second frame, is refused, naming the frame step it found.
  Smooth the table before anything subsamples it.
- A row depends on about `4 * sigma` frames on each side with `gaussian` and on half of
  `window` with `median`. With `min_bout_frames` above 1 the dependence is unbounded.
  It matters only for a classifier stacked over the step, where `cv_gap_frames` must
  cover it.
- **Out-of-fold scores stay out of fold.** A fold's network did not train on the frames
  it scored, and smoothing combines only frames of one fold. The step copies the
  input's `cv_folds.parquet`, so `tune-threshold` and a fitted-weight
  `probability-ensemble` downstream find the folds where they find a classifier's.

Cuts chosen by `tune-threshold` were scored on the probabilities of its input, and
smoothing changes them. A `decision_threshold` that names a `tune-threshold` run is
therefore accepted only with `method` set to `none`, and only for the run tuned on this
step's input. Tune over the smoothing step, then apply the cuts and merge bouts in a
second step that smooths nothing:

```json
{"id": "smooth", "type": "feature", "feature": "smooth-probabilities",
 "inputs": [{"step": "classifier"}],
 "params": {"method": "gaussian", "sigma": 2.0, "default_class": 3}},
{"id": "smooth-tune", "type": "feature", "feature": "tune-threshold",
 "inputs": [{"step": "smooth"}],
 "params": {"labels": {"step": "labels"}, "default_class": 3,
            "metric": "macro_f1", "average_over": [0, 1, 2]}},
{"id": "smooth-apply", "type": "feature", "feature": "smooth-probabilities",
 "inputs": [{"step": "smooth"}],
 "params": {"method": "none", "default_class": 3, "min_bout_frames": 5,
            "decision_threshold": {"step": "smooth-tune"}}}
```

Score `smooth-apply` with `evaluate-labels`, scoped to the test groups, as for the cuts
of the previous section. `smooth` and `smooth-apply` take the scope of the classifier,
and `smooth-tune` the training groups.

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
