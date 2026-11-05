# Cross-validate a classifier

Score the settings you try on the training groups, by folds of sequences, animals or
frame blocks. The steps change the recipe of
[Train a behavior classifier](train-a-classifier.md).

## Choose a setting by cross-validation

Scoring every setting you try on the test groups spends the test groups. `cv_folds` on
`xgboost` scores the training sequences instead, each by a model that never saw it.
Changed steps of that recipe:

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
  sequences come from one animal, one session or one recording, a fold that splits them
  leaks across them. Name that link with `cv_group_by` (below).

## Hold out the sequences of one animal together

`cv_group_by` on `xgboost`, `sequence-classifier` or `lightning-action` maps an entry
(`group:sequence`, or a bare sequence of the empty group) to a tie key or a list of
them, such as an animal, a session or a recording:

```json
{"id": "classifier", "type": "feature", "feature": "xgboost",
 "inputs": [{"step": "scaler"}],
 "params": {"templates": {"step": "labeled"}, "default_class": 3, "cv_folds": 3,
            "cv_group_by": {"train:s1": "mouse-a", "train:s2": ["mouse-a", "day-2"],
                            "train:s3": "day-2"}}}
```

- Training entries that share a key fall into one fold, and so do entries linked
  through a chain of shared keys: `s1`, `s2` and `s3` above share a fold. Only the
  training entries link: a test entry that shares a key with two training entries does
  not tie them. A training entry the mapping leaves out is a tie by itself.
- Each tie goes whole to the fold with the fewest entries so far, the largest tie first,
  so the folds can differ in size by more than one sequence. `cv_folds` is at most the
  number of ties. `cv_folds.parquet` keeps its columns.
- The mapping may list entries outside the training set, and the fit counts them in a
  warning. In mosaic-api the animal and session of an entry are its sequence tags, and
  the mapping passes them to the folds.
- `cv_group_by` needs `cv_folds` and the sequence unit, and enters the run id. Two
  spellings of one mapping (key order, value order, a string or a one-item list) are one
  run.

## Cross-validate one training sequence by blocks

Folds of whole sequences need more training sequences than folds. A dataset with one
training sequence has no score of that kind, and `cv_folds` refuses it. Set `cv_unit` to
`block` on `xgboost`, `sequence-classifier` or `lightning-action` to hold out stretches
of frames instead:

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
  one of its output rows depends on (`mosaic pipeline catalog <name>` lists it, and for
  `sequence-classifier` the reach of each `head`), and the plan
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
- `lightning-action` and `sequence-classifier` cut the held-out frames out of their runs
  of labeled frames, so they need no frames in the templates.
