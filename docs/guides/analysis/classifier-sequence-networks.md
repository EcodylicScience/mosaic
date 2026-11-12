# Sequence context and networks

Give a classifier the frames around each frame, or train a network that reads frames in
order. Start from [Train a behavior classifier](train-a-classifier.md).

## Give each frame its context

A single frame rarely identifies a behavior. `temporal-stack` widens each row into a
window before the classifier sees it:

```python
from mosaic.behavior.feature_library import TemporalStackingFeature
from mosaic.core.pipeline.types import Inputs

stacked_result = ds.run_feature(TemporalStackingFeature(
    Inputs((scaler_result,)),
    params={"half": 2, "skip": 1, "use_temporal_stack": True, "sigma_stack": 2},
))
```

`half=2, skip=1` gives each frame the two before and the two after. Feed
`stacked_result` to `ExtractLabeledTemplates` and to the classifier in place of the raw
features.

`individual-wavelet` gives each individual a spectral context instead: the log power of
the columns `cols` names at `n_freq` frequencies from `f_min` to `f_max`. Wavelets of
speeds and signed velocities (`keypoint-kinematics`) improved a classifier on CalMS21,
and wavelets of posture PCs did not. The band should suit the frame rate and the
behavior: 0.5 to 15 Hz served 30 fps mouse video, and the default is 0.2 to 5 Hz. A pair
model reads individual wavelets through a join to pair rows. A dropout (a missing frame
or a NaN, such as a keypoint the tracker lost) blanks each frequency for half its
wavelet's support on each side: about 40 s at 0.2 Hz and 1.6 s at 5 Hz with the default
wavelet. Raising `f_min` shortens it.

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
- **`cv_folds` works as for `xgboost`, at one more GPU training run per fold.** The
  training entries are dealt into folds by the same hash of `cv_seed`, group and
  sequence, so the classifiers on one templates run share their folds. One package model
  is trained without each fold, and the full model last, so the fit takes K + 1 training
  runs. A fold's model trains on the runs of the entries outside its fold and
  standardizes by their frames alone. A training entry is scored by the model of its
  fold, and any other entry by the full model. `cv_folds.parquet` lists the fold of
  every entry and its training frames per class. With `cv_unit: block` a fold's model
  trains on every run less its blocks and `cv_gap_frames` frames around them, each piece
  that is left cut to whole windows of `sequence_length`, and a row is scored by the
  model of the fold whose block holds its frame. `cv_group_by` ties entries into one
  fold as it does for `xgboost`. Each fold's model validates on a tenth of its own
  windows, so it warns sooner than the full model does.

## Train a sequence network

`sequence-classifier` trains a neural network on frames in order and predicts each series
whole. A head from a registry is the network: `tcn` (dilated convolutions, centered on
each frame), `gru` (a bidirectional GRU), `mlp` (each frame by itself), `transformer`
(attention within blocks of frames) or `unet` (convolutions at halving time
resolutions). It needs the `[deep-learning]` extra and is offered to a GPU worker.

Its defaults are those of networks trained on CalMS21 Task 1, a published benchmark,
where a bidirectional GRU and a dilated TCN on 37 per-frame columns scored 0.859 and
0.855 on its test split, macro F1 over attack, investigation and mount, the
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
  same run, as it is for `xgboost` under
  [cross-validation](classifier-cross-validation.md). Only the test groups are computed,
  with the final model, and the training entries keep their out-of-fold output. Ask the
  first request over a scope that holds the training entries, and the step refuses one
  that does not. A fitted-weight `probability-ensemble` behaves the same way, because its
  members' fold table names the entries its weights were chosen on.
- **The templates name the training entries, and the frames come from the step's
  inputs.** As with `lightning-action`, the sample that `labeled` drew is not read, an
  entry it held out never trains, and the step refuses a scope that leaves out a training
  entry that has labels. With `test_fraction` above 0, the sequences the sample sets
  aside are neither trained on nor dealt into a fold, and the full model scores them.
  Leave it at 0 to cross-validate every training sequence.
- **Heads and their options.** `head` is `tcn`, `gru`, `mlp`, `transformer` or `unet`,
  and `head_config` holds the options of that head: `hidden`, `blocks`, `kernel_size`
  and `dropout` for `tcn`, `hidden`, `layers`, `bidirectional` and `dropout` for `gru`,
  `hidden`, `layers` and `dropout` for `mlp`, `hidden`, `layers`, `attention_heads`,
  `window`, `kernel_size` and `dropout` for `transformer`, and `hidden`, `depth`,
  `kernel_size` and `dropout` for `unet`. An option the head does not have is refused.
  A `tcn` reads `(kernel_size - 1) * (2 ** blocks - 1) + 1` frames around each frame,
  127 by default.
- **A transformer attends within blocks.** The series is cut into blocks of `window`
  frames (64 by default), and each frame attends to its block and the blocks on either
  side, so a long series costs time and memory in proportion to its length and predicts in
  one pass. Each layer reaches one block further, and the network reads
  `kernel_size // 2 + (layers + 1) * window - 1` frames on each side, 321 by default. A
  convolution of `kernel_size` frames before the layers is its only sense of position,
  since a predicted series is longer than the crops it trains on. `attention_heads`
  divides `hidden`.
- **A U-Net reads at several time resolutions.** Each of its `depth` levels (4 by
  default) convolves twice and halves the frames, and the way back up doubles them and
  joins the convolutions of the same level, so a frame is labeled from coarse context
  and fine detail at once. The series is padded to a multiple of `2 ** depth` frames by
  repeating its last frame. It reads at most
  `(4 * (kernel_size // 2) + 1) * (2 ** depth - 1)` frames on each side, 135 by
  default, and that bound is the step's `temporal_reach`.
- **Refinement stages.** Every head takes `stages` in `head_config`, 1 by default. Each
  stage after the head is a `tcn` at its defaults that reads the softmax of the logits
  before it, and the last stage predicts. Each stage trains on a loss of its own, and the
  loop takes their mean. A stage adds 63 frames of reach on each side, and a model of
  several stages takes longer to train and to apply.
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
  validation split. `label_smoothing` spreads that share of each target evenly over the
  classes, 0 by default; 0.1 is a common choice when labels at bout edges disagree.
- **A series is predicted whole.** The two perspectives of a pair table are two series, a
  series is cut where its frames are not consecutive, and a row of no series takes
  `default_class`. Frames are standardized by the training frames' mean and standard
  deviation, which the model stores.
- **`cv_folds` works as for `xgboost`.** The training entries are dealt into folds by the
  same hash of `cv_seed`, group and sequence, so two classifiers on one templates run
  share their folds. One network is fitted without each fold, and the full model last. A
  training entry is scored by the network of its fold, and any other entry by the full
  model. `cv_folds.parquet` lists the fold of every entry and its labeled frames per class.
  With `cv_unit: block`, as for
  [`xgboost`](classifier-cross-validation.md#cross-validate-one-training-sequence-by-blocks),
  it holds one row per block with its `start` and `stop`, and the network of a fold
  trains on the labeled runs of every entry less its blocks and `cv_gap_frames` frames
  around them. `cv_group_by` ties entries into one fold as it does for `xgboost`.
- **Device and repeatability.** `device` is `cuda` or `cpu` and is part of the run id,
  because a GPU and a CPU need not give one model. `apply_device` is `auto` by default and
  is not part of the run id: `auto` predicts on `device` when CUDA is available there
  and on the CPU otherwise. `cpu` and `cuda` force a device. A prediction on a GPU can
  differ from the CPU's in the low bits. A CPU fit repeats at one thread count
  (`n_threads`, 1 by default, which an apply also uses), and a GPU fit repeats on the
  same machine. The torch and CUDA versions are recorded in the model's bundle.
- **Start from another run's network.** `init_from` names a `sequence-classifier` run,
  as `{"step": "pretrain"}`, and every model of the fit starts from that run's weights
  and keeps its standardization: the mean and standard deviation the source learned,
  since the weights fit inputs scaled by them. The head, its options other than
  `dropout`, and `columns` must equal the source's. The labels may contain a subset of
  the source's classes. The classes, output layer and `prob_<class>` columns stay the
  source's, and a class the runs lack gets weight 0 and is not a target. A class the
  source lacks is refused. The fit is refused when the source, or a run it started from,
  trained on an entry this run withholds: a test entry of the templates run, or under
  `cv_folds` also a train entry. A source cannot be combined with `model`.
- **A head of your own.** Register one in Python with `register_head(HeadSpec(...))`,
  giving its name, the model of its options and a function that builds a torch module
  from the options, the number of input columns and the number of classes. Raise
  `head_revision` when its code changes, so that the change is a new run. A head registered
  in Python exists only in that process. The command line and a queue's workers start their
  own, and cannot name it.
- **An installed head.** A package that declares an entry point in the
  `mosaic.sequence_heads` group gives every process that has it a head. The entry point's
  name is the head's name, and its object is the `HeadSpec`:

  ```toml
  [project.entry-points."mosaic.sequence_heads"]
  my-head = "my_package.heads:MY_HEAD"
  ```

  A recipe then sets `"head": "my-head"`, and the command line and a queue's workers load
  the entry point when the head is first named. The spec's name must equal the entry point's,
  and a registered or built-in head of that name takes precedence. A head that fails to load,
  is not a `HeadSpec`, has another name or shares its name with a second entry point is
  refused, naming the entry point and its package. Raise `head_revision` when the head's code
  changes.
