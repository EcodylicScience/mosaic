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
entry's own score. A row no label covers is left out and counted in `n_unlabeled`.
`perspective` chooses which rows of a pair output are scored: one perspective, both
(`"pooled"`), or the mean of a pair's two probability vectors
(`"mean_probabilities"`). It refuses pair rows of both perspectives unless it is set.

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
