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

## The XGBoost recipe

This recipe converts CalMS21 tracks and labels, builds pair features, draws a labeled,
class-balanced sample, fits `xgboost` and scores it with `evaluate-labels`. Save it as
`supervised.json`:

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

Scan the dataset's sources before you run it (`mosaic scan`). Planning refuses a
conversion step until its format has a scanned file.

```bash
mosaic pipeline run -m dataset.yaml --recipe @supervised.json
```

Run this way, every step covers every entry, and the score includes the sequences the
model trained on. To score held-out groups, narrow the steps with `--step-scope`
([Train and score in one request](classifier-score-and-evaluate.md#train-and-score-in-one-request)).

`n_per_class` caps the frames each behavior contributes to the sample, and
`default_class` names the majority or "other" class. `strategy` is `"multiclass"` by
default, and `"one_vs_rest"` suits behaviors that co-occur. Class weights, the GPU and
the rest are in [XGBoost parameters](classifier-score-and-evaluate.md#xgboost-parameters).

The `labeled` and `classifier` steps in Python, where `features_result` is the scaler's
result and `ds` the open dataset:

```python
from mosaic.behavior.feature_library import (
    ExtractLabeledTemplates, GroundTruthLabelsSource, XgboostFeature,
)
from mosaic.core.pipeline.types import Inputs

labeled_result = ds.run_feature(ExtractLabeledTemplates(
    Inputs((features_result,)),
    params={"labels": GroundTruthLabelsSource(), "n_per_class": 500},
))
xgb_result = ds.run_feature(XgboostFeature(
    Inputs((features_result,)),
    params={
        "templates": ExtractLabeledTemplates.LabeledTemplatesArtifact().from_result(labeled_result),
        "default_class": 3,
    },
))
```

## Where next

- [Cross-validate a classifier](classifier-cross-validation.md): choose a setting on the
  training groups, with folds of sequences, animals or frame blocks.
- [Thresholds, smoothing and blends](classifier-thresholds-and-blends.md): decision
  thresholds, smoothed probabilities, and averages and blends of classifier runs
  (`probability-ensemble`).
- [Sequence context and networks](classifier-sequence-networks.md): `temporal-stack`,
  wavelets, `lightning-action` and `sequence-classifier`.
- [Score and evaluate a classifier](classifier-score-and-evaluate.md): XGBoost
  parameters, held-out scores, label remaps and masks, predictions made outside mosaic,
  and reading the predictions.
- [Classify mouse behavior with HiDRA](hidra.md): pretrained heads for mouse social
  behavior, without labels of your own.
- [Train a FERAL video classifier](classifier-feral.md): `feral`, from video, in an
  environment of its own.
- [`calms21-template.ipynb`][template] runs features, wavelets, scaler, temporal stack,
  labeled templates and XGBoost on CalMS21, then draws the predictions onto the t-SNE
  embedding.

[template]: https://github.com/EcodylicScience/mosaic/blob/main/notebooks/calms21-template.ipynb
