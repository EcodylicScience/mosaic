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

`strategy="multiclass"` fits one model over all behaviors. `"one_vs_rest"` fits one
model per behavior, which is what you want when behaviors can co-occur.

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
