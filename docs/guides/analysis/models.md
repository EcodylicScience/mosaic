# Unsupervised and supervised models

Two questions, two paths. **What are the behaviors?** is unsupervised: fit a model
over the whole collection and let it find structure. **Is this frame behavior X?** is
supervised: you have labels, and you train against them.

Both are features. A trained model is a feature run with the same kind of `run_id` as
a per-frame speed — fitted state under `models/<name>/<run_id>/`, the per-sequence
application under `features/<name>/<run_id>/`. There is no separate model API and no
separate cache, which is what lets a model sit mid-chain.

## Unsupervised: embed, then cluster

The usual chain is four steps. Each one is a feature reading the previous one's
output:

1. **`extract-templates`** subsamples per-sequence feature data into a representative
   matrix. Fitting a global model on every frame of every sequence is neither
   affordable nor better.
2. **`global-scaler`** fits a standard scaler on those templates and applies it per
   sequence.
3. **`global-tsne`** fits an openTSNE embedding on the scaled templates and maps every
   sequence into it.
4. **`global-kmeans`** or **`global-ward`** clusters the embedding, assigning each
   sequence's frames by nearest centroid or 1-NN.

```python
from mosaic.behavior.feature_library import (
    ExtractTemplates, GlobalScaler, GlobalTSNE, PairEgocentricFeatures, Inputs,
)

ego_result = ds.run_feature(PairEgocentricFeatures())

# 1. Sample templates from the upstream feature
templates = ExtractTemplates(Inputs((ego_result,)), params={"n_templates": 2000})
templates_result = ds.run_feature(templates)

# 2. Fit the scaler on those templates, apply per sequence
scaler = GlobalScaler(
    Inputs((ego_result,)),
    params={"templates": ExtractTemplates.TemplatesArtifact().from_result(templates_result)},
)
scaler_result = ds.run_feature(scaler)

# 3. Re-sample from the scaled output, farthest-first for coverage
scaled_templates = ExtractTemplates(
    Inputs((scaler_result,)),
    params={"n_templates": 2000, "strategy": "farthest_first"},
)
scaled_templates_result = ds.run_feature(scaled_templates)

# 4. Fit t-SNE on the scaled templates, map every sequence
tsne = GlobalTSNE(
    Inputs((scaled_templates_result,)),
    params={
        "perplexity": 50,
        "templates": ExtractTemplates.TemplatesArtifact().from_result(scaled_templates_result),
    },
)
tsne_result = ds.run_feature(tsne)
```

Note the entity level: `ego_result` is one row per *pair* per frame. Everything
downstream of it stays at the pair level. Mixing in a per-individual result would be
refused rather than silently cross-joined.

`compute_cluster_label_agreement` scores the result against manual labels when you
have them. On Linux with CUDA, installing `faiss-gpu` accelerates the t-SNE kNN step.

## Behavioral syllables

Two features fit a sequence model over pose dynamics and emit a syllable per frame.

**`arhmm`** is an autoregressive hidden Markov model implemented in mosaic. It runs
in process and needs nothing extra.

**`kpms`** drives [keypoint-MoSeq](https://keypoint-moseq.readthedocs.io/), which
cannot share an environment with mosaic and so runs in one you build yourself. See
[Installation](../../installation.md) for the setup and the environment variable it
requires before the first run.

## Supervised: classify from labels

Convert manual annotations first — BORIS and CalMS21 formats are registered — then
train:

- **`xgboost`** on labeled templates (`extract-labeled-templates` is the feature that
  produces them), multiclass or one-vs-rest, with thresholds, class weighting and
  resampling.
- **`lightning-action`** for temporal action segmentation, which sees the sequence
  rather than the frame.
- **`feral`** for the FERAL V-JEPA video classifier, which reads egocentric crops
  rather than pose. It wants an environment of its own — see
  [Installation](../../installation.md).

Pair `temporal-stack` with any of them to give a frame its context window.

## Refitting is cheap, and misapplication is impossible

Because the `run_id` covers parameters and inputs, refitting with identical settings
is a no-op and a sweep organizes itself under `models/<name>/`. A *changed* upstream
feature produces a different `run_id` — which is what stops a fitted model from being
applied over data it was not fitted on.
