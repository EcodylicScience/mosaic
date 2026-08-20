# Find behaviors without labels

Unsupervised discovery groups frames by how the animals move, with no annotations
involved. Two routes: embed and cluster, or fit a state model directly.

Both are ordinary features, so both cache and both take a `run_id`.
[`calms21-template.ipynb`][template] runs the whole chain end to end.

## Embed, then cluster

The chain is: build per-frame features, sample templates from them, fit a scaler and an
embedding on the sample, then cluster.

```python
from mosaic.behavior.feature_library import (
    ExtractTemplates, GlobalScaler, GlobalTSNE, GlobalKMeansClustering,
)
from mosaic.core.pipeline.types import Inputs

# 1. Sample a representative subset to fit on.
templates = ExtractTemplates(Inputs((wavelet_result,)), params={"n_templates": 2000})
templates_result = ds.run_feature(templates)

# 2. Fit a scaler on that sample, apply to everything.
scaler = GlobalScaler(
    Inputs((wavelet_result,)),
    params={"templates": ExtractTemplates.TemplatesArtifact().from_result(templates_result)},
)
scaler_result = ds.run_feature(scaler)

# 3. Resample from the scaled features, spreading the picks to cover the space.
scaled_templates_result = ds.run_feature(ExtractTemplates(
    Inputs((scaler_result,)),
    params={"n_templates": 2000, "strategy": "farthest_first"},
))

# 4. Embed.
tsne_result = ds.run_feature(GlobalTSNE(
    Inputs((scaled_templates_result,)),
    params={
        "perplexity": 50,
        "templates": ExtractTemplates.TemplatesArtifact().from_result(scaled_templates_result),
    },
))

# 5. Cluster.
kmeans_result = ds.run_feature(GlobalKMeansClustering(
    Inputs((scaled_templates_result,)),
    params={
        "k": 100,
        "templates": ExtractTemplates.TemplatesArtifact().from_result(scaled_templates_result),
    },
))
```

**Why templates.** Fitting an embedding on every frame of every sequence is neither
affordable nor more informative than fitting on a representative sample.
`strategy="random"` takes a plain subsample; `"farthest_first"` spreads the picks to
cover the feature space, which is what an embedding wants. A template set is an
*artifact* — `ExtractTemplates.TemplatesArtifact().from_result(...)` hands the same
sample to every consumer without recomputing it.

`GlobalWardClustering` is the agglomerative alternative to k-means. It builds a
hierarchy, so you can change `n_clusters` and cut it again without refitting.

**What feeds this.** Any per-frame feature stack. The template notebook builds two
branches — `pair-posedistance-pca` → `pair-wavelet` for posture, and `pair-egocentric`
→ `pair-wavelet` for relative motion — and passes both into `ExtractTemplates` and
`GlobalScaler` together. The wavelet expansion gives each frame a multi-scale view of
its own recent history, which is what makes nearby points in the embedding mean
similar movement.

Those are pair features, so everything downstream stays pair-level and carries
`perspective`. See [What identifies a pair
row](../../concepts/features.md#what-identifies-a-pair-row).

!!! warning "NaN in, failure out"

    t-SNE refuses input containing NaN, and several features legitimately produce
    gaps — `speed-angvel` at the first and last differenced frames,
    `local-order-metrics` where no neighbor falls inside the radius. Feed it wavelet
    features, or drop the incomplete rows first.

## Fit a state model directly

`arhmm` fits an autoregressive hidden Markov model over the features and returns a
discrete state per frame — no embedding, no clustering step.

```python
from mosaic.behavior.feature_library import ArHmmFeature

ds.run_feature(ArHmmFeature(Inputs((scaler_result,)),
                            params={"n_states": 25, "n_lags": 3, "pca_dim": 10}))
```

`kpms` runs keypoint-MoSeq for the same purpose, fitting on keypoints rather than on
derived features. It runs in an environment you build and is licensed for
non-commercial research and academic use only — see
[installation](../../installation.md#keypoint-moseq). `arhmm` is mosaic's own
implementation and carries no such restriction.

## Score the result

When you do have annotations, `compute_cluster_label_agreement` scores clusters against
them:

```python
from mosaic.core.analysis import compute_cluster_label_agreement
```

That is a check on the clustering, not a classifier. To predict labels, see
[Train a behavior classifier](train-a-classifier.md).

[template]: https://github.com/EcodylicScience/mosaic/blob/main/notebooks/calms21-template.ipynb
