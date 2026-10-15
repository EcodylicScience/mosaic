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

## The measurements a fit reads

`extract-templates`, `extract-labeled-templates`, `arhmm` and `temporal-stack` read
every numeric column of their input except the identity columns and bookkeeping
columns such as `fps`, `label` and `nn_id`. The scaler, the embedding, the clustering
and the classifiers fit the columns of the templates. To fit on chosen measurements,
name them in `columns`. Leave out a column that encodes something you do not want to
cluster on, such as `vx` and `vy`, which encode the direction of motion:

```json
{"id": "templates", "type": "feature", "feature": "extract-templates",
 "inputs": [{"step": "speed"}],
 "params": {"n_templates": 2000, "columns": ["speed", "angvel", "nn_dist"]}}
```

From Python, pass the same list: `params={"n_templates": 2000, "columns": [...]}`.

Name the columns once, on the first templates step. `global-scaler` writes the columns
that it fitted beside the identity and bookkeeping columns, and drops the others. The
`farthest_first` templates resampled from its output, and every fit on them, read the
named columns alone.

The order of the names and repeated names leave the run id unchanged. A name that is
an identity or bookkeeping column, or that the input lacks, fails the step with an
error that lists the input's measurements. `mosaic pipeline validate` does not check
the names.

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

## Embed with VAME

`vame` also fits directly on keypoints, training a recurrent variational
autoencoder on windows of egocentrically aligned motion and embedding each
window in a continuous latent space, at the window's center frame. It runs in
an environment you build. See [installation](../../installation.md#vame).

Fit on some sequences, then apply the fitted model to others with `model` set,
the same pattern `kpms` uses:

```python
from mosaic.behavior.feature_library import VameFeature
from mosaic.core.scope import Scope

pose_settings = {
    "pose": {
        "pose_n": 5,
        "keypoint_names": ["snout", "left_ear", "right_ear", "center", "tail_base"],
    },
    "centered_reference_keypoint": "center",
    "orientation_reference_keypoint": "snout",
}

fitted = ds.run_feature(
    VameFeature(VameFeature.Inputs(("tracks",)), params=pose_settings),
    scope=Scope(entries=fit_entries),
)
applied = ds.run_feature(
    VameFeature(
        VameFeature.Inputs(("tracks",)),
        params={
            **pose_settings,
            "model": VameFeature.VameModelArtifact().from_result(fitted),
        },
    ),
    scope=Scope(entries=other_entries),
)
```

Apply refuses a run whose pose, reference keypoint, session (`time_window`,
`max_gap_frames`) or preprocessing settings differ from the model's.

Cluster the latents the same way as the embedding above:

```python
from mosaic.behavior.feature_library import ExtractTemplates, GlobalKMeansClustering

templates = ds.run_feature(ExtractTemplates(
    Inputs((fitted,)), params={"n_templates": 2000},
))
clusters_result = ds.run_feature(GlobalKMeansClustering(
    Inputs((fitted,)),
    params={
        "templates": ExtractTemplates.TemplatesArtifact().from_result(templates),
        "k": 20,
    },
))
other_clusters_result = ds.run_feature(GlobalKMeansClustering(
    Inputs((applied,)),
    params={
        "model": GlobalKMeansClustering.ModelArtifact().from_result(clusters_result),
    },
))
```

The second call applies the first call's model, placing `other_entries` in the
fit's cluster space instead of fitting a new one.

Fitting concatenates the chunks of the training set before it samples training
windows. A chunk is a session, or under the default `split_mode`, `mode_2`, one
of the two parts of a session around its test chunk. Some windows straddle a
join between two chunks. The fit report's `straddling_fraction` is the fraction
of training windows that straddle a join. A session too short to train or embed
on, or lacking a usable keypoint, is excluded instead and named in the same
report. The report is part of the
model bundle (`vame_model.joblib`), and each exclusion is also reported as a
run-log phase.

Render the motifs on the video with `overlay`:

```python
from mosaic.behavior.visualization_library import Overlay

ds.run_feature(Overlay(
    Inputs((clusters_result,)),
    params={"color_by": clusters_result.feature, "hide_unlabeled": True},
))
```

`color_by` names an input result by its feature storage name, which
`clusters_result.feature` contains, and colors each individual by its cluster.
Without `hide_unlabeled`, an individual with no cluster label at a frame draws
in its fallback identity color instead of being skipped.

## Score the result

When you do have annotations, `compute_cluster_label_agreement` scores clusters against
them:

```python
from mosaic.core.analysis import compute_cluster_label_agreement
```

That is a check on the clustering, not a classifier. To predict labels, see
[Train a behavior classifier](train-a-classifier.md).

[template]: https://github.com/EcodylicScience/mosaic/blob/main/notebooks/calms21-template.ipynb
