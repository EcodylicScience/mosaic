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
    Inputs((scaler_result,)),
    params={
        "perplexity": 50,
        "templates": ExtractTemplates.TemplatesArtifact().from_result(scaled_templates_result),
    },
))

# 5. Cluster.
kmeans_result = ds.run_feature(GlobalKMeansClustering(
    Inputs((scaler_result,)),
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
sample to every consumer without recomputing it. A template run writes only that
sample. A consumer reads the template step's inputs (here `scaler_result`) and fits on
the sample through `templates`.

`GlobalWardClustering` is the agglomerative alternative to k-means. It builds a
hierarchy, so you can change `n_clusters` and cut it again without refitting.

**What feeds this.** Any per-frame feature stack. The template notebook builds two
branches — `pair-posedistance-pca` → `pair-wavelet` for posture, and `pair-egocentric`
→ `pair-wavelet` for relative motion — and passes both into `ExtractTemplates` and
`GlobalScaler` together. The wavelet expansion gives each frame a multi-scale view of
its own recent history, which is what makes nearby points in the embedding mean
similar movement.

**Where this comes from.** The chain follows the approach of Berman et al. (2014),
MotionMapper ([doi:10.1098/rsif.2014.0672](https://doi.org/10.1098/rsif.2014.0672)),
which maps the stereotyped behavior of one animal from wavelet spectra of its posture.
Mosaic differs in four ways:

- The posture basis is rotation-invariant keypoint distances within and between the two
  animals of a pair, not image eigenmodes of a single animal.
- `pair-wavelet` defaults to a 0.2 to 5 Hz band, where the original used 1 to 50 Hz.
- Nothing normalizes each time point's spectrogram to sum to one. `pair-wavelet` writes
  the clamped log power, and `global-scaler` standardizes each column across the
  templates.
- States come from k-means or Ward clustering of the scaled templates, not from a
  watershed segmentation of a density estimate over the embedding.

The credit is in
[NOTICE](https://github.com/EcodylicScience/mosaic/blob/main/NOTICE).

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

The model reads each sequence on its own, so it never reads one animal's frame after
another's. Fed per-individual features, a sequence is one individual's frames. Fed pair
features, as `scaler_result` is here, it is one ordered pair seen from one perspective.
The states come with `id1`, `id2` and `perspective`, so they join back to the pair
features they were computed from. A sequence that holds the same frame twice fails the
step. A sequence with no more rows than `n_lags` gets no states, and its rows are left out
of the result. A missing value reads as the mean of its column, or as 0 when
`standardize` is off, and the standardizing counts only the values that are present.

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

agreement = compute_cluster_label_agreement(
    ds, kmeans_result.feature, kmeans_result.run_id, perspective=0
)
print(agreement["metrics"]["purity"], agreement["metrics"]["ARI"])
```

Each clustered row is scored against the label of that row. Rows nobody annotated
are unlabeled; they are dropped and counted in `n_unlabeled_dropped`, not scored as a
class.

- **Purity** (`purity`, the same number as `majority_accuracy`) maps each cluster to
  its most frequent behavior and asks how many rows that gets right. Read it when there
  are more clusters than behaviors, which is the usual case.
- **Hungarian accuracy** (`hungarian_accuracy`) pairs behaviors and clusters one to
  one. With more clusters than behaviors, every cluster beyond the pairing counts as
  wrong, so a clustering that splits each behavior cleanly still scores low.
  `hungarian_one_to_one` says whether the pairing covered every cluster.
- **ARI, AMI, homogeneity and completeness** compare the two labelings without
  mapping one onto the other.

On a pair feature, pass `perspective=0` to score each pair once, from the focal
individual of the lower id: a label that describes one animal's behavior, such as
CalMS21's, labels only that animal's rows. `perspective="each"` scores every
perspective separately, and `perspective_consistency` reports how often the two rows
of one pair at one frame fall in the same cluster.
`mosaic.core.evaluation.align_predictions_to_labels` returns the aligned rows
themselves.

### Score it as a recipe step

`evaluate-labels` in `mode: "clusters"` scores the same numbers as a step of a recipe,
so a clustering and its score run in one request, and its report is kept beside the
run as `metrics.json`:

```json
{"id": "score", "type": "feature", "feature": "evaluate-labels",
 "inputs": [{"step": "kmeans"}],
 "params": {"mode": "clusters", "prediction_column": "cluster", "perspective": 0}}
```

Without a `label_map`, each cluster is mapped to its majority behavior among the rows
being scored, so the report's `accuracy` is the purity and no row is held out. The
report's `label_map_source` says which map it used: `scored_rows` for this one,
`fit-label-map` for a map fitted elsewhere. To ask how well the clusters predict
behavior on groups they were not mapped on, fit the map on one set of groups and score
on another, with step scopes narrowing each step:

```json
{"id": "map", "type": "feature", "feature": "fit-label-map",
 "inputs": [{"step": "kmeans"}], "params": {"perspective": 0}},
{"id": "score", "type": "feature", "feature": "evaluate-labels",
 "inputs": [{"step": "kmeans"}],
 "params": {"mode": "clusters", "prediction_column": "cluster",
            "perspective": 0, "label_map": {"step": "map"}}}
```

Narrow `map` to the training groups and `score` to the test groups (`--step-scope
'map={"groups": ["train"]}'`). A cluster the map does not name scores as wrong.

`fit-label-map` gives each cluster the behavior with the most rows in it by default, so
the map follows how common each behavior is. A behavior that is a small share of the
training frames can get no cluster and score an F1 of 0 on the test groups, whatever the
clustering holds.
`"weighting": "class_balanced"` gives a cluster to the behavior whose frames it holds the
largest share of, which treats every behavior alike.

The run also writes `cluster_confusion.parquet`, the number of rows of each behavior in
each cluster, and the report's `clusters` holds `homogeneity`, `completeness` and
`v_measure` beside `ARI`, `AMI` and `purity`. These compare the clusters with the
behaviors without a map, so they do not depend on which groups the map was fitted on.

That is a check on the clustering, not a classifier. To predict labels, see
[Train a behavior classifier](train-a-classifier.md).

[template]: https://github.com/EcodylicScience/mosaic/blob/main/notebooks/calms21-template.ipynb
