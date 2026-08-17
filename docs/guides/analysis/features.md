# Features

Everything mosaic computes is a **feature**: a registered plugin that reads
standardized tracks, or another feature's output, and writes parquet under
`features/<name>/<run_id>/`.

**The library is meant to be extended.** The features that ship with the
distribution are the ones that have been needed so far, not a closed set — they are
plugins registered through a decorator, and yours registers the same way. When the
quantity you want is not here, [write it](write-your-own-feature.md); it will chain,
cache and appear in the reference exactly like the shipped ones.

## Running one

```bash
mosaic run -m dataset.yaml --feature speed-angvel
```

or from Python, which is what a notebook does:

```python
from mosaic.core.dataset import open_dataset
from mosaic.behavior.feature_library import SpeedAngvel

ds = open_dataset("dataset.yaml")
result = ds.run_feature(SpeedAngvel(), entries=[("", "trial01")])
```

`ds.run_feature(...)` is the method form of `mosaic.core.pipeline.run.run_feature`,
which takes the dataset as its first argument. Either works; the method reads better
in a notebook.

Run it twice and the second call does nothing: the `run_id` is a hash of the
parameters, inputs and frame range, so identical work is already on disk. That is
also what makes a parameter sweep organize itself — see [Reproducibility, run_id and
caching](../../concepts/reproducibility.md).

## What the library holds

[The features reference](../../reference/features.md) is the complete list with every
parameter, generated from the registry. Grouped by the question they answer:

**One individual at a time.** `speed-angvel` (speed and angular velocity),
`heading` (body orientation from keypoints, under a method you name),
`trajectory-smooth`, `body-scale`, `scale-to-cm`, `track-subsample`,
`temporal-stack` (gives a frame its context window).

**Individuals relative to each other.** `nearest-neighbor` and its
`nn-delta-response` / `nn-delta-bins` chain for social forces; the `pair-*` family —
`pair-position`, `pair-egocentric`, `pair-facing`, `pair-wavelet`,
`pair-posedistance-pca`; `orientation-rel`, `approach-avoidance`, `attention-target`.

**The group as a whole.** `collective-motion-metrics` (polarization and rotation
order parameters, and the discrete states they imply), `local-order-metrics`,
`ffgroups` and `ffgroups-metrics`, `frame-aggregate`, `social-motion-summary`.

**Models fitted over a collection.** t-SNE, k-means, Ward, AR-HMM, keypoint-MoSeq and
the supervised classifiers — see [Unsupervised and supervised models](models.md).

**Image and video artifacts.** `egocentric-crop` and `interaction-crop-pipeline` cut
animal-centered clips; `overlay` renders an annotated video. These are features so
that a pipeline can end on the deliverable rather than one step short of it.

## Chaining

A feature reads tracks by default, or another feature's output when you name one:

```python
from mosaic.core.pipeline.types import Inputs, Result

ds.run_feature(SpeedAngvel(inputs=Inputs((Result(feature="trajectory-smooth"),))))
```

The upstream identity enters the downstream `run_id`, so a result computed from
smoothed tracks can never be mistaken for the same feature computed from raw ones.

For chains of more than two steps, and for chains that also contain tracking or media
work, use a pipeline: [Chain steps into a recipe](../pipelines/chain-steps.md).

## The rule that catches most mistakes

**Inputs must align at one entity level.** A per-individual result has one row per
individual per frame; a pair result has one row per *pair* per frame. They share only
`frame`, so merging them would pair every individual with every pair and fit
downstream on rows that never existed.

mosaic refuses that merge and names both levels rather than performing it. Pick one
level, or run two chains and compare them. [Features and
composition](../../concepts/features.md) has this and the tracks-variant rule in
full.
