# Analyze

Everything mosaic computes is a **feature**: a registered plugin that reads
standardized tracks, or another feature's output, and writes parquet under
`features/<name>/<run_id>/`.

There are 45 of them. [The features reference](../reference/features.md) lists every
one with its parameters; this page is about how they fit together.

## Running one

```bash
mosaic run -m dataset.yaml --feature speed-angvel
```

or from Python, which is what a notebook does:

```python
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline import run_feature
from mosaic.behavior.feature_library import SpeedAngvel

ds = Dataset("dataset.yaml").load()
result = run_feature(ds, SpeedAngvel(), entries=[("", "trial01")])
```

Never write into `features/` by hand. `run_feature` owns the output layout, the index
rows and the `run_id` registration; a side-loaded file desynchronizes the index and
breaks every downstream cache decision.

## Two flavors

**Per-frame and per-sequence features** are stateless transforms. `speed-angvel`,
`nearest-neighbor`, `pair-egocentric`, `pair-wavelet`, `temporal-stack`, `heading`,
`scale-to-cm` — each reads one sequence and writes one sequence.

**Global features** fit once over a collection and then apply per sequence:
`global-scaler`, `global-tsne`, `global-kmeans`, and every trained model. Those are
[Model](../model/index.md)'s subject.

Two features that look like transforms are worth calling out because they exist to
make a choice explicit rather than to compute something new. `heading` derives an
angle from keypoints under a *named* method, and `scale-to-cm` converts pixels to
centimetres using the per-video factor on the media index. Both put the choice into a
run identifier, which is the whole reason they are features and not columns a
converter writes.

## Chaining

A feature reads tracks by default, or names an upstream feature's output as its
input. The [Pipeline guide](../guide-pipeline.md) covers the declarative form, where
steps wire together and the whole chain reports what is cached and what must run.

One rule catches most mistakes: **inputs must align at one entity level.** A
per-individual result has one row per individual per frame; a pair result has one row
per *pair* per frame. Merging them shares only `frame`, so it would pair every
individual with every pair and fit on rows that never existed. mosaic refuses that
merge and names both levels rather than performing it.

## Which tracks a feature reads

A dataset can hold several *tracks variants* — the same entries converted by
different recipes, each in its own `tracks/<variant>/` directory. When one entry
carries exactly one variant, it resolves silently. When it carries two genuinely
different recipes, mosaic **raises rather than guessing**, and `--tracks-run-id`
answers the refusal.

The resolved variant enters the feature's `run_id` but never the storage directory
name, so `features/<name>/` stays one directory however many recipes a dataset holds.

## Overlap and neighbors

A windowed feature needs frames from beyond its sequence's boundary. `overlap_frames`
supplies them, but only within a `continuous group` — a group whose sequences are
declared to be time divisions of one recording rather than independent trials.

The declaration is checked, not trusted: mosaic verifies that the recorded frame
extents are disjoint and increasing, and names both sequences and both ranges when
they are not. No measurement can establish that two sequences divide one recording,
and no declaration can be trusted about an axis that exists to be read, so it
requires both.

## Worked examples

See [Examples](examples.md) for end-to-end notebooks.
