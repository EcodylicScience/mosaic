# Mosaic

**Behavior analysis for any animal, end to end.**

![The mosaic pipeline](assets/pipeline-light.svg#only-light){ width="880" }
![The mosaic pipeline](assets/pipeline-dark.svg#only-dark){ width="880" }

mosaic drives the pose trackers you already use — TRex, SLEAP, DeepLabCut, Lightning
Pose, Ultralytics — and turns their output into behavioral features, unsupervised
syllables, trained classifiers, and annotated video. One dataset, one CLI, one
standardized table underneath.

Every result is content-addressed: the same inputs and parameters produce the same
`run_id`, so re-running is a no-op, parameter sweeps organize themselves, and
`mosaic inventory` can tell you exactly what a dataset already holds.

<div class="grid cards" markdown>

-   **Get started**

    ---

    Install mosaic, build a dataset from video you already have, and run a feature
    over it.

    [Installation and first run](getting-started.md)

-   **Reference**

    ---

    Every feature, op, CLI command and track format, generated from the registries
    so it cannot fall behind the code.

    [Browse the reference](reference/index.md)

-   **Extend**

    ---

    mosaic is plugins most of the way down. Add a track converter for a format it
    does not read, or wire in a tracker it does not drive.

    [Add a converter](adding-a-converter.md) &middot;
    [Add a tracker](adding-a-tracker.md)

</div>

## What mosaic is for

Group-living animals. Identities, pairs and neighbors are first-class throughout —
not an afterthought bolted onto single-animal tracking. A sequence is a recording
with several individuals in it, and most of the feature library is about what they
do relative to each other.

It is a platform rather than a tracker. Pose estimation is a *step*, and mosaic
drives four different tools for it rather than replacing them. What it owns is
everything around that step: where the data comes from, what the output means, what
was computed from it, and whether any of that is now stale.

## The parts

| Section | What it covers |
| --- | --- |
| [**Track**](track/index.md) | Run an integrated tracker, import tracks you already have, or train a pose model first |
| [**Analyze**](analyze/index.md) | Compute features over standardized tracks, and chain them into pipelines |
| [**Model**](model/index.md) | Embed, cluster, and classify behavior; train visual identity models |
| [**Visualize**](visualize/index.md) | Overlay tracks and predictions onto video; cut egocentric crops |
| [**Operate**](operate/index.md) | Run things, see what ran, and keep a dataset honest as it grows |

## How a dataset is organized

`dataset.yaml` is what makes a directory a mosaic dataset. It names the roots that
hold data, and the **sources** each scan reads:

```
dataset.yaml          what this dataset is, and where its files come from
media_raw/            the originals index -- rows may point outside the dataset
media/                transcodes, extracted frames
tracks_raw/           raw tracker output as uploaded
tracks/<variant>/     standardized <group>__<sequence>.parquet, one dir per recipe
labels/<kind>/        converted manual annotations
features/<name>/<run_id>/   one directory per feature run
models/<name>/<run_id>/     one directory per trained model
```

Roots live **inside** the dataset, so an `index.csv` travels with it when the dataset
is copied or archived. Sources deliberately do not: a source may point at a NAS or
another volume, and its files are recorded by absolute path into an index that stays
inside.

## Reproducibility

Every feature run is tagged `run_id = "<version>-<hash>"`, where the hash covers the
feature's parameters, its inputs, and the frame range. Identical inputs and
parameters give an identical `run_id`, so:

- re-running costs nothing — the run is already there;
- a parameter sweep lands in `features/<name>/<run_id>/` per setting, with no naming
  scheme to invent;
- `mosaic inventory` can report an artifact as `complete-but-drifted` when the code
  that produced it has moved.

Throughput knobs — worker counts, batch sizes — are tagged out of the hash, so
retuning them never invalidates a cache.
