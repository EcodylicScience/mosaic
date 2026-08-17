# Guides

Three stages, in order: get **media** into a dataset, turn it into **tracks**, and
compute an **analysis** from those tracks.

![The mosaic pipeline](../assets/pipeline-light.svg#only-light){ width="880" }
![The mosaic pipeline](../assets/pipeline-dark.svg#only-dark){ width="880" }

Every guide below sits somewhere on that picture.

## Media

Video, imgstore recordings, and where they live. This is dataset setup rather than
analysis, so it is covered in [The mosaic dataset](../dataset.md): declare a source,
scan it, and the media index is built.

Two operations act on media once it is indexed, and both are ops you run rather than
pages you read: `transcode` produces a derivative a tracker can read, and
`export-store` writes an imgstore recording out as plain video, which is how TRex
reads a store. Both are in [the ops reference](../reference/ops.md).

## Tracking

Getting from media to a standardized table of who was where in every frame. There
are two ways in and they end in the same place — `tracks/<variant>/`, one
`<group>__<sequence>.parquet` per entry, validated against a registered schema.

| Guide | For |
| --- | --- |
| [Import tracks you already have](tracking/import.md) | Something else did the tracking, and you have the files |
| [Run a tracker](tracking/run-a-tracker.md) | You have video, and one of four integrated tools can track it |
| [Train a pose model](tracking/train-a-pose-model.md) | No off-the-shelf model works on your animal |
| [Write a converter](tracking/write-a-converter.md) | Your format is not one mosaic already reads |

## Analysis

Everything computed from those tracks is a **feature** — the same kind of object
whether it is a per-frame speed, a t-SNE embedding or a trained classifier. That
uniformity is what lets them compose.

| Guide | For |
| --- | --- |
| [Features](analysis/features.md) | What the library holds and how to run one |
| [Write your own feature](analysis/write-your-own-feature.md) | The library does not have the quantity you need |
| [Unsupervised and supervised models](analysis/models.md) | Discovering behaviors, or classifying against labels |

## Pipelines

Everything above is a **step**, and steps compose into one graph.

This is why pipelines are their own section rather than a corner of Analysis. A step
may be an op or a feature, so a single recipe can transcode a video, run TRex on the
result, derive speeds, sample templates and fit an embedding — spanning all three
stages of the diagram, not just the last one.

| Guide | For |
| --- | --- |
| [Chain steps into a recipe](pipelines/chain-steps.md) | Composing steps, in a notebook or as a portable file |
| [Run work and see what ran](pipelines/run-and-see-what-ran.md) | Executing, scoping, and reading the outcome |
| [Keep a dataset organized](pipelines/keep-organized.md) | Inventory, re-indexing, and reclaiming space |

## Before you analyse anything

Two Concepts pages exist because not knowing them produces a wrong number rather than
an error: [what a tracker reports, and in what units](../concepts/tracks.md), and
[the entity-level rule](../concepts/features.md) that decides which features can be
combined.
