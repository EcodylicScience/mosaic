# Guides

![The mosaic pipeline](../assets/pipeline-light.svg#only-light){ width="880" }
![The mosaic pipeline](../assets/pipeline-dark.svg#only-dark){ width="880" }

## Media

Video and imgstore recordings are dataset setup rather than analysis.
[The mosaic dataset](../dataset.md) covers declaring a media source, scanning it, and
what the media index records.

## Tracking

| Guide | For |
| --- | --- |
| [Import tracks you already have](tracking/import.md) | Something else did the tracking, and you have the files |
| [Run a tracker](tracking/run-a-tracker.md) | You have video, and one of four integrated tools can track it |
| [Train a pose model](tracking/train-a-pose-model.md) | No off-the-shelf model works on your animal |
| [Write a converter](tracking/write-a-converter.md) | Your format is not one mosaic already reads |

Every tracks table is video pixels with `X`/`Y` at the body centre, whichever tracker
produced it. [What a tracker reports](../concepts/tracks.md) covers the units, and why
a tracker may not give you a speed.

## Analysis

Everything computed from tracks is a **feature**, and there are five kinds.
**Per-frame** features put a value on every row — kinematics, heading, neighbors, pair
geometry. **Global** features are fitted once across many sequences and then applied —
scalers, t-SNE, AR-HMM, keypoint-MoSeq, the classifiers. **Summary** features reduce
many rows to few. **Media** features write an image or video that something else reads,
such as an annotated overlay. **Tag** features add identity columns. All five run and
cache the same way, which is what lets them chain.

| Guide | For |
| --- | --- |
| [Features](analysis/features.md) | What the library holds and how to run one |
| [Write your own feature](analysis/write-your-own-feature.md) | The library does not have the quantity you need |
| [Unsupervised and supervised models](analysis/models.md) | Discovering behaviors, or classifying against labels |

## Pipelines

Everything that mosaic does can be composed into a **pipeline**, which is a step-by-step processing recipe that includes the parameters used.  For example, features can be chained together to define an analysis pipeline.

| Guide | For |
| --- | --- |
| [Chain steps into a recipe](pipelines/chain-steps.md) | Composing steps, in a notebook or as a portable file |
| [Run work and see what ran](pipelines/run-and-see-what-ran.md) | Executing, scoping, and reading the outcome |
| [Keep a dataset organized](pipelines/keep-organized.md) | Inventory, re-indexing, and reclaiming space |


