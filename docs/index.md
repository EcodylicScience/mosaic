# Mosaic

**A platform for behavior analysis.**

![The mosaic pipeline](assets/pipeline-light.svg#only-light){ width="880" }
![The mosaic pipeline](assets/pipeline-dark.svg#only-dark){ width="880" }

Mosaic drives trackers (TRex, SLEAP, Lightning Pose, Ultralytics) or imports already produced tracks, then gives a flexible and expandable analysis framework for working with the data using the "features" construct: e.g. individual behavioral metrics, collective motion, unsupervised methods, and supervised model fitting (e.g. training classifiers) using annotated behaviors.  A core focus is enabling **multi-animal analysis**: features be calculated on both individual, pairs or generally on all animals.

Results are organized using a content-address schema for run-caching and reproducability: the same inputs and parameters produce the same `run_id`, so parameter sweeps are easily organized and `mosaic inventory` can tell you exactly what a dataset already holds.

<div class="grid cards" markdown>

-   **Get started**

    ---

    [Install mosaic](installation.md) · [The mosaic dataset](dataset.md)

-   **Guides**

    ---

    Tracking, analysis, and composing both into one pipeline.

    [Start with the overview](guides/index.md)

-   **Reference**

    ---

    Feature, Ops, CLI command and track formats

    [Browse the reference](reference/index.md)

</div>

<!-- ## Mosaic concepts 


| | |
| --- | --- |
| [What a tracker reports, and in what units](concepts/tracks.md) | Pixels, body centre, and why a tracker may not report a speed |
| [Features and composition](concepts/features.md) | How features chain, and the entity-level rule that catches most mistakes |
| [Datasets, roots and sources](concepts/datasets.md) | What lives inside a dataset, what a scan claims, and what it leaves alone |
| [Reproducibility, run_id and caching](concepts/reproducibility.md) | Why re-running costs nothing, and what that does not promise |
| [Pipelines as documents](concepts/pipelines.md) | Why a pipeline is a file, and what that buys | -->
