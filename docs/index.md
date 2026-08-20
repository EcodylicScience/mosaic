# Mosaic

**A platform for behavior analysis.**

![The mosaic pipeline](assets/pipeline-light.svg#only-light){ width="880" }
![The mosaic pipeline](assets/pipeline-dark.svg#only-dark){ width="880" }

Mosaic drives trackers — **TRex, SLEAP, Lightning Pose, Ultralytics** — or imports
tracks something else already produced, then gives a flexible and expandable analysis
framework for working with the data through the **feature** construct: individual
behavioral metrics, collective motion, unsupervised methods, and supervised model
fitting on annotated behaviors. A core focus is enabling **multi-animal analysis**: a
feature can be calculated on individuals, on pairs, or on all animals at once.

Results are organized by a content-addressed scheme for run-caching and
reproducibility: the same inputs and parameters produce the same `run_id`, so
re-running is a no-op, parameter sweeps organize themselves, and `mosaic inventory`
tells you exactly what a dataset already holds.

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

    Features, ops, CLI commands and track formats.

    [Browse the reference](reference/index.md)

-   **Concepts**

    ---

    What a tracker reports, how features compose, and what a `run_id` promises.

    [Read the concepts](concepts/tracks.md)

</div>
