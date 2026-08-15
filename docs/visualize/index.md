# Visualize

Two things mosaic renders: video with the tracks drawn on it, and egocentric crops
that other features read.

There are two ways to get the first, and the difference is whether you want an
*artifact* or a *look*.

## As a feature: `overlay`

`overlay` renders one annotated video per sequence into its run root, addressed by
`run_id` like any other result:

```python
from mosaic.behavior.visualization_library.overlay_feature import Overlay

# The tracks alone.
ds.run_feature(Overlay(), sequences=["hex_3"])

# A classifier's predictions drawn over them -- by naming the classifier as an
# input, not by a second configuration language.
ds.run_feature(
    Overlay(Overlay.Inputs(("tracks", Result(feature="behavior-xgb__from__tracks")))),
    sequences=["hex_3"],
)
```

It is categorized `media` rather than visualization, for the same reason the crop
features are: the output is an artifact something else consumes — here, a person.
Being a feature is what lets a pipeline **end on the deliverable** rather than one
step short of it, and what puts the rendered video under the same caching and
provenance as everything upstream of it.

## As a look: `play_video`

`play_video` composes tracks, poses, identities and any per-frame label column onto
the video they came from:

```python
from mosaic.behavior.visualization_library.playback import play_video

play_video(
    ds,
    group="",
    sequence="trial01",
    color_feature="global-kmeans",   # color each individual by its cluster
    color_mode="pred",
)
```

It draws what a tracker actually reported. Keypoints are drawn when the table has
them and the body centre is used when it does not, so a centroid-only tracker still
renders. Pair-level features can drive a box around an interacting dyad rather than
around each individual.

`render_stream` is the same renderer writing to a file instead of a window, and
`prepare_overlay` is the step in between if you want the per-frame draw data without
the drawing.

**This path is plain functions, and nothing it draws enters a run identifier.** That
is the point of keeping it beside `overlay`: checking that a tracker actually worked
is not a result worth caching, and a headless "draw this sequence and let me look" is
something a CLI or a queue context cannot get from a web interface. Reach for
`overlay` when the video is the deliverable, and for `play_video` when it is the
question.

## Egocentric crops

`egocentric-crop` cuts a fixed-size, animal-centered, rotation-normalized clip per
individual, and `interaction-crop-pipeline` does the same for detected interaction
segments. Both *are* features, categorized `media`, because their output is an input
— all three identity models and the FERAL classifier read crops rather than pose.

They are cached and addressed like any other feature run, which matters because
cutting crops is expensive and is usually done once for several downstream models.

## Static plots are deliberately absent

There is no plotting feature. `viz-timeline` and `viz-global-colored` existed once,
wrote matplotlib PNGs from inside the compute backend, and were retired.

Read the values back and plot them however you like:

```python
from mosaic.core.pipeline import load_values

df = load_values(ds, feature="global-kmeans", entries=[("", "trial01")])
```

A figure is a choice about presentation, it changes far more often than the numbers
under it, and a PNG in a run directory is not something another feature can consume.
The notebooks in [Examples](../analyze/examples.md) plot straight from `load_values`,
and every figure in them is a dozen lines of matplotlib or seaborn.
