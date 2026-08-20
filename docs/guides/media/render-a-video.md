# Render an annotated video

`overlay` draws tracks, identities and behavior labels back onto the recording, one
video per sequence. It is a feature like any other, so it caches, takes a `run_id`, and
can be the last step of a pipeline.

```bash
mosaic run -m dataset.yaml --feature overlay
```

```python
from mosaic.behavior.visualization_library import Overlay

ds.run_feature(Overlay(params={"downscale": 0.5, "end": 3000}))
```

Output lands in `features/overlay/<run_id>/`, one video per sequence.

## Choosing what it draws

| Parameter | Effect |
| --- | --- |
| `label_kind` | Which converted label kind to draw. `"behavior"` by default; `None` draws none |
| `color_by` | Column to color individuals by — an identity or a cluster assignment |
| `hide_unlabeled` | Draw only the frames that carry a label |
| `start` / `end` | Render a frame range rather than the whole sequence |
| `downscale` | Shrink the output. Use it while you are still choosing the other settings |
| `show_individual_bboxes` | Per-animal boxes on or off |
| `pair_box_feature` | Draw a box around an interacting pair, from a pair feature's output |

To draw a model's predictions rather than manual labels, pass the classifier's result
as the overlay's input:

```python
from mosaic.core.pipeline.types import Inputs

ds.run_feature(Overlay(Inputs((xgb_result,)), params={"color_by": "predicted_label"}))
```

[The features reference](../../reference/features.md#overlay) lists every parameter.

## Animal-centered crops

`egocentric-crop` cuts one clip per individual, centered and rotated so the animal
holds still and the world moves around it. These are what the identity models read, and
they are useful on their own for inspecting a single animal.

```python
from mosaic.behavior.visualization_library import EgocentricCrop

ds.run_feature(EgocentricCrop(params={"crop_size": (256, 256), "rotate_to_heading": True}))
```

`crop_size` is a field of view in pixels, not a resize target: a larger value takes in
more of the surrounding scene. `rotate_to_heading` needs keypoints, and
`heading_points` names the two that define the body axis; without keypoints the crop
falls back to the body center and no rotation.

`interaction-crop-pipeline` does the same for a *pair*, framing both animals in one
clip.

Next: [Identify individuals by appearance](../analysis/identify-individuals.md) trains
a model on these crops.
