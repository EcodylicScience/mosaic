# Identify individuals by appearance

When a tracker loses identities across an occlusion, a model trained on how each animal
*looks* can recover them. Three identity models ship, all reading egocentric crops.

All three need `pip install -e ".[deep-learning]"` (PyTorch and timm), which `[all]`
includes.

## 1. Cut the crops

```python
from mosaic.behavior.visualization_library import EgocentricCrop

crops_result = ds.run_feature(EgocentricCrop(params={
    "crop_size": (256, 256),
    "rotate_to_heading": True,
}))
```

Rotating to heading is what makes the crops comparable: the animal holds a consistent
pose in frame, so the model learns markings rather than orientation. It needs
keypoints. See [Render an annotated video](../media/render-a-video.md#animal-centered-crops)
for the crop parameters.

## 2. Pick a model

| Feature | What it does | Reach for it when |
| --- | --- | --- |
| `global-identity-model` | Fine-tunes a timm image classifier over a fixed set of identities | You know the individuals and have crops of each |
| `global-identity-embedding` | Learns an embedding rather than a fixed label set | Individuals vary between recordings, or you want a similarity space |
| `global-identity-dinov2-temporal` | A DINOv2 backbone with a temporal head over short clips | A single frame is ambiguous and motion helps |

```python
from mosaic.behavior.feature_library import GlobalIdentityModel
from mosaic.core.pipeline.types import Inputs

ds.run_feature(GlobalIdentityModel(
    Inputs((crops_result,)),
    params={
        "identities": {
            "mouse_A": ["cage1/day1_mouseA_alone", "cage1/day3_mouseA_alone"],
            "mouse_B": ["cage1/day1_mouseB_alone"],
            "mouse_C": ["cage1/day2_mouseC_alone"],
        },
        "epochs": 20,
    },
))
```

`identities` maps an identity name to the sequences that show that animal alone, each
written `"group/sequence"`. That is the training signal: recordings where you already
know who is in frame. `group_as_identity=True` is the shortcut when each group *is* one
animal.

`model_name` selects the timm backbone and defaults to a large Swin transformer. Mosaic
ships no weights — whatever you name is downloaded at run time under its own license.
Set `image_size` as `(height, width)` if you need to override the backbone's default.

Fitted weights land in `models/`, per-frame predictions in `features/`, addressed by
`run_id` like everything else.

## Identity inside the tracker instead

TRex does its own visual identification during tracking, which resolves identities as
it links detections rather than afterwards. That is usually the better route when you
are running TRex anyway — see [Run a tracker](../tracking/run-a-tracker.md).
[`calms21-pose-training-and-tracking.ipynb`][calms21-pose] does it that way.

The identity features above are for tracks you already have, whatever produced them.

[calms21-pose]: https://github.com/EcodylicScience/mosaic/blob/main/notebooks/calms21-pose-training-and-tracking.ipynb
