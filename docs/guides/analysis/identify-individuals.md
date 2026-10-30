# Identify individuals by appearance

When a tracker swaps identities across an occlusion, a model of how each animal *looks*
is one way to tell them apart again. Three identity features fit such a model, all from
egocentric crops of recordings where you already know who is in frame. They stop there:
no feature yet applies a fitted model to new crops or rewrites the `id` of a tracks
table, so a swap the tracker made stays in the tracks.

All three need `pip install "mosaic-behavior[deep-learning]"` (PyTorch and timm), which `[all]`
includes.

## 1. Cut the crops

```python
from mosaic.behavior.visualization_library import EgocentricCrop

crops_result = ds.run_feature(EgocentricCrop(params={
    "crop_size": (256, 256),
    "rotate_to_heading": True,
    "output_mode": "frames",
    "frame_format": "png",
}))
crop_root = ds.get_root("features") / crops_result.feature / crops_result.run_id
```

The identity features read the crops as images,
`<crop_root>/<group>__<sequence>/frames_id<id>/frame_*.png`. `output_mode` defaults to
`"video"`, which writes one `egocentric_id<id>.mp4` per individual and no images, so set
it to `"frames"`, or to `"both"` to keep the videos as well. Keep `frame_format` at
`"png"`: the identity features read no other format.

`crop_root` is the crop run's own directory, `features/<storage>/<run_id>/`. The storage
is `crops_result.feature`, such as `egocentric-crop__from__tracks`, not the bare
`egocentric-crop`. If you gave the crop run an `output_root`, that directory is the crop
root instead.

Rotating to heading is what makes the crops comparable: the animal holds a consistent
pose in frame, so the model learns markings rather than orientation. The heading comes
from `angle_col` when you name one, and otherwise from the two keypoints of
`heading_points`. With neither, the crop is centered on the animal but not rotated, and
nothing warns you. See [Render an annotated video](../media/render-a-video.md#animal-centered-crops)
for the crop parameters.

## 2. Pick a model

| Feature | What it does | Reach for it when |
| --- | --- | --- |
| `global-identity-model` | Trains a linear classification head over a pretrained timm backbone, which stays frozen unless you set `freeze_backbone=False` | You know the individuals and have crops of each |
| `global-identity-embedding` | Trains nothing: embeds each crop with a frozen timm backbone and keeps the mean embedding of each identity as its prototype | You want a baseline before training anything |
| `global-identity-dinov2-temporal` | Trains a small temporal head over short clips, on a frozen DINOv2 backbone | A single frame is ambiguous and motion helps |

All three fit a closed set: the identities you name at fit time. None can say that an
animal is one it was not given.

```python
from mosaic.behavior.feature_library import GlobalIdentityModel
from mosaic.core.pipeline.types import Inputs

identity_result = ds.run_feature(GlobalIdentityModel(
    Inputs((crops_result,)),
    params={
        "identities": {
            "mouse_A": ["cage1/day1_mouseA_alone", "cage1/day3_mouseA_alone"],
            "mouse_B": ["cage1/day1_mouseB_alone"],
            "mouse_C": ["cage1/day2_mouseC_alone"],
        },
        "crop_root": str(crop_root),
        "epochs": 20,
    },
))
```

`identities` maps an identity name to the sequences that show that animal alone, each
written `"group/sequence"`. That is the training signal: recordings where you already
know who is in frame. `group_as_identity=True` is the shortcut when each group *is* one
animal.

`crop_root` is required. The features find no crops from their input alone, and a fit
without it stops with `No images collected` (`No clips collected` for the temporal
model), naming the usual causes.

`model_name` selects the timm backbone and defaults to a large Swin transformer. Mosaic
ships no weights — whatever you name is downloaded at run time under its own license.
Set `image_size` as `(height, width)` if you need to override the backbone's default.

## 3. What a run leaves

The fitted model lands in the run's own directory, `features/<storage>/<run_id>/`, beside
the run's tables. For `global-identity-model` that is the checkpoint
`identity_classifier.pth` (named by `weights_name`), the sidecar
`identity_classifier_model.joblib` that a later run names as `model`, the identity order
in `identity_names.joblib`, and the loss and accuracy of each epoch in
`training_history.joblib`. The validation accuracy there comes from frames held out at
random from the training sequences, so neighboring frames sit on both sides of the split
and it reads higher than accuracy on a new recording would.

The run's per-entry tables are a copy of the crop metadata it read: the features pass
their input through and write no per-frame predictions. A run given a fitted `model`
loads it, fits nothing, and passes its input through the same way. Nothing in mosaic
applies these models to crops or tracks yet, and nothing joins an identity back to the `id` of a tracks table.

## Identity inside the tracker instead

TRex does its own visual identification during tracking, which resolves identities as
it links detections rather than afterwards. That is usually the better route when you
are running TRex anyway — see [Run a tracker](../tracking/run-a-tracker.md).
[`calms21-pose-training-and-tracking.ipynb`][calms21-pose] does it that way.

The identity features above fit a model from crops of any tracks, whatever produced
them, and leave those tracks as they are.

[calms21-pose]: https://github.com/EcodylicScience/mosaic/blob/main/notebooks/calms21-pose-training-and-tracking.ipynb
