# Identify individuals by appearance

A tracker numbers the animals of one video 0, 1, 2. Nothing makes track 0 of one video
the same animal as track 0 of another. This guide names the individuals. You point
at one reference track per animal, and each other track takes the name of the reference
it looks most like, or stays unassigned when it looks like none. Every feature then
reads the same `identity` for the same animal in every video.

The path has five steps, all on SLEAP tracks:

1. Track with SLEAP.
2. Prepare embedding data from those tracks.
3. Train an embedding model on it.
4. Track by appearance with the embedding model.
5. Assign identities from reference tracks, then pin the relabeled variant.

Steps 2 and 3 take no identity labels. The path needs SLEAP detection models, a GPU for
training, and sleap-nn 0.3.5 or later in the SLEAP environment
([installation](../../installation.md#tools-that-run-in-their-own-environment)).

## 1. Track with SLEAP

```bash
mosaic track sleap -m dataset.yaml --set 'model_paths=["<detection model run id>"]'
```

See [Run a tracker](../tracking/run-a-tracker.md). This run supplies the training
frames, and its tracks stay as they are.

## 2. Prepare embedding data

An embedding model learns from tracks which crops show one animal. It trains on frames
cut from the first run. `prepare-embedding-data` cuts short windows from it and writes a
training file and a validation file.

```bash
mosaic run -m dataset.yaml --kind prepare-embedding-data --params '{
  "sleap_run_id": "sleap.1.6-<digest>",
  "train_entries": [{"group": "cage1", "sequence": "20250922"}],
  "val_entries":   [{"group": "cage1", "sequence": "20250929"}]
}'
```

The two lists share no recording, and the sequences of a divided recording are one
recording. Validation then measures another recording. A window is 300 frames
(`window_frames`), and 16 of its frames (`frames_per_window`) are kept, taken from those
where at least `min_tracks_present` tracks have an instance. Trackers swap identities
now and then. Short windows bound how far apart two crops of one track are taken as one
animal. No track continues from one window into the next.
`split_tracks_at_gap` cuts a track in two inside a window where its instances are
further apart than that many frames.

The run refuses an entry whose predictions file is missing, and a preparation with
fewer than two tracks in either file. The frames are embedded in its output. Removing
the tracker's working files later leaves it usable.

## 3. Train the embedding model

```bash
mosaic run -m dataset.yaml --kind train-sleap --params '{
  "labels": "prepare-embedding-data.0.1-<digest>",
  "head": "embedding",
  "backbone": "convnext",
  "device": "0"
}'
```

`labels` is the preparation's run id. The model maps the crop of each detection to an
appearance vector and trains on one GPU. In the trial below, ConvNeXt at a 256 pixel
crop used about 11 GB of GPU memory and took about a minute per epoch.
[Train a pose model](../tracking/train-a-pose-model.md#train-an-embedding-model)
lists what the run refuses and the settings you can change.

## 4. Track by appearance

Track the videos again with a model list that ends in the embedding model. The models
before it detect, and the embedding model scores each detection against the tracks.

```bash
mosaic track sleap -m dataset.yaml \
    --set 'model_paths=["<detection model run id>", "<embedding model run id>"]' \
    --set features=embeddings --set scoring_method=cosine_sim \
    --set candidates_method=local_queues
```

The run publishes the tracks as before, with an appearance table beside each tracks
table: the vector of every tracked detection. See
[Run a tracker](../tracking/run-a-tracker.md#track-by-appearance) for the required
settings and the tables. Embedding adds time. A 30-minute entry at 30 fps took about 20
minutes more than tracking by pose, and sleap-nn prints nothing during that time.

## 5. Assign identities

A reference says where one individual is seen: an entry and the track the individual has
there. An individual that is alone in its entry needs no `id`, and every row of the
entry is that individual.

```bash
mosaic run -m dataset.yaml --kind assign-identities --params '{
  "source_tracks_run_id": "sleap.1.6-<digest>",
  "references": [
    {"identity": "A", "entry": {"group": "cage1", "sequence": "20250922"}, "id": 0},
    {"identity": "B", "entry": {"group": "cage1", "sequence": "20250922"}, "id": 1},
    {"identity": "C", "entry": {"group": "cage1", "sequence": "20250922"}, "id": 2},
    {"identity": "D", "entry": {"group": "cage1", "sequence": "20250922"}, "id": 3}
  ]
}'
```

`source_tracks_run_id` is the run id of step 4. The run builds one prototype per
identity from its references, whatever entries you scope the run to. It splits each
track into tracklets where consecutive rows are more than `max_gap` frames apart (15
by default) and scores each tracklet against each prototype by the mean cosine of its
detections. A tracklet takes its best identity when all of these are true:

- Its best cosine is at least `min_similarity` (0.5).
- Its best cosine exceeds the second best by at least `min_margin` (0.05).
- No tracklet that shares a frame with it already has that identity in the same entry.

A tracklet that fails a test stays unassigned. In an entry that a reference names, the
reference's track takes the declared identity without these tests, and the other
tracklets are scored as above.

The result is a new tracks variant that keeps every row of the source and relabels its
`id`:

| Column | Content |
| --- | --- |
| `id` | An assigned tracklet takes its identity's position among the sorted names, 0 to k-1. Each unassigned tracklet takes a distinct negative id, counting down from -2 |
| `identity` | The identity name, null when unassigned |
| `identity_score`, `identity_margin` | The tracklet's best cosine and its lead over the second, recorded for an unassigned tracklet too |
| `identity_source_id` | The `id` the row had in the source |

A negative id is numbered within its entry, so -2 in one entry is a different animal from
-2 in another. `identity` is the key across entries. Compare animals across videos by
that column, and leave negative ids out of any comparison between entries.

The run refuses a source entry whose row records no appearance table, which a tracker run
without an embedding model leaves. It also refuses a reference whose entry or track does
not exist, and a reference without an `id` whose entry has two rows in one frame. When
some entries fail, the run names them and raises. Run it again to resume, because a
partly relabeled variant is not a place to stop.

`mosaic run` compares the source tables on every run, so a changed source table, of an
entry or of a reference, recomputes the labels it affects. A recipe step does not: it
treats a variant that covers its scope as done. After a source run is overwritten, run
`assign-identities` directly or give its recipe step `overwrite`.

### Pin the relabeled variant

The source variant and the relabeled variant both contain every entry. A step that
reads tracks without naming a variant therefore stops with an error naming both. Pass
the relabeled variant's run id to every step that follows:

```bash
mosaic run -m dataset.yaml --feature speed-angvel \
    --tracks-run-id assign-identities.0.1-<digest>
```

From Python, pass `tracks_run_id=` to `run_feature`.

## How well it separates

A trial on four similar mice, 30-minute recordings at 30 fps with no identity labels,
separated them weakly. The model trained without labels on windows cut from tracker
tracks:

| Measure | Value |
| --- | --- |
| Validation rank-1 retrieval | 0.24 at the first epoch, 0.13 by epoch 60 |
| Tracklets nearest their own track's prototype (chance 0.25) | 0.37 and 0.33, on two sessions |
| Cosine between the prototypes of two tracks | 0.93 to 1.0 |
| Cross-session detections with margin of at least 0.05 | 9% |
| Cross-session detections with margin of at least 0.10 | 4% |

In the cross-session rows, the prototypes came from one session's tracks and the
assignment was of another session's tracklets. Their best prototype cosine had a median
of 0.975 and a 5th percentile of 0.70.

Read the margin as your evidence of confidence. A tracklet with a margin near zero
looks as much like one animal as like another. On animals this similar, the default
`min_margin` leaves most tracklets unassigned, and lowering it assigns more tracklets
and makes more of them wrong. Check `identity_margin` and the share of unassigned rows
before you rely on the identities, and compare some assigned tracklets with the video.
The trial does not say how animals that differ more, or a model trained with identity
labels, would do.

## Identity inside the tracker instead

TRex does its own visual identification during tracking, which resolves identities as
it links detections rather than afterwards. That is usually the better route when you
are running TRex anyway. See [Run a tracker](../tracking/run-a-tracker.md).
[`calms21-pose-training-and-tracking.ipynb`][calms21-pose] does it that way.

## Crop identity models

Three identity features fit a model of how each animal looks, all from egocentric crops
of recordings where you already know who is in frame. They stop there: no feature yet
applies a fitted model to new crops or rewrites the `id` of a tracks table. A swap the
tracker made stays in the tracks. They fit a model from crops of any tracks, whatever
produced them, and leave those tracks as they are. To name individuals in tracks, use the
path above.

All three need `pip install "mosaic-behavior[deep-learning]"` (PyTorch and timm), which
`[all]` includes.

### 1. Cut the crops

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

### 2. Pick a model

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

### 3. What a run leaves

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
applies these models to crops or tracks yet, and nothing joins an identity back to the
`id` of a tracks table.

[calms21-pose]: https://github.com/EcodylicScience/mosaic/blob/main/notebooks/calms21-pose-training-and-tracking.ipynb
