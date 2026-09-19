# Train a pose model

If there are no tracks and no off-the-shelf model that works on your data, you can use
mosaic for the model training process: sample frames, annotate them, train a detector,
and hand the weights to a tracker.

This is the most involved path in the section. Read [Run a
tracker](run-a-tracker.md) first: training produces the model a tracker then uses.

## 1. Sample frames to annotate

```bash
mosaic run -m dataset.yaml --kind extract-frames \
    --params '{"n_frames": 20, "method": "kmeans"}'
```

Writes PNGs into `media/frames`. `uniform` spreads the sample evenly across each
video; `kmeans` picks diverse frames, so the annotation budget is not spent on near
duplicates of the same pose. Those two spellings are matched exactly. Sample from
part of the dataset with `--entries`, `--groups` or `--sequences` rather than a key
inside `--params`, which is refused.

## 2. Annotate

In CVAT, or bring annotations you already have. Three input formats are read: CVAT
XML, COCO, and Lightning Pose. Converters live in
`mosaic/tracking/pose_training/converters/`.

## 3. Train

| Op | Trains |
| --- | --- |
| `train-pose` | A YOLO pose model — keypoints per detected animal |
| `train-points` | A POLO point-detection model — points without boxes |
| `train-localizer` | A PyTorch heatmap localizer |
| `train-sleap` | SLEAP's own trainer |
| `train-litpose` | Lightning Pose's own trainer |

```bash
mosaic run -m dataset.yaml --kind train-pose --params @train.json
```

A trained model is registered as an artifact directory under
`models/<kind>/<run_id>/`, so a later step can name a prior training run instead of
carrying a weights path around.

`train-pose` and `train-points` each drive an environment mosaic does not install:
`ultralytics-env/` for pose, `polo-env/` for points. Build whichever you need and name
it with `MOSAIC_ULTRALYTICS_BIN` or `MOSAIC_POLO_BIN` — see
[installation](../../installation.md#ultralytics-and-polo). One machine can hold both.

**The first pose training run fetches its base weights.** The default `model` is the
bare asset name `yolo11n-pose.pt`, which Ultralytics downloads from a GitHub release
when the environment does not already hold it. On an air-gapped machine, or a queued
job that must not write outside the dataset, pass `model` as a path to weights that are
already there. Point training fetches nothing: `polo26n.yaml` is package data inside
the fork.

**Cancelling a training run stops it at the next epoch boundary**, leaving `last.pt`
and `results.csv` complete up to the epoch that finished. Ultralytics cannot be
interrupted inside an epoch, so on a long one a cancel is not immediate.

**SLEAP's two identity heads need labels that carry identity.** `multi_class_topdown`
and `multi_class_bottomup` classify which animal each detection is, so they train
against the tracks in the `.slp`. Labels with no identity give them no classes, and
training then succeeds and produces a model that learned none. Annotations reaching
mosaic as COCO carry identity in each annotation's `track_id`; CVAT point exports
carry none, so a set built from one trains the other four heads only.

## 4. Use it

Either run the model directly over scoped videos, which bridges into `tracks/` like a
tracker does:

```bash
mosaic run -m dataset.yaml --kind infer-pose --params '{"model": "<run_id>"}'
```

`infer-points` and `infer-localizer` are the same shape for the other two model types.
`infer-pose` and `infer-points` need their environment built and named first, and are
handed a video path -- so an imgstore recording has to be exported with
`mosaic run --kind export-store` beforehand. `infer-localizer` is mosaic's own PyTorch
and needs neither.

Or hand the model to a tracker as its detector:

```bash
mosaic track trex -m dataset.yaml --set detect_model=<run_id>
```

`detect_model` takes the training run's `run_id`, exactly as `infer-pose` does above.
A path to a weights file works too, if the weights came from somewhere else.

The difference is identity. Inference detects animals frame by frame; a tracker links
those detections across frames into individuals. If you need to know which animal is
which, you want the tracker.

## 5. Train one model from several datasets

A detector usually gets better with annotations from more than one experiment, and a
good one is worth using everywhere. Both want the model to live somewhere that is not
any single experiment's dataset. That place is a **library**: an ordinary mosaic
dataset whose job is to hold models.

### Save annotations as revisions

Annotations reach a dataset as **revisions** of a keypoint set, under
`labels_raw/keypoints/<set>/rev1`, `rev2`, and so on. The Mosaic app saves one every
time the annotator is closed. From Python:

```python
from mosaic.core.annotations.projection import write_keypoint_set_revision

saved = write_keypoint_set_revision(
    ds, set_key="openfield", annotations=annotation_set, origin={"note": "after review"}
)
saved.revision   # 3
saved.written    # False if nothing had changed since revision 2
```

A revision is never rewritten, and saving a state that did not change writes nothing.
That is what lets a model say exactly which annotations it was trained on.

### Build the library and claim what to train on

```bash
mosaic init libraries/lab

mosaic sources add -m libraries/lab/dataset.yaml --kind labels --series keypoints \
    --id mice-openfield --path /data/mice/labels_raw/keypoints/openfield \
    --file rev3/annotations.coco.json
mosaic sources add -m libraries/lab/dataset.yaml --kind labels --series keypoints \
    --id rats-arena --path /data/rats/labels_raw/keypoints/arena \
    --file rev1/annotations.coco.json

mosaic scan -m libraries/lab/dataset.yaml --kind labels
```

Name the revisions you want with `--file`. The list is then also the record of what
this library has trained on. To train on a later revision, add it with
`mosaic sources add-files` and scan again.

### Prepare, then train

```bash
mosaic run -m libraries/lab/dataset.yaml --kind prepare-training-data \
    --params '{"sets": [{"set_key": "openfield"}, {"set_key": "arena"}], "target": "yolo-pose"}'
```

This merges the sets into one training dataset and **copies the images in**, so the
library can be moved, and the datasets it drew from can be archived, without breaking
the model. `target` is `yolo-pose`, `polo`, `sleap` or `litpose`. Frames from one
recording are kept together in one split, which is what makes a validation score
honest; `"split_by": "frame"` turns that off, and its scores are optimistic.

Leave `revision` out to take the latest one claimed, or pin it with
`{"set_key": "openfield", "revision": 3}`. Either way the run is named by what the
revision contains, so the same annotations always give the same prepared dataset.

Then hand the trainer the **run id** the preparation returned:

```bash
mosaic run -m libraries/lab/dataset.yaml --kind train-pose \
    --params '{"data": "prepare-training-data.0.1-<digest>", "epochs": 100}'
```

Pass the run id rather than a path to its `data.yaml`. A path is a location, so the
same data at two paths would count as two different models.

### Use the model from any dataset

```bash
mosaic libraries add -m /data/mice/dataset.yaml --id lab --path ../../libraries/lab
mosaic run -m /data/mice/dataset.yaml --kind infer-pose --params '{"model": "<run_id>"}'
```

Once a dataset links a library, a model there is named by its run id exactly as one
trained locally is. Give the link as a relative path when the two datasets move
together.

### Ask a model what it saw

```bash
mosaic models provenance -m /data/mice/dataset.yaml train-pose.0.2-<digest>
```

Prints the prepared dataset behind the model and each annotation revision behind
that, with whatever was recorded when the revision was saved. If a dataset it drew
from has since been archived, the answer says which revisions are no longer on disk.

## Augmentation is opt-in

Build the training environment with `uv sync --python 3.12 --extra augment` to add
`albumentations`. Ultralytics then applies Blur, MedianBlur, ToGray and CLAHE at p=0.01
during YOLO and POLO training. Nothing records which way a run went, so decide before
you build: the choice belongs to the environment, not to a mosaic extra.

## Worked examples

Two notebooks train a model from tracks nobody labelled by hand, on data they
download themselves:

- [`calms21-pose-training-and-tracking.ipynb`][calms21-pose] turns CalMS21's MARS
  keypoints into YOLO pose annotations through mosaic's `AnnotationSet`, trains with
  `train-pose`, and hands the result to TRex as a detector.
- [`shiners-polo-tracking.ipynb`][shiners-polo] does the same for POLO **point**
  labels, where a label row is `<class> <radius> <x> <y>` and the radius is derived
  from the data rather than typed in.

Both are bootstrapping rather than annotation -- the labels are as good as the
tracker that produced them -- and both say so.

[calms21-pose]: https://github.com/EcodylicScience/mosaic/blob/main/notebooks/calms21-pose-training-and-tracking.ipynb
[shiners-polo]: https://github.com/EcodylicScience/mosaic/blob/main/notebooks/shiners-polo-tracking.ipynb
