# Train a pose model

If there are no tracks and no off-the-shelf model that works on your data, you can use
mosaic for the model training process: sample frames, annotate them, train a detector,
and hand the weights to a tracker.

This is the most involved path in the section. Read [Run a
tracker](run-a-tracker.md) first — training exists to feed it.

## 1. Sample frames to annotate

```bash
mosaic run -m dataset.yaml --kind extract-frames \
    --params '{"n_frames": 20, "method": "kmeans"}'
```

Writes PNGs into `media/frames`. `uniform` spreads the sample evenly across each
video; `kmeans` picks diverse frames, so the annotation budget is not spent on near
duplicates of the same pose.

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

`train-pose` and `train-points` each drive an environment mosaic does not install,
built from `src/mosaic/tracking/external/` — `ultralytics-env/` for pose and
`polo-env/` for points. Two of them, because POLO ships under the distribution name
`ultralytics` and so cannot occupy one with upstream. Build whichever you need and name
it with `MOSAIC_ULTRALYTICS_BIN` or `MOSAIC_POLO_BIN`; there is no extra to install and
one machine can do both. See
[installation](../../installation.md#tools-that-run-in-their-own-environment).

**The first pose training run fetches its base weights.** The default `model` is the
bare asset name `yolo11n-pose.pt`, which Ultralytics downloads from a GitHub release
when the environment does not already hold it. On an air-gapped machine, or a queued
job that must not write outside the dataset, pass `model` as a path to weights that are
already there. Point training fetches nothing: `polo26n.yaml` is package data inside
the fork.

**Cancelling a training run stops it at the next epoch boundary**, leaving `last.pt`
and `results.csv` complete up to the epoch that finished. Ultralytics cannot be
interrupted inside an epoch, so on a long one a cancel is not immediate.

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

Or hand the weights to a tracker, which is what TRex's detection model expects:

```bash
mosaic track trex -m dataset.yaml --set detect_model=models/train-pose/<run_id>/best.pt
```

The difference is identity. Inference detects animals frame by frame; a tracker links
those detections across frames into individuals. If you need to know which animal is
which, you want the tracker.

## Augmentation is opt-in

Building the training environment with `uv sync --python 3.12 --extra augment` adds
`albumentations`, which Ultralytics picks up on its own and uses to apply Blur,
MedianBlur, ToGray and CLAHE at p=0.01 during YOLO and POLO training. Nothing records
which way a run went, so the choice is deliberate rather than a default, and it belongs
to whoever builds the environment — that is the process that reads it.

This was the `yolo-augment` extra until training moved out of mosaic's process. No
extra in mosaic's own `pyproject.toml` can install a package into an environment mosaic
does not build, so the name is gone rather than aliased: pip's unknown-extra warning is
what tells you the opt-in moved.
