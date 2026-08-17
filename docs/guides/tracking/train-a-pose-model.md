# Train a pose model

If there are no tracks and no off-the-shelf model that works on your animal, mosaic
covers the upstream half too: sample frames, annotate them, train a detector, and
hand the weights to a tracker.

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

`train-pose` and `train-points` cannot share an environment: both extras install
something called `ultralytics`, upstream for `pose` and the POLO fork for `polo`, so
pip resolves only one. Prefer `pose` unless you need point detection.

## 4. Use it

Either run the model directly over scoped videos, which bridges into `tracks/` like a
tracker does:

```bash
mosaic run -m dataset.yaml --kind infer-pose --params '{"model": "<run_id>"}'
```

`infer-points` and `infer-localizer` are the same shape for the other two model types.

Or hand the weights to a tracker, which is what TRex's detection model expects:

```bash
mosaic track trex -m dataset.yaml --set detect_model=models/train-pose/<run_id>/best.pt
```

The difference is identity. Inference detects animals frame by frame; a tracker links
those detections across frames into individuals. If you need to know which animal is
which, you want the tracker.

## Augmentation is opt-in

Installing the `yolo-augment` extra adds `albumentations`, which Ultralytics picks up
on its own and uses to apply Blur, MedianBlur, ToGray and CLAHE at p=0.01 during YOLO
and POLO training. Nothing records which way a run went, so the choice is deliberate
rather than a default — and `albumentations` requires `opencv-python-headless` while
mosaic requires `opencv-python`, which is why it sits outside `recommended`.
