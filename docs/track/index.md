# Track

Getting from video to a standardized table of who was where, in every frame.

There are three ways in, and they end in the same place: `tracks/<variant>/` holding
one `<group>__<sequence>.parquet` per entry, validated against a registered schema.

## Run a tracker mosaic drives

Four trackers are integrated. `mosaic track <kind>` scopes the videos, runs the tool,
and bridges its output into `tracks/` in one command.

| Kind | What it is | Environment |
| --- | --- | --- |
| `trex` | [TRex](https://trex.run) — conversion then headless tracking, with or without posture | Its own conda env; needs a display even headless |
| `sleap` | [SLEAP](https://sleap.ai) — pose inference plus identity tracking | Its own conda env; headless |
| `litpose` | [Lightning Pose](https://github.com/paninski-lab/lightning-pose) — single animal, no cross-frame identity | Its own conda env; Linux CUDA for video inference |
| `ultralytics` | Ultralytics MOT, six tracker backends | In process — no second environment |

Each tool that needs its own interpreter is located the same way: a
`MOSAIC_<TOOL>_CONDA_ENV` or `MOSAIC_<TOOL>_BIN` environment variable, or a
`<tool>_conda_env=` / `<tool>_bin=` parameter, falling back to the tool on `PATH`.

```bash
mosaic track trex -m dataset.yaml --set track_max_individuals=4
```

`--set` takes any parameter the op declares; run
`mosaic tracking describe trex` for the list, or read
[the ops reference](../reference/ops.md).

Raw tool output lands under `_tracking/<tool>/<run_id>/` and is deliberately kept out
of `tracks_raw/`, which holds only what a user uploaded. `mosaic sweep-tracking`
reclaims that working space once a run is finished and past its retention window.

## Import tracks you already have

If something else did the tracking, declare the files as a source with the format
that reads them, then convert:

```bash
mosaic sources add -m dataset.yaml --kind tracks \
    --path /data/trex_out --patterns '*.npz' --src-format trex_npz
mosaic scan -m dataset.yaml --kind tracks
mosaic convert-tracks -m dataset.yaml
```

Eight formats are registered, covering CalMS21, DeepLabCut, SLEAP, Ultralytics and
three TRex variants. Each declares the schema it emits;
[the track formats reference](../reference/track-formats.md) lists them with their
parameters. For a format nothing reads yet, see
[Adding a track converter](../adding-a-converter.md).

## Train a pose model first

If there are no tracks and no off-the-shelf model that works on your animal, mosaic
covers the upstream half too:

1. **Sample frames to annotate.** `mosaic run --kind extract-frames` writes PNGs into
   `media/frames`, chosen uniformly or by k-means diversity so the annotation budget
   is not spent on near-duplicates.
2. **Annotate** in CVAT, or bring COCO or Lightning Pose annotations.
3. **Train.** `train-pose` (YOLO pose), `train-points` (POLO point detection), or
   `train-localizer` (a PyTorch heatmap localizer), plus `train-sleap` and
   `train-litpose` for those tools' own trainers.
4. **Infer.** `infer-pose`, `infer-points` and `infer-localizer` run a trained model
   over scoped videos and bridge into `tracks/` — or hand the weights to a tracker
   above, which is what TRex's detection model expects.

A trained model is registered as an artifact directory under
`models/<kind>/<run_id>/`, so inference can name a prior training run instead of a
weights path.

## What comes out

Whatever the route, the result obeys the same two rules:

- **Spatial columns are video pixels**, and `X`/`Y` are the individual's body centre.
  A physical unit is obtained downstream by the `scale-to-cm` feature, never stored
  in the table — because the conversion factor is per video and is sometimes not
  recoverable at all.
- **A tracker reports; a feature derives.** The standard schema *forbids* `SPEED`,
  `ANGLE`, `VX` and their relatives, so a converter cannot compute a quantity and
  present it as a measurement. Heading is the sharpest case: run the `heading`
  feature and choose the method, which then enters the run identifier.

Keypoints are optional. A centroid-only tracker emits no `poseX*`/`poseY*` at all,
and the features that are defined on keypoints refuse such a table by name rather
than fabricating one.
