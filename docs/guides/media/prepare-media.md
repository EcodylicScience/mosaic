# Prepare and index your media

Everything downstream reads video through the media index, so indexing comes first.
[The mosaic dataset](../../dataset.md) covers declaring a media source and scanning it;
this page covers what happens after, and the media work a dataset sometimes needs.

## What the index records

`ds.scan_media()` probes each file with `ffprobe` and writes one row per recording:
where the file is, its container and codec, resolution, duration, frame rate, frame
count, and a `video_uuid`. Features and trackers resolve video through those rows, not
by walking directories.

To inspect one file without a dataset:

```bash
mosaic media probe recording.mp4
mosaic media compare original.mp4 copy.mp4
```

`probe` prints the measured facts as JSON. `compare` reports whether two files are the
same recording, and returns the verdict as its exit code.

!!! note "Raw `.h264` files"

    A bare H.264 stream has no container, so its header metadata is unreliable. mosaic
    measures the true frame count and frame rate by scanning packets, and decodes such
    files sequentially. Nothing extra is required of you — just index them like any
    other recording.

## Transcode

Some recordings are awkward to seek or decode repeatedly. Transcode produces a
derivative beside the original and links the two, so features can read the derivative
while the index still points at what you recorded.

```bash
mosaic run -m dataset.yaml --kind transcode \
    --entries day1:trial01 --params '{"target": "analysis"}'
```

`target` is `analysis` (seekable, for repeated frame access) or `playback` (browser
friendly). Derivatives land in `media/` with their own index. `mosaic prune-media`
deletes any that nothing links to.

For a one-off file outside a dataset:

```bash
mosaic media transcode recording.mp4 --target analysis --output derivatives/
```

## imgstore recordings

Motif and Loopbio stores are directories, not single files. `scan_media()` finds them
natively — one entry per store — and every frame-reading feature opens them directly.

The four subprocess trackers (TRex, SLEAP, Lightning Pose, Ultralytics) and the
`infer-pose` / `infer-points` ops take a video *path*, so export a store first:

```bash
mosaic run -m dataset.yaml --kind export-store --entries day1:trial01
```

`infer-localizer` and every analysis feature read a store without exporting.

## Extract frames to annotate

```bash
mosaic run -m dataset.yaml --kind extract-frames \
    --params '{"n_frames": 20, "method": "kmeans"}'
```

Writes PNGs into `media/frames`. `uniform` spreads the sample evenly across each video;
`kmeans` picks visually diverse frames, so the annotation budget is not spent on near
duplicates. The method names are matched exactly — `"KMeans"` is refused rather than
folded to lower case. Narrow with `--entries`, `--groups` or `--sequences`; naming a
selector inside `--params` is refused.
[Train a pose model](../tracking/train-a-pose-model.md) picks up from here.

### Extract the frames you choose

To annotate particular frames — where a tracker loses an animal, say — list them with
`method: "list"`. Each item names an entry and the frame indices to write for it:

```json
{
  "method": "list",
  "frames": [
    {"sequence": "trial01", "indices": [1745, 3487, 14237]},
    {"group": "day2", "sequence": "trial07", "indices": [0, 600]}
  ]
}
```

```bash
mosaic run -m dataset.yaml --kind extract-frames --params @frames.json
```

```python
from mosaic.tracking import extract_frames

extract_frames(ds, method="list", frames={("", "trial01"): [1745, 3487, 14237]})
```

An index counts from 0 across a recording's clips in order, which is the axis of its
tracks' `frame` column, so frame numbers read off a track table can be listed as they
are. Order and repeats do not matter. The list chooses the entries: without a selector
every listed entry is extracted, and `--entries` narrows the list. `n_frames`,
`start_frame` and `end_frame` are refused under `list`, and an index past the end of a
recording is refused before anything is written.

### A recording whose clips differ in frame rate

A recording split into clips at different rates (30, 29.95 and 31 fps, say) cannot be
read as one timeline from its clips. Frame extraction reads it from its joined video
instead, under every method, and refuses the recording until one exists:

```bash
mosaic run -m dataset.yaml --kind export-joined --entries day1:trial01
```

## Calibrate for physical units

Tracks are always in video pixels. To get centimeters, record a per-video scale on the
media index, then run the `scale-to-cm` feature:

```python
ds.set_media_calibration(0.043)                              # every recording
ds.set_media_calibration(0.051, group="day2", sequence="trial07")   # one of them
```

```python
from mosaic.behavior.feature_library import ScaleToCm

ds.run_feature(ScaleToCm(params={"mode": "convert"}))
```

`mode="convert"` returns a whole track-shaped table in centimeters, which downstream
features can read like any other tracks input. The default `mode="derive"` adds
suffixed copies of the length columns instead. The feature refuses to run on an
uncalibrated dataset rather than assuming a factor.

## Reference

[Ops](../../reference/ops.md) has every parameter for `transcode`, `export-store` and
`extract-frames`; [CLI](../../reference/cli.md) has the `mosaic media` commands.
