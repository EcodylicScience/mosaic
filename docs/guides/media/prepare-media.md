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

`probe` prints the measured facts as JSON. Given an imgstore directory, it also prints
the store's camera serial, chunk size, chunk count and the codec of its first chunk.
`compare` reports whether two files are the same recording, and returns the verdict as
its exit code.

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

To crop, mask, trim or adjust a recording before a tracker reads it, see
[Pre-process media for a tracker](preprocess.md).

## Recordings in several files

A recording split into clips is one entry, and so is a Motif or Loopbio imgstore, a
directory of chunk files. `scan_media()` finds a store natively, one entry per store,
and every frame-reading feature opens it directly.

What a tool needs before it reads such an entry depends on how it reads:

| Tool | An entry of several clips | An imgstore |
| --- | --- | --- |
| TRex, SLEAP, Lightning Pose | `export-joined` | `export-store` |
| Ultralytics tracking, `infer-pose`, `infer-points` | Nothing | Nothing when its chunks are video, otherwise `export-store` |
| `infer-localizer` | `export-joined` only for clips at different frame rates | Nothing |

TRex, SLEAP and Lightning Pose open one video file, so mosaic writes the clips' join or
the store's video first. Ultralytics and the two inference ops read the entry's files
one after another, so nothing is copied. A store whose chunks hold images, raw arrays,
or Bayer or YUV data is still read through its export. The store's video has the
store's frame rate, measured from its frame times. The export ignores the rate that the
chunk files declare, and Motif declares 25 fps on every chunk.

A store's row in the media index records the codec of its chunks and whether its frame
times keep a constant rate, both measured from the store, and the total size and newest
time of its metadata, chunk videos and chunk indexes, not of an image store's frame
files. A scan or `write_media_index` measures them again. `mosaic reprobe-media` leaves
them on a store row whose identity is unchanged, and rewrites them when it patches the
row for another reason, such as a new or changed store uuid or a stale facts cell.

[`mosaic prerequisites`](../../reference/cli.md#mosaic-prerequisites) takes the
`--kind`, `--params` and scope flags of a `mosaic run` command and prints the runs to
make first:

```bash
mosaic prerequisites -m dataset.yaml --kind trex
```

```text
kind: trex
media: the entry media
runs: 1
  1. export-joined export-joined-67dcc35d06
     mosaic run -m /data/ds/dataset.yaml --kind export-joined --entries day1:trial01 --params '{"reencode":false}'
refusals: 0
command:
  mosaic run -m /data/ds/dataset.yaml --kind trex
```

Run the commands under `runs` in order, then the one under `command`. The verb does not
run them and leaves the dataset unchanged. An entry that cannot be made readable, such
as clips of two frame sizes, is listed under `refusals` with its cause and remedy, and
the op's command then names only the entries that the op can read. `--json` prints the
same as one object. A run that needs a file it does not have stops before it starts and
names the command that makes it.

For `mosaic track <kind>`, pass `--kind <kind>` and give its `--set` settings as one
`--params` object, such as `--params '{"media": "<run_id>"}'`. Of a tracker's settings,
`mosaic prerequisites` reads only `media`.

### Imgstores from several cameras

The synchronized cameras of one Motif recording are one entry. `scan_media()` groups the
stores that share a `synchronizationuuid` under their shared base name (`rec` for
`rec.CAM1` and `rec.CAM2`), and each store becomes a `camera` row of that entry, named by
its `camera_serial`. A store recorded by an independent camera is an entry of its own.
The index records each store's serial in `camera_serial`.

A scan refuses stores it cannot group, and writes nothing:

- The stores of one synchronized recording have different base names.
- A synchronized recording has two or more stores and one of them has no
  `camera_serial`, whether its folders are scanned together or one after another.
- Two stores that are not one synchronized recording derive the same entry, as when
  each camera's folder holds a store named by its start time alone
  (`cam1/20210219_112253` and `cam2/20210219_112253`). This holds whether the folders
  are scanned together or one after another, and for two stores of one camera as well.
  Rename the store directories apart, or assign each store its entry with
  `write_media_index`.

`write_media_index` refuses a scope holding unsynchronized stores of different cameras.
The stores of one camera under one scope join its entry as clips.

### An imgstore divided into sequences

A long recording, such as a day of one Motif camera, can be divided into sequences of
whole chunks, each read as an entry. Declare the media source with
`--divide-frames` ([Divide a recording](../../dataset.md#divide-a-recording)), or give a
`MediaIndexScope` a `division`. Each sequence's row in the media index has the store's
`abs_path` and the trailing cells of its chunks, and the sequence reads only those
chunks. Its frame 0 is the first frame of its first chunk. A tool that opens the files
itself is handed those chunks, or the `export-store` video of the sequence. A
synchronized set of stores is not divided.

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

### Leave out frames you already annotated

`exclude` takes the same items as `frames` and keeps those frames out of a `uniform` or
`kmeans` sample. `exclude_spacing` also removes the candidates at most that many
frames from one:

```json
{"method": "kmeans", "n_frames": 20,
 "exclude": [{"sequence": "trial01", "indices": [1745, 3487]}]}
```

An entry the run does not cover is ignored, and `list` refuses `exclude`.

### Extract frames that `select-frames` chose

`select-frames` proposes frames from a tracks variant: where an id jumps, keypoint
confidence is low or a keypoint is missing, two rows are one animal, or fewer animals
than `n_expected` are found. It reads a variant that a tracker or an infer op
published from the full media, and refuses a converted, resampled or trimmed one,
because only the full media has a frame axis that matches the video.

```bash
mosaic run -m dataset.yaml --kind select-frames \
    --params '{"source_tracks_run_id": "<tracks run id>", "n_frames": 20}'
mosaic run -m dataset.yaml --kind extract-frames \
    --params @selections/select-frames/<run id>/extract-frames.params.json
```

The second command extracts the selection with method `list`. Pass `exclude` to
`select-frames` with the frames already annotated, and the selection stays clear of
them.

### A recording whose clips differ in frame rate

A recording split into clips at different rates (30, 29.95 and 31 fps, say) cannot be
read as one timeline from its clips. Frame extraction reads it from its joined video
instead, under every method, and refuses the recording until one exists. Given the
flags of the extraction, `mosaic prerequisites` prints the join to run first:

```bash
mosaic prerequisites -m dataset.yaml --kind extract-frames \
    --params '{"n_frames": 20, "method": "kmeans"}'
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
