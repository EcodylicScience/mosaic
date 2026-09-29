# Pre-process media for a tracker

A `preprocess` run writes a media variant, one new video per entry, made from the
original recording by a list of steps. A tracker or an inference op reads the variant
when its `media` parameter names it. The tracks it produces are published in the
original video's pixels and frame numbers. Everything downstream reads them as it
reads tracks of the original.

Reasons to pre-process:

- `crop` to the arena or the tank. The tracker then sees one tank of a multi-tank
  recording and none of the room around it.
- `mask` a polygon to black out a neighboring tank, a reflection or a timestamp.
- `trim` to the part of the recording you analyze, and `decimate` to fewer frames per
  second.
- `grayscale`, `adjust` (brightness, contrast, gamma) or `clahe` (local contrast) to
  match the images a detector was trained on.

## Making a variant

A variant needs a list of steps and the entries to cover.

=== "Python"

    ```python
    from mosaic.core.pipeline.ops import run_op
    from mosaic.core.scope import Scope

    scope = Scope(entries=[("day1", "trial01"), ("day1", "trial02")])
    variant = run_op(
        ds,
        "preprocess",
        {
            "steps": [
                {"step": "crop", "x": 120, "y": 40, "width": 800, "height": 800},
                {"step": "trim", "start": 900, "stop": 54900},
                {"step": "decimate", "every": 2},
            ]
        },
        scope=scope,
    )
    print(variant)  # prints preprocess.0.1-0e4c6091ef for these steps
    ```

=== "CLI"

    ```bash
    mosaic run -m dataset.yaml --kind preprocess \
        --entries day1:trial01 --entries day1:trial02 \
        --params '{"steps": [
            {"step": "crop", "x": 120, "y": 40, "width": 800, "height": 800},
            {"step": "trim", "start": 900, "stop": 54900},
            {"step": "decimate", "every": 2}]}'
    ```

    The variant's name is printed as `run_id`. `--params @recipe.json` reads the same
    object from a file.

The steps apply in order. Every coordinate and frame number is a pixel or a frame of
the original video, whatever the step's position in the list. The `trim` above keeps
original frames 900 to 54899 although it follows the crop. A crop must lie inside the
frame, and its width and height must be even. A recipe that does not fit an entry is
refused before any video is written. [Media steps](../../reference/media-steps.md)
lists every step and its parameters.

`preprocess` refuses to run without a scope, because an unscoped run would re-encode
every entry in the dataset. Name entries, or whole groups with
`Scope(groups=["day1"])` or `--groups day1`.

The run id names the recipe. Running the same recipe again reuses the entries already
written and encodes only the missing ones. Changing a step, `codec`, `quality` or
`fps` gives a new variant beside the old one, under `media/preprocess/<run_id>/`.

A variant reads imgstore recordings and multi-clip entries directly. It needs no
`export-store` or `export-joined` run first, and the tracker reads the one variant
file.

### Chaining a variant from another

`media` on `preprocess` names an upstream variant, and the new steps apply after its
steps. This tries several contrast settings on one crop without cropping again:

```python
contrast = run_op(
    ds,
    "preprocess",
    {
        "media": variant,
        "steps": [{"step": "grayscale"}, {"step": "clahe", "clip_limit": 3.0}],
    },
    scope=scope,
)
```

Coordinates in a chained variant are still pixels of the original video. A second
`crop` names a rectangle in original coordinates, inside the first crop.

## Tracking the variant

Set `media` to the variant's run id on `trex`, `sleap`, `litpose`, `ultralytics`,
`infer-pose`, `infer-points` or `infer-localizer`.

=== "Python"

    ```python
    from mosaic.tracking.sleap import SleapParams

    run_op(
        ds,
        "sleap",
        SleapParams(model_paths=["models/sleap_bottomup"], media=variant),
        scope=scope,
    )
    ```

=== "CLI"

    ```bash
    mosaic track sleap -m dataset.yaml \
        --entries day1:trial01 --entries day1:trial02 \
        --set model_paths='["models/sleap_bottomup"]' \
        --set media=preprocess.0.1-0e4c6091ef
    ```

The variant is part of the tracker's run id. Two variants of one entry therefore
track as two runs. An entry the variant does not cover fails with a message naming the
`preprocess` command that writes it, and the other entries run.

### In a recipe

A tracker or inference step names a variant with `"media": {"step": "<id>"}`:

```json
{
  "schema_version": 1,
  "name": "crop, track, speed",
  "steps": [
    {"id": "crop", "type": "op", "kind": "preprocess",
     "params": {"steps": [
       {"step": "crop", "x": 120, "y": 40, "width": 800, "height": 800},
       {"step": "trim", "start": 900, "stop": 54900}]}},

    {"id": "sleap", "type": "op", "kind": "sleap",
     "params": {"model_paths": ["models/sleap_bottomup"],
                "media": {"step": "crop"}}},

    {"id": "speed", "type": "feature", "feature": "speed-angvel",
     "inputs": ["tracks"], "tracks": {"step": "sleap"}}
  ]
}
```

The reference is replaced by the variant's run id when the recipe is planned. It
also orders the tracker after the `preprocess` step without an `after` list.
A `media` reference must name a `preprocess` step. `media` on a `preprocess` step
takes the same reference, to chain two variants.
[Chain steps into a recipe](../pipelines/chain-steps.md) covers recipes in general.

## Tracks come back in the original video

A tracker run on a variant reports positions in the variant's pixels and frames. mosaic
maps them back when it publishes the table: positions are shifted by the crop's
offset, frame numbers are mapped back through `trim` and `decimate`, and `time` is
recomputed from the original recording. Overlays, egocentric crops, `scale-to-cm` and
a feature's `frame_start` and `frame_end` therefore work on these tracks as on tracks
of the original video, and tracks from two variants of one entry line up.

An `infer-*` table from a variant that trims, decimates, sets `fps` or covers an
entry of several clips records `time` in seconds. An `infer-*` table from the
original video keeps its frame-valued `time`, a known limit until the inference ops
number their frames on the video's frame axis.

A column that describes the variant image itself, such as TREx's distance to the
image border, cannot be mapped back. It is dropped, and the run-log records the
dropped columns for each entry.

## Frame ranges belong in the variant

A frame window set on a tracker or an inference op cannot be combined with `media`.
The variant is already cut to its range, and a second window would count variant
frames. These settings are refused when `media` is set:

| Op | Setting refused with `media` |
| --- | --- |
| `infer-pose`, `infer-points`, `infer-localizer` | `start_frame`, `end_frame`, `frame_step`, `max_frames` |
| `ultralytics` | `start_frame`, `end_frame`, `frame_step` |
| `trex` | `analysis_range` |
| `trex` | `analysis_range`, `analysis_stop_after`, `gui_stop_after` or `video_conversion_range` in `convert_extra_settings` or `track_extra_settings` |
| `sleap` | `analysis_range` |
| `sleap` | `frames` in `sleap_extra_settings` |

The refusal comes before any work starts:

```text
infer-pose: `start_frame` cannot be combined with `media`. `media` names
preprocess.0.1-0e4c6091ef, which is derived media: a video already cut to its own
frame range. Put the range in that variant with a `trim` or `decimate` step, or
leave `media` empty to read the original recording from `media_raw` with this frame
range.
```

Put the range in the variant with `trim` and `decimate`. A feature's `frame_start`
and `frame_end` are not affected. They select frames of the published tracks, which
are numbered in original frames.

## Tracker settings apply to the variant

Every other tracker setting applies to the video the tracker reads:

- A setting counted in frames counts variant frames. After `decimate` with
  `every: 2`, a SLEAP `tracking_window_size` of 5 spans 10 frames of the original.
- A pixel threshold applies to the variant image. A crop does not rescale, and an
  animal keeps its size in pixels.
- A setting in seconds reads the variant's frame rate, described under
  [Frame rate](#frame-rate).

When `cm_per_pixel` is unset, TREx derives it as `meta_real_width / video_width`. A
crop narrows the video. That changes the factor, and with it every TREx threshold
given in centimeters, such as `track_max_speed`. When tracking a cropped variant with
TREx, set `cm_per_pixel` to `meta_real_width` divided by the original video's width
in pixels. TREx takes `meta_real_width` as 30 when none was set. A recording 1920
pixels wide then gives `"cm_per_pixel": 0.015625`.

## Frame rate

`fps` sets the frame rate the variant file is labeled at. Unset, it is the first
clip's rate divided by the decimation. A 30 fps recording decimated by 2 is labeled
15 fps, and a tracker's settings in seconds keep their meaning. Set `fps` only for a
tool that needs a particular rate. Under any other label, a tracker's per-second
columns, such as TREx's speeds, stop measuring real time. They are dropped when the
tracks are mapped back. The `speed-angvel` feature derives speeds from the
published tracks.

An entry whose clips were recorded at different frame rates is supported. Its variant
is labeled at one rate, and the published `time` column is recovered from each clip's
own rate. TREx's per-second columns are dropped for such an entry.

## Codec

Choose the codec when you make the variant. The tracker checks it when it runs, after
the encode.

Variants are AV1 by default. TREx, Ultralytics and the `infer-*` ops read AV1 with no
setting. mosaic refuses an AV1 variant for SLEAP and for Lightning Pose by default,
because their decoders often cannot read it. To track a variant with either tool:

- Make the variant with `"codec": "h264"`. H.264 needs an ffmpeg built with libx264,
  and the run is refused when the ffmpeg on `PATH` lacks it.
- Where the tool's environment does decode AV1, set `MOSAIC_ALLOW_TOOL_CODECS=av1`
  before tracking. SLEAP decodes AV1 with conda-forge's OpenCV installed in its
  environment, as [Installation](../../installation.md#sleap) describes. Lightning
  Pose decodes AV1 on a GPU of compute capability 8.6 or newer.

`quality` sets the encoder's constant rate factor, where lower is better. Unset, it is
14 for AV1 and 16 for H.264.

## When the recording changes

A variant records which media it was made from. If an entry's recordings change after
its variant was written, a tracker refuses that entry and names the `preprocess`
command that rewrites it. Run the same `preprocess` again. It rewrites the entries
whose media changed and reuses the rest.

## Reference

[Media steps](../../reference/media-steps.md) has every step and its parameters.
[Ops](../../reference/ops.md#preprocess) has the parameters of `preprocess` and of
every tracker.
