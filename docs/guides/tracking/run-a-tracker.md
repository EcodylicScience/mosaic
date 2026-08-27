# Run a tracker

Four trackers are integrated. `mosaic track <kind>` scopes the videos, runs the
tool, and bridges its output into `tracks/` in one command, e.g.

```bash
mosaic track trex -m dataset.yaml --set track_max_individuals=4
```

| Kind | What it is |
| --- | --- |
| `trex` | [TRex](https://trex.run) — conversion then headless tracking, with or without posture |
| `sleap` | [SLEAP](https://sleap.ai) — pose inference plus identity tracking |
| `litpose` | [Lightning Pose](https://lightning-pose.readthedocs.io) — single animal, no cross-frame identity; video inference needs a Linux CUDA GPU |
| `ultralytics` | Ultralytics MOT, six tracker backends |

A Lightning Pose run therefore yields one `id=0` track per video, and its
DeepLabCut-style CSV is read by the built-in `deeplabcut` converter.

None of the four is installed by mosaic.
[Installation](../../installation.md#tools-that-run-in-their-own-environment) has the
build for each and how mosaic locates it — in short, `MOSAIC_TREX_CONDA_ENV`,
`MOSAIC_SLEAP_CONDA_ENV`, `MOSAIC_LITPOSE_CONDA_ENV` and `MOSAIC_ULTRALYTICS_BIN`.

Raw tool output lands under `_tracking/<tool>/<run_id>/`, which is working space:
`mosaic sweep-tracking` reclaims it once a run is finished and past its retention
window. The standardized tables land under `tracks/` and stay.

## Tracking parameters

Each tracker declares its own parameters, and `--set` takes any of them:

```bash
mosaic track sleap -m dataset.yaml --set peak_threshold=0.3 --set max_instances=4
```

`mosaic tracking describe <kind>` lists one tracker's keys with their types and
defaults, and the ops reference has all four —
[trex](../../reference/ops.md#trex), [sleap](../../reference/ops.md#sleap),
[litpose](../../reference/ops.md#litpose),
[ultralytics](../../reference/ops.md#ultralytics). A `--set` value is read as JSON when
it parses as JSON, so a list or a dictionary passes on one flag.

Parameters are part of the `run_id`, so changing one produces a new run rather than
overwriting the old one. The exceptions are the knobs describing *how* a run happened
rather than what it produced — `batch_size`, `device`, `precision`, `idle_timeout` —
so retuning throughput costs you no recompute.

### Settings mosaic does not declare

Every tracker has a passthrough for the settings its tool takes and mosaic has no field
for. The four differ, because the tools do:

| Tracker | Parameter | Reaches the tool as |
| --- | --- | --- |
| `trex` | `convert_extra_settings`, `track_extra_settings` | entries in TRex's settings, one dictionary per phase |
| `sleap` | `sleap_extra_settings` | `--key value` arguments appended to `sleap-track` |
| `litpose` | `litpose_overrides` | Hydra `key=value` overrides on the model's own config |
| `ultralytics` | `tracker_overrides` | fields of the tracker backend's config table |

**Lightning Pose is the one that is not a command-line passthrough.** mosaic drives its
Python API rather than a console verb, so an override edits the model's Hydra
configuration and its keys are config paths rather than flags.

Whatever a passthrough carries is part of the `run_id` like any other parameter.

## From Python

Each tracker has a function that does over a dataset what `mosaic track` does from a
shell:

```python
from mosaic.tracking import run_trex, run_sleap, run_litpose, run_ultralytics
from mosaic.tracking.litpose import LitposeParams
from mosaic.tracking.sleap import SleapParams
from mosaic.tracking.trex import TrexParams

run_trex(ds, TrexParams(track_max_individuals=4))
run_sleap(ds, SleapParams(model_paths=["models/sleap_bottomup"]))
run_litpose(ds, LitposeParams(model_path="models/litpose_model"))
```

All four take their settings as one parameter model, which declares each field's
prose and constraint — and, for TRex, the phase that consumes it. The same model
reaches the run identifier and the tool. A field added to it therefore cannot
change what runs without changing what the run is called.

TRex additionally exposes its two phases separately, which is useful when you want to
convert once and re-track several times:

```python
from pathlib import Path

from mosaic.tracking.trex import TREX_ENV, TrexParams, run_trex_convert, run_trex_track

params = TrexParams(track_max_individuals=4)
where = TREX_ENV.placed(conda_env="trex")

conv = run_trex_convert(
    "video.mp4", "out/",
    params=params, detect_model_path=Path("yolo.pt"), env=where,
)
trk = run_trex_track(conv.pv_path, "out/", params=params, env=where)
```

Both phases read one `TrexParams`, and each sends only the fields its own phase
declares — which is why re-tracking with a track-only setting changed reuses the
conversion.

`TREX_ENV.placed(conda_env=..., bin_path=..., display=...)` overrides the
`MOSAIC_TREX_*` variables for one call, and naming one aspect states nothing about
the others. Placement describes a machine rather than a result, so it reaches no
`run_id`. SLEAP, Lightning Pose and Ultralytics take the same thing as
`<tool>_conda_env=` / `<tool>_bin=` keyword arguments.

## What comes out

One parquet table per sequence, validated against a registered schema, with every
spatial column in video pixels and `X`/`Y` at the body center.

That normalization is the point of the standard, and the rules behind it are in
[What a tracker reports, and in what units](../../concepts/tracks.md).

## Notes on TRex parameters and output

??? note "TRex: an unset parameter means TRex's own default"

    Every tool-facing TRex parameter — `detect_type`, `detect_conf_threshold`,
    `detect_iou_threshold`, `cm_per_pixel`, `meta_encoding`, `track_max_individuals`,
    `track_max_speed`, `track_max_reassign_time`, `track_trusted_probability` —
    defaults to unset and is then absent from the command entirely, so **TRex's own
    default applies**. Set the ones you care about and leave the rest alone.

??? note "TRex: asking for extra columns"

    TRex decides what its per-individual `.npz` holds with `output_fields`, and mosaic
    does not set it, so you get TRex's default export. That default does **not**
    include `tracklet_id` (which tracklet a frame belongs to) or `blobid`. Setting
    `output_fields` *replaces* the list rather than adding to it, so pass TRex's own
    default plus what you want:

    ```python
    TREX_DEFAULT_OUTPUT_FIELDS = [
        ["X", ["RAW", "WCENTROID"]], ["Y", ["RAW", "WCENTROID"]],
        ["X", ["RAW", "HEAD"]], ["Y", ["RAW", "HEAD"]],
        ["VX", ["RAW", "HEAD"]], ["VY", ["RAW", "HEAD"]],
        ["AX", ["RAW", "HEAD"]], ["AY", ["RAW", "HEAD"]],
        ["ANGLE", ["RAW"]], ["ANGULAR_V", ["RAW"]], ["ANGULAR_A", ["RAW"]],
        ["MIDLINE_OFFSET", ["RAW"]], ["normalized_midline", ["RAW"]],
        ["midline_length", ["RAW"]], ["midline_x", ["RAW"]], ["midline_y", ["RAW"]],
        ["midline_segment_length", ["RAW"]],
        ["SPEED", ["RAW", "WCENTROID"]], ["SPEED", ["RAW", "PCENTROID"]],
        ["SPEED", ["RAW", "HEAD"]], ["BORDER_DISTANCE", ["PCENTROID"]],
        ["time", []], ["timestamp", []], ["frame", []], ["missing", []],
        ["num_pixels", []],
        ["ACCELERATION", ["RAW", "PCENTROID"]], ["ACCELERATION", ["RAW", "WCENTROID"]],
        ["visual_identification_p", ["RAW"]],
    ]

    run_trex(
        ds,
        TrexParams(
            track_extra_settings={
                "output_fields": [
                    *TREX_DEFAULT_OUTPUT_FIELDS,
                    ["tracklet_id", []], ["blobid", []],
                ]
            }
        ),
    )
    ```

    Pose keypoints need no entry — TRex appends every keypoint the model reports that
    you did not name yourself, so an override cannot lose them.

## Worked examples

Two notebooks run a tracker end to end on data they download themselves, and both
drive TRex:

- [`calms21-pose-training-and-tracking.ipynb`][calms21-pose] tracks two mice with
  a pose model as the detector and visual identification on, then renders an
  annotated video.
- [`shiners-polo-tracking.ipynb`][shiners-polo] tracks the same footage two ways --
  with a trained point detector, and with no model at all using TRex's own
  background subtraction -- and measures where the two disagree.

Both probe for TRex before spending a conversion pass, which is worth copying: the
not-found error is otherwise raised inside the first entry's convert phase, after a
run root and a failed run-log have already been written.

[calms21-pose]: https://github.com/EcodylicScience/mosaic/blob/main/notebooks/calms21-pose-training-and-tracking.ipynb
[shiners-polo]: https://github.com/EcodylicScience/mosaic/blob/main/notebooks/shiners-polo-tracking.ipynb
