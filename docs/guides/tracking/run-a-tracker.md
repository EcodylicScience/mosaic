# Run a tracker

Four trackers are integrated. `mosaic track <kind>` scopes the videos, runs the
tool, and bridges its output into `tracks/` in one command.

```bash
mosaic track trex -m dataset.yaml --set track_max_individuals=4
```

| Kind | What it is | Needs |
| --- | --- | --- |
| `trex` | [TRex](https://trex.run) — conversion then headless tracking, with or without posture | A display, even headless |
| `sleap` | [SLEAP](https://sleap.ai) — pose inference plus identity tracking | Nothing beyond the install |
| `litpose` | [Lightning Pose](https://lightning-pose.readthedocs.io) — single animal, no cross-frame identity | Linux CUDA for video inference |
| `ultralytics` | Ultralytics MOT, six tracker backends | Nothing beyond the install |

Raw tool output lands under `_tracking/<tool>/<run_id>/`, which is working space:
`mosaic sweep-tracking` reclaims it once a run is finished and past its retention
window. The standardized tables land under `tracks/` and stay.

## Finding the tool

None of the four is installed by mosaic, and none can share its environment — each
pins a Python version or a framework stack that would fight with mosaic's own. Install
each one yourself, then tell mosaic where it went.

All four are located the same way, first match winning:

1. a `<tool>_conda_env=` or `<tool>_bin=` argument on the call,
2. `MOSAIC_<TOOL>_CONDA_ENV` or `MOSAIC_<TOOL>_BIN` in the environment,
3. the tool on `$PATH`.

**Where a tool lives never enters a `run_id`.** It is a property of the machine, not of
the run, so two machines that install a tracker differently still agree on what a run
is called.

### TRex

TRex's conda package pins `python=3.11` and `numpy=1.26`, so it needs an environment of
its own:

```bash
conda create -n trex -c conda-forge -c trexing trex -y
export MOSAIC_TREX_CONDA_ENV=trex
```

On a headless server TRex still needs a display. Run **one** persistent virtual
framebuffer and point mosaic at it — do not wrap `trex` in `xvfb-run`, which fork-bombs,
because TRex relaunches itself:

```bash
Xvfb :99 &
export MOSAIC_TREX_DISPLAY=:99
```

### SLEAP

SLEAP 1.6 brings PyTorch and Qt, so it installs on its own:

```bash
uv tool install "sleap[nn]"
```

Nothing to export: this puts `sleap-track` and `sleap-convert` on `$PATH`, where mosaic
finds them. If you install SLEAP into a conda environment instead, name it with
`export MOSAIC_SLEAP_CONDA_ENV=sleap`.

Inference is headless, and mosaic reads the analysis HDF5 written by SLEAP.

### Lightning Pose

Lightning Pose brings PyTorch, Lightning and NVIDIA DALI, and its video inference needs
a Linux CUDA GPU:

```bash
conda create -n litpose python=3.10 -y
conda activate litpose
pip install lightning-pose
export MOSAIC_LITPOSE_CONDA_ENV=litpose
```

Single-animal and per-frame, so each video yields one `id=0` track. Its
DeepLabCut-style CSV is read by the built-in `deeplabcut` converter.

### Ultralytics

Ultralytics runs in an environment the mosaic repository defines, so build it from your
checkout:

```bash
cd src/mosaic/tracking/external/ultralytics-env
uv sync --python 3.12
export MOSAIC_ULTRALYTICS_BIN="$PWD/.venv/bin/yolo"
```

The export is needed here, and not for SLEAP, because `uv sync` builds a `.venv` inside
that directory rather than putting anything on `$PATH`.
`MOSAIC_ULTRALYTICS_CONDA_ENV` names a conda environment holding the same packages
instead.

This environment runs `mosaic run --kind infer-pose` as well as the tracker, so
building it once covers both. Point detection needs a second one: POLO ships under the
distribution name `ultralytics` and so cannot share an environment with upstream. Build
it the same way in `polo-env/` beside this one, and name it with `MOSAIC_POLO_BIN` --
both install the same `yolo` script, so a `$PATH` lookup cannot tell them apart.

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
from mosaic.tracking import run_sleap, run_litpose

run_sleap(ds, model_paths=["models/sleap_bottomup"])
run_litpose(ds, model_path="models/litpose_model")
```

TRex additionally exposes its two phases separately, which is useful when you want to
convert once and re-track several times:

```python
from mosaic.tracking.trex import run_trex_convert, run_trex_track

conv = run_trex_convert("video.mp4", "out/", detect_model="yolo.pt",
                        track_max_individuals=4, trex_conda_env="trex", display=":99")
trk = run_trex_track(conv.pv_path, "out/", track_max_individuals=4,
                     trex_conda_env="trex", display=":99")
```

## What comes out

One parquet table per sequence, validated against a registered schema, with every
spatial column in video pixels and `X`/`Y` at the body centre.

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
        track_extra_settings={
            "output_fields": [
                *TREX_DEFAULT_OUTPUT_FIELDS,
                ["tracklet_id", []], ["blobid", []],
            ]
        },
    )
    ```

    Pose keypoints need no entry — TRex appends every keypoint the model reports that
    you did not name yourself, so an override cannot lose them.
