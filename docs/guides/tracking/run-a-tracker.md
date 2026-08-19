# Run a tracker

Four trackers are integrated. `mosaic track <kind>` scopes the videos, runs the
tool, and bridges its output into `tracks/` in one command.

```bash
mosaic track trex -m dataset.yaml --set track_max_individuals=4
```

| Kind | What it is | Environment |
| --- | --- | --- |
| `trex` | [TRex](https://trex.run) — conversion then headless tracking, with or without posture | Its own conda env; needs a display even headless |
| `sleap` | [SLEAP](https://sleap.ai) — pose inference plus identity tracking | Its own conda env; headless |
| `litpose` | [Lightning Pose](https://github.com/paninski-lab/lightning-pose) — single animal, no cross-frame identity | Its own conda env; Linux CUDA for video inference |
| `ultralytics` | Ultralytics MOT, six tracker backends | Its own environment, built from the repository; headless |

`--set` takes any parameter the op declares. `mosaic tracking describe <kind>` lists
them, and so does [the ops reference](../../reference/ops.md).

Raw tool output lands under the temporary directory `_tracking/<tool>/<run_id>/` (note: `mosaic sweep-tracking`
reclaims that working space once a run is finished and past its retention window).  Finished and standardize output lands under `tracks/`.  

## Finding the tool

All four run in an interpreter of their own, and all four are located the same way: a
`MOSAIC_<TOOL>_CONDA_ENV` or `MOSAIC_<TOOL>_BIN` environment variable, or a
`<tool>_conda_env=` / `<tool>_bin=` parameter, falling back to the tool on `PATH`.

### TRex

TRex's conda package pins `python=3.11` and `numpy=1.26`, so it needs an environment
of its own:

```bash
conda create -n track -c conda-forge -c trexing trex   # dedicated TRex env (py3.11)
pip install ultralytics torch                          # into `track`, for TRex's own YOLO detection
```

Point mosaic at it with `trex_conda_env="track"` or `MOSAIC_TREX_CONDA_ENV=track`.

TRex needs an OpenGL/GLFW display even headless. Run **one** persistent virtual
framebuffer and pass its display — do not wrap `trex` in `xvfb-run`, which fork-bombs
because TRex relaunches itself:

```bash
Xvfb :99 -screen 0 1280x1024x24 &     # then pass display=":99"
```

### SLEAP

SLEAP 1.6 is heavy (PyTorch and Qt), so install it separately:

```bash
uv tool install "sleap[nn]"    # puts sleap-track / sleap-convert on $PATH
```

Inference is headless and needs no `Xvfb`. Reading SLEAP's analysis HDF5 on the
mosaic side needs `h5py`, which the `recommended` extra bundles; no SLEAP package is
imported into the mosaic process.

### Lightning Pose

Heavy (PyTorch, Lightning, NVIDIA DALI) and its video inference needs a Linux CUDA
GPU:

```bash
pip install lightning-pose     # in a dedicated env; puts `litpose` on $PATH
```

Single-animal and per-frame, so each video yields one `id=0` track. Its
DeepLabCut-style CSV is read by the built-in `deeplabcut` converter.

### Ultralytics

Ultralytics is AGPL-3.0, and a program that imports it is one work with it, so mosaic
drives it as a separate program: nothing on the tracking path imports it. The
repository carries that environment's definition, so building it is one command in its
directory:

```bash
cd src/mosaic/tracking/external/ultralytics-env
uv sync --python 3.12
export MOSAIC_ULTRALYTICS_BIN="$PWD/.venv/bin/yolo"
```

`MOSAIC_ULTRALYTICS_BIN` names the `yolo` console script, and the `python` beside it
in the same `bin/` is what mosaic runs; `MOSAIC_ULTRALYTICS_CONDA_ENV` names a conda
environment holding the same packages instead. Python 3.12 is pinned deliberately;
[Installation](../../installation.md#tools-that-run-in-their-own-environment) says why,
and what building the environment costs you. Nothing here needs the `pose` extra: that
one installs an Ultralytics into mosaic's own environment for pose model training and
inference, and the tracker does not use it.

One process per entry, handed a video path to open. An imgstore recording is a
directory of chunk files rather than a video, so it has to be exported first:

```bash
mosaic run -m dataset.yaml --kind export-store --params '{"entry": ["group", "seq"]}'
```

That step is new. TRex, SLEAP and Lightning Pose already needed it; Ultralytics read a
store directly while it ran inside mosaic, and no longer can. The error message names
the command.

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
                        track_max_individuals=4, trex_conda_env="track", display=":99")
trk = run_trex_track(conv.pv_path, "out/", track_max_individuals=4,
                     trex_conda_env="track", display=":99")
```

## TRex parameters

**mosaic sets no TRex parameter you did not ask for.** Every tool-facing parameter —
`detect_type`, `detect_conf_threshold`, `detect_iou_threshold`, `cm_per_pixel`,
`meta_encoding`, `track_max_individuals`, `track_max_speed`,
`track_max_reassign_time`, `track_trusted_probability` — defaults to unset and is then
absent from the argv entirely, so **TRex's own default applies**. Set the ones you
care about and leave the rest alone.

That matters when translating a hand-written `.settings` file into a mosaic run. A
TRex `.settings` records only *non-default* values, so the file is exactly the set of
knobs its author chose; anything it omits was deliberately left to TRex, and passing
only what the file names reproduces it. `detect_iou_threshold` is the sharpest case:
it has no numeric default at all, and TRex documents unset as preserving the upstream
model's postprocessing and set as possibly disabling NMS-free inference. Passing a
number there is a decision about your detector, not just a threshold.

### Asking TRex for extra columns

TRex decides what its per-individual `.npz` holds with `output_fields`, and mosaic
does not set it, so you get TRex's default export. That default does **not** include
`tracklet_id` (the identifier of a consecutively tracked frame segment) or `blobid`.
Setting `output_fields` *replaces* the list rather than adding to it, so pass TRex's
own default plus what you want:

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

Dropping entries from that list is a real choice, not tidying: without
`["X", ["RAW", "HEAD"]]` and the `midline_*` family you lose the columns the `trex_v2`
schema exists to preserve. **Pose keypoints need no entry** — TRex's
`add_missing_pose_fields()` appends every keypoint the model reports that you did not
name yourself, so an `output_fields` override cannot lose them.

Whatever TRex exports reaches `tracks/<variant>/*.parquet` unchanged: the converter
flattens every field in the `.npz` rather than a known list, and the standardized
schema accepts additional columns. `output_fields` is part of the tracking parameters,
so changing it correctly invalidates an existing run rather than silently reusing one
exported with different columns.

## What comes out

One parquet table per sequence, validated against a registered schema, with every
spatial column in video pixels and `X`/`Y` at the body centre — on every tracker,
including TRex, whose own export is in centimetres with the head in `X`.

That normalization is the point of the standard, and the rules behind it are in
[What a tracker reports, and in what units](../../concepts/tracks.md).
