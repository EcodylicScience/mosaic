# Getting Started

This guide walks through a typical mosaic workflow from raw data to feature extraction.

## Installation

mosaic is not published on PyPI, so it installs from a checkout. The clone is part
of the instructions, not an assumed step -- `pip install -e .` has nothing to install
from an empty directory.

```bash
git clone https://github.com/EcodylicScience/mosaic.git
cd mosaic
conda create -n mosaic python=3.12 -y
conda activate mosaic
conda install -c conda-forge ffmpeg av py-opencv -y
pip install -e ".[all]"
```

Frame decoding runs in-process via `av`, so no `ffmpeg` binary is required to
read video. System `ffprobe` is still used for media indexing and probing, and
system `ffmpeg` >= 5.1 is required for transcoding. Installing `ffmpeg` via conda
covers both.

`av` and `py-opencv` come from conda so the environment holds **one** ffmpeg.
Their PyPI wheels each bundle a complete build of their own, and two in one
process crash it nondeterministically. Nothing is pinned -- the `pip install`
that follows finds both requirements already satisfied and installs neither
wheel. Order matters: conda first, pip second.

`pip install -e .` alone installs the whole analysis pipeline; `[all]` adds the
deep-learning surface (YOLO pose, `mosaic track ultralytics`, the localizer and
the identity models), which means PyTorch and, on Linux, about 4 GB of CUDA
wheels. [Installation](installation.md) has the full extras table, a CPU-only
PyTorch line, the two components that want an environment of their own (`feral`
and `kpms`), and the platform-support matrix for Windows and WSL2.

## Create a dataset

A mosaic dataset is a directory with a `dataset.yaml` manifest naming the roots
that hold media, tracks, labels, features and models.

```bash
mosaic init my_project --name "Cage A 2026"
```

Every root lives **inside** the dataset, so its `index.csv` travels with it when
the dataset is copied, archived or synced. Video that lives elsewhere is not a
root pointing outward -- it is a *scan source*, below, whose files are recorded
by absolute path into an index that stays inside.

## Declare where the data is

```bash
# A directory to walk. It may be anywhere -- a NAS, another volume.
mosaic sources add -m my_project/dataset.yaml --kind media \
    --path /Volumes/behavior-nas/cage_a --extensions .mp4,.h264

# Raw tracking output, with the format that reads it.
mosaic sources add -m my_project/dataset.yaml --kind tracks \
    --path /Volumes/behavior-nas/trex_out --patterns '*.npz' --src-format trex_npz
```

A source can also claim an explicit list of files rather than everything a glob
matches, which is what importing some of a folder's contents needs:

```bash
mosaic sources add -m my_project/dataset.yaml --kind media --id pilot \
    --path /Volumes/behavior-nas/pilot --file trial_03/cam0.mp4 --file trial_07/cam0.mp4
```

## Scan

```bash
mosaic scan -m my_project/dataset.yaml
```

With no arguments this rescans **everything the manifest declares** -- media,
tracks and labels -- writing each root's `index.csv`. Each source is read with
its own recipe, so one dataset can draw from a folder of `.mp4` and a folder of
CalMS21 arrays at once.

A scan replaces what its sources claim and preserves everything else, so
scanning one source never deletes another's rows. `mosaic sources list` shows
what is declared and whether it is currently reachable.

The same passes are available from Python (`ds.scan_media()`,
`ds.scan_tracks()`, `ds.scan_labels()`), as is a one-off scan of
directories you do not want to declare (`ds.index_media(search_dirs=[...])`).

## Describe the dataset

Notes and tags live in the manifest, so they travel with the data:

```bash
mosaic notes set -m my_project/dataset.yaml "Cage A pilot, Feb-Apr 2026."
mosaic tags define -m my_project/dataset.yaml cohort --type categorical \
    --options 2026-spring,2026-fall
mosaic tags set -m my_project/dataset.yaml cohort 2026-spring
```

Tags are typed: `label`, `text`, `int`, `float`, `bool` and `categorical`, with
the constraints each type allows. A value outside them is refused when you set
it, not discovered later.

## Convert tracks

```python
from mosaic.core.dataset import open_dataset

ds = open_dataset("my_project/dataset.yaml")
ds.convert_all_tracks()   # -> tracks/<variant>/<group>__<seq>.parquet
```

### Tracking videos with TRex (optional)

If you don't already have tracks, mosaic can drive [TRex](https://trex.run) to
detect and track animals, producing per-id `.npz` you then convert with
`src_format="trex_npz"`:

```python
from mosaic.tracking.trex import run_trex_convert, run_trex_track

conv = run_trex_convert("video.mp4", "out/", detect_model="yolo.pt",
                        track_max_individuals=4, trex_conda_env="track", display=":99")
trk  = run_trex_track(conv.pv_path, "out/", track_max_individuals=4,
                      trex_conda_env="track", display=":99")
```

Over a whole dataset, `run_trex(ds, ...)` does the same as a tracked job and
bridges the results into `tracks/`, or from the command line:
`mosaic track trex -m dataset.yaml --set detect_model=yolo.pt --set track_max_individuals=4`.
`mosaic track` takes its flags from each tracker's parameter schema, so
`mosaic tracking describe trex` lists what `--set` accepts.

**mosaic sets no TREx parameter you did not ask for.** Every tool-facing
parameter — `detect_type`, `detect_conf_threshold`, `detect_iou_threshold`,
`cm_per_pixel`, `meta_encoding`, `track_max_individuals`, `track_max_speed`,
`track_max_reassign_time`, `track_trusted_probability` — defaults to unset and is
then absent from the argv entirely, so **TREx's own default applies**. Set the
ones you care about and leave the rest alone.

This matters when translating a hand-written `.settings` file into a mosaic run.
A TREx `.settings` records only *non-default* values, so such a file is exactly
the set of knobs its author chose; anything it omits was deliberately left to
TREx. Passing only what the file names now reproduces it. Note in particular that
`detect_iou_threshold` has no numeric default at all: TREx documents unset as
preserving "the upstream model's default postprocessing behaviour" and set as
possibly disabling end-to-end NMS-free inference, so passing a number is a
decision about your detector, not just a threshold.

**Asking TREx for extra columns.** TREx decides what its per-individual `.npz`
holds with its `output_fields` parameter, and mosaic does not set it, so you get
TREx's default export. That default does **not** include `tracklet_id` (the
identifier of a consecutively tracked frame segment) or `blobid`. Setting
`output_fields` *replaces* the list rather than adding to it, so pass TREx's own
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

Dropping entries from that list is a real choice, not tidying: without
`["X", ["RAW", "HEAD"]]` and the `midline_*` family you lose the columns the
`trex_v2` schema exists to preserve. **Pose keypoints need no entry** — TREx's
`add_missing_pose_fields()` appends every keypoint the model reports that you did
not name yourself, so an `output_fields` override cannot lose them.

Whatever TREx exports reaches `tracks/<variant>/*.parquet` unchanged: the
converter flattens every field in the `.npz` rather than a known list, and the
standardized schema accepts additional columns. `output_fields` is part of the
tracking parameters, so changing it correctly invalidates an existing run rather
than silently reusing one exported with different columns. Check TREx's
[parameter reference](https://trex.run/docs/parameters_trex.html) — or, since that
page is incomplete, `default_config.cpp` in the TREx source — if you need to
confirm the current default list.

**Two-env setup (recommended).** TRex's conda package pins `python=3.11` /
`numpy=1.26`, so install it in its **own** env rather than the mosaic env:

```bash
conda create -n track -c conda-forge -c trexing trex      # dedicated TRex env (py3.11)
pip install ultralytics torch                              # into `track`, for YOLO detection
```

Then point the mosaic caller at it with `trex_conda_env="track"` (or set
`MOSAIC_TREX_CONDA_ENV=track`); use `trex_bin=`/`MOSAIC_TREX_BIN` for an explicit
binary, or omit both for a `trex` already on `$PATH`. TRex needs an OpenGL/GLFW
display even headless — run **one** persistent virtual framebuffer and pass its
display (don't wrap `trex` in `xvfb-run`, which fork-bombs since TRex relaunches
itself):

```bash
Xvfb :99 -screen 0 1280x1024x24 &     # one persistent display; pass display=":99"
```

### Tracking videos with SLEAP (optional)

If you already have a trained [SLEAP](https://sleap.ai) model, mosaic can drive
`sleap-track` to run pose inference + identity tracking and bridge the result
into standardized tracks — one `sleap` op, the same shape as TRex:

```python
from mosaic.tracking import run_sleap

# one model directory, or two for a top-down model (centroid, then centered-instance)
run_sleap(ds, model_paths=["models/sleap_bottomup"])          # standalone / notebook
```

or from the command line: `mosaic track sleap -m dataset.yaml --set model_paths='["models/sleap_bottomup"]'`.

**Own-environment setup.** SLEAP 1.6 is heavy (PyTorch + Qt), so install it in
its **own** environment rather than the mosaic env:

```bash
uv tool install "sleap[nn]"                 # puts sleap-track / sleap-convert on $PATH
# or a dedicated conda env:  conda create -n sleap ... ; then MOSAIC_SLEAP_CONDA_ENV=sleap
```

mosaic finds the console scripts on `$PATH` by default; point it elsewhere with
`sleap_conda_env=`/`MOSAIC_SLEAP_CONDA_ENV` or `sleap_bin=`/`MOSAIC_SLEAP_BIN`.
Unlike TRex, SLEAP inference is headless and needs no `Xvfb`. Reading SLEAP's
analysis HDF5 in the mosaic env needs `h5py`, which is a base dependency; no
SLEAP package is imported on the mosaic side.

### Tracking videos with Lightning Pose (optional)

If you already have a trained
[Lightning Pose](https://lightning-pose.readthedocs.io) model, mosaic can run its
pose inference and bridge the result into standardized tracks — one `litpose` op,
the same shape as TRex and SLEAP. Lightning Pose is single-animal and per-frame
(no cross-frame identity), so each video yields one `id=0` track:

```python
from mosaic.tracking import run_litpose

run_litpose(ds, model_path="models/litpose_model")            # standalone / notebook
```

or through the op runner:
`mosaic track litpose -m dataset.yaml --set model_path=models/litpose_model`.

**Own-environment setup.** Lightning Pose is heavy (PyTorch + Lightning + NVIDIA
DALI) and its video inference needs a Linux CUDA GPU, so install it in its **own**
environment rather than the mosaic env:

```bash
pip install lightning-pose      # in a dedicated env; puts `litpose` on $PATH
# or a conda env:  conda create -n litpose ... ; then MOSAIC_LITPOSE_CONDA_ENV=litpose
```

mosaic finds the `litpose` script on `$PATH` by default (and runs inference through
that environment's `python`); point it elsewhere with
`litpose_conda_env=`/`MOSAIC_LITPOSE_CONDA_ENV` or
`litpose_bin=`/`MOSAIC_LITPOSE_BIN`. Lightning Pose inference is headless and needs
no `Xvfb`. Its DeepLabCut-style CSV is read by the built-in `deeplabcut` converter;
no Lightning Pose package is imported on the mosaic side.

## Run features

Features are composable pipeline stages. Each produces per-sequence
parquet files versioned by `run_id`:

```python
from mosaic.behavior.feature_library import (
    SpeedAngvel, PairEgocentric, Inputs,
)

# Basic kinematic features (reads from tracks by default)
speed = SpeedAngvel()
speed_result = ds.run_feature(speed)

# Pair-egocentric features
ego = PairEgocentric()
ego_result = ds.run_feature(ego)
```

### Chain features together

```python
from mosaic.behavior.feature_library import (
    ExtractTemplates, GlobalScaler, GlobalTSNE,
)

# 1. Extract templates from an upstream feature
#
# Inputs at the same entity level, always. `speed_result` is one row per
# individual per frame and `ego_result` is one row per *pair* per frame, so they
# share only `frame`: merging them would pair every individual with every pair and
# the templates would be fitted on rows that never existed. mosaic refuses that
# merge rather than performing it, so pick one level -- here the pair features --
# or run two chains and compare them.
templates = ExtractTemplates(
    Inputs((ego_result,)),
    params={"n_templates": 2000},
)
templates_result = ds.run_feature(templates)

# 2. Fit scaler on those templates, apply per-sequence
scaler = GlobalScaler(
    Inputs((ego_result,)),
    params={"templates": ExtractTemplates.TemplatesArtifact().from_result(templates_result)},
)
scaler_result = ds.run_feature(scaler)

# 3. Re-extract templates from the scaled output (farthest-first for coverage)
scaled_templates = ExtractTemplates(
    Inputs((scaler_result,)),
    params={"n_templates": 2000, "strategy": "farthest_first"},
)
scaled_templates_result = ds.run_feature(scaled_templates)

# 4. Fit t-SNE on the scaled templates, map every sequence
tsne = GlobalTSNE(
    Inputs((scaled_templates_result,)),
    params={
        "perplexity": 50,
        "templates": ExtractTemplates.TemplatesArtifact().from_result(scaled_templates_result),
    },
)
tsne_result = ds.run_feature(tsne)
```

The full pattern (with k-means/Ward clustering, ground-truth alignment, and
XGBoost training on top) is shown in the
[CalMS21 template notebook](https://github.com/EcodylicScience/mosaic/blob/main/notebooks/calms21-template.ipynb).

If your tracking output is in a format mosaic does not ship a converter for, see
[Adding a converter](adding-a-converter.md). The
[collective-motion notebook](https://github.com/EcodylicScience/mosaic/blob/main/notebooks/collective-motion-shiners.ipynb)
works one end to end on a published fish-schooling dataset, then runs the
polarization, rotation, local-order and nearest-neighbour measurements on the
result. The
[zebrafish notebook](https://github.com/EcodylicScience/mosaic/blob/main/notebooks/collective-motion-zebrafish.ipynb)
does the same for a tracker that reports no pose keypoints at all, and goes on to
build social force maps -- how a fish turns and changes speed depending on where
its nearest neighbour is.

### Declarative pipelines

For multi-step workflows, use the `Pipeline` class instead of manual
chaining. It handles caching, staleness detection, and dependency tracking
automatically:

```python
from mosaic.core.pipeline import Pipeline, FeatureStep
from mosaic.behavior.feature_library import TrajectorySmooth, SpeedAngvel

pipe = Pipeline(default_run_kwargs={"parallel_workers": 8})
pipe.add(FeatureStep("smooth", TrajectorySmooth, {"window": 5}))
pipe.add(FeatureStep("speed", SpeedAngvel, {}, ["smooth"]))

pipe.status(ds)         # check what's cached
results = pipe.run(ds)  # execute — cached steps are skipped
```

See the [Pipeline Guide](guide-pipeline.md) for the full API and examples.

## Next steps

- See the [Pipeline Guide](guide-pipeline.md) for declarative multi-step pipelines
