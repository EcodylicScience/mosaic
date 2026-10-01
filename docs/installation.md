# Installation

Mosaic needs Python 3.12 or newer, and installs from a checkout:

```bash
git clone https://github.com/EcodylicScience/mosaic.git
cd mosaic
conda create -n mosaic python=3.12 -y
conda activate mosaic
conda install -c conda-forge ffmpeg av py-opencv -y
pip install -e ".[all]"
```

`av` and `py-opencv` are installed from conda-forge, before pip runs, so that the
environment holds a single ffmpeg build. `ffmpeg` and `ffprobe` are used for media
indexing and for `mosaic media transcode`.

On Linux, apt does the same job: `apt install ffmpeg python3-av python3-opencv` links
both against the distribution's ffmpeg.

Confirm the install:

```bash
mosaic --help
python -c "from mosaic.core.dataset import Dataset; print('OK')"
```

## Optional extras

`pip install -e .` on its own is a complete analysis install: every track and label
converter, all per-frame and social features, wavelets, scaling, t-SNE, k-means, Ward,
ARHMM, the XGBoost classifier, overlays and crops. What `[all]` adds is the
deep-learning surface — the heatmap localizer and the identity models — which means
PyTorch, and on Linux about 4 GB of CUDA wheels.

| Extra              | Adds                                                            |
| ------------------ | --------------------------------------------------------------- |
| `all`              | `deep-learning` + `faiss`: the documented install               |
| `deep-learning`    | `torch` + `timm` — the heatmap localizer and all three identity models |
| `faiss`            | The `"faiss"` kNN backend for `global-tsne`; its default `"annoy"` backend needs nothing |
| `movement`         | The [movement](https://movement.neuroinformatics.dev/) smoothing and filtering features, and the xarray / netCDF4 / pynwb / sleap-io stack they sit on |
| `lightning-action` | Lightning-Action temporal action classifier                     |
| `feral`            | FERAL V-JEPA behavior classifier, training and inference        |

One thing to know before choosing:

- **`faiss` installs `faiss-cpu`.** On Linux with CUDA, install `faiss-gpu` yourself
  instead.

Note: for an install without CUDA, take PyTorch from its own index:

```bash
pip install -e ".[all]" --extra-index-url https://download.pytorch.org/whl/cpu
```

Spectral features, the SLEAP analysis reader and the DeepLabCut HDF5 reader are **not**
extras: PyWavelets, h5py and PyTables are base dependencies, because each one gates
reading a file you already have rather than an integration you opted into.

## Tools that run in their own environment

Six of the tools mosaic drives are **not installed by mosaic**. Install each one
yourself, then tell mosaic where it went, and mosaic launches it there.

| Install it yourself | Used by | Tell mosaic where it is |
| ------------------- | ------- | ----------------------- |
| [**TRex**](https://trex.run) | `mosaic track trex` | `MOSAIC_TREX_CONDA_ENV`, or `MOSAIC_TREX_BIN` for the binary itself |
| [**SLEAP**](https://sleap.ai) | `mosaic track sleap`, the `train-sleap` op | `MOSAIC_SLEAP_CONDA_ENV`, or `MOSAIC_SLEAP_BIN` |
| [**Lightning Pose**](https://lightning-pose.readthedocs.io) | `mosaic track litpose`, the `train-litpose` op | `MOSAIC_LITPOSE_CONDA_ENV`, or `MOSAIC_LITPOSE_BIN` |
| [**Ultralytics**](https://github.com/ultralytics/ultralytics) | `mosaic track ultralytics`, the `infer-pose` and `train-pose` ops | `MOSAIC_ULTRALYTICS_CONDA_ENV`, or `MOSAIC_ULTRALYTICS_BIN` for the environment's `yolo` script |
| [**POLO**](https://github.com/mooch443/POLO) | the `infer-points` and `train-points` ops | `MOSAIC_POLO_CONDA_ENV`, or `MOSAIC_POLO_BIN` for the environment's `yolo` script |
| [**keypoint-MoSeq**](https://keypoint-moseq.readthedocs.io) | the `kpms` feature | `MOSAIC_KPMS_PYTHON`, the environment's interpreter |

A `_CONDA_ENV` variable names a conda environment, which mosaic activates with `conda
run`; a `_BIN` variable names a path directly. With neither set, mosaic looks on
`$PATH`. Where a tool is installed never enters a `run_id`, so two machines that place
it differently still agree on what a run is called.

### Video codecs each tool reads

Every tool reads H.264. Mosaic's analysis transcodes, media variants and imgstore
exports are AV1 by default, and the software in a tool's environment decides whether
the tool reads AV1:

| Tool | Decodes video with | Reads AV1 |
| ---- | ------------------ | --------- |
| TRex | the ffmpeg of its conda environment, which links libdav1d | Yes |
| Ultralytics, `infer-pose`, `infer-points` | PyAV, in the tool's environment | Yes |
| `infer-localizer` | PyAV, in mosaic's process | Yes |
| SLEAP | OpenCV, through sleap-io | With conda-forge's OpenCV, as [below](#sleap) |
| Lightning Pose | NVIDIA DALI's `fn.readers.video` | No |

The GPU does not change the AV1 column, although two error messages name hardware:

- OpenCV prints "Your platform doesn't support hardware accelerated AV1 decoding"
  when its build lacks a software AV1 decoder. The Linux OpenCV wheel from PyPI
  lacks one, and fails on every GPU.
- DALI raises "Unhandled codec 225" for an AV1 file on every GPU, including GPUs
  whose hardware decodes AV1. Its reader accepts H.264, HEVC, MPEG-4, VP8, VP9 and
  MJPEG.

SLEAP, Lightning Pose and Ultralytics were measured on a GeForce GTX 1080 Ti (compute
capability 6.1) and an RTX 4000 Ada (8.9), and TRex on the GTX 1080 Ti. The versions
were SLEAP 1.6, Lightning Pose 2.3.1 and 2.4.2, DALI 1.50.0 and 2.3.0, and TRex 2.0.
The `infer-*` ops read through the same PyAV reader as Ultralytics.

Before SLEAP or Lightning Pose is handed an AV1 file, mosaic reads one frame of it
with the tool's reader in the tool's environment, once per run. The tool is handed
the file when the frame decodes. Otherwise the run is refused with the reader's
error. `MOSAIC_ALLOW_TOOL_CODECS=av1` skips the test.

A GPU older than Turing (compute capability below 7.5), such as a GTX 1080 Ti, limits
every codec. It needs builds that still contain kernels for it:

- DALI's `cuda130` build fails on it with `cudaErrorNoKernelImageForDevice`. The
  `nvidia-dali-cuda110` build that `pip install lightning-pose` installs works.
- PyTorch from PyPI fails with "no kernel image is available". Install PyTorch from
  the cu126 index into the tool's environment, as the
  [Ultralytics environments README](https://github.com/EcodylicScience/mosaic/blob/main/src/mosaic/tracking/external/README.md#older-gpus-need-torch-from-another-index)
  describes.

### TRex

TRex's conda package pins `python=3.11` and `numpy=1.26`, so it needs an environment of
its own:

```bash
conda create -n trex -c conda-forge -c trexing trex -y
export MOSAIC_TREX_CONDA_ENV=trex
```

### SLEAP

SLEAP 1.6 brings PyTorch and Qt, so it installs on its own:

```bash
uv tool install "sleap[nn]"
```

This puts `sleap-convert` on `$PATH`, where mosaic finds it. mosaic runs inference
with `sleap-nn track` from the same environment, so `sleap-nn` itself does not need
to be on `$PATH`.
Installed into a conda environment instead, name it with
`export MOSAIC_SLEAP_CONDA_ENV=sleap`.

**Give it an OpenCV that decodes AV1.** SLEAP reads video through OpenCV, and the
Linux OpenCV wheel from PyPI lacks an AV1 decoder. In a conda environment, replace
it with conda-forge's headless build:

```bash
pip uninstall -y opencv-python opencv-python-headless
conda install -c conda-forge "py-opencv=*=headless*" "libopencv=*=headless*"
```

Remove the pip wheels first. conda does not remove them, and a wheel left
installed provides a second `cv2` beside conda's build. If the solve fails on
packages that the environment's history pins, such as `ffmpeg`, add `--update-all`
to the `conda install`.

That build links the conda ffmpeg beside it, which carries `libdav1d`, and
satisfies SLEAP's unpinned `opencv-python` requirement. Take the headless build.
conda-forge's default build loads conda's Qt, and SLEAP's PySide6 then fails to
import with "No Qt bindings could be found" whenever the two Qt versions differ.
`sleap-io` reads through OpenCV whenever OpenCV is importable, and a working PyAV in
the same environment does not help.

### Lightning Pose

Lightning Pose brings PyTorch, Lightning and NVIDIA DALI, and its video inference needs
a Linux CUDA GPU:

```bash
conda create -n litpose python=3.10 -y
conda activate litpose
pip install lightning-pose
export MOSAIC_LITPOSE_CONDA_ENV=litpose
```

**Hand it H.264 video.** Lightning Pose reads video through DALI's
`fn.readers.video`, which reads AV1 on no GPU. Give it an H.264 [media
variant](guides/media/preprocess.md#codec), made with `"codec": "h264"`, or media
that was not transcoded for analysis. mosaic's test before each run allows AV1
under a DALI release whose reader adds it.

### Ultralytics and POLO

These are two separate environments, and one machine can hold both. Build whichever
you need:

```bash
cd src/mosaic/tracking/external/ultralytics-env
uv sync --python 3.12
export MOSAIC_ULTRALYTICS_BIN="$PWD/.venv/bin/yolo"
```

```bash
cd src/mosaic/tracking/external/polo-env
uv sync --python 3.12
export MOSAIC_POLO_BIN="$PWD/.venv/bin/yolo"
```

Pass `--python 3.12` rather than letting uv choose. The committed lock was resolved
for that interpreter and a newer one may not resolve at all.

Set `MOSAIC_POLO_BIN` and `MOSAIC_ULTRALYTICS_BIN` explicitly rather than relying on
`$PATH`. Both environments install a `yolo` script, and a `$PATH` lookup cannot tell
them apart.

Add `--extra augment` to either `uv sync` to install `albumentations`. Ultralytics then
applies Blur, MedianBlur, ToGray and CLAHE at p=0.01 during training. Nothing records
which way a run went, so opt in deliberately.

Both tools take a video path, so an imgstore recording has to be exported first with
`mosaic run --kind export-store` — as TRex, SLEAP and Lightning Pose also require. The
error message names the command. `infer-localizer` reads a store directly and needs no
export.

### keypoint-MoSeq

Built the same way, from its own directory:

```bash
cd src/mosaic/behavior/feature_library/external
uv sync --python 3.13
export MOSAIC_KPMS_LICENSE_ACCEPTED=1
```

Built here, mosaic finds the interpreter on its own. Built anywhere else, name it:
`export MOSAIC_KPMS_PYTHON=/path/to/env/bin/python`.

Mosaic will not start keypoint-MoSeq until `MOSAIC_KPMS_LICENSE_ACCEPTED` is set to
exactly `1`. Harvard OTD licenses keypoint-MoSeq for non-commercial research and
academic use only, and setting the variable asserts that your use is permitted;
[`external/README.md`](https://github.com/EcodylicScience/mosaic/blob/main/src/mosaic/behavior/feature_library/external/README.md)
has the terms in full.

### FERAL

FERAL runs inside mosaic's own process, but install it in an environment of its own:
FERAL pins exact dependency versions, and installing `[feral]` beside a normal mosaic
environment downgrades several packages in it. Build a second environment holding
mosaic and `[feral]`, and point it at the same datasets.

Third-party licenses, and what mosaic does about each, are recorded in
[NOTICE](https://github.com/EcodylicScience/mosaic/blob/main/NOTICE).

## Platform support

mosaic runs natively on **macOS** and **Linux**. On **Windows** the core analysis
pipeline runs natively, but several capabilities depend on components with no
native-Windows build and need **WSL2**:

| Capability                                                        | Native Windows          | WSL2 / Linux | macOS |
| ----------------------------------------------------------------- | ----------------------- | ------------ | ----- |
| Core analysis (indexing, tracks, features, clustering, ARHMM, XGBoost, visualization) | Yes | Yes | Yes |
| keypoint-MoSeq (`kpms`) -- JAX + Unix sockets                     | No                      | Yes          | Yes   |
| FERAL (`feral`) -- `decord`                                       | No                      | Yes          | Yes   |
| GPU kNN (`faiss`, `faiss-gpu`)                                    | No (`faiss-cpu` works)  | Yes          | n/a   |
| imgstore recordings (reading is native)                           | Partial                 | Yes          | Yes   |
| TRex tracking and pose-model training                             | Partial                 | Yes          | Yes   |

For any **No** or **Partial** capability, install **WSL2** (`wsl --install` in an
admin PowerShell) and follow the Linux setup above inside Ubuntu. Under WSL, keep the
repository and your datasets under home (`~/`) rather than `/mnt/c`.
