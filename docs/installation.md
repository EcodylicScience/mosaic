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

This puts `sleap-track` and `sleap-convert` on `$PATH`, where mosaic finds them.
Installed into a conda environment instead, name it with
`export MOSAIC_SLEAP_CONDA_ENV=sleap`.

### Lightning Pose

Lightning Pose brings PyTorch, Lightning and NVIDIA DALI, and its video inference needs
a Linux CUDA GPU:

```bash
conda create -n litpose python=3.10 -y
conda activate litpose
pip install lightning-pose
export MOSAIC_LITPOSE_CONDA_ENV=litpose
```

### Ultralytics and POLO

The repository carries both environments' definitions — a `pyproject.toml` and a lock
file each — so building one is a single command in its directory. Build whichever you
need:

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

Pass `--python 3.12` rather than letting uv choose: the committed lock was resolved for
that interpreter, and a newer one resolves a different set of wheels or none at all.

**Two environments, because POLO ships under the distribution name `ultralytics`** and
so cannot occupy one with upstream. Both install a `yolo` script, which is why
`MOSAIC_POLO_BIN` is worth setting rather than leaving to `$PATH` — the last step of the
search cannot tell the two builds apart.

Add `--extra augment` to either `uv sync` to install `albumentations`, which Ultralytics
picks up on its own and uses to apply Blur, MedianBlur, ToGray and CLAHE at p=0.01
during training. Nothing records which way a run went, so it stays deliberate.

Both tools open a video path, so an imgstore recording has to be exported first with
`mosaic run --kind export-store`, as TRex, SLEAP and Lightning Pose already require. The
error message names the command. `infer-localizer` is unaffected — it is mosaic's own
PyTorch and reads a store natively.

### keypoint-MoSeq

Built the same way, from its own directory:

```bash
cd src/mosaic/behavior/feature_library/external
uv sync --python 3.13
export MOSAIC_KPMS_LICENSE_ACCEPTED=1
```

Mosaic will not start keypoint-MoSeq until `MOSAIC_KPMS_LICENSE_ACCEPTED` is set to
exactly `1`. Harvard OTD licenses keypoint-MoSeq for non-commercial research and
academic use only, and setting the variable asserts that your use is permitted;
[`external/README.md`](https://github.com/EcodylicScience/mosaic/blob/main/src/mosaic/behavior/feature_library/external/README.md)
has the terms in full.

### FERAL

A mosaic extra rather than an external tool — the classifier runs in mosaic's own
process — but it wants an environment of its own too. It pins its dependencies to exact
versions where mosaic's are lower bounds, so installing `[feral]` alongside downgrades
several of them. Build a second environment holding mosaic and `[feral]`, and point it
at the same datasets.

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
| TREx tracking and pose-model training                             | Partial                 | Yes          | Yes   |

For any **No** or **Partial** capability, install **WSL2** (`wsl --install` in an
admin PowerShell) and follow the Linux setup above inside Ubuntu. Under WSL, keep the
repository and your datasets under home (`~/`) rather than `/mnt/c`.
