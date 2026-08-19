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
deep-learning surface — YOLO pose training and inference, the heatmap localizer and
the identity models — which means PyTorch, and on Linux about 4 GB of CUDA wheels.
`mosaic track ultralytics` is not in that list: it drives Ultralytics in an
environment you build, covered under
[Tools that run in their own environment](#tools-that-run-in-their-own-environment).

For that surface without CUDA — a laptop, a CPU-only node, a container you want to keep
small — take PyTorch from its own index:

```bash
pip install -e ".[all]" --extra-index-url https://download.pytorch.org/whl/cpu
```

| Extra              | Adds                                                            |
| ------------------ | --------------------------------------------------------------- |
| `all`              | `pose` + `faiss`: the documented install                        |
| `deep-learning`    | `torch` + `timm` — the heatmap localizer and all three identity models |
| `pose`             | `deep-learning`, plus Ultralytics YOLO pose model training and single-model inference |
| `polo`             | `deep-learning`, plus POLO point-model training and inference   |
| `faiss`            | The `"faiss"` kNN backend for `global-tsne`; its default `"annoy"` backend needs nothing |
| `movement`         | The [movement](https://movement.neuroinformatics.dev/) smoothing and filtering features, and the xarray / netCDF4 / pynwb / sleap-io stack they sit on |
| `lightning-action` | Lightning-Action temporal action classifier                     |
| `feral`            | FERAL V-JEPA behavior classifier, training and inference        |
| `yolo-augment`     | `albumentations`, adding photometric augmentation to YOLO and POLO training |

Four things to know before choosing:

- **`pose` and `polo` cannot share an environment.** Both install a distribution named
  `ultralytics` — upstream against a fork — and pip resolves only one. Both build on
  `deep-learning`, so taking the fork does not cost you the identity models. Neither
  reaches `mosaic track ultralytics`, whose Ultralytics lives in an environment of its
  own, so one machine can hold the fork here and upstream there.
- **`yolo-augment` changes what a training run does.** Ultralytics adds Blur,
  MedianBlur, ToGray and CLAHE at p=0.01 whenever `albumentations` is importable, and
  nothing records which way a run went.
- **`faiss` installs `faiss-cpu`.** On Linux with CUDA, install `faiss-gpu` yourself
  instead.
- **`pose`, `polo` and therefore `all` install Ultralytics, which is AGPL-3.0.** Pose
  and point model training and inference import it inside mosaic's own process, which
  is what those extras are for. A bare `pip install -e .` carries no AGPL dependency,
  and neither does `mosaic track ultralytics`, which runs Ultralytics as a separate
  program. [NOTICE](https://github.com/EcodylicScience/mosaic/blob/main/NOTICE) says
  what that means for a networked deployment.

Spectral features, the SLEAP analysis reader and the DeepLabCut HDF5 reader are **not**
extras: PyWavelets, h5py and PyTables are base dependencies, because each one gates
reading a file you already have rather than an integration you opted into.

`recommended`, `identity`, `localizer` and `gpu` still resolve as aliases and are
removed in 0.13. `wavelets`, `sleap`, `hdf5` and `imgstore` are gone.

## Tools that run in their own environment

Five of the tools mosaic drives are **not installed by mosaic**. You install each one
yourself and then tell mosaic where it is, and mosaic launches it there. Four of them
pin a Python version or a framework stack that cannot share an environment with
mosaic; Ultralytics is separate for a licensing reason, given below.

| Install it yourself | Used by | Tell mosaic where it is |
| ------------------- | ------- | ----------------------- |
| [**TRex**](https://trex.run) | `mosaic track trex` | `MOSAIC_TREX_CONDA_ENV`, or `MOSAIC_TREX_BIN` for the binary itself |
| [**SLEAP**](https://sleap.ai) | `mosaic track sleap`, the `train-sleap` op | `MOSAIC_SLEAP_CONDA_ENV`, or `MOSAIC_SLEAP_BIN` |
| [**Lightning Pose**](https://lightning-pose.readthedocs.io) | `mosaic track litpose`, the `train-litpose` op | `MOSAIC_LITPOSE_CONDA_ENV`, or `MOSAIC_LITPOSE_BIN` |
| [**Ultralytics**](https://github.com/ultralytics/ultralytics) | `mosaic track ultralytics` | `MOSAIC_ULTRALYTICS_CONDA_ENV`, or `MOSAIC_ULTRALYTICS_BIN` for the environment's `yolo` script |
| [**keypoint-MoSeq**](https://keypoint-moseq.readthedocs.io) | the `kpms` feature | `MOSAIC_KPMS_PYTHON`, the environment's interpreter |

```bash
export MOSAIC_TREX_CONDA_ENV=trex
export MOSAIC_SLEAP_CONDA_ENV=sleap
export MOSAIC_KPMS_PYTHON=/path/to/kpms-env/bin/python
```

A `_CONDA_ENV` variable names a conda environment, which mosaic activates with `conda
run`; a `_BIN` variable names a path directly. With neither set, mosaic looks on
`$PATH`. Where a tool is installed never enters a `run_id`, so two machines that place
it differently still agree on what a run is called.

**Ultralytics** and **keypoint-MoSeq** are the two mosaic helps you build. The
repository carries each environment's definition, a `pyproject.toml` and a lock file,
so building one is a single command in its directory.

Ultralytics is AGPL-3.0, and a program that imports it is one work with it, so mosaic
drives it as a separate program and imports it nowhere on that path:

```bash
cd src/mosaic/tracking/external/ultralytics-env
uv sync --python 3.12
export MOSAIC_ULTRALYTICS_BIN="$PWD/.venv/bin/yolo"
```

The interpreter is pinned rather than left to uv, which would take the newest one it
can find. That directory's `pyproject.toml` admits `>=3.12` and its `uv.lock` was
resolved for one interpreter, so a build on a newer one resolves a different set of
wheels -- or none at all, for the first months after a Python release. Two machines
tracking one video under one run identifier should be running the same code, which is
what the committed lock is for. `MOSAIC_ULTRALYTICS_BIN` names the `yolo` console
script; the `python` beside it in the same `bin/` is what mosaic runs.

That costs two things, against an Ultralytics installed into mosaic's own
environment. Building an environment is more work than adding an extra. And the
tracker is handed a video path like every other external tool, so an imgstore
recording has to be exported to plain video first, with
`mosaic run --kind export-store`, exactly as TRex, SLEAP and Lightning Pose already
require; the error message names the command.

**keypoint-MoSeq** is built the same way, from
`src/mosaic/behavior/feature_library/external/`:

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

**FERAL** is a mosaic extra rather than an external tool — the classifier runs in
mosaic's own process — but it wants an environment of its own too. It pins its
dependencies to exact versions where mosaic's are lower bounds, so installing
`[feral]` alongside downgrades several of them. Build a second environment holding
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
| TREx tracking and pose-model training                             | Partial                 | Yes          | Yes   |

For any **No** or **Partial** capability, install **WSL2** (`wsl --install` in an
admin PowerShell) and follow the Linux setup above inside Ubuntu. Under WSL, keep the
repository and your datasets under home (`~/`) rather than `/mnt/c`.
