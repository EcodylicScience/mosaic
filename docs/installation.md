# Installation

`pyproject.toml` is the canonical dependency source; this page explains what it
declares and why.

mosaic is not published on PyPI, so it installs from a checkout. The clone is
part of the instructions, not an assumed step -- `pip install -e .` has nothing
to install from an empty directory.

```bash
git clone https://github.com/EcodylicScience/mosaic.git
cd mosaic
conda create -n mosaic python=3.12 -y
conda activate mosaic
conda install -c conda-forge ffmpeg av py-opencv -y
pip install -e ".[recommended]"
```

Frame decoding runs in-process via `av`, so no `ffmpeg` binary is required to
read video. System `ffprobe` is still used for media indexing and probing
(`mosaic_media.probe_media`), and system `ffmpeg` >= 5.1 is required for the
transcode path (`mosaic media transcode`). Installing `ffmpeg` via conda covers
both.

**`av` and `py-opencv` come from conda so the environment holds one ffmpeg.**
Their PyPI wheels each bundle a complete ffmpeg build of their own, and two of
those in one process crash it at a different point on every run -- on macOS the
Objective-C runtime warns about it on every import. The conda-forge builds link
the `ffmpeg` on the same line instead. Nothing is pinned: `av` satisfies mosaic's
requirement, and one `py-opencv` registers both `opencv-python` and
`opencv-python-headless`, so the `pip install` that follows finds them satisfied
and installs neither wheel. Order matters -- conda first, pip second.

The `recommended` extra bundles wavelets, YOLO pose training/inference, and the
PyTorch localizer. It deliberately excludes `yolo-augment`: `albumentations`
requires `opencv-python-headless` while mosaic requires `opencv-python`, and pip
installs both without complaint even though they ship the same `cv2` package.
Installing `yolo-augment` also changes what a YOLO or POLO training run does,
and nothing records which way a run went, so it is opt-in. For lighter or
alternative installs, select extras individually:

| Extra              | Adds                                                                                |
| ------------------ | ----------------------------------------------------------------------------------- |
| `recommended`      | `wavelets` + `pose` + `localizer`                                                   |
| `wavelets`         | PyWavelets for spectral features                                                    |
| `pose`             | Ultralytics YOLO pose training and inference, and `mosaic track ultralytics` (six multi-object trackers, in process) |
| `polo`             | POLO point detection (mutually exclusive with `pose`; different ultralytics fork)   |
| `localizer`        | PyTorch heatmap localizer training                                                  |
| `identity`         | Image-backbone identity models (trained classifier, frozen timm backbones, DINOv2 + temporal); `torch` + `timm` |
| `lightning-action` | Lightning-Action temporal action classifier                                         |
| `gpu`              | faiss for GPU-accelerated kNN in `global-tsne` (use `faiss-gpu` on Linux + CUDA)    |
| `imgstore`         | Native imgstore (Motif / Loopbio) video support (directory-based stores as media)   |
| `sleap`            | `h5py`, to read the SLEAP analysis `.h5` its converter consumes (SLEAP itself is an external binary) |
| `hdf5`             | PyTables, which pandas dispatches `read_hdf` to, for DeepLabCut's HDF5 export (`.h5` / `.hdf5` / `.hdf`); the `.csv` form needs nothing extra. Installs no DeepLabCut code |
| `feral`            | FERAL V-JEPA behavior classifier (`FeralFeature`, training + inference); install it in an environment of its own, see below |
| `yolo-augment`     | `albumentations`, which Ultralytics picks up on its own to add Blur / MedianBlur / ToGray / CLAHE at p=0.01 to YOLO and POLO training |

`pose`, `polo`, and `recommended` install Ultralytics, which is AGPL-3.0. That
matters more than it looks: AGPL section 13 extends the duty to offer
Corresponding Source to anyone who interacts with the program over a network, so
a commercial deployment is reached without ever redistributing a copy.
Ultralytics sells an Enterprise license for use that cannot meet those terms,
but it covers Ultralytics' own distribution only — `polo` is a third-party fork,
so it is AGPL-only. Mosaic is AGPL-3.0-or-later itself, so nothing here is
incompatible; the question is whether *your* use of the combined work can meet
the obligations. See [Licensing](licensing.md).

`feral` wants an environment of its own. FERAL pins its dependency versions
exactly — `opencv-python`, `pandas`, `scikit-learn`, `timm`, `matplotlib`,
`transformers` — while every mosaic requirement is a lower bound, so pip resolves
the two together without complaint and downgrades each one. The `opencv-python`
pin is the one that does damage: it installs a wheel over the conda-forge
`py-opencv` the setup above asks for, which puts a second ffmpeg build in the
process beside `av` and crashes it nondeterministically. Install `feral` into a
separate environment and point it at the same datasets.

There is deliberately no `kpms` extra. keypoint-MoSeq cannot share an
environment with mosaic, so the `kpms` feature drives it in a separate one that
you build yourself — and it is licensed for non-commercial research and academic
use only. See [Licensing](licensing.md) for the terms and
[`external/README.md`](https://github.com/EcodylicScience/mosaic/blob/main/src/mosaic/behavior/feature_library/external/README.md)
for the setup.

## Platform support

mosaic runs natively on **macOS** and **Linux**. On **Windows**, the core
analysis pipeline runs natively, but several features depend on components with
no native-Windows build and need **WSL2** (or Linux):

| Capability                                                        | Native Windows          | WSL2 / Linux | macOS |
| ----------------------------------------------------------------- | ----------------------- | ------------ | ----- |
| Core analysis (indexing, tracks, features, clustering, ARHMM, XGBoost, visualization) | Yes | Yes | Yes |
| keypoint-MoSeq (`kpms`) -- JAX + Unix sockets                     | No                      | Yes          | Yes   |
| FERAL (`feral`) -- `decord`                                       | No                      | Yes          | Yes   |
| GPU kNN (`gpu`, `faiss-gpu`)                                      | No (`faiss-cpu` works)  | Yes          | n/a   |
| imgstore read/write (`imgstore`)                                  | Partial                 | Yes          | Yes   |
| TREx tracking and pose-model training                             | Partial                 | Yes          | Yes   |

Native-Windows support for the core is new; for any **No** / **Partial**
capability, or if anything misbehaves natively, use **WSL2** (`wsl --install` in
an admin PowerShell), then follow the Linux setup above inside Ubuntu. Keeping
the repository and your datasets on the WSL filesystem (for example `~/mosaic`)
rather than under `/mnt/c` is still much faster -- `drvfs` I/O is roughly an
order of magnitude slower than ext4 -- but it is no longer a correctness
requirement: index writes work on a `/mnt/*` mount. One caveat remains: a lock
taken from WSL and one taken by a native-Windows process are different lock
namespaces and do not see each other, so do not run both against one dataset at
the same time.
