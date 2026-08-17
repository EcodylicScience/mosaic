# Installation

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

## Two components need an environment of their own

`feral` wants one. FERAL pins its dependency versions exactly —
`opencv-python`, `pandas`, `scikit-learn`, `timm`, `matplotlib`, `transformers` —
while every mosaic requirement is a lower bound, so pip resolves the two together
without complaint and downgrades each one. The `opencv-python` pin is the one
that does damage: it installs a wheel over the conda-forge `py-opencv` the setup
above asks for, which puts a second ffmpeg build in the process beside `av` and
crashes it nondeterministically. Install `feral` separately and point it at the
same datasets.

There is deliberately no `kpms` extra, for the same reason: keypoint-MoSeq cannot
share an environment with mosaic, so the `kpms` feature drives it in one you
build yourself. Two things to know before the first run:

- [`external/README.md`](https://github.com/EcodylicScience/mosaic/blob/main/src/mosaic/behavior/feature_library/external/README.md)
  is the bootstrap, and `MOSAIC_KPMS_PYTHON` points mosaic at the interpreter.
- mosaic will not start it until `MOSAIC_KPMS_LICENSE_ACCEPTED=1` is set. Exactly
  `1` is accepted. Without it the feature refuses to spawn, which is by design
  rather than a fault to debug.

Third-party licenses, and what mosaic does about each, are recorded in
[NOTICE](https://github.com/EcodylicScience/mosaic/blob/main/NOTICE).

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

For any **No** / **Partial** capability, or if anything misbehaves natively, use
**WSL2** (`wsl --install` in an admin PowerShell), then follow the Linux setup
above inside Ubuntu.

Keep the repository and your datasets on the WSL filesystem (for example
`~/mosaic`) rather than under `/mnt/c`: `drvfs` I/O is roughly an order of
magnitude slower than ext4. Index writes work on a `/mnt/*` mount either way, so
this is a speed choice, not a correctness one. One caveat: a lock taken from WSL
and one taken by a native-Windows process are different lock namespaces and do
not see each other, so do not run both against one dataset at the same time.
