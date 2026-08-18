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
pip install -e ".[all]"
```

`pip install -e .` on its own is a complete install of the analysis pipeline:
media indexing, every track and label converter, all per-frame and social
features, wavelets, scaling, t-SNE, k-means, Ward, ARHMM, the XGBoost
classifier, overlays and crops. What it leaves out is the deep-learning surface
-- YOLO pose training and inference, `mosaic track ultralytics`, the heatmap
localizer, and the identity models -- because that means PyTorch, and on Linux
PyTorch means about 4 GB of CUDA wheels. `[all]` adds it.

If you want that surface without CUDA -- a laptop, a CPU-only node, a container
you would rather keep small -- take PyTorch from its CPU index:

```bash
pip install -e ".[all]" --extra-index-url https://download.pytorch.org/whl/cpu
```

Frame decoding runs in-process via `av`, so no `ffmpeg` binary is required to
read video. System `ffprobe` is still used for media indexing and probing
(`mosaic_media.probe_media`), and system `ffmpeg` >= 5.1 is required for the
transcode path (`mosaic media transcode`). Installing `ffmpeg` via conda covers
both.

## Why conda, and when you do not need it

The rule is **one ffmpeg build per process**. Two of them crash it at a different
point on every run. `av` and the `opencv-python` wheel each bundle a complete
build, so a pure-wheel install holds two.

Conda is the route this page recommends because it is the one that also solves
the second hazard below: conda-forge's `av` and `py-opencv` link the `ffmpeg`
installed on the same line rather than vendoring their own, and one `py-opencv`
registers *both* the `opencv-python` and `opencv-python-headless` distribution
names for its single build. Order matters -- conda first, pip second, so the
`pip install` finds both requirements satisfied and installs neither wheel.

It is not the only route, and on Linux it is often not needed at all. mosaic's
own CI installs `av` and `opencv-python` as plain wheels against an apt `ffmpeg`,
on every job, and has never crashed. `apt install python3-av python3-opencv`
links both against the distribution's single ffmpeg, and `pip install av
--no-binary av` builds against yours. Treat conda as the recommended route for a
macOS workstation and as one option among several elsewhere.

## Extras

Install nothing extra, or `[all]`, and you are done. The rest of this table
exists for the cases where that is not what you want.

| Extra              | Adds                                                                                |
| ------------------ | ----------------------------------------------------------------------------------- |
| `all`              | `pose` + `faiss`. The documented install: everything that belongs in one general-purpose environment |
| `deep-learning`    | `torch` + `timm`: the heatmap localizer and all three identity models               |
| `pose`             | `deep-learning` plus Ultralytics YOLO pose training and inference, and `mosaic track ultralytics` (six multi-object trackers, in process) |
| `polo`             | `deep-learning` plus POLO point detection. Mutually exclusive with `pose` -- a different `ultralytics` fork, and pip resolves only one |
| `faiss`            | The `"faiss"` kNN backend for `global-tsne`. The default backend is `"annoy"` and needs nothing. Installs `faiss-cpu`; on Linux + CUDA install `faiss-gpu` instead |
| `movement`         | The [movement](https://movement.neuroinformatics.dev/) filtering and smoothing features. Heavy for two features -- xarray, netCDF4, pynwb, sleap-io -- which is why it is not in `all` |
| `yolo-augment`     | `albumentations`, which Ultralytics picks up on its own to add Blur / MedianBlur / ToGray / CLAHE at p=0.01 to YOLO and POLO training. Opt-in because it changes what a training run does |
| `lightning-action` | Lightning-Action temporal action classifier. Capped at `<1.1`: 1.1.0 requires `nvidia-dali-cuda110`, unconditionally and sdist-only, so it needs a CUDA machine |
| `feral`            | FERAL V-JEPA behavior classifier (`FeralFeature`, training + inference). Install it in an environment of its own, see below |

Spectral features, the SLEAP analysis reader and the DeepLabCut HDF5 reader are
**not** extras. PyWavelets, h5py and PyTables are base dependencies: they are
numpy-only and about 30 MB together, and each gates reading a file you already
have rather than an integration you opted into.

`recommended`, `identity`, `localizer` and `gpu` still resolve, as aliases, and
are removed in 0.13. `wavelets`, `sleap`, `hdf5` and `imgstore` are gone: the
first three are base dependencies now, and `imgstore` was only ever needed to
*write* stores, which the test suite does and mosaic does not -- reading one is
native.

### Two hazards, told apart

An earlier version of this page ran these together, which is how `yolo-augment`
came to carry a packaging justification it does not need.

**Two `cv2` distributions.** `albumentations`, `lightning-action` and
`movement` (via `pyvideoreader`) require `opencv-python-headless`, while mosaic
and Ultralytics require `opencv-python`. pip installs both without complaint --
they are different distributions -- and they then ship the same `cv2` import
package, so one overwrites the other's files and both leave their bundled ffmpeg
builds in a single `cv2/.dylibs`. **This is a plain-pip hazard only.** The conda
environment above is immune, because one `py-opencv` satisfies both names.
`tests/conftest.py` refuses to start when it finds two builds.

**Two vendored ffmpeg builds.** `av` and any `opencv` wheel each bundle a
complete ffmpeg, so two of them collide even when only one provides `cv2`. This
one is **unaffected by which `cv2` flavor you pick** -- the wheels carry the same
payload -- and it does not arise on Windows, where OpenCV's ffmpeg is a separate
lazily-loaded DLL. See "Why conda, and when you do not need it" above.

`pose`, `polo`, and therefore `all` install Ultralytics, which is AGPL-3.0. That
matters more than it looks: AGPL section 13 extends the duty to offer
Corresponding Source to anyone who interacts with the program over a network, so
a commercial deployment is reached without ever redistributing a copy.
Ultralytics sells an Enterprise license for use that cannot meet those terms,
but it covers Ultralytics' own distribution only — `polo` is a third-party fork,
so it is AGPL-only. Mosaic is AGPL-3.0-or-later itself, so nothing here is
incompatible; the question is whether *your* use of the combined work can meet
the obligations. A bare `pip install -e .` carries no AGPL dependency at all.
See [Licensing](licensing.md).

`feral` wants an environment of its own, and for a different reason from
anything above. FERAL pins its dependency versions exactly — `opencv-python`,
`pandas`, `scikit-learn`, `timm`, `matplotlib`, `transformers` — while every
mosaic requirement is a lower bound, so pip resolves the two together without
complaint and downgrades each one. No install layout fixes that. Install `feral`
into a separate environment and point it at the same datasets.

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
| GPU kNN (`faiss`, `faiss-gpu`)                                    | No (`faiss-cpu` works)  | Yes          | n/a   |
| imgstore reading (native) / writing (`--group test`)              | Partial                 | Yes          | Yes   |
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
