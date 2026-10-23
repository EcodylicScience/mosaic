<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/EcodylicScience/mosaic/main/docs/assets/pipeline-dark.svg">
    <img alt="Two ways in, one table out. Video feeds the trackers mosaic runs; tracks produced by other tools are imported instead. Both arrive as one standardized tracks table, which feeds an analysis stage where individual, pair and group features compose with unsupervised and supervised models." src="https://raw.githubusercontent.com/EcodylicScience/mosaic/main/docs/assets/pipeline-light.svg" width="880">
  </picture>
</p>

<h1 align="center">mosaic</h1>
<p align="center"><strong>A platform for behavior analysis.</strong></p>

<p align="center">
  <img alt="Python 3.12+" src="https://img.shields.io/badge/python-3.12%2B-blue">
  <img alt="AGPL-3.0-or-later" src="https://img.shields.io/badge/license-AGPL--3.0--or--later-blue">
</p>

Mosaic drives trackers — **TRex, SLEAP, Lightning Pose, Ultralytics** — or imports
tracks something else already produced, then gives a flexible and expandable analysis
framework for working with the data through the **feature** construct: individual
behavioral metrics, collective motion, unsupervised methods, and supervised model
fitting on annotated behaviors. A core focus is enabling **multi-animal analysis**: a
feature can be calculated on individuals, on pairs, or on all animals at once.

Results are organized by a content-addressed scheme for run-caching and
reproducibility: the same inputs and parameters produce the same `run_id`, so
re-running is a no-op, parameter sweeps organize themselves, and `mosaic inventory`
tells you exactly what a dataset already holds.

## What it does

| Step | What mosaic does |
| --- | --- |
| **Track** | Run **4 integrated trackers** over your videos from one command — or import tracks you already have, from **8 formats** |
| **Standardize** | Every tracker's output becomes one validated schema: video pixels, `X`/`Y` at the body centre, keypoints optional |
| **Feature** | **51 registered features** — kinematic, social, collective-motion, spectral, reduction, temporal context |
| **Model** | t-SNE, k-means, Ward, AR-HMM, keypoint-MoSeq, VAME; XGBoost / Lightning-Action / FERAL classifiers; three visual identity models |
| **Annotate** | Overlay video with identities, poses and predicted behavior; egocentric crops for identity work |
| **Train pose** | No tracks yet? Sample frames for annotation and train YOLO pose, POLO point detection, or a heatmap localizer from CVAT / COCO / Lightning Pose |
| **Operate** | **32 CLI commands** and **22 ops** behind one job contract, with a run log, cancellation, and a dataset inventory |

## Install

```bash
conda create -n mosaic python=3.12 -y
conda activate mosaic
conda install -c conda-forge ffmpeg av py-opencv -y
pip install "mosaic-behavior[all]"
```

The distribution is `mosaic-behavior` and the import name is `mosaic`. An unrelated
project named `mosaic` exists on PyPI, so never run `pip install mosaic`.

`pip install mosaic-behavior` on its own is a complete analysis install: converters,
features, clustering, classifiers, overlays. `[all]` adds the deep-learning surface:
the heatmap localizer and the identity models, which means PyTorch. YOLO pose and POLO
point work is not in it. Tracking, inference and training alike run in an environment
you build, so no install of mosaic carries an AGPL-licensed dependency.

`ffmpeg` and `ffprobe` must be on `PATH`. `av` and `py-opencv` are installed from
conda-forge, before pip runs, so that the environment holds a single ffmpeg build. A
plain pip install brings the PyPI wheels of both, which collide on macOS.

### Install from source

To work on mosaic itself, or to build the Ultralytics, POLO, keypoint-MoSeq or VAME
environment, clone the repository and install it editable into the same kind of conda
environment:

```bash
git clone https://github.com/EcodylicScience/mosaic.git
cd mosaic
pip install -e ".[all]"
```

The
[environment notes](https://github.com/EcodylicScience/mosaic/blob/main/CONTRIBUTING.md#environment-notes)
of CONTRIBUTING.md cover working against an unreleased `mosaic-media`.

[Installation](https://ecodylicscience.github.io/mosaic/installation/) has the extras table, a CPU-only PyTorch line, the
tools mosaic drives but does not install, and Windows/WSL2 support. The Ultralytics,
POLO, keypoint-MoSeq and VAME environments are built from files in the repository that
the pip package does not carry.

## In 60 seconds

```bash
# Work in a directory of your own, not inside a mosaic source checkout.
cd ~
mosaic init study --name "Experiment 1"

# Point the dataset at video that can live anywhere -- a NAS, another volume.
mosaic sources add -m study/dataset.yaml --kind media \
    --path /data/day1 --extensions .mp4
mosaic scan -m study/dataset.yaml

# Track it, then derive something from the result.
mosaic track trex -m study/dataset.yaml --set track_max_individuals=4
mosaic run  -m study/dataset.yaml --feature speed-angvel

# What does this dataset now hold, and is any of it stale?
mosaic inventory -m study/dataset.yaml
```

Already have tracks? Skip the tracker: declare them as a `--kind tracks` source with
their `--src-format`, and `mosaic convert-tracks` standardizes them.

## Documentation

| | |
| --- | --- |
| [**Documentation**](https://ecodylicscience.github.io/mosaic/) | What mosaic is, and where everything lives |
| [**Install**](https://ecodylicscience.github.io/mosaic/installation/) | The environment, the extras, and the tools that run in their own environment |
| [**Reference**](https://ecodylicscience.github.io/mosaic/reference/) | Every feature, op, CLI command and track format — generated from the code |
| [**AI coding agents**](https://ecodylicscience.github.io/mosaic/agents/) | Run `mosaic agent install` to give an agent the mosaic guide |

Five worked examples live in [`notebooks/`](https://github.com/EcodylicScience/mosaic/tree/main/notebooks)
in the repository, not in the pip package. Each fetches its own data from Hugging Face,
so they run as written:

| Notebook | What it shows |
| --- | --- |
| [CalMS21 template](https://github.com/EcodylicScience/mosaic/blob/main/notebooks/calms21-template.ipynb) | The fullest path: ingest, features, embedding, clustering, an XGBoost classifier |
| [Collective motion: shiners](https://github.com/EcodylicScience/mosaic/blob/main/notebooks/collective-motion-shiners.ipynb) | A converter written in the notebook, then order parameters across four group sizes |
| [Collective motion: zebrafish](https://github.com/EcodylicScience/mosaic/blob/main/notebooks/collective-motion-zebrafish.ipynb) | A keypoint-free tracker, and social-force maps from neighbour positions |
| [CalMS21 pose training](https://github.com/EcodylicScience/mosaic/blob/main/notebooks/calms21-pose-training-and-tracking.ipynb) | Tracks become pose annotations, train a YOLO model, track with it in TREx |
| [Shiners point training](https://github.com/EcodylicScience/mosaic/blob/main/notebooks/shiners-polo-tracking.ipynb) | A POLO point model, and two ways to give TREx a body |

The two tracking notebooks drive external tools mosaic does not install; they say so
and skip those sections when the tool is absent.

## Status

Early development (0.x). Public APIs, feature names, and on-disk layouts may change
between releases; [CHANGELOG.md](https://github.com/EcodylicScience/mosaic/blob/main/CHANGELOG.md) records what moved and why.

## How to cite

Cite the tools that produced your results first: the tracker, the pose model and the
method behind each feature you used. Cite mosaic second, as the software that ran them.
The [citing page](https://ecodylicscience.github.io/mosaic/citing/) says how, and
[NOTICE](https://github.com/EcodylicScience/mosaic/blob/main/NOTICE) records the
origin, license and citation of each component.

## Contributing

See [CONTRIBUTING.md](https://github.com/EcodylicScience/mosaic/blob/main/CONTRIBUTING.md), and please open an issue before making large
changes. To contribute with an AI coding agent,
[CLAUDE.md](https://github.com/EcodylicScience/mosaic/blob/main/CLAUDE.md) is the
developer orientation.

## License

GNU Affero General Public License v3 or later. See [LICENSE](https://github.com/EcodylicScience/mosaic/blob/main/LICENSE), and
[NOTICE](https://github.com/EcodylicScience/mosaic/blob/main/NOTICE) for the third-party attributions that must travel with it.

mosaic bundles no third-party source and no model weights, but it drives tools whose
terms differ from its own — keypoint-MoSeq prohibits commercial use, TRex requires a
paid license for company use, and Ultralytics is AGPL-3.0. [NOTICE](https://github.com/EcodylicScience/mosaic/blob/main/NOTICE) records how
each component reaches you and what that means for your use of it.
