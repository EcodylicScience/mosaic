<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/pipeline-dark.svg">
    <img alt="Two ways in, one table out. Video feeds the trackers mosaic runs; tracks produced by other tools are imported instead. Both arrive as one standardized tracks table, which feeds an analysis stage where individual, pair and group features compose with unsupervised and supervised models." src="docs/assets/pipeline-light.svg" width="880">
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
| **Feature** | **45 registered features** — kinematic, social, collective-motion, spectral, reduction, temporal context |
| **Model** | t-SNE, k-means, Ward, AR-HMM, keypoint-MoSeq; XGBoost / Lightning-Action / FERAL classifiers; three visual identity models |
| **Annotate** | Overlay video with identities, poses and predicted behavior; egocentric crops for identity work |
| **Train pose** | No tracks yet? Sample frames for annotation and train YOLO pose, POLO point detection, or a heatmap localizer from CVAT / COCO / Lightning Pose |
| **Operate** | **24 CLI commands** and **17 ops** behind one job contract, with a run log, cancellation, and a dataset inventory |

## Install

```bash
git clone https://github.com/EcodylicScience/mosaic.git
cd mosaic
conda create -n mosaic python=3.12 -y
conda activate mosaic
conda install -c conda-forge ffmpeg av py-opencv -y
pip install -e ".[all]"
```

`pip install -e .` on its own is a complete analysis install — converters, features,
clustering, classifiers, overlays. `[all]` adds the deep-learning surface: the heatmap
localizer and the identity models, which means PyTorch. YOLO pose and POLO point work
is not in it — tracking, inference and training alike run in an environment you build,
so no install of mosaic carries an AGPL-licensed dependency. `av` and `py-opencv` are installed from conda-forge, before pip runs, so that
the environment holds a single ffmpeg build.

[Installation](docs/installation.md) has the extras table, a CPU-only PyTorch line, the
tools mosaic drives but does not install, and Windows/WSL2 support.

## In 60 seconds

```bash
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
| [**Documentation**](docs/index.md) | What mosaic is, and where everything lives |
| [**Install**](docs/installation.md) | The environment, the extras, and the tools that run in their own environment |
| [**Reference**](docs/reference/index.md) | Every feature, op, CLI command and track format — generated from the code |

Worked examples live in [`notebooks/`](notebooks/): an end-to-end
[CalMS21 template](notebooks/calms21-template.ipynb), and collective motion on
[shiners](notebooks/collective-motion-shiners.ipynb) and
[zebrafish](notebooks/collective-motion-zebrafish.ipynb). They read data that is not
bundled with the repository, so treat them as illustrated results rather than as
runnable tutorials.

## Status

Early development (0.x). Public APIs, feature names, and on-disk layouts may change
between releases; [CHANGELOG.md](CHANGELOG.md) records what moved and why.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md), and please open an issue before making large
changes. Working with an AI coding agent? [CLAUDE.md](CLAUDE.md) is the repository
orientation.

## License

GNU Affero General Public License v3 or later. See [LICENSE](LICENSE), and
[NOTICE](NOTICE) for the third-party attributions that must travel with it.

mosaic bundles no third-party source and no model weights, but it drives tools whose
terms differ from its own — keypoint-MoSeq prohibits commercial use, TRex requires a
paid license for company use, and Ultralytics is AGPL-3.0. [NOTICE](NOTICE) records how
each component reaches you and what that means for your use of it.
