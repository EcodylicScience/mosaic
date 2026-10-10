# The VAME environment

The `vame` feature drives [VAME](https://github.com/EthoML/VAME) in a separate
Python environment. mosaic does not install VAME and does not import it. This
directory contains the environment's definition (`pyproject.toml`, `uv.lock`).
The program that mosaic starts in the environment is `vame_runner.py` in the
directory above, with `vame_protocol.py` and `runner_wire.py` beside it. You
build the environment yourself, once.

## License

VAME is licensed GPL-3.0. The runner is part of mosaic, under AGPL-3.0-or-later,
and imports VAME in this environment's process. Section 13 of each license
permits combining a work under the other.

Building the environment installs VAME from its publisher under VAME's terms. If
you convey a built environment, for example in a container image, you convey
VAME under the GPL and must offer its source. Mosaic's
[NOTICE](../../../../../../NOTICE) records the other third-party terms.

## Install

From the repository root:

```bash
cd src/mosaic/behavior/feature_library/external/vame-env
uv sync --python 3.12
```

`pyproject.toml` admits Python 3.12 and 3.13. hmmlearn 0.3.3, which VAME
requires, publishes wheels for Python 3.13 and earlier only. On Python 3.14 uv
builds it from source, with a C++ compiler.

`uv sync` resolves from the committed `uv.lock`, which pins `vame-py` 0.14.4 and
every package below it. It also compiles each installed module to bytecode.
VAME's package turns off bytecode writing when it is imported. Without bytecode
compiled at install, every process compiles VAME again.

The build works from a source checkout only. This directory is excluded from the
uv workspace, and its files are not part of the installed wheel.

### Platforms

- Linux and Windows build with torch from PyPI. PyPI's Windows build runs on
  the CPU only. For a GPU on Windows, install torch from PyTorch's CUDA index,
  as for older GPUs below.
- macOS needs Apple silicon and macOS 14 or later. torch and netCDF4 publish
  macOS wheels for macOS 14 and later only. VAME declares torch for Apple
  silicon only. On an Intel Mac the environment builds without torch, and VAME
  fails at import.

### Older GPUs need torch from another index

The lock resolves torch from PyPI, whose Linux build targets CUDA 13 and
contains kernels for current GPUs only. GPUs older than Turing (compute
capability below 7.5), such as the GTX 1080 Ti, report
`torch.cuda.is_available()` as `True`, and the first kernel then fails with
`no kernel image is available for execution on the device`. Keep the lock's
version and take its build from PyTorch's cu126 index, the newest that still
ships kernels for them:

```bash
uv sync --python 3.12
uv pip install --reinstall-package torch \
    --index-url https://download.pytorch.org/whl/cu126 \
    "torch==<the lock's version>"
```

`--reinstall-package torch` replaces torch. Of the other locked packages, it
changes only `cuda-bindings` and `cuda-toolkit`, which the cu126 build requires
at other versions. Do not use `--reinstall`, which reinstalls every dependency
of torch from PyTorch's index and moves shared packages such as setuptools and
networkx off the lock. The cu126 build adds its CUDA 12 libraries
(`nvidia-*-cu12`). The lock's CUDA 13 libraries stay installed beside them, and
do not interfere with it.

A later `uv sync` reinstalls PyPI's build. Repeat the second step after one. On
such a card you can also run on the CPU with `device="cpu"`.

## Use an environment you already have

Point mosaic at another interpreter instead of building this one, either as a
conda environment:

```bash
export MOSAIC_VAME_CONDA_ENV=vame
```

or as a direct path:

```bash
export MOSAIC_VAME_PYTHON=/path/to/env/bin/python
```

The environment must contain `vame-py` 0.14.4 and pydantic. The feature refuses
a runner that reports another VAME version.

mosaic uses the first of the feature's `vame_python` parameter,
`MOSAIC_VAME_CONDA_ENV` and `MOSAIC_VAME_PYTHON` that is set, and otherwise
`.venv/bin/python` here. The run identifier does not include them, and a fitted
model is not recomputed on another machine.

## Check it works

```bash
.venv/bin/python -c "import importlib.metadata as m; print(m.version('vame-py'))"
```

The command prints `0.14.4`.

The integration tests fit and apply a small model through the feature, on the
CPU. From the repository root, in mosaic's environment:

```bash
pytest -m slow tests/behavior/feature_library/test_vame_integration.py
```

They find the environment as the feature does, and skip when none resolves.

## Devices

The feature's `device` parameter chooses where VAME trains and embeds: `auto`,
which selects CUDA, then MPS, then the CPU, as VAME does, or `cpu`, `cuda` or
`mps`. It does not enter the run identifier. One recipe therefore runs under one
name on a Mac and on a GPU server. The model bundle and the run log record the
device that ran.

The device changes the model. CPU and MPS train materially different models from
the same data and parameters. Only training on the CPU is bit-reproducible. Two
CPU fits of one recipe give identical models.
