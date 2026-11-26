# The lmt-analysis environment

The `rebuild-lmt-events` op runs
[lmt-analysis](https://github.com/fdechaumont/lmt-analysis) on a copy of each Live
Mouse Tracker database, in a separate Python environment. mosaic does not install
lmt-analysis and does not import it. This directory contains the environment's
definition (`pyproject.toml`, `uv.lock`). The program that mosaic starts in the
environment is `lmt_rebuild_runner.py` in `../runner/`. You build the environment
and clone lmt-analysis yourself, once.

## License

lmt-analysis is licensed GPL-3.0. The runner is part of mosaic and imports
lmt-analysis only in this environment's process, through `importlib`. mosaic does
not copy, vendor or reimplement any of its code.

Building the environment fetches lmt-analysis from its publisher under its own
terms. If you convey a built environment, for example in a container image, you
convey lmt-analysis under the GPL and must offer its source. mosaic's
[NOTICE](../../../../../NOTICE) records the other third-party terms.

## Install

From the repository root:

```bash
cd src/mosaic/tracking/external/lmt-env
uv sync --python 3.12
git clone https://github.com/fdechaumont/lmt-analysis.git lmt-analysis
git -C lmt-analysis checkout 3ac90ec842b38e945a5d226dd2486880afbd766d
```

That commit is the one the import guide names. A run names the commit it uses, and
the runner refuses a checkout at another commit or with a modified tracked file.

`pyproject.toml` admits Python 3.11 and 3.12. `uv sync` resolves from the committed
`uv.lock`, which pins the packages that lmt-analysis lists in its
`python packages needed.txt` without versions. lmt-analysis declares no package
metadata, so it is not a dependency here: it is the checkout you clone.

The script imports `tkinter` at module scope. The CPython builds that `uv` installs
include it. A system Python without it fails the import, and the runner reports
that as an environment failure.

The build works from a source checkout only. This directory is excluded from the
uv workspace, and its files are not part of the installed wheel.

## How mosaic finds it

The environment's interpreter is the first of these that is set:

1. `MOSAIC_LMT_CONDA_ENV`, a conda environment name or prefix.
2. `MOSAIC_LMT_PYTHON`, the path of a `python`.
3. `.venv/bin/python` here.

The checkout of lmt-analysis is `MOSAIC_LMT_ANALYSIS_DIR` when it is set, and
`lmt-analysis/` here otherwise. Neither enters a run identifier: they are where the
tool is on this machine. The commit does.

```bash
export MOSAIC_LMT_PYTHON=/path/to/env/bin/python
export MOSAIC_LMT_ANALYSIS_DIR=/path/to/lmt-analysis
```

Without an environment, `rebuild-lmt-events` refuses with a message that names this
file.

## Check it works

```bash
.venv/bin/python -c "import numpy, scipy, matplotlib, pandas; print('ok')"
git -C lmt-analysis rev-parse HEAD
```

The first command prints `ok`, and the second prints the commit that you checked
out.
