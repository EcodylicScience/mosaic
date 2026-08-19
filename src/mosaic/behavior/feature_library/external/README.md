# The keypoint-MoSeq environment

The `kpms` feature drives [keypoint-MoSeq](https://github.com/dattalab/keypoint-moseq)
in a separate Python environment. mosaic does not install it, does not import
it, and never ships it. This directory holds the environment's definition
(`pyproject.toml`, `uv.lock`) and the server script mosaic talks to over a
socket (`kpms_server.py`). You build the environment yourself, once.

## Before you install: the license

> **keypoint-MoSeq is licensed by the Harvard University Office of Technology
> Development for non-commercial research and academic use only. Commercial use
> is expressly prohibited.**

Harvard's definition of commercial use reaches:

- use in fee-for-service arrangements;
- use by core facilities or laboratories providing research services to, or in
  collaboration with, for-profit third parties for a fee;
- use in industry-sponsored or collaborative research projects where any
  commercial rights are granted to the sponsor or collaborator.

This is not a copyleft license, so no paid exception cures it. It restricts who
may run the software and for what purpose. Read the terms in full before you
install: <https://github.com/dattalab/keypoint-moseq/blob/main/LICENSE.md>

If your use is not permitted, the `arhmm` feature fits a comparable
autoregressive hidden Markov model in mosaic's own code and carries no such
restriction.

Mosaic's [NOTICE](../../../../../NOTICE) records the wider picture, including
why this environment is separate at all.

## Install

From the repository root:

```bash
cd src/mosaic/behavior/feature_library/external
uv sync --python 3.13
```

Python 3.13 matches the `basedpyright` execution environment declared for this
directory in the root `pyproject.toml`. The build works from a source checkout
only: this directory is excluded from the uv workspace and its non-Python files
are not part of the installed wheel.

`uv sync` resolves from `uv.lock` and will refresh that file if it has drifted
from `pyproject.toml`; a diff there after installing is expected.

## Confirm the license terms are met

`KpmsFeature` will not spawn the server until you say the terms apply to your
use:

```bash
export MOSAIC_KPMS_LICENSE_ACCEPTED=1
```

Exactly `1`, and nothing else, is accepted. Setting it is an assertion that
your use of keypoint-MoSeq is permitted under Harvard's terms.

## Use an environment you already have

If keypoint-moseq is installed somewhere else, point mosaic at that
interpreter instead of building this one:

```bash
export MOSAIC_KPMS_PYTHON=/path/to/env/bin/python
```

Resolution order, first match wins: the feature's `kpms_python` parameter, then
`MOSAIC_KPMS_PYTHON`, then `.venv/bin/python` here. The path is deliberately
excluded from the run identity hash, so moving between machines does not
recompute a fitted model.

## Check it works

```bash
pytest -m slow tests/test_kpms_integration.py
```

These tests skip unless both an interpreter resolves and the acceptance is set,
so a skip is the answer to "which precondition is missing" rather than a
failure.
