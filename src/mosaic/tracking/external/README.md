# The Ultralytics tracking environments

Mosaic's `ultralytics` tracker drives [Ultralytics](https://github.com/ultralytics/ultralytics)
in a separate Python environment. Mosaic does not install it and never ships it.
This directory holds the environments' definitions (`ultralytics-env/`, and a
`polo-env/` beside it for the POLO fork) and the two files that run inside them
(`runner/`). You build the environment yourself, once.

## Why the separation exists

> **Ultralytics is licensed AGPL-3.0**, and so is the
> [mooch443/POLO](https://github.com/mooch443/POLO) fork.

A program that imports Ultralytics is one work with it, and cannot be offered
under terms of its own without also licensing Ultralytics. Mosaic has to remain
distributable under its own terms, so what it does is spawn a second program in
the environment you build and exchange JSON files and command-line arguments
with it -- and two programs exchanging files are two programs.

`src/mosaic/tracking/ultralytics_track/run.py` is mosaic's side of that
exchange: it locates the environment and launches `runner/` inside it, and it
imports no Ultralytics.

Tracking is what runs out of process. `src/mosaic/tracking/pose_training/` --
YOLO and POLO training, and single-model inference -- still imports Ultralytics
in mosaic's own process, so the environments here do not yet cover every path
that reaches it.

The same reasoning puts keypoint-MoSeq in
[`src/mosaic/behavior/feature_library/external/`](../../behavior/feature_library/external/README.md);
[NOTICE](../../../../NOTICE) records both, and the other third-party terms.

**Building the environment installs Ultralytics from its own publisher under its
own terms.** Mosaic asks for none of them on your behalf: you are the licensee,
and AGPL-3.0 obligations -- notably the network-use clause -- attach to your
deployment, not to mosaic's.

## Install

From the repository root:

```bash
cd src/mosaic/tracking/external/ultralytics-env
uv sync --python 3.12
```

Python 3.12 is pinned rather than left to uv, which would otherwise take the
newest interpreter it can find. The `basedpyright` execution environment
declared for `runner/` in the root `pyproject.toml` names
`ultralytics-env/.venv/lib/python3.12/site-packages` as an extra path, so an
environment built on 3.13 puts its packages where nothing looks and the runner
silently type-checks against no Ultralytics at all.

The build works from a source checkout only: this directory is excluded from the
uv workspace, and its non-Python files are not part of the installed wheel.

`uv.lock` is committed and `uv sync` resolves from it. That is not bookkeeping:
an Ultralytics patch release can re-tune the detection defaults that
`runner/ultralytics_runner.py` passes explicitly for exactly that reason, so two
machines resolving the environment freshly would track the same video under one
run identifier and two sets of numbers.

## How mosaic finds it

The same ladder every other external tool uses, first match wins:

1. `MOSAIC_ULTRALYTICS_CONDA_ENV` -- a conda environment name or prefix.
2. `MOSAIC_ULTRALYTICS_BIN` -- the `yolo` console script. The `python` beside it
   in the same `bin/` is what mosaic runs, so pointing at any one of the
   environment's scripts names the directory the interpreter lives in.
3. `yolo` on `$PATH`.

```bash
export MOSAIC_ULTRALYTICS_BIN=src/mosaic/tracking/external/ultralytics-env/.venv/bin/yolo
```

The ladder resolves **placement, never identity**: which environment a tool ran
in is a property of the machine, so none of these values reaches a `run_id`, and
two machines with Ultralytics installed differently still agree on what a run is
called.

## The POLO fork lives beside this one

POLO is a full fork of Ultralytics -- it keeps every upstream task and adds
`locate` (point detection) -- and it ships under the **same distribution name**,
`ultralytics`. Installed into one environment the two cannot coexist: pip
resolves one and the other silently is not there. Two environments is exactly
what makes them coexist, which is why `runner/` is a sibling of
`ultralytics-env/` rather than inside it: both environments install the same two
files, and the program that runs is chosen by which interpreter mosaic spawns.

The directory is `ultralytics-env` and not `ultralytics` for a smaller reason: a
hyphen is not a valid Python identifier, so the directory can never be read as a
package shadowing the real `ultralytics` if it ever reaches `sys.path`.

## What runs inside

- `runner/ultralytics_protocol.py` -- the request and response models, and the
  row extraction. Imported from **both** environments, so it depends on the
  standard library, numpy and pydantic and nothing else. It imports neither
  `ultralytics` nor `mosaic`.
- `runner/ultralytics_runner.py` -- the program that imports Ultralytics, and
  the only one `mosaic track ultralytics` reaches that does. Two subcommands,
  each reading one JSON request file and writing one JSON response file:

  ```bash
  .venv/bin/python ../runner/ultralytics_runner.py probe --request req.json --out resp.json
  .venv/bin/python ../runner/ultralytics_runner.py track --request req.json --out result.json
  ```

  The response goes to a file rather than to standard output because importing
  torch and Ultralytics prints banners there. Standard output carries one JSON
  line when the weights finish loading and one per decoded batch after that,
  which is what mosaic's inactivity watchdog reads.

`probe` reports and never refuses: it says what the environment holds -- whether
Ultralytics and `lap` import, the version, the known backends, the model's task
and keypoint count -- and mosaic decides what to refuse, because the refusal
messages name mosaic commands and mosaic's own installation documentation.
