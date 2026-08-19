# The Ultralytics environments

Mosaic drives [Ultralytics](https://github.com/ultralytics/ultralytics) and the
[mooch443/POLO](https://github.com/mooch443/POLO) fork in separate Python
environments. Mosaic does not install either and never ships them. This directory
holds the two environment definitions (`ultralytics-env/`, `polo-env/`) and the
one program that runs inside them (`runner/`). You build them yourself, once
each.

| Environment | Built for | Located by |
| --- | --- | --- |
| `ultralytics-env/` | `mosaic track ultralytics`, `mosaic run --kind infer-pose`, `--kind train-pose` | `MOSAIC_ULTRALYTICS_CONDA_ENV` / `MOSAIC_ULTRALYTICS_BIN` |
| `polo-env/` | `mosaic run --kind infer-points`, `--kind train-points` | `MOSAIC_POLO_CONDA_ENV` / `MOSAIC_POLO_BIN` |

You need only the one your work reaches.

## Why the separation exists

> **Ultralytics is licensed AGPL-3.0**, and so is the
> [mooch443/POLO](https://github.com/mooch443/POLO) fork.

A program that imports Ultralytics is one work with it, and cannot be offered
under terms of its own without also licensing Ultralytics. Mosaic has to remain
distributable under its own terms, so what it does is spawn a second program in
the environment you build and exchange JSON files and command-line arguments
with it -- and two programs exchanging files are two programs.

`src/mosaic/tracking/common/ultralytics_env.py` locates either environment and
launches `runner/` inside it; `ultralytics_track/run.py`,
`pose_training/ultralytics_infer.py` and `pose_training/ultralytics_train.py`
are the tracker's, the inference ops' and the training ops' sides of that
exchange. None of the four imports Ultralytics.

**Every path that reaches Ultralytics runs here**, so no mosaic install carries
it: there is no extra to install, and `pip install -e ".[all]"` resolves no
AGPL-licensed dependency. Training was the last path to move, and the hardest,
because a cancelled run had to keep stopping the way it always had -- see
"Cancelling a training run" below.

The same reasoning puts keypoint-MoSeq in
[`src/mosaic/behavior/feature_library/external/`](../../behavior/feature_library/external/README.md);
[NOTICE](../../../../NOTICE) records both, and the other third-party terms.

**Building the environment installs Ultralytics from its own publisher under its
own terms.** Mosaic asks for none of them on your behalf: you are the licensee,
and AGPL-3.0 obligations -- notably the network-use clause -- attach to your
deployment, not to mosaic's.

## Install

From the repository root, for whichever you need:

```bash
cd src/mosaic/tracking/external/ultralytics-env
uv sync --python 3.12
```

```bash
cd src/mosaic/tracking/external/polo-env
uv sync --python 3.12
```

Python 3.12 is pinned rather than left to uv, which would otherwise take the
newest interpreter it can find. `pyproject.toml` here admits `>=3.12`, and
`uv.lock` beside it was resolved for one interpreter: a build on a newer one
resolves a different set of wheels, or none at all for the first months after a
Python release, which is exactly the reproducibility the committed lock is
there to give.

### Augmentation is opt-in, and lives here

```bash
uv sync --python 3.12 --extra augment
```

Ultralytics builds its `Albumentations` transform whenever the package is
importable, and swallows the ImportError when it is not -- so a build carrying
`--extra augment` additionally applies Blur, MedianBlur, ToGray and CLAHE at
p=0.01 to every training run, and nothing records which way a run went. That is
why it is a choice rather than a default, and why the choice belongs to whoever
builds the environment: this is the process that reads it. It was a mosaic extra
(`yolo-augment`) until training moved here, and no extra in mosaic's own
`pyproject.toml` can install a package into an environment mosaic does not build.

`opencv-python-headless` is declared outright and the GUI wheel Ultralytics asks
for is excluded. They are two distributions shipping one import package, so
installed together they overwrite each other's files and leave two vendored
ffmpeg builds in a single `cv2` -- which crashes the process nondeterministically.
Nothing here needs a GUI build.

### Training fetches its base weights the first time

`train-pose`'s default `model` is the bare asset name `yolo11n-pose.pt`, which
Ultralytics resolves as a path, then under its own weights directory, and
otherwise **downloads** from the `ultralytics/assets` GitHub release into the
working directory the run inherits. A machine with no network, or a queued job
that must not write outside the dataset, wants `model` given as a path to weights
that are already there. Point training fetches nothing: `polo26n.yaml` is package
data, resolved inside the fork.

The build works from a source checkout only: this directory is excluded from the
uv workspace, and its non-Python files are not part of the installed wheel.

`uv.lock` is committed and `uv sync` resolves from it. That is not bookkeeping:
an Ultralytics patch release can re-tune the detection defaults that
`runner/ultralytics_runner.py` passes explicitly for exactly that reason, so two
machines resolving the environment freshly would track the same video under one
run identifier and two sets of numbers.

## How mosaic finds it

The same ladder every other external tool uses, first match wins:

1. `MOSAIC_<TOOL>_CONDA_ENV` -- a conda environment name or prefix.
2. `MOSAIC_<TOOL>_BIN` -- the `yolo` console script. The `python` beside it in
   the same `bin/` is what mosaic runs, so pointing at any one of the
   environment's scripts names the directory the interpreter lives in.
3. `yolo` on `$PATH`.

```bash
export MOSAIC_ULTRALYTICS_BIN=src/mosaic/tracking/external/ultralytics-env/.venv/bin/yolo
export MOSAIC_POLO_BIN=src/mosaic/tracking/external/polo-env/.venv/bin/yolo
```

**Set the variable for the fork rather than relying on `$PATH`.** POLO installs
the same `yolo` and `ultralytics` console scripts as upstream, under the same
distribution name, so the third rung cannot tell the two apart: it resolves to
whichever is on the path. Point detection therefore checks what the environment
reported -- whether its `ultralytics` defines the `locate` task -- and refuses an
upstream build by name rather than running it.

The ladder resolves **placement, never identity**: which environment a tool ran
in is a property of the machine, so none of these values reaches a `run_id`, and
two machines with Ultralytics installed differently still agree on what a run is
called.

## Why the fork has an environment of its own

POLO is a full fork of Ultralytics -- it keeps every upstream task and adds
`locate` (point detection) -- and it ships under the **same distribution name**,
`ultralytics`. Installed into one environment the two cannot coexist: pip
resolves one and the other silently is not there. Two environments is exactly
what makes them coexist, which is why `runner/` is a sibling of both rather than
inside either: they run the same program, and which one runs is chosen by the
interpreter mosaic spawns.

`polo-env/pyproject.toml` carries no version floor where its sibling pins
`ultralytics>=8.4.63`, because pairing a direct git reference with a version
specifier for one distribution is unresolvable. The `uv.lock` beside it pins the
commit that was resolved, which is what the floor would have been for.

The directories are `ultralytics-env` and `polo-env`, not `ultralytics` and
`polo`, for a smaller reason: a hyphen is not a valid Python identifier, so
neither can ever be read as a package shadowing the real `ultralytics` if it
reaches `sys.path`.

## What runs inside

The directory holds three files: the two below, and an `__init__.py` whose whole
body is a docstring -- which is what lets mosaic locate the program by importing
the package without importing the module that imports Ultralytics.

- `runner/ultralytics_protocol.py` -- the request and response models, and the
  row extraction. Imported from **both** environments, so it depends on the
  standard library, numpy and pydantic and nothing else. It imports neither
  `ultralytics` nor `mosaic`.
- `runner/ultralytics_runner.py` -- the program that imports Ultralytics, and
  the only one mosaic reaches that does. Seven subcommands, each reading one JSON
  request file and writing one JSON response file:

  ```bash
  .venv/bin/python ../runner/ultralytics_runner.py probe --request req.json --out resp.json
  .venv/bin/python ../runner/ultralytics_runner.py tracker-defaults --request req.json --out resp.json
  .venv/bin/python ../runner/ultralytics_runner.py track --request req.json --out result.json
  .venv/bin/python ../runner/ultralytics_runner.py infer-pose --request req.json --out result.json
  .venv/bin/python ../runner/ultralytics_runner.py infer-points --request req.json --out result.json
  .venv/bin/python ../runner/ultralytics_runner.py train-pose --request req.json --out result.json
  .venv/bin/python ../runner/ultralytics_runner.py train-points --request req.json --out result.json
  ```

  The two inference subcommands write their predictions parquet themselves, the
  way `track` does, and convert each batch as it arrives rather than keeping the
  results: a video's worth of live tensors cannot cross the boundary, and holding
  them was a memory profile that grew with the recording.

  The response goes to a file rather than to standard output because
  Ultralytics' own logger is a stream handler on standard output -- a weights
  download, a warning, anything `verbose=False` does not suppress lands there --
  so a response parsed from that stream would be fragile. Importing torch and
  Ultralytics, by contrast, writes nothing there at all, which is why `probe`
  and `tracker-defaults` are silent from spawn to answer and mosaic gives them a
  deadline floor rather than an inactivity bound.

  Under `track` and the two inference subcommands, standard output carries one
  JSON line as soon as Ultralytics is imported and one per decoded batch after
  that, which is what mosaic's inactivity watchdog reads, what it reports position
  from, and what keeps an entry's claim refreshed through a long video. Loading the
  weights happens between the first line and the second, so the `idle_timeout` a
  run is given has to exceed a cold model load: that is the longest silence a
  healthy run contains.

`probe` reports and never refuses: it says what the environment holds -- whether
Ultralytics and `lap` import, whether this build is the fork, the version, the
known backends, the model's task and keypoint count, and **why the weights would
not load** when they do not -- and mosaic decides what to refuse, because the
refusal messages name mosaic commands and mosaic's own installation
documentation.

That last field is what lets a checkpoint from the wrong fork be refused by name.
POLO pickles its weights under a class upstream does not define, so an upstream
build fails inside `torch.load` before the task the checkpoint declares can be
read -- reporting the failure instead of raising it is what makes the refusal
that routes those weights to `infer-points` reachable at all.

`tracker-defaults` reports every backend's shipped configuration table in one
process. No run calls it: mosaic transcribes those tables so that an upstream
retune cannot silently re-mean an identifier already on disk, and this is how
the transcription is compared against the release that will run it.

## Cancelling a training run

Ultralytics cannot be interrupted inside an epoch. It reads a stop flag between
them, and the epoch that was running writes `last.pt` and appends to
`results.csv` before it ends -- so a run stopped at a boundary leaves a complete
checkpoint and a complete curve, where a killed process loses whichever epoch was
in flight.

Every other tool mosaic drives is cancelled by killing its process group, because
the unit of loss there is one video that will simply be redone. Training is the
exception, so `mosaic cancel` on a training run does something else: mosaic writes
a file inside the run root, the runner stats it at each epoch boundary, and the
run ends the way it always did. The kill is still there as a backstop, after a
grace long enough for an epoch -- a tool that ignores the file is not immortal.

Two consequences worth knowing. A cancel takes effect at the *next* epoch
boundary, so on a long epoch it is not immediate; that was equally true when
training ran in mosaic's own process. And the grace has to stay shorter than
whatever the substrate running mosaic allows -- a container runtime that SIGKILLs
the process tree on its own timer will win, and the ordering wants to be one
epoch, then mosaic's grace, then the runtime's.
