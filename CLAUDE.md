# CLAUDE.md

This file guides coding agents that change mosaic's source. If the user is *using*
mosaic (building a dataset, running the CLI, writing notebooks or analysis code) rather
than changing it, read [src/mosaic/agent/SKILL.md](src/mosaic/agent/SKILL.md) and follow
it instead of the rules below, and suggest running that work in a project directory
outside this checkout.

## Project overview

`mosaic-behavior` (imported as `mosaic`) turns per-frame pose tracks into standardized
parquet track tables, behavioral features, embeddings and clusters, classifiers,
identity models and annotated videos. It also samples frames for annotation and trains
pose models. See [README.md](README.md) and [docs/index.md](docs/index.md).

## Development environment

```bash
conda create -p ./.venv python=3.13 -y      # the path [tool.basedpyright] expects
conda activate ./.venv
conda install -c conda-forge ffmpeg av py-opencv -y
pip install -e ".[all]"
```

Install from conda before pip. `ffmpeg` and `ffprobe` must be on `PATH`. Every CI job
runs Python 3.13.

**One ffmpeg build per environment.** Two ffmpeg builds in one process crash it at a
different point on every run. [CONTRIBUTING.md](CONTRIBUTING.md#environment-notes) gives
the combinations that collide and what to install on each platform.

## Commands and definition of done

```bash
ruff check --fix && ruff format            # configured in pyproject.toml
basedpyright                               # strict; whole project or one file
pytest                                     # everything not marked slow, in parallel
pytest -m "slow or not slow"               # everything
mkdocs serve                               # or: mkdocs build --strict
python scripts/gen_docs_reference.py --write
```

- basedpyright has a large existing backlog. New code type-checks cleanly.
- Test selection and placement: [CONTRIBUTING.md](CONTRIBUTING.md#running-the-tests) and
  [tests/CLAUDE.md](tests/CLAUDE.md).
- `docs/reference/` is generated from the registries and the Typer app. Do not edit it.
  After changing a registry or a CLI command, run the script with `--write`
  ([CONTRIBUTING.md](CONTRIBUTING.md#environment-notes) gives its install) and commit
  the result. The docs workflow runs it with `--check`.
- Do not run `mkdocs gh-deploy`. Pages deploys from `.github/workflows/docs.yml`.
- Before reporting work done, run `basedpyright` and `pytest -m "slow or not slow"`. CI
  skips the type check and runs the unmarked slow tests only weekly. An agent runs the
  first through `.claude/skills/tools/mosaic-check`, which compares each changed file
  with a per-file baseline, and the second on the task machine through `mosaic-suite`
  (the `mosaic-testing` skill). A hook refuses a full suite on this machine.

## Layering

`core` is the foundation: the data model, schemas, the pipeline engine and media I/O.
`behavior` and `tracking` import `core` and do not import each other. They exchange data
only through files on disk. The deprecated `behavior.tracking` alias in
`behavior/__init__.py` is the one exception
(`tests/meta/test_behavior_tracking_layering.py`).

`mosaic.agent` is an adapter beside `mosaic.cli`. It may import `core`, `behavior` and
`tracking`, and no module under those three imports it
(`tests/meta/test_agent_layering.py`).

`core` imports upward only inside functions, at the sites listed in
[src/mosaic/core/CLAUDE.md](src/mosaic/core/CLAUDE.md#upward-imports). Any other upward
import is a layering break.

## Registries

New functionality is almost always a new plugin.

| Registrar | Registry | Lives in |
| --- | --- | --- |
| `@register_feature` | `FEATURES` | `behavior/feature_library/registry.py` |
| `register_track_converter` | `TRACK_CONVERTERS` | `core/track_converter.py` |
| `register_label_converter` | `LABEL_CONVERTERS` | `core/label_converter.py` |
| `@register_op` | `OPS` | `core/pipeline/ops.py` |
| `@register_media_step` | `MEDIA_STEPS` | `core/media/preprocess/registry.py` |

Label converters register by an explicit call to `register_label_converter` (see the
behavior guide). The `MediaStepSpec` union (`core/media/preprocess/specs.py`) lists
every registered step (`tests/core/media/preprocess/test_media_steps.py`).

## Cross-cutting invariants

- **One parquet writer.** Outputs go through `write_parquet_atomic`
  (`core/pipeline/writers.py`) or `atomic_savez`. A `to_parquet` call on a final path
  fails `tests/core/pipeline/test_tracks_table_atomicity.py`.
- **Feature output goes through `run_feature()`** (`core/pipeline/run.py`), which owns
  the index, the `run_id` and the layout.
- **Params are Pydantic** (`core/params.py`). Add a param as a typed field with a
  default. `HASH_EXCLUDE` marks throughput knobs only. A field that changes meaning
  bumps the feature's `version` (`tests/core/test_params.py`). Do not pass a raw
  `dict[str, Any]` across a feature boundary. Use a Pydantic model or a dataclass.
- **Determinism.** The same params and inputs give the same `run_id` and outputs. Seed
  every random state and keep results independent of filesystem order.
- **A run reports its losses.** A failed entry emits an `entry_error` run-log event and
  is listed in `Result.failed_entries`, and the CLI then reports `"status": "partial"`.
  Losing every entry raises `AllEntriesFailed` (`core/pipeline/run.py`,
  `tests/core/pipeline/test_partial_entry_failure.py`).
- **`runlog.TERMINAL_STATUSES` is `finished`, `failed`, `cancelled`.** Do not add
  `partial`, a refusal or `orphaned`. `alive` and `phase` are events, not statuses.
  Three repositories read the set, and mosaic-api's sweeper reaps it
  (`tests/core/pipeline/graph/test_pipeline_partial_failure.py`).
- **A read writes nothing.** A read-only verb or a dry run opens its dataset with
  `load_dataset(..., read_only=True)` (`cli/_context.py`), and a dry run takes no
  lock (`index_lock_unless_dry_run`). A new read joins the cases of
  `tests/cli/test_cli_read_only.py`.

## Docstrings

Google style. A public object is documented only by its docstring, because the site has
no rendered API reference.

## Where the rest lives

| When you are | Read |
| --- | --- |
| Changing a dependency, an extra or `pyproject.toml` | [Dependency invariants](CONTRIBUTING.md#dependency-invariants) |
| Changing CI, or a test's marker or job | [CI](CONTRIBUTING.md#ci) |
| Looking for the module that owns something | [Package map](CONTRIBUTING.md#package-map) |
| Changing a repository skill or hook | [Agent skills and hooks](CONTRIBUTING.md#agent-skills-and-hooks) |
| Changing `runlog.py`, a lock or an unpinned `Result` | [core.pipeline guide](src/mosaic/core/pipeline/CLAUDE.md#cross-package-rules) |
| Changing entry names or a `core` upward import | [core guide](src/mosaic/core/CLAUDE.md) |
| Publishing a variant's tracks, handing a tool a file, naming a model, or hashing a revision | [tracking guide](src/mosaic/tracking/CLAUDE.md#cross-package-rules) |

## Area guides

Each loads when an agent works in its directory.

- [src/mosaic/core/CLAUDE.md](src/mosaic/core/CLAUDE.md): dataset, manifest, scan
  sources, track schemas, entry names, imgstore, media I/O.
- [src/mosaic/core/pipeline/CLAUDE.md](src/mosaic/core/pipeline/CLAUDE.md): inventory,
  tracks index, label series, locks, run ids, alignment, claims, compositions.
- [src/mosaic/core/pipeline/graph/CLAUDE.md](src/mosaic/core/pipeline/graph/CLAUDE.md):
  the recipe graph.
- [src/mosaic/behavior/CLAUDE.md](src/mosaic/behavior/CLAUDE.md): feature protocol,
  label converters, the sandboxed kpms and VAME runners.
- [src/mosaic/tracking/CLAUDE.md](src/mosaic/tracking/CLAUDE.md): trackers, Ultralytics
  environments, frame axis and joins, model references, training provenance.
- [src/mosaic/agent/CLAUDE.md](src/mosaic/agent/CLAUDE.md): the users' agent guide and
  the MCP server: its import budget, tool contract and standard output.
- [tests/CLAUDE.md](tests/CLAUDE.md): agent detail for the suite.

User documentation: [docs/installation.md](docs/installation.md),
[docs/dataset.md](docs/dataset.md), [docs/guides/](docs/guides/),
[docs/concepts/](docs/concepts/). Contributor workflow:
[CONTRIBUTING.md](CONTRIBUTING.md).
[src/mosaic/agent/SKILL.md](src/mosaic/agent/SKILL.md) guides the agents of mosaic's
users. Update it when a change alters what a user does.
