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
conda create -p ./.venv python=3.12 -y      # the path [tool.basedpyright] expects
conda activate ./.venv
conda install -c conda-forge ffmpeg av py-opencv -y
pip install -e ".[all]"
```

Install from conda before pip. Conda-forge's `av` and `py-opencv` then satisfy pip's
requirements, and pip installs neither wheel. A `uv venv` cannot take conda packages.
`ffmpeg` and `ffprobe` must be on `PATH`. Every CI job runs Python 3.12.

**One ffmpeg build per environment.** Two ffmpeg builds in one process crash it at a
different point on every run. Two `cv2` wheels (`opencv-python` beside
`opencv-python-headless`) collide on every platform, and `tests/conftest.py` refuses to
start with them. The `av` wheel beside a `cv2` wheel collides on macOS whichever `cv2`
flavor is installed, and a dependency edit cannot fix it. Use conda on macOS. On Linux,
CI's wheels against an apt `ffmpeg` work. The Ultralytics environments declare headless
OpenCV and override the GUI wheel away, and
`tests/tracking/ultralytics_track/test_ultralytics_preflight.py` checks their locks.

To work against an unreleased `mosaic-media`, install it editable over the released
wheel ([CONTRIBUTING.md](CONTRIBUTING.md#environment-notes)). A `[tool.uv.sources]`
path breaks `pip install -e .`, because pip does not read that table.

## Dependency invariants

- A bare `pip install -e .` is a complete analysis install.
- `deep-learning` (`torch`, `timm`) serves mosaic's PyTorch code, the heatmap localizer
  and the identity models. The `feral` and `lightning-action` packages also require
  PyTorch.
- `all` is `mosaic-behavior[deep-learning,faiss]` and resolves only non-AGPL
  distributions. It leaves out `lightning-action` (its DALI requirement installs only
  with CUDA), `movement` (heavy and single-purpose) and `feral` (its exact pins
  downgrade mosaic's floors).
- `lightning-action` stays uncapped. Every release requires the DALI sdist that builds
  only with CUDA. A version cap leaves the extra on an older release with the same
  requirement. The `lightning-action` CI job therefore installs the package with
  `--no-deps`, beside the requirements its import needs.
- Retired extras stay retired, without aliases
  (`tests/meta/test_optional_dependency_messages.py`).
- `feral>=1.0,<2`: mosaic imports FERAL symbols outside its `__all__`, and the `feral`
  CI job fails on a release that moves one.
- `imgstore` is in the `test` dependency group, because only the suite writes stores.
- `mcp` (`mcp>=2.3,<3`) is outside `all`. The `test` CI job installs `.[mcp]`, and
  the extra stays out of the `test` group, which the feral job installs over FERAL's
  pins. `mosaic.agent.mcp.server` calls `MCP_SDK.load()` before importing
  `_binding`, the one module that imports the SDK.
- Ultralytics, POLO, keypoint-MoSeq and VAME run only as separate programs in
  environments the user builds. No extra or dependency group declares them. The
  [behavior](src/mosaic/behavior/CLAUDE.md) and [tracking](src/mosaic/tracking/CLAUDE.md)
  guides give the runners and the separation test. keypoint-MoSeq's non-commercial
  license cannot combine with the AGPL. Do not bundle it into a packaged installer.
- Model weights come from their hubs or from the user. [NOTICE](NOTICE) states their
  terms.

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
  skips the type check and runs the unmarked slow tests only weekly.

## CI

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs six jobs on a push to
`main` and on pull requests. The five test jobs install the `test` dependency group.

- `test`: `-m "not slow and not tracker and not identity and not feral and not
  lightning_action"`.
- `identity`, `feral`: every test of that marker, in a `deep-learning` or `feral`
  environment.
- `lightning-action`: every `lightning_action` test, which fits the real package on CPU.
- `tracking`: every `tracker` test against built Ultralytics and POLO environments.
- `lint`: `ruff check` and `ruff format --check` over `src`, `tests` and `tools`.

[`.github/workflows/slow.yml`](.github/workflows/slow.yml) runs the other slow tests
weekly and on dispatch, against a built VAME environment. Each test runs in exactly one
job (`tests/meta/test_pytest_config.py` reads both workflows). A change that breaks only
an unmarked slow test goes green on push.

## Layering

`core` is the foundation: the data model, schemas, the pipeline engine and media I/O.
`behavior` and `tracking` import `core` and do not import each other. They exchange data
only through files on disk. The deprecated `behavior.tracking` alias in
`behavior/__init__.py` is the one exception. No test checks this rule.

`mosaic.agent` is an adapter beside `mosaic.cli`. It may import `core`, `behavior` and
`tracking`, and no module under those three imports it
(`tests/meta/test_agent_layering.py`).

`core` imports upward only inside functions, at the sites below. Any other upward
import is a layering break.

- `core/__init__.py` `__getattr__` re-exports `register_feature` on first access, which
  keeps `core` and `behavior` out of an import cycle.
- `label_converter.ensure_label_converters_registered()` imports
  `behavior.label_library` to fill `LABEL_CONVERTERS`.
- `core/pipeline/graph/resolve.py` reads `FEATURES` (`tests/meta/test_graph_imports.py`)
  and calls `mosaic.tracking.register_ops` to fill `OPS`.
- `core/pipeline/reconcile_features.py` `_feature_class` looks a feature class up by
  name in `FEATURES`.
- `core/pipeline/run.py` imports a feature's module by its `__module__` to rebuild the
  feature in a worker process.

`tracking` adds its runs to `core`'s views through `register_inventory_contributor`
(`tests/core/pipeline/inventory/test_inventory_contributors.py`) and
`register_reconcilable_index`. `mosaic.tracking.register_ops` registers both, and
importing `mosaic.tracking` registers neither.

## Package map

```
src/mosaic/
  cli/                          every `mosaic` verb (Typer)
  agent/                        the guide for users' agents; `agent/mcp/` the MCP tools
  runlog.py                     append-only JSONL run-log; stdlib-only leaf
  lazy_exports.py               the body of a package `__getattr__`; stdlib-only leaf
  core/                         Dataset, manifest, schema, Params, Scope, converters
  core/pipeline/                run_feature, indexes, locks, inventory, ops, placement
  core/pipeline/graph/          a pipeline as a JSON recipe: validate, plan, run a step
  core/media/                   frame read/decode/encode; preprocess media steps
  behavior/feature_library/     registered features; sandboxed kpms and VAME runners
  behavior/label_library/       label converters (BORIS, CalMS21)
  behavior/model_library/       identity networks
  behavior/visualization_library/  overlay renderer and the media features
  tracking/ops/                 the @register_op layer behind `mosaic run --kind`
  tracking/common/              the run loop shared by every tracker
  tracking/{trex,sleap,litpose,ultralytics_track}/  one package per tracker
  tracking/external/            the Ultralytics and POLO environments and runner
  tracking/pose_training/       annotation prep and pose-model training
  tracking/frame_extraction/    frame sampling for annotation (`extract_frames`)
```

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
- **An entry name is one path component.** `validate_entry_name` (`core/helpers.py`)
  refuses `/`, `\` and NUL at write boundaries and on no read path
  (`tests/core/pipeline/test_tracks_index.py`).
- **A media variant is consumed by name and published in source space.**
  `publish_tracks_table` (`tracking/common/bridge.py`) maps a table through
  `to_source_space` (`core/pipeline/placement.py`,
  `tests/core/media/preprocess/test_placement_map_back.py`).
- **A tool is handed only a file it decodes.** `refuse_undecodable_codec`
  (`tracking/common/tool_input.py`) applies the tool's `ToolDecoder`
  (`core/pipeline/tracking_roots.py`, `tests/tracking/common/test_decoder_probe.py`).
- **A run reports its losses.** A failed entry emits an `entry_error` run-log event and
  is listed in `Result.failed_entries`, and the CLI then reports `"status": "partial"`.
  Losing every entry raises `AllEntriesFailed` (`core/pipeline/run.py`,
  `tests/core/pipeline/test_partial_entry_failure.py`).
- **A run reports its coverage in its run-log.** `RunLogSnapshot` (`runlog.py`) folds
  `entries_written`, `cache_hit` and `tracks_variant`. Emit them before the terminal
  event (`tests/core/pipeline/test_job_contract.py`).
- **`runlog.TERMINAL_STATUSES` is `finished`, `failed`, `cancelled`.** Do not add
  `partial`, a refusal or `orphaned`. `alive` and `phase` are events, not statuses.
  Three repositories read the set, and mosaic-api's sweeper reaps it
  (`tests/core/pipeline/graph/test_pipeline_partial_failure.py`).
- **`track_universe.current_run_id` is the one rule for an unpinned `Result`**
  (`tests/core/pipeline/test_unpinned_result_rule.py`).
- **A lock the filesystem refuses raises `IndexLockUnsupported`**
  (`core/pipeline/index_lock.py`) and is not degraded to an unlocked write. The lock
  serializes writers on one mount only
  (`tests/core/pipeline/test_index_lock_capability.py`).
- **A read writes nothing.** A read-only verb or a dry run opens its dataset with
  `load_dataset(..., read_only=True)` (`cli/_context.py`), and a dry run takes no
  lock (`index_lock_unless_dry_run`). A new read joins the cases of
  `tests/cli/test_cli_read_only.py`.
- **Name a model by its run id.** Across datasets, link the library with `libraries:`
  instead of a path into another dataset's `models/`. The reference string enters the
  inference identity (`resolve_model` in `tracking/model_refs.py`,
  `tests/core/test_library_links.py`).
- **A revision's content is hashed, and its number is not**
  (`tests/tracking/ops/test_prepare_training_data.py`).

## Docstrings

Google style. A public object is documented only by its docstring, because the site has
no rendered API reference.

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
