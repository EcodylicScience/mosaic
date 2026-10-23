# mosaic.behavior

This package owns the registered features, the label converters, the identity networks
and the media features, plus the runners that drive keypoint-MoSeq and VAME in their
own environments.

## Feature protocol

- **A feature is a class with four methods and seven attributes.** `load_state`, `fit`,
  `apply` and `save_state`, plus `name`, `version`, `parallelizable`, `scope_dependent`,
  `accepts_overlap`, `emits` and `consumed_roots`. The protocol and its optional hooks
  are `Feature` in `core/pipeline/types/feature.py`.
  `tests/behavior/feature_library/test_feature_conformance.py` fails on a feature that
  leaves `accepts_overlap` or `emits` undeclared.
- **The column and prerequisite declarations** (`requires`, `provides`,
  `prerequisites`) are checked by the graph, whose rules are in
  `src/mosaic/core/pipeline/graph/CLAUDE.md`.
  `tests/behavior/feature_library/test_column_contract_conformance.py` runs every
  registered feature against them. New code imports an optional package through its
  prerequisite (`OptionalModule.load()` in `src/mosaic/optional_dependency.py`), and
  the catalog and the run then name one extra.
- **Keypoints are optional in `mosaic_v1`.** A feature defined on them names them in
  `requires` or checks for them in `apply`, and refuses a table without them in an
  error naming the feature. A feature that only prefers them, such as `egocentric-crop`
  or `overlay`, falls back to `X`/`Y`, the body center.
- **A feature registers only when its module is imported.** `@register_feature` keys
  `FEATURES` by class name (`feature_library/registry.py`), and a recipe or the CLI
  finds it by its `name` slug. A new module joins the imports of
  `feature_library/__init__.py`, or `visualization_library/__init__.py` for media.
- **Two flavors.** A per-frame or per-sequence feature is a stateless transform. A
  global feature fits over a collection of sequences, then applies per sequence
  (`global-kmeans`, `xgboost`, `arhmm`, `kpms`, `vame`, `feral`, the identity models).
  `docs/reference/features.md` lists every registered feature.
- **A choice a converter could make is a feature, and its parameters enter the run
  id.** `heading` derives `ANGLE` under a chosen `method`. `scale-to-cm` hashes an
  explicit `cm_per_pixel` but not the scale it reads from the media index, and its
  module docstring says what a recalibration then serves.

## Category

- **`category` is an optional class attribute outside the protocol.** The values in use
  are `per-frame`, `global`, `summary` (per-sequence aggregations such as
  `frame-aggregate`), `tag` (`id-tag-columns`) and `media`. It colors the pipeline
  diagram (`core/pipeline/viz.py`) and fills a column of `mosaic features list`.
- **It also places a run.** A feature with `resource_class = "gpu"` and category
  `global` is offered to the GPU training lane (`lane_for` in
  `core/pipeline/graph/lanes.py`). A GPU feature that fits a model declares `global`.
- **Every `global` and `media` feature declares `costly_fit`**, a bool outside the
  protocol (`test_every_global_and_media_feature_declares_costly_fit`). `True` when a
  refit costs a GPU or hours. The superseded-run cleanup keeps every run of such a
  feature, and the catalog's `Declaration.costly_fit` carries it.
- **Every `global` or `scope_dependent` feature declares `fits`**, a bool outside the
  protocol (`test_every_global_or_scope_dependent_feature_declares_fits`). `True` when
  the feature learns something from its inputs, whether its scope names its run
  (`extract-labeled-templates`, `arhmm`) or not (`xgboost`, the `global-*` fitters).
  `evaluate-labels` depends on scope and learns nothing, so it declares `False`. A feature
  that fits only without a `model` or with `weights="fit"` declares `True` on the class and
  narrows `self.fits` in `__init__`, beside `self.scope_dependent` when that narrows too
  (`test_a_feature_that_clears_scope_dependent_per_instance_clears_fits`). A
  `scope_dependent` fitter with a `model` param narrows `fits` and leaves
  `scope_dependent` alone when its run id must not change (`arhmm`, `kpms`, `vame`, the
  identity models; `test_a_fitting_feature_given_a_model_param_narrows_fits_per_instance`),
  or the held-out guard refuses the apply step over the test group. The catalog's
  `Declaration.fits` and `StepTypeDetail.fits` carry the class value. It is not hashed.
- **A rendered artifact is `media`.** `overlay`, `egocentric-crop` and
  `interaction-crop-pipeline` live in `visualization_library/` and use the feature
  protocol and its caching. Their videos and crops are addressed by run id. mosaic has
  no `visualization` category.

## Label converter registration

`register_label_converter` is called in `label_library/__init__.py`, which imports each
converter module and rebinds the class to the registrar's result. Importing a converter
module registers nothing. A new converter adds a call to that file.

## Imports

- **torch is imported lazily under `behavior`.** `model_library/__init__.py` imports
  no network, each network imports torch inside its methods, and every `nn.Module`
  subclass is defined in a factory function.
  `tests/meta/test_behavior_import_is_torch_free.py` enforces it.
- **`feral` runs in process, unlike the sandboxed runners below.** `feral_feature.py`
  declares the `FERAL` prerequisite but imports `feral` directly inside its methods.

## Sandboxed runners in `feature_library/external/`

- **mosaic does not import keypoint-MoSeq or VAME, or declare either as a dependency.**
  For keypoint-MoSeq the process boundary is a license requirement (see NOTICE).
  `tests/meta/test_external_tool_separation.py` checks the imports, the declared
  dependencies, and that no runner program imports mosaic.
- **One wire serves both runners.** `external/runner_wire.py` imports only the standard
  library and pydantic, and its module docstring gives the messages and the exchange of
  arrays as files. `kpms_protocol.py` and `vame_protocol.py` declare the payloads, and
  `kpms_server.py` and `vame_runner.py` are the programs. A program calls
  `claim_stdout()` first in `main()` and imports its tool only inside a handler.
- **`RunnerProcess` (`feature_library/runner_client.py`) is mosaic's side.** It starts
  the runner at the first request, keeps it across requests and starts a new one after
  any exit. A feature that owns one implements `close()` (`ClosesResources`).
- **A cancel stops the runner.** A fired `cancel_token` sends SIGTERM to the runner's
  process group, then SIGKILL after 5 s, and the request raises `Cancelled`. Both tools
  discard a fit's state on any exit. A signal loses what a stop request would lose.
  `tests/behavior/feature_library/test_runner_client.py` covers it.
- **The kpms license is checked per request, and in `load_state` when it finds a model
  to apply.** `KpmsFeature._request` calls `check_license_accepted()`
  (`MOSAIC_KPMS_LICENSE_ACCEPTED=1`). `__init__` skips the check because
  `mosaic reconcile` constructs features without running them.
  `tests/behavior/feature_library/test_kpms_license.py` tests it.
- **Each runner's interpreter comes from a `ToolEnv` ladder** (`KPMS_ENV` in `kpms.py`,
  `VAME_ENV` in `vame_feature.py`). The user builds both environments from
  `external/README.md` and `external/vame-env/README.md`. Both are excluded from the uv
  workspace and share one basedpyright execution environment (`pyproject.toml`).

## Notebooks

The notebooks under `notebooks/` are examples that the test suite does not run. A
change to a feature name, parameter or public signature updates every notebook that
uses it.
