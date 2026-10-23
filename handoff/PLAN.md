# Plan: S1–S6 of the agent interface after the CalMS21 demo

Source brief: `mosaic-planning/mosaic/plans/2026-10-21-the-agent-interface-after-the-calms21-demo.md`
(written against `main` 6411b6e6). Paths below are relative to the mosaic repo.

## Context

Two unattended agent runs scored well on CalMS21 (Task 1 test macro F1 0.870, Task 3 0.454), but only
because the operator had scaffolding mosaic lacks: a train-only dataset, a held-out scoring script,
a Docker fence, and hand-written 39- and 71-step recipes that work around one defect (scope-dependent
fits refit when a request widens). This plan moves that scaffolding and the gap fixes into mosaic.

**Order.** S1 → S2 → S3 are one design (fold table, answer steps, run-id check) and go first, in
one session. S6, S4 and S5 are independent of them; do S6, then S4, then S5 (S5 is the largest).
Each slice lands as its own commits on `main`-bound branches; nothing is pushed unless asked.

**Decisions taken (user answers, 2026-10-21):**
- Sealed groups: their labels stay out of the manifest; `pipeline score` adds them on a copy.
- Manifest: `splits:` is a new top-level key with **no `MANIFEST_VERSION` bump** (code's own rule,
  `continuous_groups` precedent: `manifest.py:970`, `test_adding_it_did_not_move_the_manifest_version`).
  Consequence to document: an older mosaic ignores `splits` and does not enforce roles.
- S5 is **one `manage_dataset` tool with an `action`**. The 15-tool shared cap is Jacob's own setting
  and may be raised when needed. See "Cap" under S5/S6: S6's `patch_recipe` (15th) plus
  `manage_dataset` (16th) exceeds it, so raise `SHARED_TOOL_LIMIT` to 16 in whichever slice lands
  second, unless `manage_dataset` can live in a local-only tuple that the cap does not count.

## The contract the plan must deliver: fit once on a chosen set, then only apply

Wanted (Jacob, 2026-10-21): a fit (PCA, features, classifier) is made on a specified set of sequences
(train, or all available). Test sequences, and sequences that arrive later, never trigger a refit; they
are projected through / scored by the stored fit, so their result is well defined. Checked against
the code, per feature:

| Feature | Today | After this plan |
| --- | --- | --- |
| `global-scaler`, `global-kmeans`, `global-ward`, `global-tsne`, `xgboost`, `feral` | Already scope-free; fit set = the `templates` run; new sequences reuse the stored fit | unchanged (S1 guard covers the `extract-*templates` step that sets the fit set) |
| `pair-posedistance-pca` | `model` param exists (0.3): fit step on train + apply step `model: {step}`; the apply's id is scope-free (`scope_dependent = model is None`, `pairposedistancepca.py:234`). A single step over a wider scope **refits** | S1 guard refuses a fitting PCA step that would include a `test`/`sealed` group; the two-step pattern is the supported path (documented) |
| `sequence-classifier`, `lightning-action`, `probability-ensemble` (fit weights) | Scope-dependent while fitting: widening **refits**; the apply needs a third step | **S2:** scope-free; one step fits on its templates' train entries, new/test entries use the stored final model, no refit |
| `arhmm`, `kpms`, `vame`, `identity-*` models, `fit-label-map` | `scope_dependent = True` on the class even when `model` is given, so an apply over new sequences is a new run id that hashes the scope (model is not refit, but the identity is scope-dependent) | **New S2b** (below): per-instance `scope_dependent = (model is None)` like the PCA, after a per-feature spike |

Two honest qualifications:
- A cross-validated fit gives **out-of-fold** rows for the *training* entries (that is the point of
  `cv_folds`) and final-model rows for every other entry. Test/new entries are always final-model rows.
- A scope-free fit does not move when a *training* entry is re-converted or edited (same as `xgboost`
  today). New sequences never move it; changing the fit set means a new `templates`/fit step.
- S2's earlier note "an upstream `scope_dependent` input still moves the id" is exactly the PCA hole:
  a classifier downstream of a single wide-scope PCA step still changes id on widening. The fix is the
  PCA two-step pattern (fit on train, apply `model: {step}` to all), which S1's guard now enforces for
  test groups.

**Definition of done for every slice** (root `CLAUDE.md`): `ruff check --fix && ruff format`,
`basedpyright` clean on new code, `pytest -m "slow or not slow"`,
`python scripts/gen_docs_reference.py --write` when a registry/CLI/Declared text changes, update
`src/mosaic/agent/SKILL.md` and `CHANGELOG.md` "Unreleased". Do not edit `docs/reference/`.

---

## S1. A held-out protocol

### Findings that shape it
- `scope_dependent` is **not** "fits". Scope-free fitters (`xgboost`, `global-*`, `feral`) learn but
  are not scope-dependent; `evaluate-labels` is scope-dependent and learns nothing
  (`evaluate_labels.py:167`). So a separate declaration is required.
- Leak guard only needs to cover *stream fitters* (a fitter whose training set is its request scope).
  Scope-free fitters take their data from a pinned `templates`/`model` artifact, so the leak is at the
  templates step (`extract-labeled-templates`, `scope_dependent` + fits). Applying a fitted model to the
  test group must stay allowed (`SKILL.md:326` documents it).
- `_step_scope_problems` (`graph/plan.py:846`) returns early when there are no step scopes; the guard
  must not, because the dangerous case is a whole-dataset request with none.

### Work
1. **Roles in the manifest.** `splits: {group: train|test|sealed}` as a field on `DatasetManifest`
   (`core/manifest.py`), copying `continuous_groups` end to end: field, validator, payload write
   (`manifest_payload`), `manifest_header` doc paragraph, `Dataset.splits` / `set_splits` via
   `Dataset.mutate_manifest` (`dataset.py:1191`), `dataset_summary.py` + `cli/describe.py`. Update
   `tests/data/dataset_summary_golden.json`. Add a small operator verb `mosaic splits list|set|remove`
   (modelled on `cli/tags.py`). **No MCP tool may write `splits`** (an agent could unseal otherwise);
   say so in `agent/CLAUDE.md` and S5.
2. **`fits` declaration.** Optional class attribute `fits: bool`, `costly_fit`-style (not a
   `Feature` protocol attribute): read by `_feature_declaration` (`graph/resolve.py:439`) into
   `Declaration.fits` (`compatibility.py`), exposed in `step_type_detail` (`wire.py`) and
   `cli/_render.py`. `fits=True`: `extract-templates`, `extract-labeled-templates`, `fit-label-map`,
   `arhmm`, `kpms`, `vame`, the three identity models; set per instance in `__init__` (beside
   `scope_dependent`) for `pair-posedistance-pca`, `sequence-classifier`, `lightning-action`,
   `probability-ensemble` (fit mode). `fits=False`: `evaluate-labels`. A conformance test
   (beside `test_every_global_and_media_feature_declares_costly_fit`) requires every
   `scope_dependent` feature to declare `fits`. Not a hashing change; `test_feature_run_id_golden.py`
   must not move. Update `behavior/CLAUDE.md` (attribute list) and `core/pipeline/types/feature.py` doc.
3. **Planner guard.** One helper `refit_set(plan_inputs, groups)` in `core/pipeline/graph/` returns the
   steps that would refit if `groups` joined their target: `scope_dependent` steps whose step scope
   would select the group (`Scope.selects` with a placeholder sequence; `sequences` selectors are
   unknowable → conservative), propagated along `edges()` to steps whose parent moved, marked
   `fits` vs score-only. Used three ways: (a) the guard in `plan_pipeline` (`plan.py:528`): a
   `fits and scope_dependent` step whose target intersects a `test`/`sealed` group is a `Problem`
   (so `plan_recipe` shows it and `submit_pipeline`/`RecipeRefused` refuses); roles read from
   `ds.manifest.splits` (add a roles field to `Plan`/`plan_wire`). (b) the whole-dataset caution in
   `_plan_cautions` (`serialize.py:203`) names the groups and roles it would cover. Keep the current
   caution text pinned by `test_a_whole_dataset_submission_asks_nothing_and_its_plan_cautions_it` or
   change it deliberately. (c) the score verb's run-id check and `assume_added` (below).
4. **`assume_added` stability check.** `plan_recipe` gains `assume_added: [group]`; the result
   reports refit vs re-score steps **through the existing `cautions` strings** because the
   `plan_recipe` output schema is at 8103/8192 bytes (headroom ~89) — a new `PlanWire` field fails
   `test_mcp_server.py:193`. Input schema has room. Same `refit_set` function.
5. **`mosaic pipeline score`** (new `cli/pipeline_score.py` registered at `cli/pipeline.py:1255`,
   core logic in `core/pipeline/graph/score.py` so the API/agents can reuse it). Port
   `SB/score_heldout.py` minus its CalMS21-specific parts. Flags:
   `--heldout <file | group>`, `--answers FILE` (`{"models": {name: [step ids]}}` for now; the
   [approach issue] replaces it later), `--protocol` (perspective, `average_over`),
   `--out DIR` (copy of the study; refuse existing unless `--reuse`), `--expect-md5`, `--dry-run`,
   `--step-scope`, `--expect-plan`. Steps: copy study → plan recipe and record
   `{step: (run_id, status)}` → `Dataset.add_scan_source` (tracks + labels) → `scan_tracks`/`scan_labels`
   → `convert_all_tracks`/`convert_all_labels` called with **the recipe's own convert-step params**
   (variant ids derive from them; CLI defaults would mint a different variant) → add one
   `evaluate-labels` step per answer (`labels: {"step": convert-labels id}`) with a step scope for the
   held-out group → plan again, compare `Plan.run_ids` for the original step ids and fail with
   `RUN IDS CHANGED` listing them (exit 1; `--dry-run` stops here, takes no lock via
   `index_lock_unless_dry_run`) → `submit_request` + `run_pipeline` → read `metrics.json`
   (`EvaluationReport`) → write `test_scores.json` + a human table to `--out`. Works on a copy, so the
   scores never enter a resource the agent can read (`mcp/resources.py:344-377` serves any
   `evaluate-labels` report). When `--heldout` names a group already in the dataset with role `test`,
   skip the add-source/convert half. Use the explicit convert calls until the
   [convert issue] is fixed. Add to `tests/cli/test_cli_read_only.py` (`--dry-run`).
6. **`score_sealed` confirmation.** New `ConfirmationCondition` + `CONFIRMATION_CONDITIONS` entry
   (`agent/mcp/spec.py:289-327`), added to `submit_pipeline.confirm_when` (`:1316`), computed in
   `_submission_findings` (`agent/mcp/local.py:1256`): an `evaluate-labels` step whose target hits a
   `sealed` group. Headless hook declines by default for free (`confirm_hook.py`,
   `cli/agent.py:281`); update `--approve` help (`cli/agent.py:~323`) and `test_tool_spec.py:143/163`,
   `test_confirmation_gate.py`, `test_cli_agent.py:~418`. **Honest note:** with sealed labels outside
   the manifest this condition is defensive (reachable if an operator or S5 later puts sealed rows
   in the dataset), not the main protection.
7. **Docs/guide.** `SKILL.md` (~299-332: train/test + sealed rule), `docs/agents.md`
   (confirmation conditions list), `docs/dataset.md` (new `splits` section beside `continuous_groups`,
   :277), `docs/concepts/pipelines.md`, `docs/guides/pipelines/chain-steps.md`,
   `src/mosaic/core/CLAUDE.md` (manifest), `graph/CLAUDE.md` ("Planning").

### Tests
Manifest round-trip + unknown-key + no-version-move (`test_dataset_manifest.py`); guard cases on the
existing `split` fixture (`test_pipeline_step_scopes.py`: train `s1,s2`, test `s3`) — fitting step
including test fails, `evaluate-labels` on test passes, applying a model to test passes, whole-dataset
caution names groups; `refit_set` unit tests; `assume_added`; `score` end-to-end with
`tests/helpers/tracks.py:calms21_dataset`; `test_guide_recipes.py` stays green (roles are opt-in).

### Acceptance (operator-run on kn-cgpu02)
Rerun the Task 1 prompt on a dataset with `splits` set and the test file outside the manifest;
add it only through `mosaic pipeline score`. A fitting step scoped to include the test group fails to
plan. No operator script.

---

## S2. One fit, apply and cross-validation contract

### Findings
- Feasible: training rows come only from `train_entries_of_templates` ∩ labeled entries;
  non-train stream entries are skipped before any state is touched (`sequence_classifier.py:716-718`),
  folds depend only on train entries (`deal_folds`, :595), manifest order is a pure sort. `xgboost`
  is the precedent. Apply already uses the fold model for fold entries and the full model otherwise
  (`out_of_fold_choice`, `helpers.py:803-841`).
- **Gap:** a scope-free fit first invoked over a scope that lacks train entries fails
  (`refuse_entries_outside_scope`). Decision: keep it a **loud refusal** with reworded remedy; no
  engine hook. Also an upstream `scope_dependent` input (`pair-posedistance-pca`) still moves the id
  through `_inputs`; S2 cannot fix that (document).
- Unreleased code (`b0ff600c`, `b45d6e71` have no tag) but still bump versions per the params rule;
  goldens move either way.

### Work
0. **Spike first (gates the rest).** Test: fit the same `sequence-classifier` over scope=train and
   scope=train+test via the stream; assert equal bundles/`cv_folds.parquet`
   (pattern: `test_two_fits_of_one_seed_are_equal...`, `test_sequence_classifier.py:417`). Check
   that train entries arrive in the same relative order (`build_manifest`/`iter_manifest`,
   `core/pipeline/manifest.py`) and that `_make_filter_factory` (`run.py:374`) filters match. If it
   fails, switch to the fallback (item 7).
1. **Scope-free identity.** `scope_dependent = False` at class level for `sequence-classifier`,
   `lightning-action`, `probability-ensemble` (fit weights go scope-free too — `fit_entries` come from
   the members' `cv_folds.parquet`, otherwise a blend changes id on widening). Delete the
   `__init__` assignments (`sequence_classifier.py:497`, `lightning_action_feature.py:423`,
   `probability_ensemble.py:322`), fix module/class docstrings and Declared text
   (`sequence_classifier.py:12-17,329-331`, `lightning_action_feature.py:279-288`). Bump versions;
   regenerate `tests/data/feature_run_id_golden.json` (`MOSAIC_UPDATE_GOLDEN=1`). Exempt these from the
   AST check `test_stream_fitting_features_declare_scope_dependent`
   (`tests/core/pipeline/test_hashing_rules.py:88-104`) via a declared marker, not an allowlist of names.
   `fits=True` (S1) stays declared so S1's guard sees the templates step, not these.
2. **Invert tests that pin the old behavior:** `test_sequence_classifier_pipeline.py:138`,
   `test_lightning_action_pipeline.py:213-249`, `test_lightning_action_feature.py:791-801`,
   `test_probability_ensemble.py:259-262`, `test_guide_recipes_torch.py:65,84`. Add: widening creates no
   new run id; a cv run gives out-of-fold rows for train entries and final-model rows for others.
3. **Ensemble member check.** Fixed/uniform weights never need a fold table (removes the demo
   blocker even for genuinely mixed members); `weights: "fit"` keeps the mixed (`:439-446`) and
   differing-folds (`:448-456`) refusals. Fix the spurious "no table" on an empty member lookup
   (`:424-428`).
4. **Pooling.** `evaluate-labels` multi-input with **stack (union of entries) semantics**, declared
   like `CROSS_JOIN_FEATURES` (`core/pipeline/loading.py:41`, used `run.py:1303`,
   `graph/resolve.py:441,479`); an entry present in two inputs is refused (no double counting). Today
   it merges column-wise and silently scores only input 0 (`evaluate_labels.py:351-352`). Reuse
   `pool_entry_counts` / `entry_counts` (`core/evaluation.py`). Bump `evaluate-labels` if identity moves.
5. **`mosaic pipeline verify <request>`** (`cli/pipeline.py`, logic `core/pipeline/verify.py`).
   For each fitting step of the plan: construct the feature (`build_step_feature`), call
   `load_state`/`fit`/`save_state` on a **scratch root** with the stream built by a helper extracted from
   `run.py:1493-1510` (no engine run-root override; scratch never touches the index; never copy a
   cached bundle in or `load_state` returns True). Compare to the stored run: probabilities within
   `--atol` (default 1e-5, n_threads pinned to 1) and report max-abs-diff + argmax agreement; compare
   `metrics.json` scores for evaluation steps; non-neural features exact. Evidence for the tolerance:
   same-thread CPU TCN is equal; across thread counts mean 7e-3 with ~99% argmax agreement; `gru`
   ~6e-7 (sequence-classifier plan). The tolerance is a **decision recorded in the verb's docs**.
   Read-only on the dataset (`read_only=True`), scratch under `--scratch` / system temp; join
   `test_cli_read_only.py`.
6. **Small fixes in scope:** unresolved `model` raises the misleading "No templates loaded"
   (`sequence_classifier.py:566-569`) → say the model reference did not resolve
   ([refit-in-place issue]). `--overwrite` docs (`SKILL.md:209-212`).
7. **Fallback if the spike fails:** the issue's design (apply step with its own identity); written up
   as its own plan, not built blind.
8. **Docs:** `SKILL.md` :209-212, :233-244, :312-336; `train-a-classifier.md` :207-223, :296-335,
   :346-352, :446-556 (these recipe blocks are executed by `tests/helpers/guide_recipes.py`);
   `docs/concepts/reproducibility.md:83-88` (verify); `CHANGELOG.md` (reword the "fits again" entry).
   Note staleness: a scope-free fit does not move when a train entry is re-converted (same as xgboost).

9. **S2b: apply-mode scope-free for the other scope-dependent fitters** (added so the contract above
   holds for unsupervised models too; the pattern is already shipped in `pair-posedistance-pca`).
   For `arhmm`, `kpms`, `vame`, `identity-model`, `identity-embedding-model`,
   `dinov2-temporal-identity-model`, `fit-label-map`: set `self.scope_dependent = self.params.model is None`
   in `__init__` (keep class-level `True` + the `test_hashing_rules.py:88` exemption). **Per-feature spike
   before each flip:** with `model` given, does the run read the scope for anything (state, per-entry
   output, normalisation) and does `load_state` fail loudly when the model reference is unresolved
   (`arhmm.py:310-320` currently fits a new model under the referenced run id — the
   [refit-in-place issue] — fix that first, it would otherwise turn a missing model into a silent
   scope-free fit). `kpms` and `vame` run in sandboxed environments (`behavior/CLAUDE.md`): confirm
   apply there is per-entry. Flip only the features whose spike passes; version bump + golden each; list
   the rest as still two-run in the docs. Test per feature: apply step over a wider scope keeps one run
   id and one model digest.

**Not solved here:** apply-only runs still queue on the training lane (`lanes.py`, the
[training lane issue]).

### Acceptance
A TCN seed = one fit step + one evaluation step; a blend = one ensemble step over cross-validated
members; widening the scope creates no new run id; `mosaic pipeline verify` passes on both demo
recipes (operator-run on kn-cgpu02; unit-level on a small fixture in CI).

---

## S3. Tuned thresholds and cross-validation inside one sequence

Depends on S2's fold table. Three phases:

**3a (quick, ship with S2).** `DECISION_THRESHOLD_DESCRIPTION` (`classifier_common.py:63-68`) says an
unlisted class gets threshold 0 and the binary form `{"0": 1.0, "1": t}`; same for
`DEFAULT_CLASS_DESCRIPTION`; fixes item 1 of the guide-gaps issue. Regenerate docs.

**3b `tune-threshold` step** (new `behavior/feature_library/tune_threshold.py`, imported in
`feature_library/__init__.py`).
- Global, scope-free feature. Inputs: classifier/ensemble run with `cv_folds.parquet`, plus `labels`.
  Params (all `Declared`): `classes`, `metric` (`pooled_f1` of one class | `macro_f1` over `average_over`),
  `grid`, `nested` (default true).
- Reuse `probability_ensemble._fit_weights` stream pattern: `fit_entries` from the cv table,
  `labels_for_rows`, `refuse_entries_outside_scope`, `_pooled_macro_f1`.
- Outputs: `thresholds.parquet` (class, threshold; loaders only support npz/parquet/joblib, so no
  JSON) and `tune_report.json` with the **plain** and the **nested** estimate (thresholds for fold k from
  the other folds' out-of-fold rows). Note the cross-validation plan's caveat: model hyperparameters
  were still chosen on those rows.
- **Cycle constraint:** a classifier cannot consume its own tuning (cycle, and a threshold change refits
  it). The consumer is a downstream cheap step: `decision_threshold` becomes
  `float | Mapping | ArtifactSpec-reference` (`graph/resolve.py:907-915` `_is_reference_field`), first
  honored by `probability-ensemble` (a one-member ensemble is the supported "apply threshold" step) and
  by per-behavior `xgboost` stages over probabilities. Document this pattern in the guide.
- Registration plumbing: `test_feature_prerequisites.py` `EXPECTED`, `test_column_contract_conformance.py`
  `PARAMS`, `test_feature_conformance.py`, golden `_TRACKS_DEFAULT_FEATURES`,
  `test_pipeline_compatibility.py`, `test_params_declaration.py`; MCP catalog detail ≤16,000 chars.

**3c Block cross-validation.** `cv_unit: "sequence"|"block"`, `cv_blocks_per_sequence`,
`cv_gap_frames` as hashed params, dealt in the shared fold code (`classifier_common.py:271-407`,
`cv_fold_assignment` `helpers.py:766-800`). Everything there is entry-granular today, so blocks change:
`cv.fold_of` rows in the bundle (`SequenceClassifierCv`, `XgboostCv`), `cv_folds.parquet` gains
`start`/`stop` frame columns, `_FOLD_COLUMNS` + equality check in `probability_ensemble.py:97,448`,
per-row (not per-entry) fold choice in `_scoring_model`/`out_of_fold_choice`, and gap frames dropped
from training. Defaults (`"sequence"`) keep today's behavior; confirm whether defaulted new fields
enter `identity_dump` (if so, bump versions + goldens).
- Order: `sequence-classifier` first (it already sees frames in `_collect_runs`), then `xgboost`.
- **xgboost needs frame numbers in templates** (`extract-labeled-templates.py:~795-830` saves group and
  sequence only; `BOOKKEEPING_COLS` `data_config.py:262`) → version bump of
  `extract-labeled-templates` and a templates-schema change. This is required for the Task 3
  acceptance (Task 3 used XGBoost; "intromission has one train video"). Treat as the largest S3 item.
- `lightning-action` has no `cv_folds` yet; add it (or defer block CV there and say so).
- **`temporal_reach`:** optional class attribute / params-function on features (read via `getattr` like
  `costly_fit`, `Declaration` field, conformance test), `int | None` (None = unknown/unbounded, e.g. a
  bidirectional `gru`). Declare for `temporal-stack`, `pair-wavelet`, `speed-angvel` smoothing,
  `trajectory-smooth`, the `tcn` head (receptive field 127 at blocks=6, kernel 3). Planner computes the
  upstream reach (sum along a path, max over paths); if all known and `cv_gap_frames` < reach → plan
  problem; if any unknown → caution requiring an explicit gap. (Speculation per the brief: not every
  feature can state one.)

**Docs/tests:** `test_classifier_common.py`, `test_cv_folds.py`, `test_xgboost_cross_validation.py`,
`test_probability_ensemble.py`; `train-a-classifier.md` (:346-352, :396-401 and the `37 columns` text,
guide-gaps item 4); `SKILL.md`.

**Acceptance (operator-run):** rerun the Task 3 prompt — intromission gets a cross-validated score, the
final recipe holds no hand-picked threshold, the report prints plain and nested estimates.

---

## S6. Recipes by reference, patching and expansion (independent; do before S4/S5)

Findings: saved recipes live at `.mosaic/pipelines/<digest>.json` (`store.recipe_path`) and are written
only by `submit_request`; digest → recipe lookup exists (`store.load_recipe`, `resources.py:215`);
`RecipeDocument` is already `dict[str, JsonValue]`; `validate_recipe` and `suggest_steps` take no
`dataset` arg today (`spec.py:578,584`); `GraphModel` is `extra="forbid"`; every reader goes through
`Recipe.model_validate` (a single hook point); `_structural_problem` (`validate.py:1031`) maps pydantic
locations to raw `steps[i]`.

1. **By reference.** `recipe` accepts `{"digest": "..."}` or a saved name for all four tools
   (`spec.py:394`, input schemas have room: 1080–5429 bytes of 8192). Add optional `dataset` to
   `validate_recipe` and `suggest_steps` (required when a reference is used). Resolve via
   `load_recipe(recipe_path(...))`; name → digest by scanning saved recipes' `name` (reuse the scan in
   `retirement.py:~251-280`); an ambiguous name is refused listing digests. An unknown digest says to
   `patch_recipe` or submit first. Digest-by-reference also shrinks `_call_digest` confirm state
   (`_binding.py:464`).
2. **`patch_recipe(dataset, base, ops)`** → `{digest, steps_total, problems}`. Ops: `add_step`,
   `remove_step` (refuse while referenced), `set_param`, `replace_step`. It **persists** the expanded
   result with `save_recipe` (content-addressed, idempotent) so the new digest resolves later and
   remotely; mark it `readOnlyHint=false, idempotent=true`, no confirmation. Update the
   `changing == confirming` pin in `tests/agent/mcp/test_tool_spec.py:95`. **Deviation from the brief:**
   `set_scope` is dropped — step scopes live on the request (`Request.step_scopes`), not in the recipe,
   so a recipe patch has nowhere to put one. Handled by item 4 instead.
3. **`repeat`.** Pure function `expand_repeats(document)` in new `graph/expand.py`, hooked as a
   `mode="before"` validator on `Recipe` so all readers get it. `{"id": "xgb_s{seed}",
   "repeat": {"seed": [1,2,3]}}` expands in declared order (several keys → cartesian product).
   Substitution in `id`, `inputs`/`tracks`/`after` references and string params (whole-string `"{seed}"`
   becomes the typed value). A list item with `repeat` expands in place, so an ensemble can take
   `{"step": "tcn_s{seed}", "repeat": {"seed": [1,2,3]}}` as three inputs. The **expanded** form is
   hashed and saved, so run ids and digests of existing recipes do not move. **`schema_version` decision:
   no bump.** `repeat` is an input-document feature; the saved/canonical form stays `schema_version: 1`,
   so older mosaic and mosaic-api read saved files, and `StepTypes.schema_version`/`CatalogIndex`
   (`resolve.py:614`, `wire.py:243,361`) are untouched. Keep problem reporting by step id (indices shift
   after expansion); an expansion error names the declared step. Pin: `test_pipeline_recipe.py:447`
   digests unchanged.
4. **One scope format.** The MCP `entries`/`step_entries` accept the library `Scope` JSON the CLI's
   `--step-scope` uses (`{"entries"|"groups"|"sequences": ...}`, `core/scope.py`); the bare pair list
   stays accepted and is normalized at the edge (`local._scope/_step_scopes`, `local.py:1202-1231`);
   one schema description. Tests: `test_tool_spec.py:384-406`, `test_local_backend.py:679-745`,
   `cli/_io.py:parse_step_scopes`.
5. **The Task 3 claim is not a bug.** `_step_target` (`plan.py:836`) intersects each step scope with
   the request scope, so a request narrowed to `train` with a step scoped to `test` correctly plans no
   entries (`_step_scope_problems`, `plan.py:846`). Fix is wording: the `step_entries`/`entries`
   descriptions and the problem message say "the request scope must cover every step's scope; leave
   `entries` unset to take their union". (Interacts with S1: do not make the guard depend on request
   narrowing.)
6. **Docs/tests.** `RECIPE_EXAMPLE` (`spec.py:362`) and `RecipeDocument` description document
   `digest` and `repeat`; `docs/agents.md`, `SKILL.md:365-368`, `docs/guides/pipelines/`. Tests for
   expansion (determinism, cartesian, reference lists, errors), digest equality of expanded vs spelled-out
   recipe, patch ops, by-name ambiguity. **Cap:** `patch_recipe` is the 15th tool; with S5 raise the cap.
   Check `INSTRUCTIONS` stays < 2048 chars (`_binding.py:109`, 1337 now) and every tool schema ≤ 8192.

**Acceptance:** a Task 3-sized recipe (synthetic equivalent in `tests/`, 71 expanded steps) is < 20
declared steps with `repeat`, and a `plan_recipe`/`submit_pipeline` call using a digest or patched
digest carries < 5 KB.

---

## S4. A remote MCP transport (independent)

`mcp 2.3.0` already ships `Server.streamable_http_app(...)` (`mcp/server/lowlevel/server.py:732-852`)
and depends on `uvicorn` and `starlette`; no new extra.

1. **Server.** `mosaic.agent.mcp.server.serve_http(...)` beside `serve()` (`server.py:62`): same
   `MCP_SDK.load()` → `LocalBackend` → `bind_server` construction (`_serve`, :106) but **no fd-1 claim**.
   Import `uvicorn`/`starlette` inside the function (import budget in `agent/CLAUDE.md`; layering
   test `tests/meta/test_agent_layering.py`). Build the app with `streamable_http_app(
   auth=AuthSettings(issuer_url=..., resource_server_url=..., validate_token_resource set to avoid
   `MCPDeprecationWarning`), token_verifier=<static bearer>, transport_security=TransportSecuritySettings(
   allowed_hosts/origins))`; run with `uvicorn.Server(Config(app, host, port, ssl_* optional))` under
   `anyio.run`.
2. **CLI.** `mosaic agent serve --http HOST:PORT --token-file FILE [--allowed-host H ...] [--tls-cert F
   --tls-key F]` extending `serve_command` (`cli/agent.py:188-278`). Safety: token file must be mode
   0600; constant-time compare (`hmac.compare_digest`); refuse a non-loopback bind without a token file;
   refuse `--http` with stdio-only options that make no sense. Loopback enables DNS-rebinding
   protection automatically; a tailnet host needs explicit `--allowed-host` or the check is off (warn).
3. **Confirmations (spike before choosing a default).** Stateless JSON mode (the hosted plan's choice)
   gives `can_send_request == False` (`streamable_http_manager.py:233`): modern clients still confirm
   via `input_required`/`request_state` (`_binding.py:380-392`); **legacy clients fall to the
   `confirm: true` path, which a model can set**. Spike with the `mcp` HTTP client and, if possible,
   Claude Science's Remote tab; default `stateless_http=False` (stateful SSE, elicitation works) unless
   the spike says otherwise, and refuse the `confirm: true` shortcut for HTTP legacy clients on the
   changing tools. `confirm_hook` is a Claude Code hook and does not apply to Claude Science — document.
4. **Tests.** Real server on an ephemeral port with `mcp.client.streamable_http.streamable_http_client`
   (pattern: `tests/agent/mcp/test_mcp_stdio.py` subprocess test): 401 without/with a wrong token,
   tool list equals stdio's, a light `plan_recipe`, a confirm round trip, a dropped connection with a
   detached request still running (drivers already survive).
5. **Docs.** `docs/agents.md` "Shared deployments" (:488) and "Every option" (:495) — one user, trusted
   network only, no TLS in the SDK (proxy or `--tls-*`); regenerate `docs/reference/cli.md`
   (:1294-1317); `agent/CLAUDE.md` "the local server promises" (S4 adds a network listener to the open
   toolkit; record the AGPL §13 note from the hosted plan). `mosaic agent install --mcp` stays stdio;
   an `--url` variant is optional.

**Acceptance:** Claude Science's Remote tab connects to the server on kn-cgpu02 over the tailnet,
lists datasets and runs a light request (manual); automated test above in CI.

---

## S5. Datasets through the MCP (largest; after S6 and S4)

Findings: `LocalBackend` holds a fixed `_manifests` and a `_served` map built once under a non-reentrant
`_served_lock` (`local.py:178,888`); `_Served.dataset` is opened once and reused by plan/inventory
(`local.py:151`), while `describe_dataset`/`submit`/`cancel` reopen per call; no hot reload. The CLI
inlines source-field validation (no shared helper); `ScanSource.path` is **not** confined by the
manifest (only `files` is). `Dataset.save()` is not locked; `mutate_manifest` is.

1. **Tool.** One `manage_dataset` tool, `action ∈ create | add_source | remove_source | scan |
   set_tags` with flat optional fields (check the 8192-byte schema budget; keep fields minimal).
   Conversion stays a recipe step. **No action writes `splits`.** Implementation in a new
   `agent/mcp/dataset_tools.py` used by `LocalBackend`, reusing: `new_dataset_manifest`
   (`dataset.py:484`), `MediaScanSource`/`TracksScanSource`/`LabelsScanSource` +
   `Dataset.add_scan_source` (`:1353`), `remove_scan_source` (`:1391`, rows kept by default),
   `scan_media/tracks/labels` (`:1550,1614,1644`), `define_tag`/`set_tag_value`/`remove_tag`, all
   through `open_dataset(..., ensure_roots=False)` and `Dataset.mutate_manifest` (race-safe), and
   `LocalBackend._changeable` (`:914`) for the managed-dataset gate. Extract the CLI's field validation
   into a shared function so CLI and tool cannot drift.
2. **Operator config.** `ServerConfig` (`config.py:84-130`) gains `dataset_roots` and `source_roots`
   (resolved relative to the config file in `read_server_config`, `:133`). Every path is `resolve()`d
   (symlinks, `..`) and must lie inside `dataset_roots` (create) / `source_roots` (source `path`); anything
   else is refused. Operators must keep sealed test files **outside** `source_roots` (S1) — document.
3. **Confirmations.** New conditions `create_dataset` and `add_source` (the latter also covers
   `remove_source`, since it changes what the dataset reads) in `ConfirmationCondition` /
   `CONFIRMATION_CONDITIONS`, per-action `confirm_when`; `scan` and `set_tags` need no confirm.
   Update pins: `test_tool_spec.py:95,143,163`, `_dispatch`, `ToolBackend` protocol
   (`spec.py:1423,1562`), `--approve` help (`cli/agent.py:288`). Everything the docs say
   contradicts S5 must change: `docs/agents.md:222-228,340-350`, `SKILL.md:~424,35-70,171`,
   `agent/CLAUDE.md` ("only tools that change a dataset"), and the local-plan statement.
4. **Hot pickup.** `LocalBackend._attach(manifest)` inserts a `_Served` (call outside the held lock;
   mind that `_register_contributors` also takes `_served_lock`); after any mutation replace
   `held.dataset` so plan and submit see one manifest; `list_datasets` (`:356`) re-runs
   `discovered_manifests` over `dataset_roots` (depth-limited) so a new dataset needs no restart.
5. **`next_steps`.** Additive field on `DatasetSummary` (`core/dataset_summary.py`, shared with
   `mosaic describe --json` and mosaic-api → must default; update the golden): an unconverted dataset
   returns strings such as "run a convert-tracks step" computed from declared sources, raw index and
   converted universe. `_plan_cautions` "covers all 0 entries" (`serialize.py:213-219`) names the
   converting step instead ([small-gaps issue] item 5). Scan runs on a worker thread and blocks (md5
   on by default); document the cost, defer progress.
6. **Cap.** `manage_dataset` is the 16th tool with S6's `patch_recipe`: raise `SHARED_TOOL_LIMIT`
   (`spec.py:147`) deliberately in the later slice, update `test_shared_tool_cap.py` and
   `test_mcp_server.py:188`, and check `INSTRUCTIONS`.
7. **Tests.** New `tests/agent/mcp/test_dataset_tools.py` (create/add/scan/tags/remove, confinement
   including symlink and `..` escapes, hot attach, managed-dataset refusal, `splits` untouchable);
   `test_server_config.py`; join `test_cli_read_only.py` lists for `describe_dataset next_steps`.
   Upload for remote clients is **not** built (waits on S4 + a hosted decision).

**Acceptance:** an agent given only a folder of `.npy` files and an allowed root creates the dataset,
converts it (recipe step) and reaches a plan, with no operator command — automated with the
`calms21_dataset`-style fixture over the in-memory MCP client; the scanned-but-unconverted plan goes
through the `IdentityUnresolved` path for scope-dependent steps (note in `graph/CLAUDE.md`).

---

## Risks and cross-slice notes
- Any change to a fit's identity moves run ids: every such change bumps `version`, regenerates the
  golden, and gets a CHANGELOG note (S2, S3c, `extract-labeled-templates`, `evaluate-labels`).
- S1's `splits` is unenforced by older mosaic (no version bump, by choice).
- `verify` depends on determinism the TCN lacks across thread counts/devices; the tolerance is a
  documented decision.
- S4 puts a network listener in the open toolkit; S5's mutating tools plus S4 widen the trust surface —
  `dataset_roots`/`source_roots`, bearer token and confirmations are the guards.
- The plan_recipe output schema has ~89 bytes of headroom: no new `PlanWire`/`PlanPage` fields in S1/S6.
- Slices touching `tests/core/pipeline/graph/test_guide_recipes*.py`: those execute the guide's recipe
  blocks, so doc edits are test-visible.

## Verification
- Per slice: new unit tests above; `pytest -m "slow or not slow"` (identity tests need the
  `deep-learning` env — run torch-marked tests with `-m identity`); `basedpyright`;
  `ruff check/format`; `python scripts/gen_docs_reference.py --check`; `mkdocs build --strict`.
- Meta tests to watch: `tests/meta/test_agent_layering.py`, `test_shared_tool_cap.py`,
  `tests/core/pipeline/test_hashing_rules.py`, `test_feature_run_id_golden.py`,
  `tests/agent/mcp/test_tool_spec.py`, `test_mcp_server.py` (schema/instruction budgets),
  `tests/cli/test_cli_read_only.py`.
- Acceptance reruns (operator, kn-cgpu02 sandbox; not CI): S1 Task 1 rerun with `pipeline score`;
  S2 `verify` on both demo recipes; S3 Task 3 rerun; S4 Claude Science Remote tab; S5 folder-of-`.npy`
  agent run; S6 Task 3 recipe with repeats.
