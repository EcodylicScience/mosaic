# The agent interface after the CalMS21 demo

A held-out protocol, one fit and apply contract, tuned thresholds, a remote transport, datasets
and recipes through the MCP. A design brief for the session that picks this work up.

Written against `main` at `6411b6e6` (0.17.0). Paths are relative to the mosaic repository and
cite `path:anchor`, where the anchor is literal text in the file. `SB` is
`code/dev/calms21-agent-sandbox/` in Dropbox and `DEMO` is `SB/run-2026-10-20-demo/`
(`RESULTS.md`, `task1/`, `task3/`, `test-scoring/`, `history-task1.md`, `history-task3.md`,
the prompts). Statements marked (estimate) or (speculation) were not measured. Nothing here is
built.

## What the demo showed

Two unattended agent runs through the local MCP server, one prompt each, on train data only. The
operator scored each final recipe on the held-out sequences afterwards, once.

| | Cross-validated | Test | Time and cost |
| --- | --- | --- | --- |
| Task 1: blend of 3 XGBoost seeds and 3 TCN seeds | macro F1 0.8138 | 0.8700 (MAP 0.932; paper Top-1 0.864 and 0.914) | 5 h, $30.46 |
| Task 3: one XGBoost classifier per behavior on a Task 1 model's probabilities | mean F1 of the behavior class 0.439 over six behaviors | 0.454 over seven (MAP 0.448; paper best 0.363 and 0.352) | 4.3 h, $18.56 |

Both agents wrote a mosaic-only `reproduce.py` whose refit matched the recorded scores to four
decimals, and logged eight (Task 1) and seven (Task 3) gaps in mosaic. The operator side needed
scaffolding that mosaic does not have:

- a tool that scores a final recipe on held-out sequences added afterwards, and checks that no
  run id moved (`SB/score_heldout.py`);
- a way to draw a recipe (`SB/draw_recipe.py`);
- a train-only dataset built by hand with `init`, `sources add`, `scan` and `convert-*`
  (`SB/build-demo.sh`);
- a Docker fence, because the test labels are otherwise in the dataset.

This brief moves the scaffolding and the gap logs into mosaic, in the order below. The issues
that are not scheduled stay in [`../issues/`](../issues/_INDEX.md) as possibilities (last
section).

## Order

| Slice | Change | Planning that exists | Size (estimate) | Depends on |
| --- | --- | --- | --- | --- |
| S1 | A held-out protocol: split roles, a fit guard in the planner, a sealed group, a score verb | step scopes built; [cross-validation plan](2026-10-18-cross-validation-by-sequence.md) ("Label-free fits are not held out") | M | none |
| S2 | One fit, apply and cross-validation contract | [fit and apply](../issues/fit-and-apply-are-one-feature.md), [sequence classifier plan](2026-10-19-sequence-classifier.md) (D8), [refit in place](../issues/refit-in-place-of-an-unresolved-model.md) | L | none |
| S3 | Tuned thresholds and cross-validation inside one sequence | `decision_threshold` built; the cross-validation plan says a second stage is "not nested" | M | S2 for the fold table |
| S4 | A remote MCP transport | [hosted server plan](../../mosaic-api/plans/2026-10-14-mcp-hosted-server.md) (deferred) | M | none |
| S5 | Datasets through the MCP | [local server plan](2026-10-14-mcp-local-server.implemented.md) (reads only), [small gaps](../issues/an-unattended-run-found-small-gaps-in-status-the-catalog-and-the-guide.md) item 5 | L | S4 for remote clients |
| S6 | Recipes by reference, patching and expansion | [follow-up plan](2026-10-17-mcp-calms21-followup.md) P3 (digests) and P5 (result size) | M | none |

A first session takes S1 to S3. They share one design: the fold table, the answer steps and the
run-id check. S4 and S6 are independent of them, and S5 is the largest.

## S1. A held-out protocol

**Problem.** The test labels were in the dataset in runs 1 to 6, and the rule was in the prompt.
The demo kept them out by building a train-only dataset and adding the test file afterwards.
Every fitting step needed a step scope naming the train group, and `pair-posedistance-pca` is
`scope_dependent` (`src/mosaic/behavior/feature_library/pairposedistancepca.py:scope_dependent = True`), so in run 6 a
whole-dataset request refit it on train and test; the agent caught that at planning. The scoring
tool copies the study, adds the file as a source, converts it, adds one `evaluate-labels` step per
answer scoped to the test sequences, compares every run id before and after, runs and prints.

**What exists.** `Request.step_scopes` (`src/mosaic/core/pipeline/graph/model.py:step_scopes`),
the guide's train and test recipe (`src/mosaic/agent/SKILL.md`), and plan cautions for a whole
dataset or a large scope (`src/mosaic/core/pipeline/graph/serialize.py:_plan_cautions`). Nothing
knows that a group is a test group.

**Design.**

1. **Roles.** Each group has a role, `train`, `test` or `sealed`. Recommended: a `splits:` mapping
   in `dataset.yaml`. The manifest keeps unknown top-level keys, and `MANIFEST_VERSION`'s rule says
   when to bump it. The alternative, dataset tags, fits the grouping guidance in
   `src/mosaic/core/CLAUDE.md` ("analysis grouping lives in mosaic-api's tags") but a tag cannot
   name a group in the local manifest.
2. **Planner guard.** A step that learns from its inputs and resolves to entries of a `test` or
   `sealed` group is a plan problem. It needs a declaration that separates fitting from
   `scope_dependent`, since `evaluate-labels` is `scope_dependent` and learns nothing
   (`src/mosaic/behavior/feature_library/evaluate_labels.py:scope_dependent = True`). The
   whole-dataset caution names the groups it would cover.
3. **Sealed groups.** The labels of a `sealed` group are not converted into the dataset the agent
   reads. Scoring them needs a new confirm condition, `score_sealed`, which the headless hook
   declines by default and the operator's own run approves.
4. **The verb.** `mosaic pipeline score --heldout <file or group>` does what the demo script does:
   adds the source, converts it with `convert-tracks` and `convert-labels` (see the
   [convert issue](../issues/a-convert-step-reports-complete-after-a-raw-file-is-added.md)),
   adds the evaluation steps for the recipe's answer steps, checks that no run id moved, runs,
   and writes a record. The answer steps come from a file at first (the demo's
   `final_models.json`) and from the
   [approach issue](../issues/a-recipe-does-not-say-which-steps-are-its-answer.md) later.
5. **A stability check in the plan.** `plan_recipe` could take `assume_added: [group]` and report
   which steps would refit if that group joined. The prompts had to ask each agent to do this by
   hand.

**Open questions.** Where roles live. Whether a sealed group's counts show in `describe_dataset`.
How a Task 3 style test set, one per behavior, maps to groups.

**Acceptance.** Rerun the Task 1 prompt and add the test file only through the verb. A fitting step
scoped to include the test group fails to plan. No operator script is needed.

## S2. One fit, apply and cross-validation contract

**Problem.** `sequence-classifier` and `lightning-action` are `scope_dependent` while they fit
(`src/mosaic/behavior/feature_library/sequence_classifier.py:self.scope_dependent = self.params.model is None`,
`lightning_action_feature.py:self.scope_dependent = self.params.model is None`), and `xgboost` is
not (`xgboost_feature.py:scope_dependent = False`). Widening a request from train to train and test
keeps XGBoost's run id and refits the TCN
([sequence classifier plan](2026-10-19-sequence-classifier.md), D8, which names the fit and apply
issue as the fix). The Task 1 agent worked around it. Each TCN seed became a train-scoped fit step
with `cv_folds`, an evaluation of it, and an apply step with `model: {step: ...}`: three steps per
seed. The blend became two ensembles. `probability-ensemble` refused to mix members with and
without `cv_folds.parquet`, even with fixed weights, because the apply members have no
out-of-fold rows. The recipes reached 39 steps (Task 1) and 71 (Task 3). The same cause blocks a
second stage over `sequence-classifier` members, and `evaluate-labels` takes one input, so
per-fold reports cannot be pooled in a recipe. `--overwrite` re-applies a stored model and never
refits it, so both `reproduce.py` scripts refit by changing a parameter that enters the run id and
not the model.

**What exists.** The [fit and apply issue](../issues/fit-and-apply-are-one-feature.md), the
[unresolved model issue](../issues/refit-in-place-of-an-unresolved-model.md), the
[training lane issue](../issues/apply-only-global-runs-on-the-training-lane.md), `cv_folds` and
`cv_seed` (built), and the ensemble's fitted weights (built).

**Design.**

1. **Scope-free fits when templates fix the training entries.** If `sequence-classifier` and
   `lightning-action` read only the entries their `templates` run names, their identity can hash
   the templates run and not the request scope, as `xgboost`'s does. Widening then adds rows to
   the same run, a cross-validated run gives out-of-fold rows for train entries and the final
   model's rows for the others (its `cv_folds` text already says so), and the mixed-member refusal
   and the step tripling go away. Spike first: does the stream read depend on the scope beyond
   `templates`?
2. **If the spike fails**, build the issue's design: an apply step with its own identity.
3. **Verify, do not refit.** A run with the same parameters and inputs must give the same
   outputs (determinism rule in `CLAUDE.md`). The wanted operation is therefore
   `mosaic pipeline verify <request>`: recompute each fit into a scratch run and compare scores or
   outputs within a stated tolerance. The TCN is not bit-identical across thread counts
   (the sequence classifier plan notes it), so the tolerance is a decision.
4. **Pooling.** `evaluate-labels` accepts several inputs and pools their per-entry counts, which
   its output already carries (`tp_<class>`, `fp_<class>`, `fn_<class>` per entry).

**Acceptance.** A TCN seed is one fit step and one evaluation step. A blend is one ensemble step
whose members were cross-validated. Widening the scope creates no new run id. `verify` passes on
both demo recipes.

## S3. Tuned thresholds and cross-validation inside one sequence

**Problem.** Task 3 ended with a threshold of 0.05, the lowest value tried, chosen on the same
out-of-fold rows it was scored on (the Task 3 agent's report says so). Cross-validation by sequence
over one to four videos was a poor guide: mount_attempt scored 0.303 in cross-validation and 0.017
on test, groom 0.529 and 0.336, and the six-behavior mean fell from 0.439 to 0.392.
Intromission has one train video, so it had no cross-validated score at all. The paper reports
that threshold tuning raises F1 (Top-1 0.319 to 0.363). On the positive class a mapping needs
`{"0": 1.0, "1": t}`, because an unlisted class gets threshold 0
(`src/mosaic/behavior/feature_library/classifier_common.py:threshold.get(cls, 0.0)`) and the
description does not say so.

**What exists.** `decision_threshold` in four steps. The cross-validation plan notes that a second
stage on the same folds is "not nested" and that nested cross-validation is the clean estimate.

**Design.**

1. **A `tune-threshold` step.** Inputs: a classifier or ensemble run with `cv_folds.parquet`, plus
   `labels`. Params: `classes`, the `metric` (pooled F1 of one class, or macro F1 over
   `average_over`), a `grid`, and `nested` (default true). It writes per-class thresholds that a
   consumer names with `decision_threshold: {"step": id}`. With `nested` it also reports the
   estimate in which the thresholds for fold k came from the other folds.
2. **Block cross-validation.** `cv_unit: "sequence" | "block"` with `cv_blocks_per_sequence` and
   `cv_gap_frames`, dealt by the shared fold code in `classifier_common.py`. The gap drops the
   frames around a held-out block. It has to cover the longest temporal reach of the features
   upstream (a stack of a few seconds, a network's receptive field), so features need a declared
   `temporal_reach` (speculation: not every feature can state one).
3. **The threshold description** says what an unlisted class does. The quick fix is in the
   [guide gaps issue](../issues/the-agent-guide-and-catalog-have-gaps-from-the-calms21-demo.md).

**Acceptance.** Rerun the Task 3 prompt. Intromission has a cross-validated score, the final
recipe holds no hand-picked threshold, and the report prints both the plain and the nested
estimate.

## S4. A remote MCP transport

**Problem.** The server is stdio only (`src/mosaic/agent/mcp/server.py:stdio_server`). The hosted
server is deferred to early 2027 ([plan](../../mosaic-api/plans/2026-10-14-mcp-hosted-server.md)),
and the roadmap rows for personal access tokens and the hosted interface are not built. Claude
Science, a macOS client, runs a local command inside a sandbox that hides most of the home folder
and denies outgoing connections (observed: `ssh: connect to host ... Operation not permitted`;
its `~/.claude-science/config.toml` grants folders through `[sandbox] user_read_paths` and
`user_write_paths`). Its Remote tab needs a streamable-HTTP URL, optionally with an OAuth or
header helper. The data and the GPUs are on a server, so that client can reach them only through
a remote HTTP server. The demo used Claude Code over ssh and a small local server in `~/mosaic-kn`.

**Design options.**

- **A.** Wait for the hosted server in mosaic-api: multi-user, ABAC, audit.
- **B.** `mosaic agent serve --http HOST:PORT --token-file FILE` in the toolkit: the same tool
  contract (`mosaic.agent.mcp.spec`), one user, a bearer token, for a trusted network such as a
  tailnet. Detached request drivers already survive a dropped connection.

B is a transport for the local server and adds nothing to the hosted design, which the placement
decision keeps in the closed repository. Recommended, as a stopgap.

**Open questions.** TLS (a tailnet or a reverse proxy). Whether the confirm prompts work over the
transport. The licence placement of a network listener in the open toolkit.

**Acceptance.** Claude Science's Remote tab connects to the server on kn-cgpu02 over the tailnet,
lists the datasets and runs a light request.

## S5. Datasets through the MCP

**Problem.** The server serves the datasets its config names at start (`datasets`, `discover`).
Every demo dataset was made by hand (`SB/build-demo.sh`). Neither an agent nor a scientist can
create a dataset, declare a source, scan, convert, tag, or add data later. A dataset that is
scanned and not converted reads as empty (the small gaps issue, item 5), and a convert step stays
complete after a raw file is added (the convert issue). A remote client cannot upload files.

**What exists.** `discover_datasets` and read-only discipline in the
[local server plan](2026-10-14-mcp-local-server.implemented.md) (T1, T10), and the CLI verbs.

**Design.** Tools `create_dataset`, `add_source`, `scan`, `set_dataset_tags` and `remove_source`.
Conversion stays a recipe step. The operator's config gains `dataset_roots` and `source_roots`,
outside which nothing can be created or read, and two confirm conditions, `create_dataset` and
`add_source`. The server picks up a new dataset without a restart. `describe_dataset` returns
`next_steps` for an unconverted dataset. Uploading for remote clients waits for S4.

**Acceptance.** An agent given only a folder of `.npy` files and an allowed root creates the
dataset, converts it and reaches a plan, with no operator command.

## S6. Recipes by reference, patching and expansion

**Problem.** `validate_recipe`, `suggest_steps`, `plan_recipe` and `submit_pipeline` take the whole
recipe inline (`src/mosaic/agent/mcp/spec.py:recipe: RecipeDocument`). The final recipes were
24 KB (Task 3, 71 steps) and each call re-sent it. In run 6 a 54-step plan result was
48,698 characters and was refused (`result_too_large`, `MAX_RESULT_CHARS`); `6411b6e6` pages it.
The agents wrote generator scripts to build recipes, because seeds, arms and folds multiply steps.
`step_entries` in the MCP takes bare pair lists and the CLI's `--step-scope` takes
`{"entries": ...}`. The Task 3 agent reported that a request cannot be narrowed to one group while
a step is scoped to another (not reproduced here).

**What exists.** Saved recipes by digest in `.mosaic/pipelines/` and digests that ignore `5.0` and
`5` (follow-up plan P3), and result bounds (P5).

**Design.**

1. `recipe` accepts `{"digest": ...}` or a saved name, for all four tools.
2. `patch_recipe(base, ops)` returns a new digest. The ops are `add_step`, `remove_step`,
   `set_param` and `set_scope`.
3. A `repeat` in a recipe, such as `{"id": "xgb_s{seed}", "repeat": {"seed": [1, 2, 3]}}`, expanded
   deterministically at parse. The digest hashes the expanded form, so run ids do not move. This
   needs a recipe `schema_version` decision.
4. One scope format.

**Acceptance.** The Task 3 final recipe is under 20 declared steps with repeats, and a plan or
submit call carries under 5 KB.

## Risks

- Any change to a fit's identity moves run ids. Each needs a `version` bump and a note, per the
  parameter rule in `CLAUDE.md`.
- S1 adds a manifest key, so `MANIFEST_VERSION` and the read-in-memory migration apply.
- `verify` depends on determinism the TCN does not fully have.
- S4 puts a network listener in the open toolkit, which changes what the local server promises.

## Not scheduled

These are issues, not slices. They stay as possibilities:

- [A request says nothing about what a step costs or where it runs](../issues/a-request-says-nothing-about-what-a-step-costs-or-where-it-runs.md)
  (estimates, honoured concurrency, a device per request, XGBoost sampling).
- [An agent cannot add a model or bring outside predictions in](../issues/an-agent-cannot-add-a-model-or-bring-outside-predictions-in.md)
  (import ops, reviewed plugins, model read tools). It builds on the
  [plugins spec](../specs/2026-10-19-mosaic-plugins.md).
- [A fitted feature model cannot be reused in another dataset](../issues/a-fitted-feature-model-cannot-be-reused-in-another-dataset.md).
- [The CalMS21 label converter names every task with the Task 1 vocabulary](../issues/the-calms21-label-converter-names-every-task-with-the-task-1-vocabulary.md).
- [A convert step reports complete after a raw file is added](../issues/a-convert-step-reports-complete-after-a-raw-file-is-added.md).
- [evaluate-labels scores pooled F1 only](../issues/evaluate-labels-scores-only-pooled-f1.md).
- [A recipe does not say which steps are its answer](../issues/a-recipe-does-not-say-which-steps-are-its-answer.md).
- [The agent guide and catalog have gaps from the CalMS21 demo](../issues/the-agent-guide-and-catalog-have-gaps-from-the-calms21-demo.md).

The roadmap rows these belong to are *Agent interface (MCP)*, *Unattended agent runs through the
MCP* and *Behavior classifiers (XGBoost, Lightning-Action, FERAL)*. No row was added.
