# mosaic.core.pipeline.graph

This directory owns pipelines as documents, from validating a JSON recipe through
planning it against a dataset to running each step of a submitted request. A test module
named without a path is in `tests/core/pipeline/graph/`.

## Files under `<dataset>/.mosaic/`

`pipelines/<digest>.json` is a recipe and `pipelines/requests/<rid>.json` one submission
of it (`store.py`). `runs/<execution_id>.jsonl` is a step attempt's run-log. `claims/`
keeps the failure records (`claims.py`), the one part of a step's status that is stored
instead of derived from artifacts.

## Recipe and request

- A recipe names neither a resolved `run_id` nor an entry list. A submission's choices go
  in `Request` (`model.py`), and `submit_request` assigns every step its execution id and
  pins `step_versions` before anything runs (`test_pipeline_submit.py`).
- A cross-step reference is written at the site it substitutes. Do not add a stored
  `edges` array. `edges()` in `topo.py` derives one, and `after` is the one explicit list,
  for ordering only (`test_edges_are_derived_from_the_bodies_with_their_sites`).
- `reject_unless_valid` (`validate.py`) runs before the dataset is touched and reports
  every problem. It refuses an `overwrite` key in `params` (`OVERWRITE_PARAM`) and the
  `extract-frames` op (`EXCLUDED_KINDS`), as `test_pipeline_validate.py` checks.

## Planning

- `plan_pipeline` (`plan.py`) resolves every step in one topological walk and submits
  nothing. A `scope_dependent` step's entries come from `intended_scope` (`scope.py`).
- `resolve_feature_identity` (`core/pipeline/run.py`) decides a step's `run_id` for the
  planner and the live `Pipeline` alike (`test_the_live_pipeline_and_the_planner_agree`).
- An op step's scope comes from the plan through `_op_scope`. An op with `scope_takes =
  "none"` is asked for no entry, and `check_scope_takes` (`core/pipeline/ops.py`) refuses
  a mismatch at plan time. A `scope_dependent` step is asked for all of its scope and a
  scope-free one for the remainder (`_entries_for`, `test_pipeline_plan.py`).

## Running a step

- `execute_step` (`step.py`) is the step body that `mosaic run --graph-request <rid>
  --step <id>` and `run_pipeline` (`run.py`) both call. Preflight, parent pinning and
  failure recording belong there and in no driver.
- A step re-plans its identity at its start. A `run_id` resolved at plan or submit
  time drives the preview, the estimate and the decision to enqueue. Execution and a
  downstream job's payload do not read it.
- `_pinned_parents` pins every feature ancestor, immediate or not, to the `run_id` that
  its attempt's run-log recorded. Preflight checks an op parent by the tracks variant it
  produced (`variant_for_producer_run`). An ancestor with no run-log is a cache hit
  (`test_two_requests_on_one_dataset_each_bind_to_their_own_upstream`).
- `mosaic run` finds the request from `--manifest`'s parent (`cli/run.py`). A second path
  flag would name a path that a queue cannot rewrite for another mount.
- A step declines by raising `StepRefused` (`preflight.py`), a `Refusal` naming a
  `RefusalReason` from `core/pipeline/refusal.py`. A new reason is a wire addition. The
  attempt exits `REFUSED_EXIT_CODE` (65) with status `failed`
  (`test_the_cli_exits_with_the_reserved_code_on_a_refusal`).
- `allow_partial` answers only a coverage shortfall, from a short upstream of a
  `scope_dependent` step or from quarantined entries (`test_pipeline_partial_failure.py`).
- An entry inside its backoff only narrows this attempt. An entry with `QUARANTINE_AFTER`
  failed attempts (`failures.py`) is quarantined and keeps the step back until
  `allow_partial` excludes it. Waiting does not exclude an entry
  (`test_pipeline_partial_failure.py`).
- Failure attempts per `(storage_name, run_id, group, sequence)` accumulate across
  resubmits, and exclusions belong to one request (`test_pipeline_failure_record.py`).
- A request is one-shot. `request_rollup` (`rollup.py`) reads run-logs alone, and a failed
  branch is not terminal while a sibling runs (`test_pipeline_step.py`).

## Declarations and the registries

- Only `resolve.py` imports `FEATURES`, and only inside functions. Parsing, ordering,
  lane decisions and status views do not pay the feature-library import
  (`tests/meta/test_graph_imports.py`). Its import of `mosaic.tracking` is listed under
  [Layering](../../../../../CLAUDE.md#layering).
- `can_connect` and `can_join` (`compatibility.py`) answer from `declaration_catalog()`
  with no dataset. `can_join` refuses a join of mismatched entity granularity unless the
  consumer cross-joins (`alignment_verdict`, `CROSS_JOIN_FEATURES`).
- A feature declares `requires` and `provides` with the references of
  `core/pipeline/column_contract.py`. Validation checks them with no dataset
  (`columns.py`), the plan against the tracks parquet footers, and `run_feature` per
  entry (`MissingInputColumns`). Each registered feature declares both
  (`tests/behavior/feature_library/test_column_contract_conformance.py`).
- A feature's `prerequisites` (`mosaic/optional_dependency.py`) are evaluated once per
  process, without importing, into `Declaration.unavailable`. Validation, the plan and
  `attach_candidates` refuse with that sentence (`test_pipeline_availability.py`).
