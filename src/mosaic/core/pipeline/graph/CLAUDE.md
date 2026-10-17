# mosaic.core.pipeline.graph

This directory owns pipelines as documents, from validating a JSON recipe through
planning it against a dataset to running each step of a submitted request. A test module
named without a path is in `tests/core/pipeline/graph/`.

## Files under `<dataset>/.mosaic/`

`pipelines/<digest>.json` is a recipe and `pipelines/requests/<rid>.json` one submission
of it (`store.py`). `runs/<execution_id>.jsonl` is a step attempt's run-log. `claims/`
keeps the failure records (`claims.py`), the one part of a step's status that is stored
instead of derived from artifacts. `drivers/<rid>.lease.json` is the lease of the process
driving a request, and `drivers/<rid>.log` the output of that driver and its steps
(`driver_lease.py`).

## Recipe and request

- A recipe names neither a resolved `run_id` nor an entry list. A submission's choices go
  in `Request` (`model.py`), and `submit_request` assigns every step its execution id and
  pins `step_versions` before anything runs (`test_pipeline_submit.py`).
- `plan_fingerprint` (`serialize.py`) digests the dataset uuid and each step's
  `run_id`, status, coverage counts and asked entries, and leaves out `reason`. Given
  `expect_plan`, `submit_request` compares it before writing either file and raises
  `PlanChanged`, a `conflict` and not a `RefusalReason`
  (`test_pipeline_plan_fingerprint.py`).
- `open_requests_matching` (`store.py`) returns the requests of one recipe digest whose
  scope overlaps (`Scope.overlaps`) and whose rollup verdict is `running`. A verdict
  of `orphaned` is not open, read as the status wires read it.
  `requests_matching` summarizes the `running` and `orphaned` ones, which
  `PlanWire.open_requests` lists outside the fingerprint (`test_pipeline_plan.py`).
  A request whose digest differs is matched by what its saved recipe digests to now, so
  a request saved under an earlier canonical form is still found.
  `submit_request` reports them in `duplicates` and, given `refuse_duplicates`,
  raises `DuplicateRequest`, a `conflict`, before writing either file
  (`test_pipeline_duplicate_requests.py`). The check runs again with the write under
  `submission_lock`. A request file that cannot be read is skipped with a warning, so
  one damaged file leaves submission available.
- A step's `params` are normalized at parse (`canonical_params`, the RFC 8785 number
  rule), and `save_recipe` writes `canonical_recipe`, so the bytes on disk are what the
  name digests. `load_recipe_for_request` also accepts the digest of the saved document
  as written, so a request whose file an earlier canonical form wrote still loads. A
  defaulted field added to a recipe model moves every digest; old requests load through
  that rule. Do not delete a `pipelines/<digest>.json` that a request names
  (`test_pipeline_recipe.py`, `test_pipeline_step.py`).
- A cross-step reference is written at the site it substitutes. Do not add a stored
  `edges` array. `edges()` in `topo.py` derives one, and `after` is the one explicit list,
  for ordering only (`test_edges_are_derived_from_the_bodies_with_their_sites`). No
  rule restricts where an `after` edge goes.
- A params field typed as a `LabelsSource` is the `labels` reference site
  (`ReferenceKind`). A `{"step": ...}` there names a step whose declaration
  `writes_labels` (`convert-labels`): the field gets its label kind and
  `StepSpec.labels_run_id` pins the variant it writes, through the plan and
  `run_feature` (`test_pipeline_converting_graph.py`, `test_pipeline_validate.py`).
- `Request.step_scopes` narrows single steps, and `save_request` leaves the key out
  when it is empty, so such a request file reads in an older mosaic. A step's target
  is the plan's entries narrowed by its scope (`step_target`), and its identity,
  coverage, asked entries, stall and shortfall are measured over it. The fingerprint
  includes the step scopes only when there are any (`test_pipeline_step_scopes.py`).
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
- `_step_scope_problems` refuses a step scope naming no step, on an op that takes no
  scope or a bound step, or selecting nothing, and a reader planned over entries its
  `inputs` or `tracks` parent is not. A params reference and `after` carry no entry
  rule, so `refuse_shortfall` asks a params parent (`PlannedStep.params_parents`) only
  to be complete.
- A writer of tracks that takes no scope (`convert-tracks`) does not make the media
  universe the target. On a dataset with no tracks, a graph holding one leaves its
  `scope_dependent` features and its scoped ops unresolved with an
  `IdentityUnresolved` naming the conversion; scope-free features keep their identity
  (`test_pipeline_converting_graph.py`). A step whose artifact reads complete after its
  attempt is not stalled.

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
- A step declines by raising `StepRefused` (`step_preflight.py`), a `Refusal` naming a
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
- `status.py` builds `RunStatusWire` and `RequestStatusWire` from run-logs alone. Their
  `partial`, `pending` and `orphaned` are derived words, and none joins
  `runlog.TERMINAL_STATUSES`. `orphaned` is a running request whose driver lease is
  stale and none of whose step processes may still run, or that has no lease and
  every one of whose started, unended steps is `dead` (`request_verdict_of`,
  `live_step_process`, `test_pipeline_status_wire.py`). A same-host pid counts only
  when its process began by the attempt's recorded start, which rules out a pid
  reused after a reboot.
- `liveness.py` is the one rule for whether an attempt's process may still run
  (`alive`, `unknown`, `dead`, `ended`): an earlier boot of this host, then a local
  pid, then ten declared intervals of silence. A job's `started` event declares the
  interval, and a thread in `JsonlRunLog` writes `alive` after each silent one.
  `may_still_run` keeps the pre-liveness answers for a run-log without an interval:
  `unknown` on another host may run, here it may not (`test_pipeline_liveness.py`).
- `run_request` (`run.py`) runs each unfinished step's `step_argv` as a process of its
  own session, with output to the driver log and never to inherited streams, and
  submits nothing. It skips a finished step, which makes running a request again a
  resume. A cancel forwards SIGTERM to the running step's group and records every step
  without a terminal record `cancelled` (`record_step_ended`). The rollup then closes.
  The driver deletes its lease when the request ends and keeps it when it raises
  (`test_pipeline_request_driver.py`, `test_pipeline_driver_lease.py`). The lease
  names the running step's process, which a killed driver leaves running in its own
  session, and a resume is refused while that process runs, even before it has
  written a run-log. A lease refresh that raises is reported and retried, and the
  driver stops with `LeaseNotRefreshed` before a step once its lease has gone
  unrefreshed for half the grace.
- `run_pipeline` (`run.py`) holds an `in_process` lease that names its own pid as the
  running step's, deletes it when the request ends, and keeps it after a shortfall
  until a later call in the same process takes it over.
  It stops before a step that a cancel recorded `cancelled`, and records its unended
  steps `cancelled` on `Cancelled` or `KeyboardInterrupt` (`test_pipeline_run.py`).
- `cancel_attempt` and `cancel_request` (`core/pipeline/cancel.py`) send SIGTERM to
  the pid that a run-log or a driver lease records. They record an attempt
  `cancelled` only when its process has exited or is provably `dead`, or when its
  step has not started and the request's driver has exited. A request with a live
  driver is cancelled through the driver alone, and an `in_process` one step by
  step. A process on another host is not signalled, and its request is left as it
  is unless liveness proves it gone or the caller passes `assume_exited`
  (`mosaic cancel --assume-exited`, CLI only), which is refused for a current
  heartbeat (`tests/core/pipeline/test_cancel.py`). Only a lease of this host names a
  pid to check here. A pid held by a process that began after the attempt's
  `started` record is reused and is not signalled. A queue that dispatches a
  request's steps holds no lease, so a step it starts after the cancel records its
  own ending after `cancelled`.
- `entry_errors_wire` (`entry_errors.py`) joins an attempt's `entry_error` events with
  the failure records keyed by its `target` and `run_id`. `held_back_by_quarantine`
  is true when any entity that the attempt lost is quarantined, listed or not. Every
  lost entity also leaves the run short for a `scope_dependent` consumer
  (`test_pipeline_entry_errors.py`).
- `list_requests` (`store.py`) and `list_runs` (`mosaic.runlog`) sort file names, which
  are ULIDs, newest first, and read only the page that they return
  (`test_pipeline_request_listing.py`, `tests/core/test_runlog_readers.py`).
- `__init__.py` binds every public name on access, and `status.py`, `store.py`,
  `claims.py`, `entry_errors.py`, `driver_lease.py` and `liveness.py` import nothing
  that loads pandas, cv2, av or the feature library
  (`tests/meta/test_status_imports.py`). The editor wires in `wire.py` import the op
  registry and the inventory, which do.

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
- A prerequisite that implements `RestrictedUse` supplies `CatalogEntry.license_note`,
  which the catalog lists whether the prerequisite is met or not. `kpms` declares one.
  `catalog_index` summarizes the editor catalog without params schemas, and
  `describe_step_type` returns one entry whole (`test_pipeline_catalog_index.py`).
- `step_type_detail` extends an index row with one `ParamRow` per params field,
  nested models as dotted paths, and gives the compacted schema only under
  `detail="schema"`. The CLI and the MCP print it, and mosaic-api reads
  `CatalogEntry` and `EditorCatalog`, which it leaves unchanged
  (`test_pipeline_step_type_detail.py`).
- `param_table.py` reads rows off a JSON Schema with the standard library only,
  because `agent/mcp/spec.py` imports its schema walker. The docs generator
  renders its parameter tables with the same helpers and Markdown markup, so a
  change there must keep `scripts/gen_docs_reference.py --check` passing.
