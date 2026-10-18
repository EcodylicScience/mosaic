# Changelog

One entry per milestone of the hashing and data-consistency program, plus any
release that moves an identifier or a name on disk, saying what moved on the
surface another repository can observe. A paragraph to read instead of a diff to
interpret.

M0 and M1 predate this file; both carried their entry in the final commit
message of their branch, and for both the answer was **nothing**.

## Unreleased: mosaic-media 0.3.5 and the conda install line without an `av` pin

**`body-scale` and `orientation-rel` compute an entry at once, and a run of them takes
seconds.** Both looped in Python over each `(frame, id)`, and `orientation-rel` also
rebuilt the ids of a frame from every pose it had cached, once per frame. On a table of
20,000 frames, two individuals and 7 keypoints, `body-scale` took 15.6 s (2,600 rows per
second) and now takes 0.02 s (1.6 million), and `orientation-rel` took 32.0 s (1,250 rows
per second) and now takes 0.06 s (700,000).

- No run id moves. Neither feature's `version` or params changed, and the outputs agree
  with the loops' to float rounding: the largest difference is 1.1e-14 on 40,000 pair
  rows, and the frame, the ids, `perspective`, the column order and the dtypes are
  identical. A cached run stays a hit.
- `body-scale` keeps the dtype of the input's `frame` and `id`. A table with no
  `group` or `sequence` column came out with float keys, because the loop read a row as
  one numeric Series. `mosaic_v1` requires both columns, so no stored run changes.
- A row with a missing heading is NaN in the `orientation-rel` columns as before, and
  no longer emits an "All-NaN slice" `RuntimeWarning` for each pair.
- Both features stay `parallelizable`, and `orientation-rel` still receives the
  `body-scale` lookup in `load_state`, so a process worker needs nothing shared. A run
  still computes entries in turn until the operator raises `max_workers`.

**The agent guide and the `parallel_mode` tool text agree on when to use `process`.**
`SKILL.md` said to use `process` only for small, fast steps, and the `submit_pipeline`
argument said it is faster for work that holds the interpreter lock. The second is what
happens. On 8 entries and 10 cores, a Python loop took 17.9 s in turn, 26.2 s in four threads and
4.8 s in four processes, and a vectorised numpy step took 2.0 s in turn, 1.1 s in four
threads and 2.1 s in four processes. Both texts now say to start with `thread` and to
use `process` for a step that stays slow in threads because its Python code holds the
interpreter lock, when the host's memory allows. Nothing recomputes.

**A step can compute several entries at once, and reports the memory it used.**
`mosaic run` and `mosaic pipeline run` take `--workers`, `--parallel-mode` and
`--min-free-memory-fraction`, and `submit_pipeline` takes `workers` and `parallel_mode`
within the limits `max_workers`, `allow_process_mode` and `min_free_memory_fraction` in the
server's configuration. The count is not part of a recipe or a run id, and nothing
recomputes. A run with several workers whose process worker is stopped raises
`WorkerProcessDied`, and the entries it wrote are kept.

- The run-log gains a `resources` event. `RunLogSnapshot`, `RunStatusWire` and
  `StepStatusWire` gain `workers`, `parallel_mode`, `peak_rss_bytes`,
  `min_available_bytes` and `total_memory_bytes`, which read 0 or `""` for a log
  without the event.
- `CatalogIndexEntry` and `StepTypeDetail` gain `parallelizable`. `PlannedStepWire`
  gains `parallelizable` and `workers`, and `PlanWire` gains `parallel`. All default, and
  `plan_fingerprint` leaves them out. `CatalogEntry` is unchanged.
- `start_detached_driver` and the `LocalBackend` `start_driver` seam take the
  parallelism as a last argument. mosaic-queue builds its own commands and does not pass
  the flags, so steps it dispatches run in turn.

**A template step writes only its sample.** `extract-templates` (0.3 to 0.4) and
`extract-labeled-templates` (0.4 to 0.5) wrote a copy of each entry's input as that
entry's table, beside the `templates.parquet` sample, and the labeled one added `split`
and `label`. Each entry's table is now empty, and its index row counts 0 rows. The
sample and `template_provenance.parquet` are unchanged.

- Every template run id moves with the version, and so does every run that pins one
  through `templates`: `global-scaler`, `global-kmeans`, `global-ward`, `global-tsne`,
  `xgboost` and `lightning-action`. An existing dataset recomputes those chains once,
  and their models are the same.
- A step that reads a template step through `inputs` is refused by
  `mosaic pipeline validate`, the plan and submission, beside another input too. The
  problem, at `inputs`, names the inputs of the template step to read instead.
  `run_feature` raises `InputWritesNoColumns` for such an input. A reader rewired to
  the template step's inputs loses the `extract-templates__from__` segment of its
  storage name and gets a new run id.
- A recipe stored in mosaic-api that reads a template step through `inputs` no longer
  validates.
- The editor catalog lists both features with `passthrough` false and no provided
  column. `Provides.is_empty` is true for a feature that declares an empty output.
- A labeled template run records each entry's `split` in
  `template_provenance.parquet`. Read it there instead of from the entry tables.
- Old template runs stay on disk. A superseded-run cleanup keeps an old run while a
  run that read its entry tables names it (`read_by`).

**A whole number in a recipe's `params` is an integer, so `5` and `5.0` name one
recipe.** A step's `params` are free JSON. A hand-written recipe file kept `5.0` while
an MCP client sent `5`, so one recipe had two digests: a CLI request and an MCP plan of
the same recipe did not see each other as duplicates, and a request's digest differed
from the plan's. Parsing now writes a finite whole-number float below 2**53 as an
integer (the RFC 8785 number rule), and `save_recipe` writes that form.

- Recipes with whole-number floats in `params` get a new digest. The next submission
  writes a second `pipelines/<digest>.json`, and the old file stays. Old requests keep
  their digest and still load, resume and cancel, because a request also accepts the
  digest of its saved file as written. The status wires and the `recipes/{digest}`
  resource show the old digest.
- A plan fingerprint taken by 0.17 for such a recipe is refused with `PlanChanged`;
  plan again.
- Run ids move only for whole-number floats that a recipe writes inside a free
  settings map: trex `convert_extra_settings` and `track_extra_settings`, sleap
  `sleap_extra_settings`, litpose and train-litpose `litpose_overrides`, train-pose and
  train-points `augmentation` and `train_overrides`, train-sleap `sleap_overrides`,
  overlay `draw_options` and nn-delta-bins `category_specs`. Recipes sent through MCP
  already carried integers. A direct `run_op`, `run_feature`, `Pipeline` or
  `mosaic track --set` call keeps the float, so for those settings it no longer shares
  a run with the same value written in a recipe.

**A feature storage name longer than 255 bytes is abbreviated, and its step runs.** A
storage name contains the whole input lineage, and a step over two multi-input parents
could pass the 255-byte limit on one path component. `validate_recipe` and
`plan_recipe` accepted it, and the step failed when it created its run directory, then
again writing its failure record. Such a name is now
`<slug>__from__<head>~<10 hex>`, at most 160 bytes, where the digest is of the full
input suffix. `run_feature`, `Pipeline`, the planner and the plan wire all derive it.
`params.json` `_inputs` lists the full lineage.

- Names of 255 bytes or fewer, which covers every name on disk, are unchanged. No
  directory, index row or run id moves. Only names that could not be created gain the
  `~` tail.
- A failure record under `.mosaic/claims` is filed under the percent-encoded storage
  name. When that passes 255 bytes, the directory is a head and a `~` digest of the
  full storage name instead. A step whose name fit on disk and overflowed once encoded
  failed when it cleared its records, and now finishes.
- Cleanup treats a run of a `~` storage whose params are unknown, including an attempt
  that has not named its run, as reading every run.

**mosaic needs mosaic-media 0.3.5, which accepts PyAV 19.** mosaic-media 0.3.4
required `av>=18,<19`, and conda-forge now ships `av` 19, so the documented conda
recipe followed by pip replaced conda's `av` with the PyPI wheel. That wheel bundles
its own ffmpeg, which printed the `AVFFrameReceiver is implemented in both` warning on
macOS. mosaic-media 0.3.5 requires `av>=18,<20`, and mosaic requires
`mosaic-media[io,cli]>=0.3.5,<0.4.0`, so the install lines in the README, the
installation page and the agent guide take `av` from conda-forge without a version.

**The MCP tools `search_step_types` and `describe_step_type` are one tool,
`describe_step_types`.** A search by words returns index rows as before. `names` (up
to 8) returns each step type's params as a flat table, nested fields as dotted paths
and a run reference typed as a recipe writes it, and `detail: "schema"` adds the JSON
Schema. The whole catalog entry it returned before ran to 27,600 characters
(`ultralytics`), and the largest table, `vame`'s, is about 15,700. The server now has 14
tools. A client that allow-lists tools by name
must allow `describe_step_types`; a call to an old name is refused with
`unknown_tool` and names the replacement in `detail.replacement`. An allow-list of the
whole server (`mcp__mosaic`) is unaffected. The `mosaic://catalog/{name}` resource
still returns the whole entry.

**A read tool's result is at most 40,000 characters.** A larger one is refused with
reason `result_too_large`, whose `detail.narrow_with` names the arguments that narrow
the call. `submit_pipeline` and `cancel_request` are never refused for their size. The
defaults shrink: `preview_output` returns 5 rows of at most 40 columns
(`max_columns`), the identity and bookkeeping columns first, and lists only the
columns read (`listed`); a `get_inventory` listing leaves out params, which a `ref`
returns, and names 5 missing entries per artifact; `plan_recipe` names 3 per step. A
successful call also returns its output as JSON text after the summary. The five
tools of the pipeline workflow carry `anthropic/alwaysLoad`, and the server
instructions are rewritten around `describe_step_types`.

**`mosaic preview` bounds a wide table and previews a labels variant.**
`--max-columns` reads at most that many columns, the identity and bookkeeping columns
first, and names the rest in `columns_omitted`. `--column` takes `fnmatch` patterns
(`W_*_f0`); a column of that exact name wins. `--listed read` lists only the columns
read. A `labels-variant/<kind>/<run_id>` ref previews the entry's `.npz` file: its
events as rows, and `labels` with each class's events, a pair event stored in both
orders counted once, and the frames labeled. The file loads without unpickling, and
`load_labels_auto` takes `allow_pickle` for that. `ArtifactPreview` gains
`n_columns`, `columns_omitted`, `columns_omitted_has_more` and `labels`, all
defaulted; the library defaults of `preview_artifact` are unchanged.

**`mosaic features describe` prints a step type's params as a table.** The `--json`
shape changes: it is a `StepTypeDetail`, whose `requires` and `provides` are labels
and whose `params` are rows, and `params_schema` is given only under `--detail
schema`. `mosaic pipeline catalog NAME` prints the same for any step type, ops and
`tracks` included; without a name its output is unchanged. In the library,
`step_type_detail` returns it, and `plan_wire(plan, missing_sample=)` and
`inventory_wire(..., with_params=, missing_sample=)` take smaller samples. Their
defaults are today's output, so `CatalogEntry`, `EditorCatalog`, `PlanWire` and the
plan fingerprint are unchanged for mosaic-api. Nothing on disk moves.

**A failed step ends only the branch below it.** `mosaic pipeline run --request`, the
detached driver that `submit_pipeline` starts, and `mosaic pipeline run --recipe`
stopped at the first step that failed. Steps on other branches did not run, and the
request closed `failed`. A mosaic-queue gate already ran them.

- Both drivers record each step below a failed one `failed`, with `error_json`
  `{"type": "ParentFailed", "message", "reason": "parent_failed", "parent",
  "parent_execution_id"}`, and run the steps that do not descend from it. A
  `version_moved`, `recipe_missing` or `digest_mismatch` refusal, which every step
  meets, and a cancel still end the run.
- `request_rollup` reads a failed request with an unstarted step `running` while its
  driver lease file exists. A poll, `open_requests_matching` and `mosaic cancel` then
  see the driver at work. A stale lease reads `orphaned`, and a resume runs the rest.
  `request_verdict` and `runlog.TERMINAL_STATUSES` are unchanged.
- `mosaic pipeline run --request` does not drive a request that has ended: it starts
  nothing, deletes a stale driver lease and reports the request's status.
  `RequestRun` gains `blocked`, and the `--json` output gains `blocked`, the steps
  the driver recorded `ParentFailed`.
- `run_pipeline` and `mosaic pipeline run --recipe` raise the first step error after
  the other branches ran, with the same exit codes. A step that raised before its
  run-log recorded an end, and a stall without `allow_partial`, leave the steps below
  them unrecorded and keep the lease. The request then resumes.
- No identifier moves.

**A request whose step died reads `orphaned`, even with no driver lease or on another
host.** A request run by `mosaic pipeline run --recipe` took no driver lease, so its
crash left it `running` for good, and a step on another host, such as a Docker
container that got a new host name, counted as running forever. Each job now
declares its liveness, and one rule (`mosaic.core.pipeline.graph.liveness`) reads
every attempt as `alive`, `unknown`, `dead` or `ended`.

- Run-log, additive. `started` records `boot_id`, `pid_ns` and `liveness_interval`
  (60 s). A thread appends an `alive` event after each interval without another line,
  and never after the terminal event. A fitted feature writes `phase` events (`fit`,
  then `apply`). `RunLogSnapshot` and `RunStatusWire` gain `boot_id`, `pid_ns`,
  `liveness_interval` and `phase`. A reader that judges an attempt by its silence
  needs a time-to-live of at least 120 s. `TERMINAL_STATUSES` is unchanged.
- A `KeyboardInterrupt` inside `job_context` records the attempt `cancelled`.
- An attempt is `dead` when this host rebooted since it started, its local pid exited,
  or it declared an interval and wrote nothing for ten of them. A request without a
  lease reads `orphaned` when every started, unended step is `dead`. A stale lease of
  another host names a live unstarted step for ten minutes, not forever. A run-log
  written by 0.17 or earlier on another host is `unknown` and keeps its request
  `running`, as before.
- `run_pipeline` takes an `in_process` driver lease (a new `DriverLease` key that
  older readers drop) and a `cancel_token`, and records its unended steps `cancelled`
  on a cancel. `mosaic cancel --request` cancels such a request step by step, and
  `mosaic pipeline run --recipe` exits 130 on SIGTERM or SIGINT.
- `mosaic cancel` records a `dead` attempt `cancelled`, also on another host.
  `--assume-exited` records an `unknown` one on the user's word, and is refused for a
  current heartbeat. A cancel no longer checks another host's lease step pid on this
  host. The MCP `cancel_request` takes no new argument.
- Wires, additive. `StepAttempt` gains `heartbeat_at`, `boot_id`, `pid_ns`,
  `liveness_interval` and `phase`. `StepStatusWire` gains `host`, `started_at`,
  `heartbeat_at`, `phase` and `liveness`, `RequestStatusWire` gains
  `poll_after_seconds` (10 to 300 s), and `PlanWire` gains `open_requests`, the
  recipe's `running` and `orphaned` requests (`requests_matching`), which the
  fingerprint leaves out. The MCP `RunSummary` gains `host`, `heartbeat_at`, `phase`
  and `liveness`.
- No identifier moves.

**`fit_scope.json` records what a fit was trained on, not where it was run.** A
scope-free fitter (`global-scaler`, `global-kmeans`, `xgboost`, ...) trains on the
templates it pins, but its record held the scope of the invocation that fitted it. Its
entries and its tracks and labels variants are now the union of the fit records of the
artifacts it pins, and a new `via` lists those artifacts. The entries are an upper
bound: `xgboost` fits only the `split == "train"` rows of its templates.

- The record gains `source` (`stream`, `artifacts` or `unknown`) and `via`.
- `entries`, `tracks_variants` and `labels_variants` are `null` when they are unknown:
  for a fitter that pins no artifact (`feral` reads a directory of videos), or one
  whose artifact has no record. `FitScope` holds `None` for them.
- An older record still reads. One from a scope-free run reads with unknown entries,
  because it held the invocation scope; one from a scope-dependent run keeps its
  entries. Nothing is rewritten.
- `write_fit_scope(run_root, scope)` takes a `FitScope`, which the new
  `derive_fit_scope` builds. No run id moves.

**The inventory measures a scope-dependent run over the scope it ran on.** A run whose
scope is in its run id, such as `extract-labeled-templates` over the training groups,
read `partial` for ever, because it was measured against every entry in the dataset
and no action could finish it. It now reads `complete` when it holds every entry of
its declared scope, and `partial` when it lost some. Its `coverage.target` counts the
declared entries. A selector that names none of its entries still reads `absent`.
`mosaic inventory`, `get_inventory` and a narrowed `inventory_wire` apply the same
rule through `feature_run_target`.

**The inventory measures every feature run over the entries it was asked for.** A
scope-free run applied to a subset, such as `xgboost` applied to the test groups
through a step scope, read `partial` against the whole dataset while `plan_recipe`
read it `complete`. `params.json` now records `_scope.asked`, the entries of every
invocation of the run, and `_scope.asked_all`, set when an invocation of a scope-free
run asked for every entry. A run reads `complete` when it has every entry it was asked
for and `partial` when it lost one, also after a later invocation over other entries. A
run invoked without a scope, directly or as a recipe step that neither the request nor
a step scope narrows, still reads `partial` once the dataset gains a sequence.
`run_feature` takes `asks_every_entry` for a caller that passes the entries it computes
and asks for every entry. A file written before these keys is measured over its latest
scope and its index rows. No run id moves.

**CalMS21 tracks no longer carry the ground-truth `label` column (`calms21_npy` and
`calms21_json` 0.4).** The converter copied each sequence's annotations into the
tracks, so every feature built on them carried the answer it was later scored on, and
a fitter that reads numeric columns by wildcard could train on it. Labels now come
only from `convert-labels`. Converting again writes a new tracks variant beside the
0.3 one; an existing dataset keeps resolving its 0.3 variant until it is reconverted.
After reconverting, pass `tracks_run_id=` or remove the old variant, since an entry
with two variants is refused.

**A frame with no label event is unlabeled (`-1`), not class 0.** The label loaders
filled every frame no event covered with `default_label=0`, which in CalMS21 is
`attack`, and in a BORIS file whose background sorted after a behavior was that
behavior. `mosaic.core.helpers.UNLABELED` (`-1`) now marks such a frame, unless the
file names a background class.

- `load_labels_auto`, `load_labels_for_feature_frames` and `expand_labels_to_dense`
  take `default_label=None`. For a sparse file it means the file's background class:
  its `background_id`, or the `background_label` its labels variant recorded, or a
  class named `none`, or `-1`. The last two warn with `LegacyLabelFileWarning`, which
  reconverting the labels silences. A dense file labels every frame it holds, so a
  frame past its end is `-1`.
- `load_labels_for_feature_frames(individual_filter=)` also takes one `(id1, id2)`
  per row, with `-1` as a wildcard on either side. An event labels the row whose
  individuals it names: a scene event every row, an individual event every row focal
  on that individual, a pair event only the row of its direction. **The tuple form is
  now directed**: `(0, 1)` no longer matches an event stored as `(1, 0)`.
- Without a filter, an individual or scene event is no longer dropped as the second
  direction of a symmetric pair, which left a BORIS focal-subject event unread.
- New `mosaic.core.pipeline.loading.labels_for_rows(path, df)` labels each row of a
  frame by its `frame` and its own ids (`row_individual_pairs`). Rows whose ids are
  not integers align by frame alone.

**Label converters record their background class (`calms21_npy` labels 0.3,
`boris_aggregated_csv` and `boris_pandas_pickle` 0.2).** Each file now carries
`background_id`, the class a frame without an event means. CalMS21 writes `-1`,
because every CalMS21 class is a behavior. BORIS puts its `background_label` at id 0
even when the background occurs in the data, where it used to take an alphabetical
id. A labels variant id moves only when the labels are reconverted. After
reconverting, the dataset holds two label variants, so pass `labels_run_id=` or
remove the old one; until then the old files still read correctly, through the
background the variant recorded. `convert_labels_custom` documents the key.

**Templates and `load_values` leave unlabeled rows out of class 0
(`extract-labeled-templates` 0.4).** `extract-labeled-templates` samples only rows
that a label event covers, and a pair row takes the label of its own direction. Its
apply writes `-1` for a row with no label. On CalMS21 the intruder-focal pair rows and
the intruder's individual rows become unlabeled, which halves the training rows of a
pair feature: a CalMS21 label describes the resident. Its run ids move, and so do the
ids of the runs downstream of it when they rerun.

- `load_values` returns `-1` for a row no event covers and for every row of an entry
  with no label file, where it returned 0. It raises when an entry has a label file
  but its rows have no `frame` column. `pair_filter` selects rows first, and labels
  then align to each surviving row.
- `xgboost` and `lightning-action` refuse templates with a negative label, which an
  older templates run can hold.

**Cluster scoring drops unlabeled rows, scores one perspective, and keys entries by
group and sequence.** `compute_cluster_label_agreement` pooled both perspectives of a
pair run, read unannotated frames as class 0, matched a run's files to label files by
sequence name alone (so two groups sharing a sequence name read each other's labels),
cut a sequence to the shorter of two arrays without saying so, and guessed a frame
source by a hardcoded feature name. The new module `mosaic.core.evaluation` holds the
alignment and the scorers, and `mosaic.core.analysis` calls it.

- `align_predictions_to_labels(ds, ResultColumn(...))` pairs each output row of a run
  with the label of that row through `load_values`, in one labels variant
  (`labels_run_id=`), filters a `perspective`, and drops and counts unlabeled rows and
  rows with no prediction (`AlignedLabels`). It reads the tracks variant the scored
  run read, so a dataset holding two variants of an entry after a reconversion is not
  refused; `tracks_run_id=` names another.
- `cluster_label_scores` (which `cluster_gt_agreement` now calls) drops rows labeled
  `-1` and reports `n_unlabeled`. It adds `majority_accuracy` (equal to `purity`),
  `majority_mapping` (cluster to class) and `hungarian_one_to_one`. With more clusters
  than classes, read `majority_accuracy`: the Hungarian pairing counts every cluster
  beyond it as wrong. `majority_mapping` and `perspective_consistency` are also
  public.
- `compute_cluster_label_agreement` takes `perspective=` (an int, or `"each"` for
  `metrics_by_perspective`), `scope=`, `labels_run_id=` and `tracks_run_id=`. It
  reports `n_unlabeled_dropped`, `n_missing_prediction` and, for a pair run,
  `perspective_consistency`, `n_perspective_pairs` and `n_perspectives`. `max_frames`
  subsamples after the perspective filter. Rows of `aligned_sequences` and
  `per_sequence` carry `group`, `sequence`, `sequence_safe` and `entry_key`, and no
  longer `cluster_path` or `label_path`. `frame_source_feature` is ignored with a
  `DeprecationWarning`; a run whose rows have no `frame` column is refused.

**`submit_pipeline` no longer asks to confirm a large or whole-dataset scope, and
an unanswered confirmation is not a decline.** A headless `claude -p` run answered
every confirmation at once, and the server reported "the user declined" for a CPU
recipe whose only condition was that it covered the dataset.

- The conditions `whole_dataset` and `large_scope` are removed. `plan_recipe` and
  `mosaic pipeline plan` report both as cautions, which refuse nothing: `PlanWire`
  gains `cautions` (additive) and `Plan` gains `covers_dataset`. Neither enters the
  fingerprint. `LARGE_SCOPE_ENTRIES` moved from `mosaic.agent.mcp.spec` to
  `mosaic.core.pipeline.graph`.
- The condition `cleanup` is declared for the superseded-run cleanup that
  `submit_pipeline` gains in this release.
- A cancelled elicitation, which a client answers when nobody can see the form, is
  the new reason `confirmation_unanswered`. A declined form, or one accepted with its
  box unticked, is `confirmation_declined`, whose message no longer says that a
  person declined. Both carry `detail.conditions`, and both tell the model to stop
  and report rather than run the same work through the CLI. The server instructions,
  the tool descriptions and the `build_pipeline` prompt say the same.
- The question's last line is `[mosaic-confirm tool=<tool> conditions=<a>,<b>]`,
  which `mosaic.agent.mcp.spec.read_confirmation_tag` reads. A finding is put on one
  line and a `bind` finding quotes each step id, so a step id cannot forge the tag. A
  2026-07-28 request state is `<call digest>.<conditions>`.
- New `mosaic agent confirm-hook --approve CONDITION --otherwise ask|cancel|decline`
  answers the confirmations of an unattended run from a Claude Code `Elicitation`
  hook (`mosaic.agent.confirm_hook.answer_elicitation`). It defaults to `decline`,
  which no other hook's accept overrides, and answers a confirmation whose tag it
  cannot read the same way. The agents page shows the settings.
- `CONFIRM_FIELD` moved from `mosaic.agent.mcp._binding` to `mosaic.agent.mcp.spec`,
  and `_binding` still exports it.
- An accepted form confirms only the conditions it named. The call is checked again
  before it runs, and one that meets another condition by then, such as an
  `open_duplicate` from a request started meanwhile, is asked about again: a
  2026-07-28 client gets a new `input_required`, an earlier one a second form. A state
  without the conditions suffix now confirms nothing and is asked again.


**Conversion, the train/test split and scoring are recipe steps.** One recipe and one
request can now convert a dataset's raw tracks and labels, fit on the training groups
and score on the test groups.

- New ops `convert-tracks` and `convert-labels` (category `convert`, domain
  `tracking`, cpu lane, `scope_takes="none"`) convert one source format. Each names
  the variant it writes from its params before it runs, through the new pure
  `Dataset.track_conversion_variant` and `label_conversion_variant`, which write
  nothing. A file that fails is an `entry_error` and fails the attempt, and a rerun
  resumes. The plan refuses a conversion whose format has no scanned row. 22 ops
  are registered; `mosaic tracking list` lists both.
- Additive on mosaic-api-visible types: `Op.writes_labels`, `OpIdentity.labels_variant`,
  `ProducerDecl.writes_labels`, `ReferenceKind` gains `"labels"` (so a labels field,
  such as `extract-labeled-templates`' `labels`, is listed in
  `CatalogEntry.reference_fields`), `ResolvedStep.labels_variant` and `labels_kind`,
  `StepSpec.labels_run_id`, `PlannedStep.labels_variant`, `labels_kind` and
  `status_parents`, `Plan.step_scopes`, and `PlannedStepWire.labels_variant` and
  `step_scope`.
- A `labels` field takes `{"step": <convert-labels step>}`: the field gets its label
  kind, and the run reads the variant that step writes. `resolve_labels_variants`
  takes a pinned `labels_run_id` as given, without reading the index, which names a
  run planned before its labels exist as it executes. Where the variant's rows
  exist, nothing moves.
- On a dataset with no tracks, a graph that converts its own leaves its
  scope-dependent steps and scoped ops unresolved, naming the conversion, rather
  than naming them over no entries. A step whose artifact reads complete after its
  attempt is no longer reported as stalled, which an op planned over no entries was.
- `Request.step_scopes` narrows single steps, keyed by step id and intersected with
  the request's scope. A step's identity, coverage, asked entries, stall and
  shortfall are measured over its narrowed target. The plan refuses a step scope
  naming no step, on an op that takes no scope or a bound step, or selecting
  nothing, and a reader narrowed wider than an `inputs` or `tracks` parent. A params
  reference or an `after` edge carries no entry rule, and a shortfall asks such a
  parent only to be complete. The request file leaves out `step_scopes` when it is empty, so such a
  request still reads in 0.17, and the plan fingerprint includes them only when
  there are any. **A request file with step scopes is unreadable by an older
  mosaic**: upgrade mosaic-queue workers and mosaic-api together before using them.
  `submit_request` and `run_pipeline` take `step_scopes=`; mosaic-api's `step_runs`
  should read `request.step_scope(id) or request.scope` if it ever submits them.
- CLI: `mosaic pipeline plan`, `run` and `submit` take `--step-scope STEP=SCOPE` and
  `--group`; `submit --json` reports `step_scopes`. `convert-tracks` takes
  `--source-format`, and its `--json` adds `variants`. `convert-labels --json`
  reports `variant`, `written` and `skipped` where it reported only a status.
- Library: `convert_all_tracks` takes `source_format`, `progress`, `cancel_check` and
  `on_failure`, and `ConversionOutcome` gains `variants`. `convert_all_labels` takes
  `progress` and `cancel_check`, returns `LabelConversionOutcome` where it returned
  `None`, and registers the conversion through `mutate_manifest`.
- MCP: `entries` takes `{"groups": [...]}` beside a list of entries (on
  `plan_recipe`, `submit_pipeline` and `get_inventory`), and `plan_recipe` and
  `submit_pipeline` take `step_entries`. A run's report is the new resource
  `mosaic://dataset/{id}/artifacts/feature/{name}/{run_id}/report`, read through
  `mosaic.core.pipeline.inventory.read_run_report`. The tool count is unchanged.
- New features `evaluate-labels` (summary, scope-dependent) and `fit-label-map`
  (global, scope-dependent). `evaluate-labels` scores a classifier, or a clustering
  through a label map, against labels read through the run's dependency lookup,
  choosing the perspective of pair rows; it writes `metrics.json` (an
  `EvaluationReport`) and `confusion.parquet`, and one row per entry.
  `fit-label-map` fits each cluster's majority class on its own entries and writes
  `label_map.parquet`. `mosaic.core.evaluation` gains `classification_scores`,
  `select_scored_rows`, `EvaluationReport` and `ClassScore`.
- No existing run id, op id, recipe digest or fingerprint moves.

**A pipeline removes the feature runs its new params superseded, when asked.** Changing
a step's params leaves the earlier runs of its storage on disk, and no verb reclaimed
them. `mosaic pipeline plan --cleanup superseded` and `plan_recipe` with
`cleanup: "superseded"` list every other run of the recipe's feature storages, each
removable or kept with its reason (`costly_fit`, `params_unknown`, `attempt_open`,
`held_by_request`, `wider_coverage`, `other_fit_scope`, `young`, `newest`, `read_by`
and others). Planning writes nothing. `mosaic pipeline submit` and `run --recipe` with
`--cleanup superseded --expect-plan <fingerprint>`, and `submit_pipeline` with
`cleanup: "superseded"`, remove the removable ones after every step has finished. Each
removal records the run's `params.json` in the cleanup's run-log before the index rows
and then the directory go, so the run can be computed again.

- The policy is `CleanupPolicy(keep_newest=1, min_age_hours=24)`, set by the CLI's
  `--keep-newest` and `--min-age-hours` or by the MCP server's configuration
  (`cleanup:`), never by a tool call. A request that ended holds its runs for at least
  an hour.
- The MCP server refuses `submit_pipeline` with a cleanup (`cleanup_not_allowed`)
  unless started with `--allow-cleanup feature-runs` or `allow_cleanup:
  [feature-runs]`, and then asks to confirm the `cleanup` condition. `submit_pipeline`
  is now annotated destructive. A managed dataset refuses a cleanup
  (`CleanupOnManagedDataset`), and a cleanup without a reviewed fingerprint is refused
  (`CleanupNeedsPlan`).
- A request with a cleanup reads `running` until the cleanup's run-log ends, and a
  failed or cancelled cleanup leaves the verdict to the steps. `request_verdict` is
  unchanged. A killed driver leaves the request `orphaned`, and
  `mosaic pipeline run --request` finishes the cleanup. A queue that dispatches the
  steps itself runs no cleanup.
- `plan_recipe` lists at most five superseded runs a step (`PLAN_SUPERSEDED_SAMPLE`,
  through `plan_wire(superseded_sample=)`), and `pipeline plan` fifty. The totals count
  every run.
- Every `global` and `media` feature declares `costly_fit`.
- New fields, all defaulted: `PlanWire.cleanup`, `PlannedStepWire.superseded` and
  `superseded_omitted`, `RequestStatusWire.cleanup`, `RequestRollup.cleanup`,
  `Request.cleanup`, `Plan.cleanup`, `Declaration.costly_fit`,
  `SubmittedPipeline.cleanup_runs` and `cleanup_bytes`, `PlanRecipeInput.cleanup`,
  `SubmitPipelineInput.cleanup`, `ServerConfig.allow_cleanup` and `cleanup`,
  `RequestRun.cleanup_status` and `PipelineRun.cleanup_status`. New run-log events
  `run_removing`, `run_removed` and `run_kept`, read by
  `mosaic.runlog.read_removal_records`; `TERMINAL_STATUSES` is unchanged.
- New library: `mosaic.core.pipeline.feature_run_removal` (`remove_feature_run`,
  `run_bytes`) and, in `mosaic.core.pipeline.graph`, `plan_with_cleanup`,
  `plan_cleanup`, `run_cleanup`, `CleanupPolicy`, `CleanupPlan`, `SupersededRun`,
  `CleanupOutcome` and `CleanupStatusWire`.
- No identifier moves, and a plan not asked for a cleanup fingerprints as before.
- An older mosaic cannot read a request file that carries a cleanup, so it cannot list
  the requests of a dataset that holds one. A request without one is written as
  before.

**An attempt ends at the last count it reported.** `JobContext.heartbeat` writes on
every tenth call or after 15 s, and the attempt ended without the count of a
throttled call. A `convert-tracks` or `convert-labels` over two raw files then
finished at 1 of 2, and `get_request_status` reported the step half done. Before its
`finished`, `failed` or `cancelled` record, an attempt now writes one `heartbeat` with
that count, unless the last `heartbeat`, `entry_end` or `epoch` line holds it or came
after the throttled call. A failed or cancelled attempt reports its last count too.

- Run-log, additive: at most one more `heartbeat` line before the terminal event.
  `TERMINAL_STATUSES` and the wires are unchanged.
- The progress of `convert-tracks` and `convert-labels` counts raw files, so
  CalMS21 reports 2 of 2 beside `entries_written` 89.
- `JsonlRunLog.progress_writes`, `JsonlRunLog.last_progress` and
  `JobContext.flush_progress` are new.
- No identifier moves.

**A cleanup keeps a run that a request of another recipe recorded.** A candidate in a
planned storage was judged whichever recipe wrote it. A sibling that another saved
recipe uses, such as a second `k` of `global-kmeans`, was therefore removable an hour
after that recipe's request ended. Recipes with the same non-empty `name` are now
versions of each other, and an unnamed recipe is a version only of itself. A run that a
request of any other recipe recorded or binds is kept as `recipe_uses` once that
request's hold lapses, and so is a run recorded by a request whose recipe file does not
read. A running request whose recipe file cannot be opened holds every run, where the
plan used to fail.

- Name a recipe and keep the name across its versions. An unnamed recipe is a version
  only of itself. A params edit of one keeps the runs that its earlier requests
  recorded.
- A run in a storage the recipe no longer names is not a candidate unless the
  submission retires the recipe that wrote it (the next entry), and a run written
  outside a request (`run_feature`, `Pipeline`) is judged as before.
- `plan_cleanup` takes a required `recipe_name`. `KeptReason` gains `recipe_uses`, and
  the wire's `reason` is a string. A fingerprint taken before the change for a plan
  with such a sibling is refused with `PlanChanged`; plan again.

**A submission can retire an earlier recipe, so a cleanup reaches the runs it wrote.**
A cleanup judged only the storages the planned recipe names. A run left in a storage the
recipe no longer names, because a step reads something else or was dropped, was outside
every later plan, and recipes iterated under new names kept each other's runs as
`recipe_uses`. `plan_recipe` and `submit_pipeline` take `retire`, and `mosaic pipeline
plan`, `submit` and `run --recipe` take `--retire`: digests of saved recipes. The runs
those recipes' requests wrote become candidates in any storage, and a request of a
retired recipe stops keeping them. It still keeps a run it only found already done, a
run it pinned, and a run that a feature call outside every request also used.

- A retirement belongs to one plan or submission. The request records it in
  `cleanup.retired`, and a later plan judges as if nothing was retired. It needs a
  cleanup. A digest that no saved recipe has, the planned recipe's own, and a recipe
  with a request that has not ended are refused (`retire_unknown_recipe`,
  `retire_own_recipe`, `retire_open_recipe`, `retire_needs_cleanup`).
- A plan asked for a cleanup lists the saved recipes whose requests wrote runs under
  `retirable`, and the digest to pass comes from there. The rest of the cleanup rules
  apply to a retired run: its hold, a costly feature, a reader, its age, and the
  `keep_newest` newest of its storage, so the default policy keeps the newest retired
  run of each abandoned storage. A step with no run id yet and the coverage rules do
  not apply to a run in a storage no step names.
- Retiring a recipe also reclaims the template runs that `extract-templates` 0.3 and
  `extract-labeled-templates` 0.4 wrote, which copied their input per entry.
- Additive on the wire: `PlanWire.retirable` and `retirable_omitted`,
  `CleanupPlanWire.retired`, `retired_runs` and `retired_omitted`,
  `SupersededRunWire.retired`, `CleanupRunWire.retired`, `SupersededRun.retired_by`,
  `CleanupPlan.retired` and `StepAttempt.target`. Each defaults empty.
- A request file without `retire` is unchanged byte for byte, and the fingerprint gains
  `cleanup.retire` only when a plan retires something. A mosaic 0.17 reader refuses a
  request file that holds `cleanup.retired`.
- A removed run's digest is recorded on the cleanup run-log's `run_removing` event, not
  on `run_removed`, which does not repeat the reason. The reason is `retired`, and the
  line of an ordinary removal is unchanged.
- New library: `retire=` on `plan_with_cleanup`, `plan_cleanup` and `submit_request`;
  `retirable_recipes`, `retired_recipe_names`, `RetirableRecipeWire`, `RETIRED_SAMPLE`,
  `RETIRABLE_REQUEST_IDS` and the errors `RetireNeedsCleanup`, `RetireUnknownRecipe`,
  `RetireOwnRecipe` and `RetireOpenRecipe` in `mosaic.core.pipeline.graph`;
  `current_digest` and `request_ids` public in `store`; `RemovalRecord.retired`, a new
  key of the published dict.
- `retirable` lists at most eight recipes over MCP, newest activity first, and only
  recipes with a run still on disk; a call retires at most eight, and
  `cleanup.retired_runs` lists five, removable first. Take a digest the list omits from
  `list_activity`. An earlier version of the planned recipe, one with the same name,
  never keeps a run, a retired one included. `retire` takes a recipe's digest in
  either canonical form.

## 0.17.0: a `vame` feature embeds keypoints, `kpms` fits and applies in one run, `mosaic agent serve` serves an MCP client, fits leave track ids out, a fit reads the columns it is given, and a TRex table is timed at its media's measured rate

**mosaic-behavior is on PyPI.** `pip install "mosaic-behavior[all]"` replaces the
clone and editable install, after the conda step that installs `ffmpeg`, `av` and
`py-opencv` (see the installation page). The Ultralytics, POLO, keypoint-MoSeq and
VAME environments are still built from files in the repository, which the wheel
does not carry. NOTICE now credits the file formats mosaic reads and the behavioral
map that follows Berman et al. (2014), and `CITATION.cff` and a citing page state
how to cite mosaic.

**A table that TRex tracks is timed at the measured rate of the media TRex
read.** TRex keeps a video's frame rate as an integer, so it timed a 29.97 fps
video at 29 fps. Its `time` ran 3.3% long, 124.0 s at the end of a 120.0 s
video, and every per-second column it exports was off by the same factor. Now
`time` is the frame divided by the measured rate, and `frame_rate` is that rate
on every row. Velocities, speeds and angular velocities are multiplied by the
measured rate over TRex's rate, accelerations and angular accelerations by its
square, and `timestamp` is divided by it. A video recorded at a whole number of
frames per second keeps its times and rates. A TRex run with
`track_enforce_frame_rate` turned off times its frames by their timestamps
instead, and its table should not be retimed.

**TRex tracking runs re-address from `0.2` to `0.3`, and the conversion cache
stays at `0.2`.** The version is a visible segment of an identifier and is not
hashed. A tracking run and its tracks variant move from `trex.0.2-<digest>` to
`trex.0.3-<digest>` with the same digest, and every cached conversion keeps its
`trex-convert.0.2-<digest>`. Existing TRex tables keep their identifier and
their timing. The next `mosaic track trex` with the same settings tracks each
entry again from its cached `.pv`, without detecting again, and publishes a
retimed table under the new identifier. An entry whose `.pv` is not in the
cache, because its media carries no content identity or the cache was swept, is
converted again. The entry then has two tracks variants, and a scope that
resolves both raises until `tracks_run_id=` names one or
`Dataset.drop_entries(..., run_id=)` retires the older. A feature computed from a TRex tracks variant
hashes that variant's identifier, so its run id moves with it.

A TRex export converted with `mosaic convert-tracks` keeps TRex's timing,
because mosaic does not know which media TRex read. The `trex_npz` converter
stays at `0.3`, and no `convert-trex_npz` identifier moves.

`mosaic.tracking.trex.version.TREX_VERSION` is replaced by `TREX_TRACK_VERSION`,
which names a tracking run, and `TREX_CONVERT_VERSION`, which names the
conversion cache.

**A TRex table with a numeric column that mosaic has not classified is refused
when the two rates differ**, as it already was when its `cm_per_pixel` is not 1.
Whether such a column is per second is unknown. Classify the field in
`mosaic.core.track_library.trex`.

`EntryHints` takes `fps`, the measured frame rate of the media a tool read, or
`None`.

**New feature `vame`.** It trains a recurrent variational autoencoder on windows
of egocentrically aligned keypoints and embeds each window in a continuous
latent space, at the window's center frame. It runs in a separate environment
you build, the same way keypoint-MoSeq does. See
[installation](docs/installation.md#vame). Mosaic's extras, dependency groups
and `uv.lock` do not install `vame-py`.

**`kpms` fits and applies in one run.** Its runner now uses the same wire as
vame's, sending requests and replies as JSON lines over standard input and
output instead of a Unix domain socket. The runner survives `save_state` instead
of shutting down before `apply` runs. After a fresh fit, every entry used to
fail at apply. The same run now applies the fitted model to every entry.

**A cancel of either runner is a hard cancel.** A fired cancel kills the
runner's whole process group at once, without a cooperative sentinel. A stopped
fit's weights are discarded either way. For every feature, an entry whose
`apply` raises after a cancel ends the run `cancelled` without recording an
`entry_error`.

**A `_CONDA_ENV` name runs the named environment's executables.** Mosaic looks
for the environment in each directory that `CONDA_ENVS_DIRS` lists, then in the
installation's `envs`, then in `~/.conda/envs`, in the Windows layout on
Windows. It reads the installation itself only for the name `base`. For an
environment found in neither `CONDA_ENVS_DIRS` nor the installation's `envs`,
every launch of the environment's `python` used to run the installation's
instead: SLEAP's probe and labels export, Lightning Pose, the Ultralytics and
POLO runners, and the decoder probe. They now run the environment's `python`, or
the bare name, which `conda run` resolves. `kpms` and `vame` refuse an
environment found in none of these directories. Add its directory to
`CONDA_ENVS_DIRS`, or set `MOSAIC_KPMS_PYTHON` or `MOSAIC_VAME_PYTHON` instead.

**External tools run without mosaic's `PYTHONPATH` and `PYTHONHOME`.** With
them, a tool's interpreter imported the packages of mosaic's environment,
`mosaic` included.

**`train-sleap` runs `sleap-nn train`.** sleap-nn 0.3.3 no longer installs the
`sleap-nn-train` script, so every `train-sleap` run against it exited at once
with code 127. The op now runs `sleap-nn train` with the same `--config-dir`
and `--config-name` flags, which sleap-nn 0.3.1 accepts as well.

**Three corrections to the features that read keypoints.**

- `pair-egocentric` used to read a negative `neck_idx` or `tail_base_idx` from
  the end of the keypoints without raising, and failed on an out-of-range one
  with a bare `IndexError`. It now refuses both before building a pair.
- `kpms` and `arhmm` now read a table with no `id` column in frame order. Their
  output changes only for such a table whose rows were not already sorted by
  frame.
- Extracting a keypoint's position now sets both of its coordinates to NaN
  when either one is not finite. `kpms` already treated such a keypoint as
  missing.

**A neighbor's track id is no measurement.** `nearest-neighbor` writes `nn_id`
and `attention-target` writes `attention_target_id`, each another individual's
track id. `feature_columns` returned both as payload, so every template,
scaler, embedding, clustering, `arhmm` fit, temporal stack and classifier over
either output fitted the id as a number, and a `farthest_first` template
selection compared rows on it. Both columns now join `BOOKKEEPING_COLS` beside
`fps` and `label`. The payload, the `Payload` requirement and recipe validation
leave them out. `temporal-stack`, `xgboost` and `lightning-action` drop them
from their output, as they drop `fps`. `global-tsne`, `global-kmeans` and
`global-ward` keep them unfitted beside `fps`. A reader of the neighbor reads
`nn_id` by name from `nearest-neighbor`'s output, as `nn-delta-response` and
`social-motion-summary` do.

**A merge keeps one copy of a bookkeeping column.** When two inputs both
contain a `BOOKKEEPING_COLS` column, the merge keeps the first input's and
drops the later one, as it drops a second identity column. It used to number
the later copy (`nn_id__1`, `fps__1`), and a numbered copy was read as payload.
Of two `nn_id` columns that name different neighbors, the later one is lost.

**A missing column names only the features that write it.** Recipe validation
and `input_candidates` no longer list a feature as a provider of a column that
it only copies from its input when the input has it.

**A fit reads the columns it is given.** `extract-templates`,
`extract-labeled-templates`, `arhmm` and `temporal-stack` take a `columns` param,
a list of payload column names. Unset, each reads every numeric payload column of
its input, as before. Set, each reads the named columns alone, and the scaler,
embeddings, clusterings and classifiers fitted on its templates fit those alone.
The list is sorted and each name kept once before it is hashed, and the order and
repeats of the names leave the run id unchanged. A name of an identity or
bookkeeping column, or of a column absent from the input, raises an error. The
error lists the input's payload columns. Recipe validation does not check the names,
because a column requirement on a list param is met by any one of its names.

**`global-scaler` writes only the columns it fitted.** Its output keeps the
identity and bookkeeping columns and the fitted columns, scaled, and drops every
other payload column. It used to keep every column of its input and scale only
the fitted ones. A `farthest_first` `extract-templates` over its output then
sampled the unfitted columns unscaled beside the scaled ones, and a templates
reference loaded with `ParquetLoadSpec(drop_columns=[...])` kept a column out of
the scaler's fit and left it in every later fit. Its declared output is no longer
a passthrough. It declares the identity and bookkeeping columns, kept when
present, and a family per payload column of its input.

**`nearest-neighbor` gives every individual a neighbor when another is missing.**
An individual without a position in a frame had a NaN distance to every other,
and `argmin` returns the first NaN it meets, so in any frame where one was
missing every row named it as the nearest and read NaN. Such frames were 16% of a
10-fish TRex recording's rows. A missing individual is now nobody's neighbor and
has none itself. `nearest-neighbor` moves to 0.3, the version segment only, and
every step reading its output moves its digest.

**A track conversion files a file's sequences under the group its source
declares.** `convert_all_tracks` and `mosaic convert-tracks` used `infile` when no
`group_from` was passed, so a source declared with `--group-from filename` had its
tracks filed under the group inside the file, such as CalMS21's `annotator-id_0`,
while `convert-labels` filed the same file's labels under the file's name. A label
lookup by group and sequence then found nothing, and every row read label 0. The
default is now `filename`, the label conversion's: the file-level group where the
scan recorded one, which it does only for a source that declares a grouping rule,
and the in-file group everywhere else. `--group-from infile` keeps the old naming.
The group is not part of a conversion's identity, so a dataset that declared a
rule and converts again keeps its run id and files its entries under the new
names.

**A plan refuses a tracker or inference step whose media awaits an analysis
transcode.** `plan_pipeline`, `mosaic pipeline plan` and the MCP tool
`plan_recipe` accepted a tracker or `infer-*` step over media that requires an
analysis transcode and lacks its analysis derivative, and the step failed when
it ran. The plan now reports a problem at that step's `after`,
naming the entries and the `transcode` step to add, unless a `transcode` step of
`"target": "analysis"` is above the step or the step names a media variant in
`media`. `submit` and `run` refuse the same recipe, and the step refuses itself
with `recipe_refused` when it starts. `Dataset.route_media_row` raises the new
`AnalysisDerivativeMissingError` in `mosaic.core.media.facts_columns`, a
`MediaProbeError`, for such an entry, and an attempt that fails on one records
that name as its error's `type`.

**Names moved or removed.**

- `mosaic.tracking.common.toolenv` is now `mosaic.core.pipeline.toolenv`.
  `mosaic.tracking.common` no longer re-exports `BinMode`, `ToolEnv`,
  `ToolExitError`, `ToolNotFoundError`, `conda_invocation`, `subprocess_env` or
  `tool_invocation`. Import them from `mosaic.core.pipeline.toolenv`.
- `kpms` declares `RunnerInterpreter(KPMS_ENV)` where it declared the removed
  prerequisite `KpmsEnvironment`. `vame` declares `RunnerInterpreter(VAME_ENV)`.
- `resolve_kpms_python` is removed. `locate_tool(KPMS_ENV, executable="python")`
  returns the invocation that starts the kpms runner's interpreter.
- The kpms runner's socket protocol is removed: the request and response models
  of `kpms_protocol`, its `encode_array`, `decode_array`, `send_message` and
  `receive_message`, and `kpms_server`'s `serve`, `recv_request` and
  `prctl_set_pdeathsig`. `kpms_protocol` now declares the payloads of the
  runner's commands, which mosaic sends in requests of `runner_wire`. The runner
  ships inside mosaic, and only mosaic builds its requests.
- `_process_apply_worker` in `mosaic.core.pipeline.run` is now
  `process_apply_worker`.

**Names added.**

- `MOSAIC_KPMS_CONDA_ENV` names a conda environment for the kpms runner,
  matching `MOSAIC_TREX_CONDA_ENV` and the rest.
- `MOSAIC_VAME_PYTHON` and `MOSAIC_VAME_CONDA_ENV` locate the vame runner's
  interpreter, matching `MOSAIC_KPMS_PYTHON` and `MOSAIC_KPMS_CONDA_ENV`.
- A feature may declare an optional `close()`, through the `ClosesResources`
  protocol. `run_feature` calls it once per attempt, in a `finally` around the
  run, to release a long-lived resource such as a subprocess.
- New API: `ClosesResources` in `mosaic.core.pipeline.types`; `locate_tool`,
  `ToolInvocation`, `ToolEnv.repo_venv`, `missing_interpreter` and
  `interpreter_refusal` in `mosaic.core.pipeline.toolenv`; `LineChild`,
  `SupervisedPopenKwargs` and `supervised_popen_kwargs` in
  `mosaic.core.pipeline.subprocess_util`; `CancelCheck` and `cancel_check` in
  `mosaic.core.pipeline.job`; `DatasetRoots` in `mosaic.core.pipeline.index`;
  `PoseConfig.names` and `PoseConfig.resolve_keypoints`; `KeypointArrays`,
  `keypoint_arrays`, `keypoint_suffixes`, `reindex_contiguous` and
  `resolve_keypoint_refs` in `mosaic.core.pose_columns`; the module
  `mosaic.core.individuals`, with `iter_individuals` and `refuse_repeated_rows`;
  `label_individuals`, `entry_identity` and `chosen_payload_columns` in
  `mosaic.behavior.feature_library.helpers`; `PayloadColumnNames` and
  `PAYLOAD_COLUMNS_DESCRIPTION` in `mosaic.behavior.feature_library.types`;
  `RunnerProcess`,
  `RunnerInterpreter`, `RunnerCommandError`, `RunnerVersionError` and `OnStart`
  in `mosaic.behavior.feature_library.runner_client`; `KPMS_ENV`,
  `KpmsExitError` and `KpmsFeature.close` in `kpms`; `VameFeature`, `VAME_ENV`,
  `VameNotFoundError` and `VameExitError` in `vame_feature`; the module
  `vame_sessions`; and the runner modules `runner_wire`, `vame_protocol` and
  `vame_runner` in `feature_library/external/`.

**Read verbs and dry runs leave the dataset unchanged.** `inventory`, `status`,
`runs`, `sequences`, `models provenance`, `pipeline plan`, `pipeline status`, the
`list` and `show` verbs of `sources`, `tags`, `notes` and `libraries`, and every
`--dry-run` used to create each declared root directory that was absent. A dry
run also took the index lock, which created a `.lock` sidecar and stamped the
index's modification time. Taking the lock no longer changes an existing index's
modification time. New in the library: `index_lock_unless_dry_run` in
`mosaic.core.pipeline.index_lock`.

**A failure under `--json` is one typed object.** It reports `kind` (`refused`,
`invalid`, `unavailable`, `not_found` or `conflict`), `reason`, `step`,
`message` and `detail`. `mosaic pipeline plan|run|submit|show`, `mosaic run` and
`mosaic track` used to write nothing to stdout on an invalid recipe, a refused
scope or model reference, an unbuildable step or a missing optional dependency.
They now write this object and still exit 1. A refused attempt's object, exit
65, keeps its keys and gains `kind`, `message` and `detail`. A refused plan
still prints the plan. New in the library: `ErrorKind` and `ClassifiedError` in
`mosaic.core.pipeline.error_kind`, and `ErrorEnvelope` and `error_envelope` in
`mosaic.core.pipeline.error_envelope`. `Refusal`, `RecipeInvalid`,
`StepBuildError`, `ScopeRefused`, `ManifestVersionError`,
`MissingOptionalDependency`, `ModelReferenceRefusedError` and
`ModelNotFoundError` subclass `ClassifiedError` and keep their other bases.
`Problem.as_json` returns a recipe problem as JSON.

**`mosaic inventory --json` prints the inventory wire.** It is
`InventoryWire`: `artifacts`, `unavailable_kinds`, `errors`, `total` and
`next_cursor`. Each artifact reports `kind`, `name`, `run_id`, `status`, a
`coverage` object (`covers_all`, `covered`, `target`, `missing_sample`,
`has_more`), `drift` as a count, `params_state`, the recorded `params`,
`tracks_variants`, `upstreams`, `execution_id`, `started_at`, `finished_at`,
`notes` and `ref`. The `dataset` key is removed. An artifact's `params` key used
to contain the params state and now contains the recorded params. `covered`,
`target`, `missing_sample` and `has_more_missing` moved into `coverage`, where
`has_more_missing` is `has_more` and each missing key is a list of its parts
instead of Python tuple text. `frame_axis_mismatch`, `frame_tail_short` and the
other per-kind sets are counted in `notes`, and their keys moved to `mosaic
inventory show`. The verb gains `--entry`, `--tracks-run-id`, `--status-not`,
`--limit` and `--cursor`, and `mosaic inventory show <ref>` reports one
artifact with its recorded params, orphaned rows and files, and drifted
entries. The plain listing prints the `ref` that `show` takes, in place of the
`kind`, `name` and `run_id` columns. A plan's JSON is unchanged.

The record of a feature run, a tracks variant or a labels variant lists its
index rows as orphaned only when at least one of its outputs matches an entry,
the rule that its status already followed. A plan measures such a run as the
inventory does, and reports a run whose outputs are all unmatched as `partial`
or `absent` instead of `inconsistent`. A variant whose tables are all gone
therefore lists no orphaned rows in `mosaic inventory show`.

New in the library: `InventoryWire`, `ArtifactWire`, `ArtifactDetailWire`,
`KeySampleWire`, `ResolvedWire`, `inventory_wire`, `coverage_wire`,
`key_sample`, `ref_address`, `InventoryQueryInvalid`, `ArtifactNotFound`,
`run_tracks_variants`, `reclassify`, `ARTIFACT_STATUSES` and
`is_artifact_status` in `mosaic.core.pipeline.inventory`, and
`dataset_inventory` and `register_inventory_contributors` in `mosaic.tracking`.
`CoverageWire` and `MISSING_SAMPLE_WIRE` moved from
`mosaic.core.pipeline.graph` to `mosaic.core.pipeline.inventory`, and
`CoverageWire.missing_sample` is typed as keys of any length.

**Importing `mosaic.tracking` leaves the trackers unloaded and the registries empty.** The
package and `tracking.common`, `tracking.frame_extraction` and `tracking.ops`
bind their public names on access, and `from mosaic.tracking import run_trex,
resolve_model, extract_frames` keeps working. Importing any tracking module used
to import every tracker, op module and frame extraction, and so registered every
tracking op, tracker index, tracks-axis reader and inventory contributor. Call
`register_ops` from `mosaic.tracking` before reading `OPS`, the reconcilable
indexes or those readers, as the CLI and mosaic-api do. `dataset_inventory`
reads the four tracking kinds without importing a tracker subpackage, a trainer
or frame extraction, and reads tracker runs from the `tracker` roots of
`TRACKING_ROOTS`. New in the library: `REGISTERING_MODULES` in
`mosaic.tracking.registration`; the module `mosaic.tracking.ops.trained_models`,
which defines `TrainedModelIndexRow`, `trained_model_index` and
`training_is_complete`, all still importable from `mosaic.tracking.ops.train`;
the module `mosaic.tracking.frame_extraction.frame_runs`, which defines the
frames index and `list_frame_runs`; `tracker_kinds` and `tracker_run_records` in
`mosaic.tracking.common.index`; and `mosaic.lazy_exports`. Removed:
`register_tracker_row_class`, `tracker_row_class`, `registered_tracker_kinds`
and `INVENTORY_CONTRIBUTOR_MODULES`. A submodule of `tracking.ops`, and
`tracking.ops` itself on `mosaic.tracking`, is bound only once it is imported
(`from mosaic.tracking.ops import train`). An unknown op's `KeyError` names
`register_ops`.

**`mosaic status --json` and `mosaic pipeline status --json` print the status
wires.** Both keep their keys. `status` prints `RunStatusWire` and adds
`outcome` (`partial` for a finished attempt that lost entries), `terminal`,
`progress_fraction`, `error` (the recorded `error_json` parsed, with a refusal as
an `ErrorEnvelope`), `failed_entries` (the first 20) and
`failed_entries_has_more`. `progress` is `null` without `--progress`. `pipeline
status` prints `RequestStatusWire` and adds `verdict` (`partial` for a finished
request with lost entries, and `orphaned` once a driver lease is recorded and
lapses), and per step `outcome`, `progress_done`, `progress_total`,
`progress_fraction`, `error`, `failed_entries` and `failed_entries_has_more`.
**One value changes:** a step with no run-log reported `status` and `run_id` as
`-`, and now reports both as `""`, with `outcome` `pending`. A reader that
matched `-` matches `""` or `outcome == "pending"` instead. The plain table
prints each step's outcome and the request's verdict, and plain `mosaic status`
prints the error's message and one line per failed entry. A progress event field
of the wrong type reads as 0, `""` or `{}`. Importing `mosaic.core.pipeline.graph` binds
its public names on access. A status read leaves pandas, cv2 and av unloaded.
New in the library: `RunStatusWire`, `RequestStatusWire`, `StepStatusWire`,
`FailedEntryWire`, `RecordedErrorWire`, `RequestVerdict`,
`FAILED_ENTRY_SAMPLE`, `run_status_wire`, `request_status_wire` and
`recorded_error` in `mosaic.core.pipeline.graph`; `AttemptOutcome`,
`attempt_outcome`, `EntryErrorRecord`, `ProgressEventRecord` and
`read_entry_errors` in `mosaic.runlog`; and `progress_done` and `progress_total`
on `StepAttempt`. `read_entry_errors` returns an `EntryErrorPage` (`items`,
`total`, `has_more`), each record gives the blob's `error_type` and `message`,
and a `limit` below 1 raises `ValueError`.

**`mosaic pipeline list` lists a dataset's pipeline requests, and `mosaic runs`
reads a page at a time.** `pipeline list` prints the requests newest first, each
with its recipe digest, owner, submission time, step count, status and verdict.
`--json` prints `RequestPageWire` (`items` and `next_after`). `mosaic runs
--limit N --after ID` reduces only the run-logs that one page reads. Under
`--json` a paged `runs` prints an object, `items` and `next_after`. Without
`--limit` and `--after`, `runs` prints the same array as before. In both verbs,
`--after` takes the previous page's `next_after`, and a request or attempt
created after that page was read leaves the next page unchanged. New in the
library: `list_requests`, `request_summary`, `rollup_verdict`,
`RequestSummaryWire` and `RequestPageWire` in `mosaic.core.pipeline.graph`, and
`list_runs` and `RunLogPage` in `mosaic.runlog`. `read_runs` keeps its signature
and its result. Importing `mosaic.core.pipeline.graph.store` leaves pandas, cv2
and av unloaded.

**`mosaic runs errors <execution_id>` lists the entities that one attempt
lost.** The report joins each entity with its failure record. It gives the
failures counted across requests, `not_before`, and a retry state of `ready`,
`waiting`, `quarantined` or `unrecorded`. `held_back_by_quarantine` is true when
any entity that the attempt lost is quarantined, listed or not, which keeps its
step back until a request with `allow_partial` proceeds without it. Every lost
entity also leaves the run short, and `allow_partial` answers that shortfall for
a `scope_dependent` step that reads the run. `--json` prints `EntryErrorsWire`,
and `--limit` lists more than the first 20. `mosaic runs` becomes a command
group, listed after the commands in `mosaic --help`. Its listing keeps its
options and its output. New in the library: `entry_errors_wire`,
`EntryErrorsWire`, `EntryErrorWire` and `RetryState` in
`mosaic.core.pipeline.graph`, and `error_summary`, `ErrorSummary`,
`EntryErrorPage`, `read_entry_error_keys` and `ERROR_MESSAGE_LIMIT` in
`mosaic.runlog`. A pipeline step records its attempt on each failure record, as
`last_execution_id`. `to_safe_name`, `from_safe_name`, `make_entry_key` and
`parse_entry_key` are defined in `mosaic.core.entry`, whose import leaves pandas
unloaded, and `mosaic.core.helpers` still exports all four.

**The step-type catalog has an index, and `kpms` is marked restricted.**
`catalog_index()` returns one row per step type of `editor_catalog()` without
its params schema, about 30 KB of JSON in place of about 480 KB. A row gives
`name`, `type`, `version`, `category`, `summary`, `lane`, `emits`, the labels of
`requires` and `provides`, `passthrough`, `payload_families`, `unavailable` and
`license_note`. `describe_step_type(name)` returns one `CatalogEntry`, and
raises `StepTypeNotFound`, a `not_found` error, for a name the catalog lacks.
`CatalogEntry` gains `license_note`, which is `None` unless a prerequisite
limits use. A prerequisite declares a limit by implementing `RestrictedUse`
(`mosaic.optional_dependency`), and `kpms` declares that keypoint-MoSeq's
license forbids commercial and fee-for-service use. The catalog lists the note
whether or not the license is accepted. `mosaic features list --json` prints
each feature's index row, keeping `name`, `version` and `category`, and the
plain table adds `lane`, `available` and `restricted`. `mosaic features describe
--json` keeps its keys and adds `summary`, `requires`, `provides`,
`unavailable`, `license_note` and `lane`. `mosaic pipeline catalog --json` adds
`license_note` to each entry.

**`mosaic preview` reads one entry's table of a feature run or a tracks
variant.** `preview_artifact(ds, ref, entry, *, columns, limit)` in
`mosaic.core.pipeline.preview` returns an `ArtifactPreview`: the parquet
footer's columns, `n_rows`, `frame_min` and `frame_max`, the first `limit` rows
(20 by default, at most 50), and the min, max, mean and NaN-or-null fraction of
each numeric column read. A column is typed `int`, `float`, `text` or `other`,
rows are positional in the order of `read_columns`, and a non-finite float is
`null`. `ref` is a `FeatureRunRef` or a `TracksVariantRef`, and `preview_ref`
reads one from the address that `mosaic inventory` prints. The preview takes no
lock, converts nothing and writes nothing. An unindexed entry or an absent file
raises `PreviewNotFound`, a `not_found` error, an indexed file that is not a
readable parquet raises `PreviewUnreadable`, a `conflict` error, and a malformed
ref, an unknown column or a limit out of range raises `PreviewInvalid`, an
`invalid` error. The statistics are read one column and one row group at a time,
and an integer column's min and max are exact.

**`read_parquet_column_names` and `output_n_rows` moved** from
`mosaic.core.pipeline.writers` to `mosaic.core.pipeline.parquet_reader`, which
holds every footer read.

**`mosaic describe` summarizes a dataset.** `dataset_summary(ds, *, found)` in
`mosaic.core.dataset_summary` returns a `DatasetSummary`: the manifest's name,
`uuid`, on-disk `manifest_version`, notes, tags, libraries and continuous
groups; the counts of sequences with a tracks table, indexed videos, tracks
variants, feature runs and trained models; the tracks tables per schema; and
the indexed videos per frame rate. A count is `null` for an artifact kind that
the inventory `found` did not report. Without `found` the summary reads an
inventory of `SUMMARY_KINDS` itself, and the CLI passes
`mosaic.tracking.dataset_inventory`, which reports models. The summary writes
nothing.

**A manifest's top-level `managed_by` key names the service that manages the
dataset.** `DatasetManifest.managed_by` returns its value, read from the
preserved keys, and `DatasetManifest.is_managed` whether it is non-empty. The
summary reports both, as `managed_by` and `api_managed`. The format version is
unchanged.

**`discover_datasets(root, *, max_depth=3)` in `mosaic.core.dataset_discovery`
finds the datasets below a directory** by their manifests and returns each as a
`DiscoveredDataset` with its manifest path, `uuid` and `name`, or the error of a
manifest that does not read. The walk follows symbolic links once and passes over
a directory it cannot read. It does not enter a hidden directory, or a directory
of a dataset named as one of its roots, such as `features` or `_tracking`.

**Opening a dataset and reading its inventory leave cv2 and av unloaded.**
`mosaic.core.media` and `mosaic.core.media.preprocess` bind their public names on
access, and `from mosaic.core.media import MultiVideoReader` keeps working.
Importing `mosaic.core.dataset`, `mosaic.core.pipeline.inventory` or a leaf of
`core.media` used to load both through the two package `__init__` files. Reading
any public name of `mosaic.core.media.preprocess` registers every built-in step in
`MEDIA_STEPS`. Importing `mosaic.core.media.preprocess.registry` alone no longer
does. An inventory read still loads pandas.

**A plan has a fingerprint, and a submission can require it.**
`plan_fingerprint(plan)` in `mosaic.core.pipeline.graph` digests the dataset's
manifest `uuid`, the recipe digest and, per step, its `run_id`, status, coverage
counts, drift count and asked entries. A new feature run or tracks entry
changes it. `PlanWire` gains `fingerprint`, which `mosaic pipeline plan --json`
prints, and the plain listing prints it on a first line, `plan <fingerprint>`.
`submit_request(..., expect_plan=)` and `mosaic pipeline submit --expect-plan`
raise `PlanChanged`, of kind `conflict`, before writing the recipe or the
request, when the plan resolved at submit has another fingerprint. `mosaic pipeline submit --json` gains
`plan_fingerprint`. `Plan` gains `dataset_uuid`. `asked_of` moved from
`mosaic.core.pipeline.graph.step` to `mosaic.core.pipeline.graph.plan`, and
`from mosaic.core.pipeline.graph import asked_of` keeps working.

**A submission reports the open requests of its recipe over its entries.**
`open_requests_matching(base_dir, recipe_digest, scope)` in
`mosaic.core.pipeline.graph` returns the ids of the requests of one recipe whose
scope overlaps *scope* (`Scope.overlaps`) and whose rollup is `running`, in
descending id order, which is creation order for ids that `submit_request` mints.
A request whose steps have not started counts as open. A feature runs without a
claim. Two such requests that run at once both compute it. A request file that
cannot be read is skipped with a warning. `submit_request` reports them as
`SubmittedRequest.duplicates`. `mosaic pipeline submit` and `mosaic pipeline run`
warn on stderr, and `submit` prints them under `duplicates` with `--json`.
`submit_request(..., refuse_duplicates=True)` and `mosaic pipeline submit
--refuse-duplicates` raise `DuplicateRequest`, of kind `conflict`, before writing
the recipe or the request. The check and the write of the request run under one
lock, `.mosaic/pipelines/requests/.submit.lock`, so two submissions on one mount
cannot both pass it.

**`mosaic pipeline run --request <id>` drives a submitted request, and
`--detach` runs the driver in the background.** The driver runs each unfinished
step as `mosaic run --graph-request` in a process of its own, in dependency
order, and submits nothing. Its steps write their output to
`.mosaic/drivers/<id>.log`, never to the command's own streams. A finished step
is skipped, and running the request again resumes it. A failed or refused step
ends the run, and the command exits 1. SIGTERM or SIGINT forwards SIGTERM to the
running step, records every step without a terminal record as `cancelled`, and
exits 130. `--detach` starts the driver in a new session, which outlives the
command, and `--json` prints `request_id`, `driver_pid` and `log_path`. The
driver holds a lease, `.mosaic/drivers/<id>.lease.json`, which it refreshes
every 10 s, which names the step whose process runs, and which it deletes when
the request ends. A running request whose lease names a process gone from this
host, or was not refreshed for 60 s, and none of whose step processes may still
run, reads `orphaned` in `mosaic pipeline status` and `pipeline list`, and is
not open to `--refuse-duplicates`. A step process counts on this host only when
it began by the start that its run-log records, so a pid reused after a reboot
does not. A second driver of a request with a live lease, or of a request with a
step process still running, is refused as a `conflict`, and so is a driver whose
lease another took over while it ran. A lease refresh that fails is reported on
standard error and retried, and a driver whose lease has gone unrefreshed for
30 s stops before its next step with `LeaseNotRefreshed`, a `conflict`. A run-log
that a resumed attempt appended a second `started` event to reports the
counters, the progress and the entry errors after that event, so the entities
that the dead process lost are not counted twice. `--recipe` is optional now, and exactly one of `--recipe` and
`--request` is given. New in the library: `run_request`,
`start_detached_driver`, `RequestRun`, `DetachedDriver`, `StepStillRunning`,
`record_step_ended`, `request_verdict_of`, `DriverLease`, `DriverRunning`,
`LeaseKeeper`, `LeaseLost`, `driver_alive`, `read_driver_lease`,
`refuse_live_driver`, `take_driver_lease`, `release_driver_lease`,
`LeaseNotRefreshed`, `live_step_process`, `StepProcess` and their paths in
`mosaic.core.pipeline.graph`, and `host`, `pid` and `started_at` on
`StepAttempt`; `record_exited_attempt` in `mosaic.runlog`; and
`create_exclusive`, `parse_instant`, `process_running` and
`process_running_since` in `mosaic.core.pipeline.markers`. `create_exclusive`, and so
`try_create_inflight`, publishes the file whole through a hard link, so a reader
never finds an in-flight marker or a lease created and still empty. `step_argv`
takes a command of several words as `program`.

**`mosaic cancel --request <id>` cancels a pipeline request.** A request whose
driver runs on this host receives SIGTERM through its lease, and the driver
stops its running step and records the others `cancelled`. Without a running
driver, each running step receives SIGTERM, a step whose process has exited is
recorded `cancelled`, and so is each unstarted step. A pid that names a process
begun after its attempt started, as after a reboot, is not signalled, and the
attempt is recorded `cancelled`. A request whose steps a queue dispatches is not
one this cancels: a step that the queue starts afterwards records its own ending
after the recorded `cancelled`. A request whose driver or a running step is on
another host is left as it is, and the command exits 1. Exactly one of
`--execution-id` and `--request` is given. Under `--json`, `mosaic cancel
--execution-id` adds `outcome` (`signalled`, `recorded`, `ended` or
`other_host`), `host` and `message` to its keys, and `pid` to an attempt that
had already ended. A failure to signal and an unknown attempt print the error
envelope. The refusal of a process on another host now reads "a cancel signals
processes on this host only". New in the library: `cancel_attempt`,
`cancel_request`, `AttemptCancel`, `RequestCancel`, `CancelOutcome`,
`AttemptNotFound`, `RequestNotFound` and `CannotSignal` in
`mosaic.core.pipeline.cancel`.

**`mosaic.agent.mcp` declares the tools an MCP client calls, and answers them
from the library.** `mosaic.agent.mcp.spec` declares fifteen tools, each with a
description for a model, an input and an output model, annotations and the
conditions under which it asks the user to confirm. It imports no MCP SDK and
neither pandas, cv2, av, the feature library nor `mosaic.tracking`.
`LocalBackend` in `mosaic.agent.mcp.local` serves the datasets of an allowlist
by manifest `uuid`, opened without creating roots, and returns every failure as
the error envelope. `submit_pipeline` requires the fingerprint of a current
plan, starts the request's detached driver, and refuses until confirmed when a
step runs on a GPU lane, trains, tracks or infers, or is asked for more than 100
entries, when `allow_partial` or `bind` is set, when the submission covers the
whole dataset, or when a request of the recipe over an overlapping scope is
open. `cancel_request` asks to confirm when a running step loses its work. A
dataset whose manifest names `managed_by` refuses both unless the backend was
built with `allow_managed`. `ArtifactKind`, `ArtifactStatus` and their value
tuples and guards are defined in `mosaic.core.pipeline.artifact_kinds`, which
imports the standard library only, and `mosaic.core.pipeline.inventory`
re-exports them. `count_cell` in `mosaic.core.pipeline.tracks_index` reads a
count cell of the tracks index, blank as `None`. `load_request` and
`list_requests` raise `RequestUnreadable`, a `ValueError` with reason
`request_unreadable` naming the request and its file, for a request file that
is not valid JSON or not a valid request, and `read_request` in
`mosaic.core.pipeline.cancel` reads a request or raises `RequestNotFound`.

**`mosaic agent serve` serves mosaic's tools to an MCP client over stdio.** It
needs the new `mcp` extra (`pip install "mosaic-behavior[mcp]"`), which `all`
does not include. The server lists the fifteen shared tools, each returning
structured content that validates against its output schema and a one-paragraph
summary, and reports a failure as an error result with the error envelope. A
tool refused with reason `confirmation_required` asks the user through form
elicitation when the client declares it, and then ignores a `confirm` that the
model passed. The answer confirms only a call with the arguments the user was
shown. Otherwise the tool refuses until the call is repeated with `confirm:
true`. A declined confirmation returns reason `confirmation_declined`.
Resources: `mosaic://guide`, `mosaic://catalog`, `mosaic://catalog/{name}`, and
per dataset `manifest`, `inventory{?kind,status}`, `recipes/{digest}`,
`requests/{rid}`, `runs/{eid}`, `runs/{eid}/log{?tail}` and
`artifacts/feature/{name}/{run_id}/params` under `mosaic://dataset/{uuid}/`.
Prompts: `build_pipeline`, `diagnose_dataset`, `explain_request`, `compare_runs`
and `summarize_dataset`. The server serves the datasets named by `--dataset` and
by a YAML `--config` file (`datasets`, `discover`, `allow_managed_writes`), and
`--allow-managed-writes` lets it change a dataset whose manifest names
`managed_by`. `mosaic agent install --mcp [--dataset PATH]` writes the `mosaic`
entry of the project's `.mcp.json` and keeps its other servers. New in the
library: `ToolSpec.input_schema` and `ToolSpec.output_schema`,
`MAX_SCHEMA_BYTES`, `CONFIRMATION_DECLINED` and `confirmation_declined` in
`mosaic.agent.mcp.spec`; `LocalBackend.dataset`, `LocalBackend.catalog`,
`ToolFailure`, `answer` and `discovered_manifests` in `mosaic.agent.mcp.local`;
the modules `config`, `prompts`, `resources`, `summaries` and `server` of
`mosaic.agent.mcp`; and `MCP_CONFIG`, `MCP_SERVER_NAME`, `mcp_server_entry`,
`with_mcp_server`, the `mcp_datasets` argument of `install_agent_files`,
`Installation.mosaic_command` and `Installation.interpreter_command` in
`mosaic.agent`.

**A step-type search ranks its matches.** `search_step_types` and the new
`mosaic features list --query TEXT` rank a step type first when its name is the
query. Next come the names whose words each query word equals or begins, then
the names that contain every query word, and last the step types matched partly
or only by their summary or category. Within one of these, a name ranks higher
with more words that equal a query word or its plural, so `track` lists `tracks`
before `track-subsample`. Case, spaces, hyphens and underscores are alike, so
`speed_angvel` finds `speed-angvel`. Each word of a hyphenated query now matches
on its own. `per-frame` used to need that text, and now needs `per` and `frame`
anywhere in the name, summary or category. `search_step_types` used to list its
matches by name, so `speed` listed `nn-delta-bins` before `speed-angvel`. A
search cursor records the ranking. Without a query, both list by name as before.
New in the library: `search_catalog` and `CatalogMatch` in
`mosaic.core.pipeline.graph`.

**[docs/agents.md](docs/agents.md#tools-for-an-mcp-client) documents the MCP
server**: its installation, its registration in Claude Code and Claude Desktop,
the datasets it serves, when it asks to confirm, the operations it leaves to
the user, and the values its tools send to the model provider. The
installation page lists the `mcp` extra.

**Identifiers that move.** `extract-templates` and `extract-labeled-templates`
move to 0.3, `arhmm` to 0.2, `temporal-stack` to 0.5, `xgboost` to 0.2,
`lightning-action` to 0.3 and `global-scaler` to 0.2. Their run ids move on every
dataset, with or without `nn_id`. The `columns` field enters the hashed params of
the first four even when unset, and their digest moves as well as their version.
The other two and `global-scaler` move the version segment only. `xgboost` and
`lightning-action` move because a templates reference that resolves to an
`extract-labeled-templates` 0.2 run would otherwise fit other columns under the
same run id. A step reading one of these outputs hashes its run id, and its
digest moves too: the t-SNE step of the plan golden moves from `0.4-9ef3a62a39`
to `0.4-3f5b267c0b`. TRex tracking runs move to `trex.0.3`, as the TRex entry above
describes, and a feature computed from one moves with it.

## 0.16.0: every feature declares its columns, an editor reads the pipeline graph, and global fits skip non-finite rows

**Every registered feature declares the columns that it reads and writes.**
`requires` names the columns that `fit` and `apply` read from the merged input,
and `provides` the columns of the output. Both are class attributes beside
`emits`, written with the references of `mosaic.core.pipeline.column_contract`: a
column by name, by a params field or by a template over the params, a prefix
family, `AnyOf`, `Keypoints`, `Payload`, `Derived` and `When`. `run_feature`
checks each entry's merged frame before `fit` and `apply`. An entry that lacks a
column fails with `MissingInputColumns` through the per-entry error path, in one
message form for every feature. A feature registered outside mosaic may declare
neither, and validation lists its steps as unchecked.

**Recipe validation checks each step's columns.** It propagates, with no
dataset, the columns that each step's output is guaranteed to contain and the
columns that it may contain. A required column absent from every input is a
problem naming the providers readable by the step. A column that an input may
contain is a caution, and a caution does not refuse a run.
`ValidationReport.unchecked` lists the steps whose feature declares no
`requires`, and the CLI prints them on one line.

**A pipeline editor reads the graph as JSON.** `mosaic.core.pipeline.graph.wire`
defines the shapes. The CLI's `--json` output and mosaic-api's responses are
these models dumped.

- `editor_catalog()`, and the new `mosaic pipeline catalog`, list every step type
  that a recipe may run: its declaration, its params JSON Schema, the params
  fields that accept a step reference, and `unavailable` when this installation
  cannot run it.
- `validation_report(document)` returns every problem, caution and unchecked
  step of a document that need not parse.
- `attach_candidates` returns the step types that may be added below a step.
  `input_candidates` returns the earlier steps that may become an input of a
  step. `insert_candidates` returns the step types that may be inserted between
  a step and the steps whose first input reads it. A refused candidate names its
  reason.
- `plan_wire(plan)`, which `mosaic pipeline plan --json` prints, tags each reason
  by kind and bounds the missing entries to a sample.
- `request_verdict` is the rule of `request_rollup`, for a control plane that
  ends an attempt before its run-log is written.

**A plan refuses a recipe that the dataset cannot run.** `plan_pipeline` raises
`RecipeRefused`, a `RecipeInvalid` that contains the plan and its problems
(`Plan.problems`, `PlanWire.problems`). Each problem is placed where the recipe
would change:

- a time filter on a dataset without `meta.fps_default`, at
  `run.filter_start_time` or `run.filter_end_time`, where a bare `ValueError` was
  raised;
- a required column that the tracks tables of a step lack, read from their
  parquet footers, as `ANGLE` on `mosaic_v1` tracks;
- two inputs of one step that resolve to one run only on the dataset.

A queued step refused when it re-plans at its start raises
`RecipeRefusedAtStep`, with the new `RefusalReason` `recipe_refused`, and exits
65. `mosaic pipeline plan`,
`submit` and `run` report a refusal and exit 1.

**A feature declares its prerequisites beyond the base install.** Its
`prerequisites` tuple of `OptionalModule`, `KpmsLicense` and `KpmsEnvironment`
is evaluated, without importing the packages, into `Declaration.unavailable`: a
sentence naming what to install or enable. Validation reports it at the step's
`feature`, and `attach_candidates` refuses the feature with it.
`movement-smooth`, `movement-filter-interpolate`, `feral`, `lightning-action`,
`kpms` and the three identity models declare theirs.

**The global fits skip non-finite rows.** A `speed-angvel` step with `step_size`
writes NaN on each sequence's edge rows, and `global-kmeans` raised
scikit-learn's "Input X contains NaN" on templates sampled from them.
`extract-templates` now samples only rows finite in every template column, and
`template_provenance.parquet` gains a `dropped` column. `global-kmeans`,
`global-ward` and `global-tsne` fit on their finite templates, the rows that
their apply reads, and the SMOTE path of `xgboost` synthesizes from the finite
rows. The farthest-point template selection compares two rows over the columns
finite in both, and takes each row once.

**Identifiers that move.** `extract-templates` and `extract-labeled-templates`
move to 0.2, the version segment only. A step reading their output hashes their
run id, and its digest moves too: the t-SNE step of the plan golden moves from
`0.4-d3058395ad` to `0.4-9ef3a62a39`.

**Features that read or write differently.**

- `approach-avoidance`, `pair-position` and `orientation-rel` accept a feature
  output beside the tracks, and read `heading`'s `ANGLE` from it.
- `ffgroups-metrics` declares `emits` `"individual"`, for the row per id and
  group size that it writes.
- `frame-aggregate` and `collective-motion-metrics` aggregate a table ordered by
  its time, where pandas raised "cannot insert time, already exists".
- The color path of `interaction-crop-pipeline` asks for NVENC, then SVT-AV1,
  then OpenCV's writer, where a build without `av1_nvenc` failed the feature.

**The toolkit does not own a params field.** The catalog lists every params
field, `filter_expr` and `egocentric-crop`'s `output_root` among them. A control
plane setting a field itself removes it from the schema that it serves, and
refuses it in a recipe, as mosaic-api does.

**New API.** `MissingInputColumns` and the reference types of `column_contract`;
the wire models and `editor_catalog`, `read_recipe`, `validation_report`,
`input_candidates`, `insert_candidates`, `resolved_columns`, `resolved_levels`,
`plan_wire`, `request_verdict`, `RecipeRefused` and `RecipeRefusedAtStep` in
`mosaic.core.pipeline.graph`; `OptionalModule`, `Prerequisite` and `unmet` in
`mosaic.optional_dependency`; `BOOKKEEPING_COLS` and `NON_PAYLOAD_COLS` in
`mosaic.core.pipeline.types`; `read_parquet_column_names` in
`mosaic.core.pipeline.writers`; and `mosaic.core.narrowing`, where
`is_str_mapping` moved from `mosaic.core.pipeline._utils`.

## 0.15.0: an H.264 copy of an entry needs no step, and a control plane can own a tracker's model

**A `preprocess` variant with no steps and `codec: "h264"` is an H.264 copy of its
source.** It holds every frame of the entry media, for a tool whose reader lacks
AV1, such as Lightning Pose's DALI reader. A step-less AV1 variant is still
refused, because it is the entry media itself. No identifier moves.

**A tracker's parameters can be narrowed by a control plane.** mosaic-api sets a
tracker's model, its `media` and its execution knobs itself, and removes those
fields from the subclass of the op's parameters that it exposes. `SleapParams`'
device check no longer requires its field, and `MediaInputParams`' frame-window
check passes a subclass without `media`.

**New API.** `TrainingProvenance.keypoint_count`, the keypoints a pose model
predicts per instance, read from the `kpt_shape` of its prepared `data.yaml`.
`as_json()`, and so `mosaic models provenance --json`, carries it as
`keypoint_count`.

## 0.14.0: the Ultralytics runner reads an entry's files, a caller can ask what an entry needs, and six extras are retired

This release carries every section below it down to 0.13.0.

**The Ultralytics runner reads an entry's files, and nothing is joined or
exported for it.** Ultralytics tracking, `infer-pose` and `infer-points` hand
their runner an entry's files in order, which it reads on one frame axis: the
clips, at any mix of frame rates, and the chunk files of an imgstore whose chunks
are video. `export-joined` no longer runs for these three. `export-store` runs
only for a store whose chunks are images, raw arrays, or Bayer or YUV data, or
whose chunks a probe finds disagreeing with the store's index. No run identifier
moves, and output tracked from a join or an export is reused.

**A tracks row from `infer-pose` or `infer-points` over a store names
`media_raw`.** `consumed_source_roots` names the store, as a tracker's row
already did, where it named the export under `media`.

**A caller can ask what an entry needs before a tool reads it.**
`required_media_ops(ds, *, kind, entries=None)` in
`mosaic.tracking.common.tool_input` returns one `MediaRequirement` for each
entry that `kind`'s run would refuse. It carries the entry, a `cause` from the
closed `MediaRequirementCause` set (`no_join`, `superseded_join`,
`several_joins`, `unidentified_clips`, `store_export`, `unjoinable`), the `op`
that meets it, `reencode` for a join, and `reason`, the run's own refusal text.
It makes the run's checks without probing a file. `TrackingRoot.reads`
(`one-file`, `entry-files`, `in-process`) declares how each tool reads an entry.

**Lightning Pose is refused an AV1 file at once.** `ToolDecoder.never_reads`
declares the codecs that a tool's reader decodes in no environment. Lightning Pose
declares AV1, which DALI's reader decodes on no GPU, so an AV1 file is refused
with the H.264 remedy before Lightning Pose's environment starts, where a DALI
decode test used to run first. `MOSAIC_ALLOW_TOOL_CODECS=av1` still hands it over.
A caller building Lightning Pose's input reads the set to make an H.264 variant.

**Names moved or added.**

- The runner's `TrackRequest` and `InferRequestBase` replace `video_path` and
  `media_facts` with `sources`, a list of `SourceFile`. The runner ships inside
  mosaic, so no other program builds these.
- New: `entry_runner_sources`, `ToolFile`, `request_sources`, `source_windows`,
  `readable_chunks`, `NativeStore.chunk_spans`, `required_media_ops`,
  `MediaRequirement`, `MediaRequirementCause` and `ToolDecoder.never_reads`.

**Six extras are retired.** `recommended`, `identity`, `localizer`, `gpu`, `pose`
and `polo` are no longer declared. pip warns about an unknown extra and installs
the base, with no torch, so replace each in a saved command:

- `recommended` with `all`;
- `identity`, `localizer`, `pose` and `polo` with `deep-learning`;
- `gpu` with `faiss`.

## Unreleased: a codec refusal exits 65, and a cancel is recorded as cancelled

**A tracker's codec refusal is a refusal.** When a tracker refuses to hand its
tool a file whose codec the tool's environment does not decode, `mosaic run`,
`mosaic track` and `mosaic pipeline run` exit 65 (`REFUSED_EXIT_CODE`) with the
reason in `error_json`, where they exited 1 with a traceback. The reason,
`undecodable_codec`, joins the closed `RefusalReason` set. The refusal comes
when the tracker reaches the entry. Earlier entries of the run may already be
published. `mosaic pipeline run` exits 65 for every refusal, a
`coverage_shortfall` included, where it exited 1, and prints the refusal JSON
under `--json`.

**A cancel ends an attempt `cancelled`.** A tool killed on a cancel was
recorded as a failed attempt, and a transcode or preprocess entry that a cancel
ended as a lost entry.

**Names moved or removed.**

- `REFUSED_EXIT_CODE` and `RefusalReason` are in
  `mosaic.core.pipeline.refusal`, and `mosaic.core.pipeline.graph` no longer
  exports them.
- `ToolCodecError` moved from `mosaic.tracking.common.tool_input` to
  `mosaic.core.pipeline.tracking_roots`.
- `TrackingRoot.joins_sources` is removed, because mosaic joins a multi-clip
  entry into one file for every tracker.

**New on disk.**

- `pose_id` and `pose_name` on the prepared-data index: the pose the annotation
  sets were narrowed to, as provenance.
- `observed.models` in a tracks variant's `params.json`: the models that made
  the variant, kept as a union across the runs that wrote it.
- `tracks/<variant>/params.json.lock`, the lock beside that record, which is
  created once and never removed, as every index lock is.

**New API.** The op declaration `model_reference`, and `resolve_op_model`,
`delete_trained_model`, `tracks_made_with`, `join_needs_reencode` and
`narrowed_pose`. `MissingJoin` gains `cause`, one of the closed
`MissingJoinCause` set (`no_join`, `superseded_join`, `several_joins`,
`unidentified_clips`). A caller can name the remedy from it without reading the
message. For `unidentified_clips` that remedy is `mosaic reprobe-media --apply`
before the join. The error behind it, `EntryJoinMissingError`, is a
`JoinedExportMissingError` with the same `cause`.

## Unreleased: the frame-axis check compares the frames each tool read

**The tracks index gains a `frames_read` cell, and the frame-axis comparison
reads it instead of a table's last frame.** A table's last frame is the last
frame an animal was seen in. A recording that ended with nobody in view read as
a short axis. `frame_axis_mismatches()`, the bridge's `frame_axis_mismatch`
run-log event and `mosaic measure-tracks` now compare how many frames the tool
read with `media_frames`, the frames it should have read. `media_frames` is now
recorded for every run that read its whole entry, one clip included, and for a
media variant that keeps every source frame. No identifier moves and no table
changes.

Names moved with the meaning: the run-log event's `tracked` field is `read`,
`FrameAxisMismatch.tracked` is `read`, and the `tracked_frames` key of `mosaic
measure-tracks --json` is `frames_read`. That command also reports
`frames_read_measured`.

The Ultralytics run index's `n_frames` is the frames the runner read on every
run. A run that reused its tracking recorded the frames with a detection
instead, and now reads the count from the response the runner left beside its
predictions, or keeps the count its row records when that response is gone.

**A TREx table short by its known tail loss is reported apart from a mismatch.**
TREx never reads the frames its decoder holds back to reorder at the end of a
file, and reads one fewer when the container omits a frame count. Measured
with TREx 4b4860187e on 26 files from 60 to 6,000 frames and on joins of three
clips, that is 2 frames of an H.264 or HEVC file with B-frames, 1 with one
B-frame, none of an AV1 file or of one without B-frames, and 1 more in Matroska.
The `trex` tracking root declares the loss as that rule of the file
(`TrackingRoot.tail_loss`). Each tracks row records in a new `known_tail_loss`
cell what the rule gives the file TREx read, from the header of the source its
`.pv` records. A `.pv` made from several files is allowed none, because TREx
loses frames at every boundary. So is a row of a `trex.0.1` run of several clips
whose `.pv` is swept. The TREx run index row of that run counts and lists the
clips it handed TREx. A row whose file's header was not read is allowed the most
TREx loses on any file, 3, and every report says so. A shortfall within the
allowance is a `frame_tail_short` run-log event, counted as
`entries_frame_tail_short` on the snapshot, named under
`extra["frame_tail_short"]` on the inventory record, and listed by
`frame_tail_shortfalls()` and under `frame_tail_short` in `mosaic measure-tracks
--json`, with the frames `allowed` and whether the allowance comes from the
file's header (`allowance_known`). A larger shortfall, an overshoot, and any
shortfall of another producer stay a `frame_axis_mismatch`. A count cannot show
that the missing frames are at the end, and each report says so. `mosaic
measure-tracks` fills `known_tail_loss` for older rows in a fourth pass,
`backfill_known_tail_loss`, and its JSON adds `known_tail_loss_measured`,
`known_tail_loss_cleared`, `known_tail_loss_not_established` and
`known_tail_loss_unregistered`. A row whose media has changed since its run
keeps its loss, because the file at the path its `.pv` records is then not the
one TREx read.

A row written before this change has no `frames_read`, and is not compared until
`mosaic measure-tracks --apply` fills it from what its run left on disk: the
`.pv` of a TREx conversion, the response the Ultralytics runner wrote for the
tracker and for `infer-pose` and `infer-points`, or a Lightning Pose table.
SLEAP records no `frames_read` at all, and its tables are not compared. Nor are
`infer-localizer` tables published before this change, whose runs left nothing
on disk that tells.

**`mosaic measure-tracks --apply` now rewrites `media_frames` and `frames_read`
to what the rule gives each row.** An earlier `--apply` filled `media_frames`
for runs under a frame window and for converted tables, which the rule leaves
blank. Such a cell beside a filled `frames_read` reads as a mismatch that is
not there. The pass clears those cells and rewrites a count that disagrees with
the rule. A row whose run cannot be established keeps its value: an unmounted or
moved media root, a lost variant record, media changed since the run, or files
the run left that are gone. So does a tracker row whose op is not registered,
because the process has not imported `mosaic.tracking`. That row is reported
apart as unregistered. `backfill_media_frames`, `backfill_frames_read`,
`Dataset.measure_media_frames` and `Dataset.measure_frames_read` return a
`CellBackfill` of the rows written, cleared, not established and unregistered,
in place of a frame of the rows filled, and `mosaic measure-tracks --json` adds
`media_frames_cleared`, `media_frames_not_established`,
`media_frames_unregistered`, `frames_read_cleared`,
`frames_read_not_established` and `frames_read_unregistered`. The
`media_frames` pass reads the media index, the sequence projection and each
variant's record once for all its rows, and the `known_tail_loss` pass reads the
sequence projection once. These three passes take the tracks-index lock only to
write. A producer publishing during a pass no longer waits for it, and a row
published while a pass runs keeps the values it was published with.

## Unreleased: inference tables are timed in seconds and numbered by source frame

**`infer-pose` and `infer-points` re-address from `0.3` to `0.4`, and
`infer-localizer` from `0.2` to `0.3`.** The version is a visible segment of the
op run and tracks variant identifiers and is not hashed. Tables written under
the older versions keep the identifiers they have. Re-run such a table where its
`time` column is read, or its `frame` column after a run with a frame window.

Two columns changed meaning. No model runner reports a time, and the bridge
filled `time` with frame numbers. A table from one clip had frame numbers there,
while a table from several clips was retimed into seconds. Every table's `time`
is now in seconds, each frame timed at the rate of the clip it came from. Under
`start_frame` or `frame_step`, a row's `frame` was its place among the frames
that the model read. A run over frames 5, 7 and 9 of the video numbered its rows
0, 1 and 2. It is now the frame that the model read, as a tracker's is. A run
without a frame window numbered its rows by the source frame already.

## Unreleased: a table from several clips is timed by each clip's rate

**A SLEAP, Lightning Pose or Ultralytics entry of several clips is tracked again
on its next run, with no `--overwrite`.** A table of such an entry made before
this change may cover only its first clip. One that covers every clip times each
frame at the first clip's rate, which is wrong when the clips were recorded at
different frame rates. Their markers record the first clip's identity alone.
The reuse gate of these trackers now compares the whole clip set, the ordered
composition of the clips' identities, and the next run replaces the table. An
entry of several clips of which one lacks a content identity is refused
instead, until `mosaic reprobe-media --apply` mints one. TRex tables were
already timed by each clip, and the inference ops predict again on every run.

An entry of several clips that includes an imgstore recording is refused before
any tool runs, for the four trackers, `infer-pose` and `infer-points`. Each
hands its tool one file, and `export-joined` does not join stores.
`infer-localizer` reads stores itself and refuses them only at different frame
rates. Either refusal names a `preprocess` command that makes a media variant of
the entry, a trim over all of its frames, to name with the `media` parameter.
Two stores whose estimated rates differ by at most 5 parts in 10,000, such as
30.0 and 30.002 fps, are one rate measured twice, at any length. The store
reader reads them as one axis, and a table read from them, directly or through a
media variant, keeps the columns that one rate computes.

## Unreleased — every inference table carries a body centre, and three inference identifiers move

**`infer-pose`, `infer-points` and `infer-localizer` re-address: `0.2 → 0.3`,
`0.2 → 0.3` and `0.1 → 0.2`.** The version is a visible segment and not a hash
term, so nothing is re-derived and existing runs keep the names they have. But
**tables written under the older versions carry no `X`/`Y` at all and should be
re-run**: nothing that reads a body centre can use them, and the bump is what
keeps a variant directory from holding both shapes under one identifier.

**All three inference ops were writing `mosaic_v1` tables without the two
columns that schema requires.** Each declares `output_schema="mosaic_v1"`, which
requires `X`/`Y` and defines them as the individual's body centre; none of them
wrote the pair. `infer-pose` emitted keypoints and no centre, while
`infer-points` and `infer-localizer` emitted the centre under the lowercase
`x`/`y` their raw prediction tables use. The bridge validated with
`strict=False`, where a missing *required* column is printed rather than raised,
so the report went out under a line reading `completed` and the table was
written and indexed. Downstream that is not an incomplete table but a silently
useless one: the overlay's `centroid_cols` yields `None`, `nearest-neighbor` and
the social-force chain find no column, and `egocentric-crop` and
`interaction-crop-pipeline` fall back *to* the centre that is missing.

The bridge now derives it — the mean of the detected keypoints for a pose model,
which is what every tracker bridge already did and what a midline skeleton's
centre is by construction; a rename for the two producers that reported the
position already. The coercion stays at the bridge: the predictions parquet
beside each entry is an audit artifact showing what a detector emitted *before*
schema coercion, and the point columns are a wire contract crossing a subprocess
boundary, so neither had to move. **That one call site now validates with
`strict=True`**, so the shortfall raises instead of printing; every tracker write
path is unchanged, and whether a missing required column should refuse there too
remains a separate question.

**One `keypoint_centroid`, in `mosaic.core.pose_columns`.** It was written out
longhand in five converters and copied into two visualization features, and the
copies had drifted. `pose_column_pairs` moves there from
`core.pipeline.loading` — a converter takes no import from `core.pipeline`, and
routing four of them through it for two functions would have put the pipeline's
loading surface on every converter's import path. **No tracks variant moves**:
the shared rule is the one four of the five converters already applied.

**A CalMS21 conversion no longer loses a frame's body centre to one unlabelled
landmark.** It averaged with a plain `mean` where the others used `nanmean`, so
a single NaN landmark made `X`/`Y` NaN for that frame — indistinguishable from a
genuinely absent animal. Corrected rather than versioned: `nanmean` is what the
column always meant, and standard CalMS21 keypoints are dense, so no published
conversion changes. `interaction-crop-pipeline` gains the matching fix on its
own side, where two independently filtered column lists could average one
keypoint's x against another's y.

## Unreleased — the SLEAP and Lightning Pose training ops can be given a device, and a short run

**No run on disk is re-addressed.** `device` is `HASH_EXCLUDE` on both ops, and
Lightning Pose's newly derived epoch keys are computed inside the trainer, below
`Params`. The golden digests in `tests/data/op_identity_golden.json` do not move,
and a model already trained stays a cache hit.

**`train-sleap --params '{"device": ...}'` works, and means what it means
everywhere else.** Any value but `auto` used to die during Hydra config
composition before a single epoch, because the key was injected bare into a
document that does not carry it. It now takes `auto`, an accelerator family
(`cpu`, `gpu`, `mps`) or a comma-separated list of CUDA indices (`0`, `0,1`),
the spelling `train-pose` and `train-localizer` already use, and translates to
sleap-nn's `trainer_accelerator` plus `trainer_device_indices`. Selecting GPU 1
was not expressible before. An unusable value is refused when the run is
submitted, so mosaic-api answers it with a 422.

**A `sleap_overrides` key needs no `+` prefix.** `train_sleap` compares each
override against the config it just wrote and appends the ones that config does
not carry, leaving an explicit `+` or `~` as written. `data_config.preprocessing.scale`
and `data_config.preprocessing.crop_size` used to fail the same way `device`
did. So did resuming from a `base_model`, which injects
`trainer_config.resume_ckpt_path`.

**`train-litpose` completes below 250 epochs.** `max_epochs` scaled only itself,
while the config carries four keys tied to it, so every short run — every smoke
test — failed, either at `ModelConfig.validate` or, between three and four
epochs, after paying for every epoch with no checkpoint to show. `min_epochs`,
the multi-step `milestones`, `check_val_every_n_epoch` and `unfreezing_epoch`
are now derived from `max_epochs`, and at the template's own 300 they reproduce
the template's own values.

**`TrainLitposeParams.device` reaches the trainer, and its schema no longer
carries `x-mosaic-unwired`.** A client that hid the control because it changed
nothing should now offer it. It names CUDA devices, as a comma-separated list of
indices, through `CUDA_VISIBLE_DEVICES` and `training.num_gpus`. There is no
`cpu` setting and one is refused: Lightning Pose fixes its trainer's accelerator
to `gpu`.

**`train-sleap` refuses two runs it used to accept.** An environment holding
`sleap-nn-train` but not `sleap_nn` — which is what `pip install sleap` without
the `nn` extra leaves, since the console script belongs to the base
distribution. And a `multi_class_` head against a `.slp` whose instances carry
no track, which trained against no classes at all and produced a model that had
learned nothing. Both are decided from one probe of the SLEAP environment, run
after the reuse gate and before the run root is claimed.

**`metrics_path` on a `models/<kind>/index.csv` row holds metrics or holds
nothing.** `train-sleap` and `train-litpose` recorded their `config.yaml` there,
which exists, so the index advertised a metrics path with no metric in it. They
now record sleap-nn's `training_log.csv` and Lightning Pose's
`predictions_pixel_error.csv`, and the empty string when the run wrote neither.

**`mosaic cancel` on an attempt whose process has already exited succeeds.** It
used to exit non-zero having touched nothing, leaving the attempt `running`
forever — and a run-log that never goes terminal is the one state
`inflight_state` reads as neither live nor reclaimable, so the run root stayed
claimed until the marker expired. It now records the terminal `cancelled` event
and reports success. Scoped to that branch: the run-log has one writer, and in
the branch where SIGTERM was delivered that writer is alive.

**`mosaic release --execution-id <id>` is new.** It frees the run roots claimed
by an attempt with no terminal record anywhere to find — one that ran untracked,
or whose run-log never reached the dataset. A claim held from another host is
refused, and one whose process is still running here is refused without
`--force`.

## 0.13.0 — one selector model, a scope that leaves op params, and a request document's key is renamed

**No run on disk is re-addressed.** A selector says which entries a caller
wants and never enters a payload. One entry set asked for five ways therefore
mints one `run_id`, which `tests/test_hashing_rules.py` pins for a scope-free
feature and a scope-dependent one.

**`Request.entries` is now `Request.scope`, and a request written before this
does not load.** Request models forbid unknown fields and `load_request` has no
migration. A `.mosaic/pipelines/requests/*.json` holding the old key therefore
fails validation with pydantic's `extra_forbidden`, and that file is what
`mosaic run --graph-request` and `mosaic pipeline status` read. No migration is
provided, on the same reasoning `TranscodeParams.entry` was removed under. A 0.x
wire key moves in one step rather than growing an alias. A submission whose
steps have finished keeps its results, which are addressed by `run_id` on disk
and reported by the run-logs. Recovery for one still in flight is to resubmit
the recipe.

**One model replaces three loose keyword arguments everywhere.**
`mosaic.core.scope.Scope` holds `entries`, `groups` and `sequences`, and
`entries` excludes the other two rather than intersecting with them. Seventeen
public entry points take `scope=Scope(...)` where they took `groups=`,
`sequences=` and `entries=`: `run_feature`, `load_values`,
`Dataset.run_feature`, `Dataset.drop_entries`, `Dataset.load_id_labels`,
`extract_frames`, `inventory`, `build_manifest`, `IndexCSV.read`,
`IndexCSV.drop_entries`, `yield_sequences`, `narrow_target`, `delete_set`,
`plan_pipeline`, `run_pipeline`, `submit_request` and `intended_scope`.

**A selector that names nothing covers nothing, and only an unset one covers
everything.** `Scope()` is every indexed entry. `Scope(entries=[])`,
`Scope(groups=[])` and a group name that matches no row all resolve to none. A
misspelled group used to read as no restriction at all and measure coverage
against the whole dataset.

**A caller that cannot enumerate a `groups` selector refuses it by name.** Only
an index can list the entries a group covers. `resolve_feature_identity`,
`Dataset.drop_entries`, `delete_set`, `IndexCSV.drop_entries` and `asked_of`
each raise a `ValueError` naming themselves and the `Scope(entries=[...])` to
pass instead, rather than guessing at everything or at nothing.

**`mosaic track --groups X --entries Y` is refused where it used to intersect.**
The two spell one narrowing twice and the message says to give either alone. A
dataset with no media index and an `--entries` beside `--groups` both come back
as a message rather than a traceback.

**`mosaic pipeline plan --json` reports what each step will be asked for.** The
per-step `entries` value used to be the step's own narrowing, which was `[]` for
a step the plan did not narrow. It is now the entries the step will actually
compute, which for an unnarrowed step is the plan's whole scope.

**An op declares how much scope it takes.** `ScopeTakes` is one of `"none"`,
`"any"`, `"at-least-one"` or `"exactly-one"`, and `check_scope_takes` is the
only place a scope refusal is raised. An unset selector, an empty entry list and
a selector that named a group and missed resolve to zero entries and mean three
different things, and each gets its own message.

**No op params model publishes a coverage or a recompute flag, and a client
drawing controls from `Op.Params.model_json_schema()` sees neither.** The
schema held `entries` on eight ops, `entry` and `camera` on `export-store`, and
`overwrite` on six more. All are gone from all seventeen. What a run covers and
whether it redoes the work belong to the attempt, and two attempts differing
only in either are one recipe under one identifier.
`tests/test_op_scope_declaration.py` pins both halves for every registered op.

**No identifier moves for that removal.** Every deleted field was
`HASH_EXCLUDE`, which `Params.identity_dump()` pops before the payload is
hashed.

**Four op params models lose their selector and their `overwrite`.**
`TrackerOpParams`, `_InferParamsBase`, `ExtractFramesParams` and
`ResampleTracksParams` keep their names and base directly on `Params`.
`Op.scoped_params` is deleted. An op step's scope comes from the plan through
`_op_scope`, and `build_op_params` validates a recipe's params as the recipe
wrote them.

**`Op.target`, `Op.plan_identity` and `Op.run` take a `ResolvedScope`, and
`Op.run` also takes `overwrite`.** `run_op` gained keyword-only `scope=` and
`overwrite=`. Anything outside mosaic that subclasses `Op` or calls those
methods positionally must be updated.

**The `transcode` and `export-store` identifiers move, and nothing on disk is
renamed.** Both dropped the sorted source uuids they carried and are now the
recipe hash namespaced: `transcode-<recipe>` and `export-store-<recipe>`. A
derivative is addressed by its own filename and reuse is gated on that plus the
forward link, so neither identifier names a file and no encode is repeated.
Both golden corpora that record a transcode identifier land on
`transcode-49bf3aa578`: the op corpus moves from `transcode-03632a82d2` and
the worked plan from `transcode-816195ccbd`, which covered two entries and so
hashed two source uuids. Both ops now declare `scope_dependent = False`, which
is what the declaration always meant: whether coverage decides the output.
What a run covered still reaches the ledger through the `runs.target` column,
and `export-store`'s label now names the cameras the selector named, which is
the only place a one-camera export differs from a whole-entry one.

**A transcode entry's failure budget now accumulates across scopes.** Attempts
are counted under `(storage_name, run_id, group, sequence)` in
`.mosaic/claims/`, and an op step's `storage_name` is empty, so the run
identifier was what separated one submission's count from another's. Varying
the scope used to mint a fresh `transcode` identifier and start the count
over, which is how an entry could keep failing without ever reaching
`QUARANTINE_AFTER`. One recipe is one counter now: three failures of an entry
quarantine it whichever scope asked for it, and a success under any scope
clears it.

**`mosaic run --kind` takes the scope flags a feature run takes, and refuses a
selector inside `--params`.** `--entries`, `--groups` and `--sequences` are
declared on both arms. The first was refused with `--kind` and the other two
did not exist. `--overwrite` is accepted on `--kind` and on a graph op step,
where both refused it. A `--params` naming `entries`, `groups` or `sequences`
is refused by name. Those keys are fields on no feature and on no op, and a run
accepting one would take a narrowing its own model never validated. A caller
that spelled an op's scope into `--params` — mosaic-api and mosaic-queue both
did — passes the flags instead.

## Unreleased — every parameter says what it means, and ten identifiers move

**Ten identifiers move, and four of them name a directory on disk.** Six name a
tracker run root. The other four belong to `resample-tracks`, whose
`plan_identity` mints its tracks variant from the same dump: every
`resample-tracks.0.1-*` variant directory and tracks-index row on an existing
dataset is renamed. `mosaic reconcile --apply` re-addresses a moved artifact
instead of recomputing it, and registers tracks as one of its three reconcilers.

```
trex/defaults                            8b7a78f4d5 -> 790baf0eff
trex/max-individuals                     126b71ad3a -> 2fdfe5dd15
sleap/single-model                       206c842d1b -> 384a016537
sleap/top-down                           40cf71a844 -> 2fd813de91
litpose/single-model                     5e79836525 -> feac95ad89
litpose/with-overrides                   9d6c0fda6b -> d941098e09
resample-tracks/30fps                    a9d8b3c46d -> 82815cdde7
resample-tracks/30fps-prefiltered        8ba92b08a1 -> ed393a19d9
tracks/resample-variant                  5456f6d608 -> 730bd96918
tracks/resample-variant-other-upstream   69965f347c -> 2fd213e23e
```

**Which media a run covered used to decide what its outputs were called.** A
tracker's `run_id` is minted from its settings dictionary, which names no entry,
and `resample-tracks` minted both its `run_id` and its tracks variant from
`identity_dump()`. Running one recipe over a second group of sequences produced
a second identifier for the same recipe. `groups`, `sequences` and `entries` are
now `HASH_EXCLUDE` on `TrackerOpParams` and on `ResampleTracksParams`. That
exclusion moves the ten digests above.

**Every parameter field a client can set now publishes a description.** 766
fields across 90 parameter models, and 46 more across the seven nested
configuration models those point at, declare their prose through `Declared`, a
marker placed in the field's `Annotated` beside pydantic's own `Field`. `mosaic
features describe`, `mosaic tracking describe` and mosaic-api's `/features`
endpoint all serve `model_json_schema()`. A form that draws a control from the
schema now has a label for every one of those fields.

Three keys are new beside `description`. `x-mosaic-unit` states the quantity's
unit. `x-mosaic-hash-exclude` marks a field the run identity ignores.
`x-mosaic-unwired` states why a declared field reaches no code path, which lets
a client refuse to offer a control that changes nothing.

**A parameter model's schema description is its prose alone, without its field
list.** pydantic copies `__doc__` into the schema verbatim, and the rendered
`Attributes:` section that `help()` reads was landing there too, publishing every
description a second time inside one multi-kilobyte string. A model that
declares no prose of its own now publishes no description at all, where it used
to publish its parent's.

**Four tracker entry points take a parameter model instead of loose keyword
arguments.** `run_trex`, `run_sleap`, `run_litpose` and `run_ultralytics` each
take one model that both the identity and the work read. A setting can no longer
reach the tool while the `run_id` stays put. `run_trex_convert` and
`run_trex_track` take it too, alongside a placement object:
`TREX_ENV.placed(conda_env=..., bin_path=..., display=...)` overrides the
`MOSAIC_TREX_*` variables for one call, and naming one aspect states nothing
about the others. Placement describes a machine, not a result, and reaches no
`run_id`. SLEAP, Lightning Pose and Ultralytics keep `<tool>_conda_env=` and
`<tool>_bin=` keyword arguments.

Two consequences reach a direct call that a `mosaic run --kind ...` submission
never had. A model reference is a string on every tracker now: `run_sleap` and
`run_litpose` accepted a `Path` and take `str`, matching `run_trex` and
`run_ultralytics`. And a mistyped scalar -- `peak_threshold=1` where a float is
meant, `track_window=5.0` where an integer is -- mints the identifier the op
path already minted for that value. One setting used to name two runs depending
on which entry point submitted it.

**`TranscodeParams.entry` is removed, and `entries` is the one selector.**
Parameter models forbid unknown fields, so the singular spelling now raises at
construction. A repeated entry is collapsed, keeping the order the caller gave.

**Three leaf modules hold what every parameter model shares.** `Params`,
`HASH_EXCLUDE` and the declaration vocabulary are `mosaic.core.params`;
`StrictModel` is `mosaic.core.strict_model`; `Entry` and `CameraEntry` are
`mosaic.core.entry`. Two import paths are retired rather than aliased.
`from mosaic.core.pipeline.types import Params` raises, and so does `from
mosaic.core.pipeline.inventory import Entry` -- that module re-exported both
aliases, and every consumer now names the leaf. `Inputs`, `Result`,
`ArtifactSpec`, `InputStream`, `DependencyLookup` and the two labels sources
stay in `mosaic.core.pipeline.types`.

**The suite runs one worker per core.** `pytest` gains `-n auto --dist
worksteal`, and `pytest-xdist` joins the `test` dependency group beside
`imgstore`. `-n0` is the opt-out a debugger or `-s` needs. A whole run takes 43
seconds where it took 358.

**`uv.lock` is regenerated, not amended.** 68 packages in, 8 out. The committed
lock had drifted from `pyproject.toml` well before this work: it resolved
`mosaic-media` 0.3.0 against a `>=0.3.3` requirement, and it still carried
Ultralytics and `albumentations`, which belong to the environments outside the
package rather than to any extra. It gains the `movement` tree, `pytest-xdist`
and the pinned Typer the reference pages are generated from. No CI job reads the
lock, so this corrects a stale file rather than changing what any job installs.

## 0.12.0 — the install has a default worth having

**No identifier moves, and no run on disk is re-addressed.** What changes is what
`pip install` brings and what the extras are called.

**A bare `pip install -e .` is now a complete analysis install.** PyWavelets,
h5py and PyTables became base dependencies: each gates *reading a file you
already have* — a spectral feature, a SLEAP analysis export, a DeepLabCut HDF5 —
and the three are about 30 MB together, which is not worth a way to get the
install wrong. `psutil` joined them because `temporal-stack`'s memory-headroom
check used to inherit it from `ultralytics` and would otherwise have become a
silent no-op. `seaborn` and `networkx` left; nothing imported either.

**Thirteen extras became nine, and the choice became two.** Install nothing, or
`[all]`. `wavelets`, `sleap` and `hdf5` are base now. `localizer` and `identity`
were both, once h5py moved, essentially "torch", so they merged into
`deep-learning`, which `pose` and `polo` both self-reference — choosing the POLO
fork no longer costs you the identity models. `gpu` became `faiss`, being what it
installs. `imgstore` left the extras entirely for the `test` dependency group:
reading a store has been native for some time, and the package is needed only to
*write* the fixture stores the suite builds.

**`recommended` became `all`, and bundles are now self-referential.**
`all = ["mosaic-behavior[pose,faiss]"]` cannot drift from its parts, where a
copied list could. `recommended`, `identity`, `localizer` and `gpu` still resolve
as aliases and are removed in 0.13 — they are kept only because pip *warns* about
an unknown extra and carries on, so a saved `.[recommended]` would otherwise
produce a working install with no torch in it and nothing to say so.

**A new `movement` extra declares a dependency that was never declared.**
`movement-smooth` and `movement-filter-interpolate` are registered
unconditionally and import `movement` lazily, so the package had no entry in
`pyproject.toml` at all, and the error message pointed at
`pip install 'mosaic[movement]'` — the wrong distribution name and an extra that
has never existed.

**`lightning-action` is capped at `<1.1`.** 1.1.0 requires `nvidia-dali-cuda110`
unconditionally and PyPI serves that as an sdist only, so the extra could not
install anywhere without CUDA, for a reason nothing named.

**One place now decides what a missing dependency says.**
`mosaic.optional_dependency.require` replaces ten hand-written `ImportError`
messages in four shapes, and `tests/test_optional_dependency_messages.py` checks
every extra a message names against what `pyproject.toml` declares. Two guards
that never existed were added — `global-tsne`'s faiss import raised a bare
`ModuleNotFoundError`, and the DeepLabCut `read_hdf` path is now covered by
PyTables being a base dependency rather than by a guard. Three `try/except
ImportError` blocks around the movement, lightning-action and feral imports were
deleted: none of those modules imports its optional dependency at module scope,
so the blocks caught nothing they were written for and would have swallowed a
real error, dropping a feature from the registry.

**The two-OpenCV story was told wrong, and is now told as two.** Installing
`albumentations`, `lightning-action` or `movement` beside mosaic is safe on the
documented conda environment — one conda-forge `py-opencv` registers both
distribution names for its single build, so they resolve with no wheel at all.
That collision is a plain-pip hazard. The separate hazard, `av` and any `cv2`
wheel each vendoring a complete ffmpeg, is unaffected by which OpenCV flavor is
installed and is not fixed by any dependency edit. `yolo-augment` stays opt-in
for the reason that was always sufficient: it changes what a YOLO or POLO
training run does, and nothing records which way a run went.

**Known, and pre-existing: `uv lock` does not resolve.** `ultralytics` publishes
nothing for win32 on Python 3.14, which `requires-python` admits, so `uv.lock`
is stale and `NOTICE` cannot be regenerated with it. No CI job reads the lock.

## Unreleased — no install of mosaic carries Ultralytics

**No identifier moves, and no run on disk is re-addressed.** The golden corpus has
no diff, no params field is added and no op version is bumped. What changes is
where `train-pose` and `train-points` run, and what `pip install` brings.

**`pip install -e ".[all]"` now resolves no AGPL-licensed dependency.** Pose and
point *training* were the last two paths that imported Ultralytics inside mosaic's
own process; both now drive the same runner the tracker and both inference ops
already use, in the environment their model belongs to. Nothing under
`src/mosaic/` imports Ultralytics outside
`src/mosaic/tracking/external/runner/`, no extra declares it, and
`tests/test_ultralytics_separation.py` asserts both with empty sets and a witness
that the detector still finds it where it really is.

**`pose` and `polo` become aliases for `deep-learning`, removed in 0.14.** A
saved `pip install -e ".[pose]"` still produces a working environment and no
longer produces an Ultralytics: build
`src/mosaic/tracking/external/ultralytics-env/` or `.../polo-env/` and name it
with `MOSAIC_ULTRALYTICS_BIN` or `MOSAIC_POLO_BIN`. `all` reaches
`deep-learning` directly, so no live bundle points into a deprecated alias.

**`yolo-augment` is removed rather than aliased.** `albumentations` is read by
whichever process runs the trainer, and after this that process is the external
environment -- so it is declared there, opted into with `uv sync --extra augment`,
and no alias in mosaic's own `pyproject.toml` could have put it where it is now
needed. Both environments also declare `opencv-python-headless` and override the
GUI wheel away, because `ultralytics` wants one build and `albumentations` the
other, and two builds of one import package in a process is what pitfall 8 is
about.

**A cancelled training run still stops at an epoch boundary.** This is the
behavior the move had to preserve rather than a new feature: Ultralytics cannot be
interrupted inside an epoch, so a killed process loses whichever one was running,
where a flag it reads between them leaves `last.pt` and `results.csv` complete.
Mosaic writes a file the runner stats at each boundary, and falls back to the kill
every other tool gets only after a grace long enough for an epoch. On a substrate
that imposes its own termination grace -- a queue pod -- that grace has to exceed
one epoch too, or the runtime's SIGKILL arrives first.

**`n_epochs` in the trained-model index now records what ran, not what was
asked.** A run stopped early by `patience` at forty of three hundred was recorded
as a three-hundred-epoch model. Anything reading that column sees a value that
means something narrower than it did.

**Two things a training run now does that it did not before.** It reports each
epoch into the run-log, so a queued job's progress is visible where previously the
denominator was set and the numerator never advanced; and it refreshes its own
run-root claim from the tool's output, so a run longer than the claim's window is
no longer read as abandoned by the next execution along.

## Unreleased — pose and point inference leave mosaic's process, and land in source pixels

**`infer-pose` and `infer-points` now run their model in an environment mosaic
does not install**, reached as a subprocess, the way the `ultralytics` tracker
already did. Nothing in `src/mosaic/tracking/pose_training/inference.py` imports
Ultralytics any more, and the separation guard's allowance is down to
`train.py`. Model training is what is left, and the entry below is where it goes.

**Both ops move to version `0.2`, because their output moved.** They used to
resize every frame at decode time to fit `imgsz` (640 by default) and return
coordinates in that smaller space — which then reached `tracks/`, where every
spatial column is supposed to be video pixels. Frames are now fed at their native
size, as the tracker's always have been. A table written by `infer-pose.0.1-*` or
`infer-points.0.1-*` holds coordinates scaled by `min(640/width, 640/height)` and
should be re-run; the two versions write into separate directories, so nothing is
overwritten and nothing is silently reinterpreted. No digest moves: the version is
a visible segment, not a hash term, and `tests/data/op_identity_golden.json` is
unchanged.

**POLO has an environment of its own**, `src/mosaic/tracking/external/polo-env/`,
located by `MOSAIC_POLO_CONDA_ENV` / `MOSAIC_POLO_BIN`. It cannot share one with
upstream: both ship under the distribution name `ultralytics`. Both environments
install the same `yolo` console script, so the `$PATH` step of the location
ladder cannot tell them apart — name the fork's environment explicitly. Point
detection refuses an upstream build by name rather than running it, which is what
that step would otherwise do silently.

**A checkpoint from the wrong fork is refused rather than crashing.** POLO pickles
its weights under a class upstream does not define, so an upstream build failed
inside `torch.load` before the task the checkpoint declares could be read, and the
refusal that already routed `locate` weights to `infer-points` was unreachable.
The probe now reports a load failure instead of raising it, and mosaic names the
environment those weights belong to. This fixes `mosaic track ultralytics` too.

**An imgstore recording has to be exported first**, for these two ops as it
already was for the tracker: they are handed a video path, so
`mosaic run --kind export-store` comes first and the error message names the
command. `infer-localizer` is unaffected — it is mosaic's own PyTorch, still runs
in process, still reads a store natively, and stays at version `0.1`.

**Two fixes the boundary forced.** Predictions are converted per batch instead of
being accumulated, so peak memory no longer grows with the length of a recording.
And a long video no longer expires its own claim: the runner's progress lines
refresh it, where an in-process op had no line to hang that on.

**One defect found and fixed in the shared runner.** Its prefetch producer offered
the end-of-stream sentinel once, with a timeout, so a consumer that had not freed
a queue slot within half a second never received it and blocked forever. Whether
that happened was decided by how long one batch took against that timer — never on
a fast GPU, reliably on a busy machine or on CPU. `mosaic track ultralytics` was
exposed to it too.

## Unreleased — a graph step names itself, and says no before doing the work

**No identifier moves.** What changes is the surface another repository sees: a
new argv form, a reserved exit code, two new file layouts under `.mosaic/`, and
one field renamed in a document nothing outside mosaic had yet written.

**A step is addressed rather than spelled out.**
`mosaic run --manifest <path> --graph-request <rid> --step <id> --execution-id
<eid>` runs one step of one submitted pipeline. It is strictly more expressive
than the spelled-out form: `--groups`, `--sequences`, the four frame filters and
`overlap_frames` all reach a feature's identity and none of them has a flag, so a
step that re-plans itself reads what a caller could not pass. **The request is
found from the manifest's parent, and there is deliberately no second path
flag** — a path mosaic-queue does not know about is one it cannot rewrite for a
substrate that mounts the dataset somewhere else, which would break precisely on
the machine a GPU step lands on.

**A refusal is a reserved exit code, not a new terminal status.** A step that
declines before doing any work exits **65** with `error_json` carrying
`{"reason": …}` from a closed set — `coverage_shortfall`, `upstream_empty`,
`schema_family_mismatch`, `variant_mismatch`, `version_moved`,
`parent_unrecorded`, `recipe_missing`, `digest_mismatch`, and later
`undecodable_codec`, a tracker's refusal of a file's codec.
`terminal_status_for_exit` maps 65 to `failed`, the status of a refusal. Nothing
joins `runlog.TERMINAL_STATUSES`: three repositories read that set and
mosaic-api's sweeper reaps it, the same reason `partial` was kept out of it.

**Two new places under `.mosaic/`.** `pipelines/requests/<request-id>.json` is
one submission — its narrowing, its `bind` pins, `allow_partial`,
`max_concurrent_steps`, the `step_id → execution_id` map and the version every
step's producer declared, all assigned before anything runs. `claims/` holds what
has been *tried*: per-entry and per-step failure counts under `entries/` and
`steps/`, and per-request exclusion decisions under `requests/`. That last split
is load-bearing — attempts are global so a resubmit does not reset the only bound
on retrying a sequence that cannot succeed, while the decision to proceed without
an entry is a scientific one and must not leak from one request to another.

**`Request.feature_versions` is now `step_versions`**, and covers op steps too.
The old name described only half of what belongs in it, and nothing outside
mosaic had written a request file yet, so the rename costs nothing now and would
have been a wire break later.

**Two behaviours that were wrong and are now refused by name.** `mosaic run
--kind ... --overwrite` accepted the flag and dropped it — `run_op` has no
overwrite — so it promised a recompute that never happened; it now fails. And a
dataset whose tracks tables span two schema families is refused with the families
named, at submit and at each step's start, instead of raising out of the middle
of an identifier hash.

`mosaic pipeline` gains `submit` and `status`. `submit` records the pipeline
against the dataset and prints the command each step is run by, so a graph can be
driven by a shell loop or a job array with no queue involved; `status` reads the
steps' run-logs and says how far the submission got, without touching the feature
registry.

**The run-log carries three more facts, because it is the only channel that can.**
`entries_written`, `cache_hit` and `tracks_variant` join `entries_failed` on
`RunLogSnapshot`. All three were already known inside a run and reached nothing:
`cache_hit` rode a non-serialized `Result` field, the entry count lived in a local
variable, and which tracks recipe a run read was recorded nowhere at all. A queue
cannot recover any of them — it spawns `mosaic run` with stdout *and* stderr on
`DEVNULL`, deliberately, because an undrained pipe deadlocks a chatty child — so
the run-log is where they have to be.

Three new event kinds rather than three new columns on an existing one, because
their knowers differ: an op reusing a trained model has a `cache_hit` and no
entry count, a partial feature run has a count and no claim to make, and the
variant is known at the start while the count is only final at the end. Nothing
joins `runlog.TERMINAL_STATUSES`, and an older reader is unaffected — an `ev` it
does not recognise falls off the end of the fold having advanced liveness and
changed nothing, which is the same property that made `entry_failed` safe to add.

**`entries_written` counts what the scope holds, not what the attempt computed.**
Cache hits count, so a resumed run and a fresh one report the same number over the
same scope — the point being that one number can drive a coverage reading without
the reader knowing which kind of run produced it. Two consequences worth stating:
it is last-write-wins where `entries_failed` accumulates, and for a tracker it is
attempted-minus-lost rather than the index-row count, because a failed bridge
still writes a row (the tool output is durable and adoptable, so a re-run redoes
only the conversion) and the two numbers therefore differ on exactly the partial
run where it matters.

`tracks_variant` is what a run **read**, never what an op produced. The queue
cannot derive it from the job spec: a step-addressed argv carries no
`--tracks-run-id`, because the step resolves its own variant out of the recipe.

`mosaic run --json` reports `entries_written` on the feature path, and the op path
stops reporting `cache_hit: null` — it now reads both back from its own log, which
is why `entry_failure_status` is now `attempt_facts`.

**Ultralytics tracking runs in an environment you build, and an upgraded worker
that has not built one fails every `mosaic track ultralytics` job.** That is the
operational fact this entry exists for, and mosaic-queue and mosaic-api deploy
this. The refusal is at the probe, before anything is minted, and it is
`UltralyticsNotFoundError` naming the two variables and the build command — so it
is loud rather than silent, and it costs nothing already computed. But it is
every job on the machine, not the first one to need something unusual.

The environment is built with `uv sync --python 3.12` in
`src/mosaic/tracking/external/ultralytics-env/`, and located by
`MOSAIC_ULTRALYTICS_CONDA_ENV` (a conda environment) or `MOSAIC_ULTRALYTICS_BIN`
(that environment's `yolo` script), the same ladder TRex, SLEAP and Lightning
Pose already use. With neither set mosaic looks for `yolo` on `$PATH`. Where the
tool sits is a property of the machine, so none of it reaches a `run_id`.

Ultralytics is AGPL-3.0, and a program that imports it is one work with it.
Mosaic now imports it nowhere on the tracking path: it spawns a program in that
environment and exchanges a JSON request file, a JSON response file and progress
lines on standard output.

**No identifier moves**, and no run on disk is re-addressed. The tracker's
settings, its version and its digest are what they were; a run recorded before
this change reads as current and reuses.

**The `pose` and `polo` extras are unchanged, and still install Ultralytics.**
The separation covers tracking only — `train-pose`, `train-points`, `infer-pose`
and `infer-points` still import Ultralytics inside mosaic's own process — so an
install that asked for one of those extras has an AGPL dependency in it exactly
as before, and a tracking run still ignores it and uses the built environment.

**An imgstore recording has to be exported first.** The tracker is handed a video
path like every other external tool, and a store is a directory of chunk files,
so `mosaic run --kind export-store` comes first — the same requirement TRex,
SLEAP and Lightning Pose already carry, and the error message names the command.
In-process tracking could open a store directly; that is the one capability this
costs.

## Unreleased — a table may declare centimetres, and three TREx readers say which

**mosaic refused data it could analyse perfectly well, over a number nobody
needed.** TRex has scaled its positional output by `cm_per_pixel` since long
before it began *recording* the factor (2025-02-18, TRex 2.0.0), so every older
export is centimetres with no record of by how much. Nothing can divide that back
out. "Tracks are pixels" therefore made a whole population of real data
unconvertible — and the workaround it invited, `cm_per_pixel = 1.0`, would have
written centimetres into a table whose schema promised pixels, which is the exact
silent lie the pixels rule exists to remove.

**New schema `mosaic_cm_v1`.** The same contract as `trex_v2` — same required
columns, same allowance for what TRex genuinely measures, `X`/`Y` the body
centre — in centimetres. It is deliberately **its own schema family**, extending
nothing: these columns mean the same *things* as `mosaic_v1`'s and not the same
numbers, so a scope resolving both is refused by `_refuse_mixed_schemas`, naming
both families and their entries. The required set is now shared through
`STANDARD_COLUMNS`, because the second family cannot inherit it — extending would
put it in the pixel family and make the two mix, which is the one thing they must
not do.

Pixels remain the default and what every modern tracker emits. `scale-to-cm`
converts px → cm as a recorded step; nothing converts cm → px without a factor,
because nothing can.

**New converter `trex_npz_cm`**, and TRex now has three readers with one recipe
each — `output_schema` is declared once per class, so a converter choosing its
schema per file would make that declaration a guess:

| the file | reader | emits |
| --- | --- | --- |
| records `cm_per_pixel` | `trex_npz` | `trex_v2`, pixels |
| does not, factor known | `trex_npz_scaled` | `trex_v2`, pixels |
| does not, factor gone | `trex_npz_cm` | `mosaic_cm_v1`, centimetres |

`trex_npz_cm` takes no parameters and converts nothing. It does still run
`name_the_body_centre`: the head-versus-centre defect is not a unit question and
does not deserve to ride along with one. `MissingTrexCalibrationError` now names
all three routes instead of one that did not exist.

**Three TREx 1.x field names were unclassified, and all of them mattered.** TRex
renamed them on the way to 2.x, and mosaic knew only the later spellings:

- `frame_segments` → `tracklets` and `segment_vxys` → `tracklet_vxys` are
  per-*tracklet*, and both are now in `OFF_AXIS_FIELDS`. Before this they were
  padded onto the frame axis — a value on a row that denies it — and, once
  `unscale_to_pixels` existed, refused the whole table as unclassified. Every
  file of a 720-file archive failed on exactly this.
- `segment_length` → `midline_segment_length` is a length in centimetres
  (`length(seg[1].pos - seg[0].pos) * cm_per_pixel`, verified in TRex's
  `OutputLibrary.cpp` at v1.1.9) and is now in `_LENGTH_FIELDS`, and in
  `scale-to-cm`'s classifier so the two stay in step.

**On disk:** nothing existing moves. `mosaic_cm_v1` is a new name on the
`std_format` column, `trex_npz_cm` writes its own
`tracks/convert-trex_npz_cm.<version>-<digest>/`, and a dataset holding both a
centimetre and a pixel conversion of one entry has two labelled variants —
`select_variant_rows` refuses to choose, and `Dataset.drop_entries(..., run_id=)`
retires one.

## Unreleased — `scale-to-cm` can be chained, and a TREx reader can be told the factor

**`scale-to-cm` could not be put in a pipeline, which is the one thing it is for.**
It returned only the scaled columns joined to the five metadata ones, so `ANGLE`,
`X#head`, the keypoint confidences and everything else non-length were dropped —
and `X`/`Y` came back as `X_cm`/`Y_cm`. A track feature chained onto that output
does not fail: `trajectory-smooth` reads `X`/`Y` literally and every positional
step is guarded on their presence, so it emits its input unchanged and reports
success. Nothing in the tree had ever chained it, and nothing had noticed.

A new `mode` decides. `"derive"` is the old behaviour and stays the default.
`"convert"` returns the **whole** table with every length column converted in
place under its own name, so the result is track-shaped and a whole pipeline can
run downstream of the conversion rather than around it. Emitting both spellings
at once — `X` in pixels beside `X_cm` — is the one thing neither mode does; that
is one table holding two coordinate systems, which is the failure the feature
exists to remove. `suffix` names nothing in `convert` mode and is refused there
rather than hashed, since two spellings would otherwise mint two identifiers for
one byte-identical table.

The invariant does not move. It is about *tracks*: the tables under `tracks/`,
which this feature never writes and which stay pixels in either mode. Its own
output is a derived table, and a derived table has always carried whatever unit
its feature computed.

**Its length classifier disagreed with the converter's, in both directions.**
`_LENGTH_NAMES` was missing `SPEED_SMOOTH`, `SPEED_OLD`, `ACCELERATION`,
`ACCELERATION_SMOOTH`, `BORDER_DISTANCE` and `NEIGHBOR_DISTANCE` — six columns the
TREx converter divides out into pixels and this never multiplied back, so a border
distance in pixels would be compared against a threshold in centimetres and keep
every frame while saying nothing. And it scaled `midline_length`, which TREx never
scales (its conversion is commented out in `OutputLibrary.cpp`), because that name
shares the `midline_` prefix with three genuine lengths. Both are fixed and two
tests now pin the correspondence in each direction; the second fails on the old
code.

**`scale-to-cm` 0.1 → 0.2.** The classifier is a module function, not a `Params`
field, so no digest can see that a default-params run now emits a different column
set. `"scale-to-cm/default"` in the golden corpus moves, and it is the only line
that does. No stored run is re-addressed: nothing in the corpus had run it.

**New `trex_npz_scaled` converter.** `MissingTrexCalibrationError` has always
ended by naming a remedy — "an older file has to be re-exported from its
`.results`, or **converted by a reader that is told the factor**" — and that
reader did not exist. It does now: the same conversion, schema and per-individual
merge as `trex_npz`, with a **required** `cm_per_pixel` param instead of a value
read off the table. Required rather than optional so a missing factor raises once,
before any file is opened, rather than being collected by the conversion loop and
reported as one skipped sequence per entry. A file that *does* record a factor and
disagrees raises `TrexCalibrationConflictError` rather than either value winning
silently.

A second `src_format`, not a flag on the first: a tracks variant identity names
exactly one producer and `converter_op` puts the format in the directory name, so
tables whose factor a human reconstructed stay addressable apart from tables whose
factor the exporter measured. Same reason `calms21_json` is a class rather than a
branch.

**On disk:** nothing existing moves. `trex_npz_scaled` writes to its own
`tracks/convert-trex_npz_scaled.<version>-<digest>/`, and two factors are two
variants. A dataset converting the same entries under both readers holds two
labelled variants, which `select_variant_rows` refuses to choose between —
retire one with `Dataset.drop_entries(..., run_id=...)`.

**For a caller with pre-2025 exports:** the factor is recoverable from the file
even though TRex did not write it down. TRex exports `tracklet_vxys` in px/s while
`SPEED#wcentroid` beside it is cm/s, so their ratio is the applied `cm_per_pixel`.
The converter deliberately does not do that division itself — it is noisy, and two
individuals of one recording would scale their halves of a merged table by subtly
different numbers. Recover it once, round it, state it; `params.json` records what
was used.

## Unreleased — `overlap_frames` reads across a continuous recording, or refuses

**`overlap_frames` did not do what it said, and said nothing about it.** It
concatenated the neighbouring sequences onto a run's input and then trimmed the
output by *row offsets measured on the input* — sound only for a feature returning
one row per input row in input order, which the `Feature` protocol has never
required and about half the library breaks by sorting, filtering or reducing. The
trim returned the right row *count* and the wrong rows. It also counted rows rather
than frames, so three individuals turned a request for three frames of context into
one; and its `core_start == 0` fast path skipped the trim outright for the first
sequence of every group whenever the feature dropped rows, writing the next
segment's rows into that entry's parquet. Two features documented a prohibition
against it; they were the only honest statements about the facility.

**The frame axis it needed did not exist.** Every converter numbers frames per file
from zero, so the three segments carried the same frame numbers: concatenating them
handed `apply` three rows for frame 7, and every feature that sorts or groups by
frame interleaved or merged three recordings. No trim can repair numbers computed
from that, so fixing only the trim would have made the result *look* correct.

**A group now declares that its sequences divide one recording**, with
`continuous_groups:` in the manifest, and mosaic verifies the claim against
`frame_min`/`frame_max` — two columns the tracks index now measures from each
parquet as it is written, the way `n_keypoints` is, blank meaning unknown. Within a
continuous group, neighbours are ordered by recorded frame extent rather than by
sequence name (which sorted `seg1, seg10, seg2` and closed silently over a missing
sequence), context is a window of N *frames*, the output is trimmed on the frame
interval the entry covers, and media resolves as one shared timeline so `frame`
still addresses the right clip. Anything else raises and names the two sequences and
their ranges. Features declare `accepts_overlap`; the two prohibitions are gone, and
both collective features now support it.

**Feature identity now covers the context width.** `compute_run_id` gains an
`_overlap_frames` term and `FEATURE_IDENTITY_SCHEME` moves 5 → 6. Before this,
`overlap_frames=300` and `overlap_frames=0` minted one identifier and one directory,
and the second run was served the first's parquet — the same shape of hole `_tracks`
and `_labels` closed, but for an *argument* that changes the numbers rather than an
input the digest failed to name. The term is omitted when zero, which every existing
run is, so the golden corpus moved zero lines and no directory is re-addressed;
`params.json` records it unconditionally so `mosaic reconcile` can still reproduce
an address. `Pipeline` step-cache prediction reads it too, where a missed term would
have reported "cached" over a directory built with different edge handling.

**On disk:** `tracks/index.csv` gains `frame_min` and `frame_max` (blank on existing
rows; `ds.measure_frame_extents()` fills them). `dataset.yaml` gains an optional
`continuous_groups`. `yield_sequences_with_overlap` is removed — a second
implementation of the same slicing with the same defects, reachable only from its
own tests, whose worked example taught the positional trim.

## Unreleased — tracks are pixels, and every tracks variant re-mints

**TREx tracks a session, not its first clip.** An entry whose media index holds
several videos — a recording a device split into clips — was tracked from
`video_order` 0 and the rest were dropped with a line on stderr. TREx now
receives all of them as one `PathArray`, converts them into a single `.pv` whose
frame index is continuous, and produces one set of identities across what used to
be an artificial boundary at every clip. SLEAP, Lightning Pose and Ultralytics
are unchanged: they still read the first clip and still say so, which is now
declared as `joins_sources` on each tracker's `TrackingRoot` rather than
assumed. Both changed later. mosaic joins a multi-clip entry into one file for
every tracker, TREx included, and `joins_sources` is removed.

Three observable consequences.

**`time` in `tracks/` is mosaic's for a joined entry, not TREx's.** TREx reads
one frame rate from the first clip and never checks the others, and it has no
per-frame timestamps for an `.mp4` at all — its timestamp path for video files is
compiled out. A real session measuring 30, then 29.95, then 31 fps was therefore
timed as if it were 30 throughout: about 3% wrong for most of the recording, and
accumulating. `time` is now reconstructed per clip from the measured rates
(`mosaic.core.media.timeline`), `frame_rate` names the rate in force at each
frame rather than one value, and the synthesised `timestamp` column is dropped
rather than re-minted from numbers mosaic never measured. `frame` is untouched.
When the clips *disagree* on rate, the per-second columns TREx derived against
the single rate — `SPEED`, `VX`, `ANGULAR_V` and their kin — are dropped rather
than rescaled; `speed-angvel` derives them from `X`/`Y` and the corrected `time`
with its method in a run identifier. A uniform-rate session keeps all of them.
The time axis assumes the clips are gapless, which mosaic cannot verify: the
probe records no creation timestamp, so a recorder that stopped between clips is
timed as though it had not.

**Four columns on `_tracking/trex/index.csv`.** `video_sources` (a JSON array of
root-relative paths, in `video_order`), `video_uuids` (comma-joined, never
sorted), `media_composition` (the digest of that arrangement) and
`n_source_videos`. `video_abs_path` still names the first clip, as every
tracker's row does. Older indexes gain the columns empty on the next write.

**A joined entry's markers record a composition digest in `source_uid`.** For a
single video it is still that video's `video_uuid`, so no existing run is
invalidated; for several it is the ordered digest, which is what notices a clip
being added, removed or reordered. A joined entry is also never adopted from a
pre-marker directory — that directory cannot say how many clips it covered.

Nothing else moves: no settings key was added, so every `run_id` and tracks
variant is unchanged, and `consumed_composition` on the tracks row stops
over-claiming, since the entry's whole media composition is now what was actually
tracked.

`mosaic sources add --kind media --layout per_sequence` is how several files
become one sequence; no new ingestion code was needed.

**Index locking moved to a sidecar, and every index directory gains one
zero-byte file.** `index_lock` held its lock on the index inode itself.
`atomic_write` renames a *new* inode over that path, which POSIX permits and
Windows does not — and WSL's `/mnt/*` mounts carry Windows semantics, so on a
dataset under `/mnt/c` every index write failed, reporting a missing temp file
that `atomic_write`'s own cleanup had already removed. The lock now lives on
`<index>.lock` beside the index — `media_raw/index.csv.lock`,
`tracks/index.csv.lock`, one per index — created on the first locked write,
never deleted, and never renamed over. Anything enumerating a dataset root will
see it. Nothing reads it, no identifier moves, and `index.csv` is no longer held
open at any point.

Two things travel with the move. The rule that a locked block may perform at
most one `atomic_write`, as its last act, is no longer load-bearing: it existed
because the first write dropped the block's grip on the inode it had locked, and
a sidecar is never renamed over. Where that shape survives in the code it is now
about throughput — not holding a lock tuned for a CSV rewrite across an ffprobe
pass. And the Windows branch, which kept its lock in `%TEMP%` keyed by a hash of
the index path, is gone: `%TEMP%` is per user, as `$TMPDIR` is per SLURM job and
per container, so a temp-directory lock silently failed to serialize the very
cases the lock exists for. Both platforms now lock the same file.

**Standardized tracks are now in video pixels, on every tracker, and `X`/`Y`
name the body centre.** Neither held before. TREx reports centimetres scaled by
`cm_per_pixel` and puts the *head* in its bare `X`; every other converter wrote
pixels and a keypoint mean. A feature reading `X` across trackers was comparing
a head to a centroid in two unit systems, and nothing on disk recorded either
fact. It went unnoticed because `cm_per_pixel` defaults to 1, where the two
units are the same number — the error appears the first time somebody calibrates.

**Every tracks variant re-mints, and every feature run built on tracks moves with
it.** All five converters bump: `trex_npz` 0.1 → 0.2, `sleap_analysis_h5`,
`deeplabcut` and `ultralytics_tracks` 0.1 → 0.2, `calms21_npy`/`calms21_json`
0.2 → 0.3. A tracks variant names the directory its tables live in, and the
resolved variants enter every feature identifier as the `_tracks` term, so
existing feature outputs become orphans rather than silently-reused caches.
Reconvert with `mosaic convert-tracks`.

**Three schemas, declared by the producer.** `mosaic_v1` is the tracker-neutral
standard; `trex_v2` is that plus what TREx genuinely measures, also in pixels;
`trex_v1` stays registered permanently because a real archived dataset is in it,
and its spatial columns are centimetres. A converter declares which it emits
through `TrackConverter.output_schema`, a tracker through
`TrackingRoot.output_schema`, and the tracks index records it in `std_format` —
a column that existed with no reader and now has one. `build_manifest` refuses a
scope resolving tables from two schema families, which is the mixture
`select_variant_rows` structurally could not see.

**Columns that were never measurements are gone.** `VX`, `VY`, `SPEED`, `ANGLE`
and a duplicated `X#wcentroid` were computed *by the converters* and presented
as tracker output. `mosaic_v1` forbids them. Heading is now the `heading`
feature (the method is a parameter, so the arbitrary-sign principal-component
fit is a choice rather than a silent fallback) and velocity is `speed-angvel`,
which gains `vx`/`vy`. CalMS21 also drops eighteen fabricated placeholder
columns, fifteen of them all-NaN floats that `feature_columns()` was pulling
into every matrix built from it.

**Three feature identifiers move.** `speed-angvel` 0.1 → 0.2 and
`nearest-neighbor` 0.1 → 0.2 move their version segment only, digests unchanged.
`nn-delta-response` 0.2 → 0.3 moves its digest too: its `speed_col` default
changes from `SPEED#wcentroid` to `SPEED`, and it now raises instead of
returning an empty frame — it was silently producing nothing on three of the
four trackers. `nearest-neighbor` gains an `nn_ego_unrotated` column recording
when its `_ego` offsets are world-frame because no heading was available.

**The media index gains a `cm_per_pixel` column.** Text, not numeric, so empty
means *uncalibrated* rather than `0.0`; set with
`Dataset.set_media_calibration(...)` and preserved across every rescan. Physical
units come from the new `scale-to-cm` feature, which refuses an uncalibrated
sequence rather than assuming 1.0. The media-index header moves, so a reader
pinning its column list sees one more.

**New: `mosaic upgrade-tracks`.** Rescales centimetre-era TREx tables whose raw
export has been reclaimed by `sweep-tracking`, reading the factor TREx wrote
into the file and refusing a table that does not record one. Dry-run by default.
Reconverting is still the better route where the `.npz` survives.

**For mosaic-api:** `parse_pose_columns` reads these tables, and
`scripts/conversion.py` globs `*.parquet` without consulting `tracks/index.csv`
— with two variants under `tracks/`, that can select the wrong recipe. Existing
`.pose` files are never regenerated (bound by filename co-location, with
`pose_frame_count` frozen at import), so they keep serving the old conversion.
A stored pipeline chain naming `"SPEED#wcentroid"` or `"X#wcentroid"` in a
free-string column parameter will start raising.

**The guppies archive stays on `trex_v1` and is deliberately not migrated.**

## Unreleased — mosaic-media 0.3.0 moves every media identifier

**Every `video_uuid` and `content_digest` re-mints.** `mosaic-media` advances its
identity scheme to 2, and the scheme is the first element hashed into both, so a
value stored under 0.2.x will not match what a re-probe now mints for the same
unchanged file. Run `mosaic reprobe-media --apply` once per dataset. It rewrites
the identity columns and re-points each derivative's `source_video_uuid` at its
source's new value, resolving the link from `source_path` rather than carrying
the old uuid, so derivative links survive. A derivative whose source no longer
resolves is reported under `derivative_links_unresolved` rather than silently
mislinked.

**Every transcode derivative is renamed.** The release changes what the command
builder emits — a stream copy that would drop frames now re-encodes, and a source
stating no frame rate raises rather than letting the muxer invent one — so
`TranscodeOp.version` moves to 0.2 and `recipe_hash` and the transcode run
identifier move with it. No dataset holds a transcode derivative, so nothing on
disk is orphaned.

**A `media_facts` cell written before this release no longer reconstructs.**
`MediaFacts` drops `timing_measured` and gains `timing_source`,
`coded_reordering_depth`, `discard_flagged_packets`,
`leading_non_keyframe_frames` and `max_timestamp_gap_frame_periods`. The same
re-probe rewrites the cells. The media-index header does not move: none of the
new fields is a flat column.

**An analysis read refuses a codec outside the measured frame-exact set**
(`h264`, `hevc`, `av1`, `vp9`, `vp8`), reporting
`unverified_frame_correspondence`. Such a source needs an analysis transcode
before tracking or frame extraction can read it, and that transcode emits AV1,
which satisfies the gate. A raw read still warns and proceeds.

### 0.11.0

**`global-identity-model` no longer builds a CNN from scratch.** It trained one
from raw crops with no prior, and individual animal identity is exactly the
regime where that loses: a few thousand egocentric crops per animal is far too
little to learn general visual features from nothing, and the features it needs —
markings, shape, texture — are ones an ImageNet-scale backbone already has. It
now puts a linear classification head on a pretrained timm backbone and trains
that instead, defaulting to the MIT-licensed
`timm/swin_large_patch4_window12_384.ms_in22k_ft_in1k` — the same starting point
`global-identity-embedding` was already using to good effect without training at
all.

The backbone is frozen by default, so a fit is fast and cannot damage the
pretrained representation. `freeze_backbone=False` fine-tunes end to end for
datasets large enough to earn it. That choice also decides what a checkpoint
holds: a frozen run stores the head alone and refetches the backbone by name, so
the file is kilobytes rather than hundreds of megabytes.

**The `V200` and `V118_3` architectures are deleted, and with them the ability to
exchange identity checkpoints with T-Rex in either direction.** Mosaic can no
longer export weights for `visual_identification_model_path`, nor read a
checkpoint T-Rex saved. That interop was never safe to rely on: T-Rex picks its
architecture from its own `visual_identification_version` setting rather than
from the checkpoint, and loads with `strict=False` and merely warns, so every way
of getting it wrong produced a randomly-initialised network and a log line
instead of an error. Nothing else about the T-Rex integration changes —
`mosaic track trex` still drives the binary, and the track converters still read
its exports.

**`global-identity-model` keeps its slug and is at `0.3`.** `predict()` still
returns `(N, num_classes)` probabilities and the training history keeps its four
keys, so anything reading either is unaffected. Gone from its params:
`input_normalization`, `export_trex_weights` and `trex_weights_name`. New:
`model_name`, `freeze_backbone`, and a pre-fitted `model` reference, which this
was the only identity feature to lack — pinning one lets an inference run carry
its training set by reference instead of retraining. `image_size` now defaults to
`None`, following whatever the backbone declares, and `channels` to 3.

**Existing `global-identity-model` runs do not carry forward.** The network
changed outright, so a checkpoint written by `0.2` cannot be read by `0.3` at
all; the version bump moves the `run_id` so a stale one is never adopted in
place of a refit. Refit. The exported checkpoint is also deliberately no longer
named `identity_model.pth`: that filename resolves against the `train-identity`
model reference a T-Rex run reads, so under the old name a file this feature
wrote could still be handed to T-Rex and fail the silent way described above. It
is `identity_classifier.pth` now, which that reference cannot match.

Shared backbone plumbing — model-id resolution, preprocessing, device selection —
moved to `model_library/timm_backbone.py`, where the classifier, the embedding
model and the DINOv2 temporal model all read it from one place, replacing three
copies. The classification head's weights are now drawn from a seeded generator:
`nn.Linear` initializes from torch's global RNG, which is unseeded, so two runs
with identical params started from different weights and ended at different
predictions while the `run_id` matched and the cache hit.

**The MABe22 converters are gone, and with them the `mabe22_npy` format name.**
Both the track converter and the behavior label converter are removed, along with
the mouse-triplets notebook and the script that generated it. Neither converter
was ever exercised against real data beyond its own fixtures, and an untested
reader of a benchmark format is a claim mosaic could not stand behind. A dataset
holding `convert-mabe22_npy.*` tracks variants keeps its tables and its index
rows — nothing on disk is rewritten — but the format can no longer be converted,
and `get_track_converter("mabe22_npy")` now raises and names the formats that do
exist. CalMS21 is unaffected and remains the reference dataset the template
notebook is built on.

Two test files used `Mabe22Converter` as their vehicle for the converter-identity
seam rather than as their subject, and now use the SLEAP converter, which carries
`fps` and keeps the underscore in `sleap_analysis_h5` that the run-identifier
parsing test depends on. One MABe22 test was the only cover anywhere for a
converter refusing to guess among several sequences in one file; CalMS21 has the
same guard and had no test for it, so that test moved across rather than being
deleted with the format.

###  The manifest says where the data comes from

**`dataset.yaml` is at version 2, and declares its scan sources.** Roots were
pinned inside the dataset so that every `index.csv` travels with it; the
replacement for an outside root -- a *search directory* whose files are recorded
by absolute `abs_path` -- was an argument typed at the command line and
remembered nowhere. It is now `sources:` in the manifest, one entry per place the
dataset draws from, each carrying its whole recipe. `mosaic scan` with no
arguments rescans exactly that set, across media, tracks and labels.

A source may claim a directory to glob or an explicit list of files. The second
is what an import that selects some of a folder's contents needs, and no glob
expresses it.

**A scan now replaces what it claims and preserves everything else.** It used to
rewrite the index from whatever it had just walked, which meant scanning
directory A and then directory B kept only B, any scan destroyed rows pointing at
files outside the dataset, and one dataset could not hold two source formats at
once. All three are fixed by the same change. `--prune-unsourced` asks for the
old whole-file rebuild. A scan also no longer overwrites an identity a caller
*assigned*: it refreshes the measured cells and keeps the identity ones, so
declaring `media_raw` as a source cannot silently repartition a project the
control plane manages.

**Observable from another repository.** Newly created manifests are version 2.
Existing ones stay version 1 on disk until something saves; reading never writes.
`format`, `index_format`, `dataset_type`, `segment_duration` and `time_column`
are no longer written -- nothing read them -- but they are not deleted either:
unknown top-level keys are now preserved through a load-and-save round trip,
where `save()` previously wrote a fixed key list and annihilated everything else.
A manifest declaring a *newer* version raises rather than being read under the
wrong rules.

`mosaic index-media` and `mosaic index-tracks` are replaced by `mosaic scan`,
with `mosaic sources` to declare what it reads. New alongside them: `mosaic init`
(no command could create a dataset before), and `mosaic notes` / `mosaic tags`
for the dataset's own description. Tags are typed, carrying the same
`type` / `type_constraints` / `value` shape as mosaic-api's sequence and
individual tags. The library entry points `index_media`, `index_tracks_raw` and
`index_labels_raw` keep their signatures.

mosaic-api and mosaic-queue need no change: the whole `Dataset` surface either
uses is methods, and all of them are unchanged.

### A sequence stops resolving another sequence's media

**`resolve_media` matched a request against the media index's *filenames*, so a
sequence with no row of its own answered with another entry's video.** The last
matching tier tested containment against a row's `name` cell: an index holding
one row named `clip_a.mp4` resolved the sequence `clip` to it, registering one
sequence's extracted frames against another's recording. The test was also
regex-enabled -- `clip.a` matched `clipXa.mp4`, and an unbalanced bracket raised
`re.PatternError`.

That tier now compares the request against each row's own `sequence` cell,
case-insensitively, and tolerates the request carrying a media extension the
entry's own name lacks. Only a real one is stripped, from `VIDEO_EXTENSIONS`, so
`trial.1` no longer answers with entry `trial` -- entry names carry dots
routinely, and `cam1.left` names a recording rather than a suffixed `cam1`. A
dataset holding files outside that set no longer bridges a request written as
such a filename; it reports no match instead of the wrong entry. Matching
identity rather than filenames also matches an entry whole: a request landing on
a multi-file recording resolves every file in `video_order` rather than the one
chunk whose name fitted, and naming a chunk's filename resolves nothing.

**Where it used to guess, it now raises `AmbiguousMediaMatchError`.** Two groups
holding a sequence of the same name, asked for without a group, previously
concatenated both groups' media into one timeline; passing the group resolves it
as before. The exception subclasses `MediaProbeError`, so a caller that already
reports resolution faults per entry keeps going; one that re-resolves inside its
own handler must catch it there.

### A blank group stops being the word "nan"

**A dataset with no group converted its tables under one.** `convert_all_tracks`
read `tracks_raw/index.csv` with a bare `pd.read_csv`, so an empty `group` cell
arrived as a float NaN, and each `str()` of it downstream spelled the word:
`nan__seq.parquet` on disk, `group=nan` in `tracks/index.csv`, and a composition
lookup under a key nothing had recorded. `group` is empty on every dataset the
control plane creates, so this was the common path rather than an edge; only the
per-sequence merge normalized, and only TRex reached it.

**What to do about a dataset already holding one.** Re-converting writes the
correct `seq.parquet` beside the stale `nan__seq.parquet` and leaves the old
index row in place, because `("nan", seq)` and `("", seq)` are different
entries. `convert_all_tracks` already reports that: it names the entries this
conversion did not rewrite and gives the remedy, `ds.drop_entries([...],
delete_files=True)`. Nothing is repaired automatically — after the fact a group
spelled `nan` is indistinguishable from one a user genuinely named that, and
deleting tables a call did not write is what the migration rule forbids.

**`tracks_raw/index.csv` now reads like every other index.** Its frame reader
pins its text columns as strings, so a sequence named `001` reads back as
`"001"` rather than as the integer `1` — the failure `IndexCSV` describes, and
the numeric names are the CalMS21 and MABe convention. An index written before a
column existed still reads.

**`source_md5` carried the same word.** With `compute_md5=False` the empty
column reached `tracks/index.csv` as the literal `nan`; it is now empty.

**`index_labels_raw` refuses a format no converter claims.** It wrote any string
verbatim, and `convert_all_labels` then skipped those rows forever without ever
naming them. It now checks the format half of the label registry and lists what
exists. Register a custom converter before indexing under its format — the
order every doc already gives.

**The label converters load on demand.** `LABEL_CONVERTERS` filled only as a
side effect of `import mosaic.behavior.label_library`, which nothing in `core`
does — so a caller who reached a Dataset through `mosaic.core` alone was told
`Available: []`, and `migrate_labels_raw` matched no row and reported zero
migrated. The registry now fills itself when it is empty. A caller who
registered converters of their own keeps exactly those.

**One exception type for a format no converter claims.** `get_track_converter`
raised `KeyError`, whose repr adds quotes when the CLI interpolates it into a
message; both resolvers now raise `ValueError`, which `convert_all_labels`
already did.

**`convert_id_tags_from_csv` wrote nothing for a sequence with no group.** Its
`"category"` and `"multi"` branches group the CSV by `(group, sequence)`, and
`groupby` drops a row whose key is missing — so with `group` blank, which is
what most datasets have, every row was removed before the loop body ran. The
only symptom was a lower `Created N id_tags files` count. Nothing was written,
so re-running the conversion is the whole remedy. The same read now keeps a
sequence named `001` from arriving as the integer `1`; the rest of the CSV is
deliberately still inferred, because `id` keys the `.npz` that a tracks table's
integer `id` column is looked up in. A row naming no sequence is refused rather
than silently dropped.

**`write_labels_row` spelled a blank group `nan`.** The labels index writer
carried the same `str(group) if group is not None else ""` the tracks one did,
and the `is not None` guard does not catch a float NaN either.

## 0.10.0 — one tracker driver, and a model that may be a directory

**`mosaic trex` is gone; `mosaic track <kind>` replaces it.** It was 211
hand-wired lines mirroring params it could drift from, and it had no equivalent
for SLEAP or Lightning Pose. The new command reads a tracker's own parameter
schema, so scope and execution are flags and everything tool-specific is
`--set key=value`, validated against the schema rather than against a second
copy of it. `mosaic run --kind <tracker>` is unchanged and is still what the
executor shells out to, so nothing queued or scripted against it moves.

**The three trackers now run through one loop.** Everything a run does around
the tool — locating it, minting the run, scoping to work items, claiming an
entry, reusing a finished phase, bridging into `tracks/` — moved into
`tracking/common/`, and each tracker supplies only what is genuinely its own.
SLEAP and Lightning Pose picked up T-Rex's content-based reuse gate on the way,
so a video replaced in place now forces a recompute for all three rather than
one. What each leaves on disk is unchanged, pinned as a normalized snapshot.

**One count is renamed.** The tracker run indexes carried `n_tracks` for SLEAP
and `n_individuals` for Lightning Pose, which were not the same quantity; both
are now `n_ids`. An index written before this reads back under the new name
without being rewritten.

**A model reference may name a directory.** The resolver was shaped around a
single `best.pt`, so SLEAP and Lightning Pose — whose models are directories,
and whose top-down form is an ordered *pair* of them — each carried a private
resolver and could not be registered as a training run at all. A reference is
now an artifact described by a per-kind spec: what shape it points at, how many,
and which files inside it identity is allowed to read. Nothing a tool writes
back into its own model directory can move that identity, which is what makes
Lightning Pose's `video_preds/` harmless. Every identifier is preserved exactly;
no run is re-derived.

**Two observable consequences.** A trained-model row can record a directory
artifact, so a future framework-native training run has somewhere to say so.
And the `models/` root was registered under the wrong index shape — naming a
file nothing has ever written — so it was invisible to `make_portable`,
`rewrite_index_paths` and `reconcile`. It is reachable now, which means
`reconcile` will prune trained-model and converted-dataset rows whose artifact
is gone. `Dataset.reindex` still defaults to a dry run.

## 0.9.0 — identity backbone, generalized

**The MegaDescriptor identity feature was a generic backbone loader wearing one
backbone's name.** `MegaDescriptorNetwork` never contained anything specific to
MegaDescriptor: it loaded whatever `model_name` named through
`timm.create_model`, froze it, probed the embedding width, and decided identity
by cosine k-NN against per-identity prototypes. The name is now the mechanism —
`global-identity-embedding`, `EmbeddingIdentityNetwork`,
`identity_embedding_model.joblib` — and the backbone is a parameter, which is
what it always was. `model_name` now takes a bare timm architecture tag as well
as a Hugging Face hub id, so one feature reaches both catalogs.

**Preprocessing follows the backbone instead of being asserted.** Normalization
statistics and input size are read from the loaded model's own `pretrained_cfg`
through `timm.data.resolve_model_data_config`, rather than hardcoded to
ImageNet's triple and 384x384. A backbone that declares nothing falls back to
exactly the old constants, so the no-information path is the previous behaviour
rather than a third one. Interpolation and crop ratio are deliberately *not*
adopted: those describe timm's evaluation transform for a full image being
centre-cropped, and this feature's input is already a tight egocentric crop, so
honouring a 0.9 crop ratio would discard the border a discriminative marking may
sit in. `image_size` stays a parameter but defaults to `None`, meaning *follow
the backbone*; a value pins an override. The resolved configuration is written
into the exported checkpoint (`format_version` 1 → 2), so reloading a fitted
model reproduces the preprocessing it was fitted with even if the upstream
repository changes underneath it.

**The default weights are permissively licensed.** `model_name` defaults to
`timm/swin_large_patch4_window12_384.ms_in22k_ft_in1k` (MIT) — the same Swin
architecture MegaDescriptor fine-tuned, carrying ImageNet-22k weights instead of
wildlife ones. `BVRA/MegaDescriptor-L-384` stays documented and is one parameter
away; it is CC-BY-NC-4.0 and remains the right choice for academic wildlife
re-identification, where it substantially outperforms an ImageNet backbone at
telling individual animals apart. Mosaic distributes no weights: whichever
backbone is named is fetched at run time under its own license. `docs/licensing.md`
gains the backbone table and says plainly that the default is the shippable
option, not the measured one. Unlike `kpms`, there is no acceptance gate — the
restricted component here is a value the user types, and the default is
permissive.

**A clean break, and an identity-neutral one.** `global-identity-megadescriptor`
resolves to nothing. The feature registry has no alias mechanism, and `mosaic
reconcile` categorically cannot carry a slug rename — it enumerates directories
under `features/`, and a run whose slug no longer resolves is reported
`unresolvable_pre_provenance` with nothing moved. But the rename moves no
digest: a feature's own slug is not in its hash payload, and every `Params`
field name survived, so a run that pins the three defaults that did change —
`model_name`, `image_size`, `weights_name` — mints byte-identical `0.1-8aebe700d2`
under the new name. A golden case pins exactly that. The feature therefore stays
at version `0.1`, and `FEATURE_IDENTITY_SCHEME` stays `5`: the shape of the
hashed payload is unchanged, only default values inside it. A dataset holding
runs under the old slug keeps them, unreachable by name; re-run under the new
slug, pinning `model_name` if the analysis used MegaDescriptor.

**Contract surface: nothing moved.** No sibling repository references any
identity slug — mosaic-api, mosaic-app, mosaic-queue and trex reference none of
the three. The `mosaic-behavior>=0.7.0` floor owed since M7 stands.

## 0.8.0 — T-Rex checkpoint interop

**The identity checkpoints mosaic wrote were not loadable by T-Rex, and the ones
T-Rex wrote were not loadable by mosaic.** Both directions failed on the module
tree. mosaic built the V200 as an `nn.Sequential`, so its state_dict keys were
positional (`0.weight`) where T-Rex's are named (`model.conv1.weight`). Reading a
real T-Rex checkpoint raised. Writing one was worse and quieter: T-Rex loads with
`load_state_dict(strict=False)` and only *warns* on a key mismatch, so a
mosaic-exported `.pth` passed to `visual_identification_model_path` produced a
**randomly-initialised network and a log line**. `GlobalIdentityModel` has
documented that export as T-Rex-loadable since it was introduced; it was not.
Both networks now mirror T-Rex's tree — a wrapper whose children are `normalize`
and `model` — so the keys agree by construction rather than by string surgery.

**`V118_3`'s `bn4` was the wrong normalization layer.** T-Rex uses
`nn.LayerNorm`; mosaic used `nn.BatchNorm1d(track_running_stats=False)`. The two
expose an identical state_dict — `weight` and `bias`, no running statistics — so
a checkpoint cannot distinguish them and the wrong one loaded clean while
computing different math. Measured against a real 4-mouse checkpoint, using
T-Rex's own TorchScript as the oracle: `max|Δlogit| = 10.36` and 12% argmax
agreement, plus a hard crash at batch size 1 and predictions that changed
depending on which other crops shared the batch. With `LayerNorm` the same
checkpoint reproduces T-Rex exactly. Anything inferred from that class's
predictions before 0.8 should be recomputed.

**Input normalization is now a stated contract, not an assumption.** T-Rex's
`Normalize.forward` differs across builds: some compute `(x / 255 - mean) / std`,
some pass raw `[0, 255]` through, and some ship the statistics in the checkpoint
as `normalize.*` buffers. mosaic assumed the first, silently. It is now the
`input_normalization` parameter (`"imagenet_scaled"` | `"raw255"`), recorded in
exported metadata and detected on load: buffers in the file win and their values
are used verbatim, metadata is consulted next, and a checkpoint that states
neither is genuinely ambiguous — mosaic keeps its previous behaviour and warns,
naming the override. Existing mosaic exports therefore keep their meaning.

**What moved, for a reader downstream:**

- `global-identity-model` is at `0.2`. Its `run_id` moves and every existing run
  recomputes. Network numerics are not part of the `run_id` payload, so without
  the bump `load_state` would have adopted a checkpoint the previous code wrote.
  One golden line moved; one was added, closing a `scope-a`/`scope-ab` pair the
  corpus claimed to keep but did not.
- A new `input_normalization` param on that feature.
- **The exported `.pth` changed shape**: named `model.*` keys, optional
  `normalize.*` buffers, and `input_normalization` / `architecture_version` /
  `model_type` / `class_labels` / `mosaic_checkpoint_version` metadata. Metadata
  stays primitive because T-Rex reads these files with `weights_only=True`.
- A TREx run that passes such a file to `visual_identification_model_path` **as a
  bare path** gets a new `run_id`: an unregistered path is identified by its
  digest, and re-exporting changes every byte. A run that passes a training
  `run_id` instead is unaffected. Same narrow population as the 0.5.0 note below.
- `input_shape` is `(W, H, C)` in both exporters. T-Rex compares it exactly, so
  the V118_3 exporter's previous `(H, W, C)` was a hard load failure there for
  any non-square crop. Files already written the other way still read correctly:
  the old exporter's `"architecture": "v200-native"` marker identifies them.
- `TRexNativeIdentityNetwork` is now `TRexV118_3IdentityNetwork`, in
  `model_library/trex_v118_3_identity.py`. **Renamed without an alias.** It was
  never a V200 — it is T-Rex's `V118_3`, as its own checkpoints say. Checkpoints
  written by mosaic ≤ 0.7 still load, through a positional-key shim that warns
  and is removed at 0.9.
- `V118_3` gained the three `Dropout2d(0.05)` layers T-Rex has and mosaic
  omitted. Inference is unaffected (dropout is identity in eval mode); training
  now regularizes as T-Rex does.

**Tests, where there were none.** Nothing previously constructed either network,
ran a forward pass, or exercised a checkpoint round trip — the whole suite stayed
green through all of the above. `tests/test_trex_checkpoint_interop.py` pins the
forward output against T-Rex's own classes, the normalization contract, batch
invariance, the key layout, the `(W, H, C)` orientation, and the legacy shim;
`tests/test_trex_checkpoint_real_weights.py` (slow, opt-in via
`MOSAIC_TREX_MODELS_DIR`) pins agreement with real deployed checkpoints using
T-Rex's TorchScript sidecar as the oracle — the only test that covers a build
whose preprocessing the current source no longer represents. torch runs in its
own CI job.

**A hazard these tests uncovered, which is not fixed here.** torch and xgboost
each bundle an OpenMP runtime, and a process holding both segfaults on macOS —
in either import order, inside whichever is first asked to do real work. xgboost
is a core dependency and torch the optional `identity` extra, so **a session that
trains an identity model and then an XGBoost classifier can crash**, with no
Python traceback. Nothing imported torch before these tests, so the suite had
never met it. `tests/conftest.py` now pins `OMP_NUM_THREADS=1`, which holds for
the suite; the usual `KMP_DUPLICATE_LIB_OK=TRUE` does *not* stop it here and is
documented as able to produce silently wrong results. The real fix is an
environment with one OpenMP runtime, which is outside this change.

**Owed.** mosaic still does not write the TorchScript `*_model.pth` sidecar that
T-Rex writes beside every checkpoint and falls back to when a state_dict load
fails. Reading one is supported; writing one needs a TorchScript-clean forward.

## 0.7.0 — M7, reconcile

**A dataset can now be brought forward after an identity change, in place.** Every
milestone up to here makes a change loud rather than silent -- a new scheme marker,
a moved digest -- but leaves the actual migration to a full recompute. `mosaic
reconcile` (and `Dataset.reconcile`) is the pass that recomputes each feature,
tracks, and label identifier from the current code, compares it against the one on
disk, and -- where the recorded provenance confirms the recipe is unchanged --
re-addresses the artifact under its new identifier rather than recomputing it: the
directory moves, the index rows and `params.json` are restamped, the scheme marker
is refreshed, and the index is backed up first. It runs bottom-up, so a moved
tracks or label variant carries its feature consumers to their new identifiers in
the same pass.

**It never guesses.** A run whose recipe cannot be confirmed unchanged -- one that
predates the scheme marker, whose upstream was never pinned, or whose digest moved
under the *current* scheme with nothing to explain it -- is reported and left where
it is, to be recomputed by an ordinary run. A version bump is a new recipe, not a
re-address: the recomputed identifier keeps the run's recorded version, so bumping a
feature or converter version leaves existing runs `ok`. The pass reads the
`.identity_scheme` marker each run was minted under, so it is idempotent and
resumable -- a re-run over an already-migrated dataset reports every run `ok`.

**Dry-run by default.** `mosaic reconcile --manifest <ds>.yaml` reports what would
move as a classified list (`ok`, `scheme_stale`, `identity_shift_relocatable`,
`identity_shift_recompute`, `unresolvable_pre_provenance`); `--apply` performs the
confirmed re-addresses and marker refreshes; `--only <kind>` narrows to one artifact
kind; `--json` emits the report. A full run also folds in the two cheap index-hygiene
passes -- dropping dangling rows (`reindex`) and rewriting non-portable `abs_path`
cells (`make_portable`). The heavier media and tracking passes stay their own
commands: `reprobe-media` (source drift), `prune-media` (stranded transcodes), and
`sweep-tracking` (expired working directories) probe or delete and carry their own
reports. `--force` is reserved for a future destructive path; the forward pass
itself produces nothing to delete.

**Contract surface: nothing moved.** M7 is mosaic-library-only. `Dataset.reconcile`
and the reconcile engine are imported by no sibling repo, and none of the five
`mosaic` symbols `mosaic-api` imports changed. The `mosaic-behavior>=0.6.0` floor
owed since M6 becomes `>=0.7.0`.

## 0.6.0 — M6, labels

**Converted labels now have the provenance and identity tracks already had.** A
converted label set used to live at `labels/<kind>/<group>__<seq>.npz`, flat, with
nothing recording which recipe produced it or what it was made from. It now lives
under `labels/<kind>/<run_id>/<group>__<seq>.npz` behind a single typed
`labels/<kind>/index.csv`, one index per kind, mirroring `tracks/`. The row carries
the converter identity, the op run behind it, and `consumed_source_roots` — which
distinguishes a **scored** set converted from an upload (`labels_raw`), a
**derived** set (the seam exists; no producer does this yet), and an **authored**
set minted in-process (id-tag columns, empty `run_id`, still flat).

**New source root: `labels_raw`.** It joins `media_raw` and `tracks_raw` as a
place that holds only what a person uploaded, with its own composition hash over
the raw-file checksums, so a label upload moves the identity of everything
converted from it exactly the way a track upload does. A dataset created before
this gains the root on load; nothing on disk moves for it.

**The label-converter extension point changed.** A `LabelConverter` now returns
`list[LabelEntry]` — data — and the `Dataset` writes the `.npz` and the typed row,
the same split `tracks` made. Converters carry a version and typed `Params`, are
keyed by `(source_format, label_kind)`, and register through
`core/label_converter.py` rather than `dataset.py`. A custom label converter
written against the old protocol must move to returning entries; the four built-in
converters and the templates already have. `convert_all_labels()` re-converts once
on a dataset converted before this release, because the output path changed.

**Feature identity now covers which label set a run read.** `compute_run_id` gains
a `_labels` term and `FEATURE_IDENTITY_SCHEME` moves 4 → 5. As with `_tracks`, the
term is omitted when there is nothing to say, so a run that reads no labels keeps
its identifier and the golden corpus moved zero lines — the shift is confined to
features that consume labels. `labels_run_id` is the additive selector, keyword-only
on `run_feature` and `Dataset.run_feature`, defaulting to today's behaviour.

**Migration both ways.** `migrate_labels_raw` copies label rows out of
`tracks_raw` into `labels_raw` (files stay where they are); `revert_labels`
flattens the variant directories back to the old flat layout and restores the
untyped index. Rearranging a sequence that carries labels is still refused — the
frame-index remap that would make it safe is not built.

**Contract surface: nothing moved.** M6 is mosaic-library only. The label-converter
registry, `LabelEntry` and the labels index are imported by no sibling repo, and
none of the five `mosaic` symbols `mosaic-api` imports changed. The
`mosaic-behavior>=0.5.0` floor owed since M5 is now `>=0.6.0`, and
`MINIMUM_MOSAIC_VERSION` in the contract test — still `"0.1.0"`, still passing
vacuously — is owed the same fix. Authoring internal scoring in Dolt and
projecting it into `labels_raw` on commit stays future work.

## 0.5.0 — M5, consolidation

**Tracker intermediates moved to `_tracking/`, and inference lost its root.**
A TREx, SLEAP or Lightning Pose run now writes under `_tracking/<tool>/<run_id>/`
instead of `tracks_raw/trex/…`, and model inference writes under
`_tracking/<infer-kind>/<run_id>/` instead of a top-level `predictions/`. So
`tracks_raw` holds only what a person uploaded, which is what lets a raw-track
scan tell source from byproduct.

**A dataset created before this keeps its own layout, deliberately.** Loading
one adds the `_tracking` roots it never declared — otherwise anything keyed on
them raises — but a `trex` root already reading `tracks_raw/trex` is **left
exactly where it is**. Repointing it would orphan every run on disk and strand
the index naming them. The consequence to know: on such a dataset the tracker
keeps writing inside `tracks_raw`, where a `*.npz` scan of that root will index
its per-individual intermediates as user raw tracks and fold them into the
sequence's composition. If that matters to you, move the directory and update
`roots.trex` in `dataset.yaml` by hand; nothing does it for you, and the sweeper
refuses to run at all while a tracker root sits inside a source root.

**New: `mosaic sweep-tracking`.** Deletes tracker working directories that are
finished and past their retention window — 14 days for tracker output, 3 for
inference audit parquets — dry-run by default. It never touches work in
progress: a directory a live execution holds, one this dataset's index does not
yet name, or one carrying no mosaic marker is reported and left alone. Rows go
before files, so an interrupted sweep leaves rows naming absent directories,
which `mosaic reindex` repairs.

**`mosaic reindex --root` covers every root, not just `features/`.** The
`_tracking` indexes were reached by no reindex, prune or portability pass at
all, so a working directory removed by hand left a row naming it forever.

**New: `promote_correction`.** A corrected track set copies into
`tracks_raw/<entry>/` as `corrected.rev<N>`, an append-only series — nothing is
overwritten, and each revision moves that sequence's composition, which is what
invalidates the artifacts built from it. Blocked while derivatives exist; forcing
promotes and deletes nothing.

**Roots must resolve inside the dataset.** `set_root` and `new_dataset_manifest`
refuse one that does not; a dataset already holding one still loads. **`abs_path`
is unchanged and stays able to point outside** — that is how a second dataset
references a video living inside a first without copying it, and it is the
mechanism that replaced shared-video membership.

**Contract surface: two additions, no removals.** `index_media` gains
`media_layout="per_sequence"`, which reads `(group, sequence)` from the entry
directory the control plane already writes — the default is unchanged, so no
existing dataset is re-identified. `Dataset.sweep_tracking` and
`Dataset.reindex` are new. `InferenceIndexRow`, `inference_index` and
`prediction_index_path` are **gone**; nothing outside mosaic imported them, and
every column of that row survives on the tracks row or in
`tracks/<variant>/params.json`.

Identity moved for one narrow population: a TREx run that sets
`visual_identification_model_path`, which is now resolved to the model's
identity rather than carried as a path. Both golden corpora otherwise gained
lines and moved none.

Owed to `mosaic-api`: raise the floor to `mosaic-behavior>=0.5.0`, and fix
`MINIMUM_MOSAIC_VERSION`, still `"0.1.0"` and so passing vacuously against the
`>=0.4.0` already declared.

## 0.4.0 — M3, source identity

**Every transcode derivative was renamed and relocated.** A derivative now lives
at `media/transcode/<video_uuid>.<recipe_hash>.<target>.mp4`, named for the video
it came from and the recipe that produced it. It used to be named for the video's
*position* within its sequence, directly under the media root — so reordering two
videos renamed both and re-encoded both, in place and without a transaction, and
a crash mid-loop left an index row pointing at another video's frames. The suffix
was additionally empty for a single-source sequence, so adding a second video
renamed the first one's derivative. Both are gone: a reorder now touches no
derivative file at all, and the transcode job is idempotent and skippable.

This is the most observable thing in the release. **A dataset transcoded before
it holds files no current name resolves to**, and clearing them is the whole
migration — `scripts/clear_transcode_derivatives.py`, dry-run by default, then a
re-run of the transcode job rebuilds whatever is wanted. Nothing is lost by
waiting; the old files are simply unreachable.

`media/index.csv` gained `recipe_hash`, additively, and reverting the release is
the reverse migration for the rows: reverted code reads the index through a
schema without that column, drops that cell and no other, and every forward link
still resolves. The *files* are the half a revert does not undo, and a revert
also reintroduces the path-keyed derivative matching this release replaced with
uuid-keyed matching — a known defect, which is the one way this differs from M2's
purely additive break.

**New: `mosaic prune-media`.** Deletes transcode derivatives that no forward link
reaches — the ones a retuned recipe leaves stranded. Dry-run by default, with an
age window so it cannot race a running encode, and it refuses to delete a
derivative whose source is no longer indexed, since that may be the last copy of
an archived video. `--relink` repairs instead of deleting where a current recipe
would reproduce the file. It does not replace the sweep script above: pre-rename
derivatives sit outside its blast radius, and the two reaches are disjoint.

**Transcoding is now refused on a dataset with no `media_raw` root.** There,
`media/index.csv` *is* the originals index, so the job was appending derivative
rows into it and writing links that routing then ignored — wasted work, and an
originals index left holding rows only `recipe_hash` distinguished from
originals.

**A new file per source root: `<root>/sequences.csv`.** One row per sequence,
holding the composition hash of what that sequence is made of — the ordered video
uids for `media_raw`, the raw-file checksums for `tracks_raw`. Written by the
four index writers as a projection of the index they just committed, and read
through `read_sequence_index`. Absent reads as empty, so a dataset that has never
been re-indexed behaves exactly as before.

**And one per dataset: `<base_dir>/sequences.csv`.** A sequence's *label*, as
distinct from its token. `Dataset.display_name` / `set_display_name` /
`display_names` are the surface; relabelling touches this file and nothing else,
so it cannot move a directory or invalidate a run. `(group, sequence)` remains
what every filename, directory and index join key is built from.

**Identifiers move for one population.** A `scope_dependent` feature that
declares it reads a source root now carries that root's per-entry composition in
its identifier (`FEATURE_IDENTITY_SCHEME` 3 → 4). No feature in the toolkit is in
that population today — the eight scope-dependent features all read tables — so
in practice nothing on disk re-derives. The golden corpus moved zero lines.

Separately, `train` and `infer` identifiers move **only** where a model was
handed in as a bare filesystem path. A path is a mutable key: swapping the
weights reused the identifier, and moving unchanged weights minted a new one.
Both are now named by the weights' content digest instead. A model referenced by
its training `run_id` is unaffected. `TRACKS_IDENTITY_SCHEME` 1 → 2 for the same
reason, on inference variants only.

**New columns, all additive.** `assignment_source` on the media index records how
each row learned its `(group, sequence)`. `consumed_roots` and
`consumed_composition` on the feature index record what each entry was made from.
`base_digest` on the trained-model index. Every one reads back `""` on a row that
predates it.

**Checksums are on by default.** `Dataset.index_tracks_raw` and
`write_tracks_raw_index` now hash each source file (`compute_md5=True`), because
the `tracks_raw` composition is over those checksums and an off-by-default column
left every sequence's composition unestablishable. A digest is carried forward
when path, size and mtime all match, so a re-index re-hashes only what changed;
`mosaic index-tracks --no-md5` opts out and accepts an unestablished composition.

**Imgstores now carry an identity.** A store's `video_uuid` is its
`__store.uuid`, marked `identity_scheme = "imgstore/1"` so a reader can tell a
declared identity from a measured one. It is a mint, not a derivation: it cannot
be re-derived, so a re-probe cannot audit it, and chunks edited in place go
undetected. Existing indexes pick it up through `index_media` /
`write_media_index`, not through `reprobe-media`. Transcoding a store is now
refused by name rather than by accident.

**Contract surface.** `MEDIA_INDEX_COLUMNS` gained `assignment_source`, and
mosaic-api imports that list in six test modules. `resolve_model` now returns a
`ResolvedModel` record rather than a `(path, run_id)` tuple; mosaic-api does not
import it. Nothing else on the five-import contract surface moved.

## 0.3.0 — M2, versioned tracks

**Tracks tables moved.** A standardized table now lives at
`tracks/<op>.<version>-<digest>/<group>__<seq>.parquet` instead of
`tracks/<group>__<seq>.parquet`. The directory names the recipe that produced
it, and is the same identity already recorded in the row's `run_id` and
described in that directory's `params.json`.

This closes a defect rather than only tidying a layout. All five producers — the
three converter branches, the TREx bridge, the inference bridge — used to target
one flat path behind an `exists() and not overwrite` skip, so a second producer
for a sequence was **discarded with a success return**. Two tracker runs with
different settings each built a complete run-addressed tree under `trex/`, and
only the first reached `tracks/`.

**Read paths are unaffected.** Every reader resolves `abs_path` out of
`tracks/index.csv`, and no glob in the package walks the tracks root. Nothing on
disk is moved or deleted: tables written before this release stay where they are
and their rows keep resolving, so reverting the release resolves both layouts.

**`convert_all_tracks()` re-converts once, on a dataset converted before this
release.** The output path changed, so the overwrite skips no longer find
anything. That call is cell one of the CalMS21 and MABe22 notebooks and of the
getting-started guide. It is not destructive — the old tables and rows stay — but
it is not free either, and it then moves every downstream identifier for
features that read tracks, because of the next item.

**Feature identity now covers which tracks recipe a run read.** A run consuming
`tracks` used to record the bare literal in its digest, so two tracker settings —
which produce different numbers in the tables — shared one identifier, one
directory and one cache entry. `compute_run_id` gains a `_tracks` term, and
`FEATURE_IDENTITY_SCHEME` moves 2 → 3.

The term is **omitted when there is nothing to say**, which is what protects
existing work: rows written before tracks carried an identity have an empty
`run_id` and contribute nothing, and an absent key digests differently from an
empty one. A dataset that is not re-converted keeps every identifier it has.
Verified against the archived guppies analysis, whose identifiers are unchanged
and are now pinned in the golden corpus so the check needs no drive.

**`tracks/index.csv` may hold several rows per entry.** Its dedup key is the
`(run_id, group, sequence)` triple its sibling indexes use. Which row an entry
*resolves* to is decided in one place: an unlabelled row loses to a labelled
one, and two genuinely different recipes for one entry raise rather than guess.

**New selector, additive.** `tracks_run_id` is keyword-only on `run_feature`,
`Dataset.run_feature` and `load_values`; `run_id` on `Dataset.load_tracks`,
`Dataset.drop_entries`, and both sequence iterators; `--tracks-run-id` on
`mosaic run` for feature runs. All default to today's behaviour.

**Contract surface: nothing moved.** The five `mosaic` symbols `mosaic-api`
imports — `Dataset`, `MediaIndexScope`, `new_dataset_manifest`, `from_safe_name`,
`FEATURES`, plus `core.pipeline.media_index` and `write_media_index` — are
untouched, and mosaic-api, mosaic-app and mosaic-queue reference neither the
`tracks` root nor its index. `mosaic-api` still declares
`mosaic-behavior>=0.1.0`; the floor is two releases stale and is doing nothing.
Bump it to `>=0.3.0`.
