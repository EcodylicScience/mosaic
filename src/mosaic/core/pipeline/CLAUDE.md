# mosaic.core.pipeline

This package owns feature execution, data loading, output writing, dependency
resolution and the typed indexes of a dataset. Features own computation only.

Layering, the one parquet writer and `run_feature` as the only path to feature outputs
are in the root `CLAUDE.md`. The upward imports are in `core/CLAUDE.md`. Source-space
publication and codec refusal are in `tracking/CLAUDE.md`. Rules for the pipeline graph
are in `graph/CLAUDE.md`.

## Cross-package rules

- **A run reports its coverage in its run-log.** `RunLogSnapshot` (`runlog.py`) folds
  `entries_written`, `cache_hit` and `tracks_variant`. Emit them before the terminal
  event (`tests/core/pipeline/test_job_contract.py`).
- **An uneven time axis is a run-log event.** A method that reads its rows as evenly
  spaced warns through `core/timebase.py`, and `run_feature` collects each warning per
  entry (`apply_noting_irregular_time`, also in a process worker) into a
  `time_irregular` event. `entries_time_irregular` counts them, `read_time_irregular`
  lists them, and the request and run status wires carry both. The event is neither a
  status nor terminal, and a reader that predates it folds the log as before
  (`tests/core/test_irregular_time_notices.py`).
- **A read site uses a per-kind lookup, and a write site an own-address function.** A
  dataset reads linked data through `tracks_rows`, `feature_rows`, `feature_run_roots`,
  `feature_run_file`, `feature_run_file_location`, `fit_state_root`,
  `readable_feature_storages`, `frame_rows`, `frame_run_file`, `label_series_rows`,
  `labels_rows`, `readable_label_kinds` and `run_coverage`, which walk
  `Dataset.lookup_order(kind)`. A site that writes, removes or reconciles stays on
  `feature_run_root`, `feature_index_path`, `read_tracks_index`, `tracks_table_path`,
  `frames_run_root`, `list_frame_runs`, `model_index_path`, `read_label_series` and
  `read_labels_index`. A linked dataset is then not written
  (`tests/meta/test_link_reads.py` names both sets).
- **A lookup passes `merge_linked_rows` the path columns it makes absolute.** A linked
  row's cell names a file of its own dataset, and an own cell stays as stored. The
  tracks lookup passes `TRACKS_ROWS_PATH_COLUMNS` (`abs_path` and the columns the
  portability passes rewrite), the converted labels lookup `LABELS_ROWS_PATH_COLUMNS`
  (`abs_path` and `source_abs_path`), the feature and label series lookups `abs_path`,
  and the frames lookup `abs_path` and `video_abs_path`. A new indexed path column joins
  its lookup's tuple (`tests/core/pipeline/test_linked_rows.py`).
- **`link_unreachable` is a refusal reason** (`LinkUnreachableError`). A plan problem at
  that `where` refuses a submission, and a step planned again at its start refuses
  itself alone (`graph/CLAUDE.md`, "Planning"). It is raised where an offline link
  decides whether data is computed again, and an answer that a reachable dataset already
  gives does not raise it.
- **`track_universe.current_run_id` is the one rule for an unpinned `Result`**
  (`tests/core/pipeline/test_unpinned_result_rule.py`). A `DefinitionModel` is the
  exception: an unpinned one is refused, naming the candidate fits
  (`definitions/fit_run.py`), since an apply reads one fit.
- **A lock the filesystem refuses raises `IndexLockUnsupported`**
  (`core/pipeline/index_lock.py`) and is not degraded to an unlocked write
  (`tests/core/pipeline/test_index_lock_capability.py`). The lock serializes writers
  across machines where the filesystem sends locks to its server (NFS, CIFS). A CIFS
  client that caches still loses rows written by other machines, so `index_lock` warns
  with `CachingMountWarning` on a Linux CIFS mount without `actimeo=0,closetimeo=0`
  (`tests/core/pipeline/test_index_lock_caching_mount.py`).
- **Entries computed at once go through `EntryPool`** (`entry_pool.py`): `run_feature`
  and each op that sets `parallelizable` (`convert-tracks`, `resample-tracks`), which
  reads its workers from `ctx.parallel`. A worker stages a tracks table under a hidden
  `.stage-*.staged` name beside its final one (`staged_tables.py`) and returns it with
  its index row (`TracksRowDraft`). The calling process, through `OrderedResults`, in
  submission order, publishes the tables in batches (`TableCommitter`: the tables of a
  batch move into place, then one write of the index records their rows, so a run is not
  quadratic in the index; a batch is due after `max_batch` tables, or after the longer
  of 10 s and four times the duration of the last write; a batch with a key collision is
  refused before any table moves), reports the failures and the progress, so the outputs
  do not depend on the worker count. The caller flushes the committer at the end of the
  run and when it stops early. Of two files naming one entry the first keeps it, or the
  last when overwriting. The op removes the staged files of an earlier run at its start,
  and a table with no row is made again, not skipped. A process worker reopens the
  dataset from its manifest path (`dataset_ref`, `worker_dataset`). An op that ends on
  an exception settles the pool first and publishes what its workers finished
  (`tests/core/pipeline/test_op_parallelism.py`). A cancel signals the run's process
  group, and an op's process workers, which keep the default dispositions, die of it. A
  pool that breaks while a cancel is requested ends the run `cancelled`, not `failed`
  with `WorkerProcessDied`, which `run_cost.stopped_for_memory` reads as a memory stop.
  A feature run's pool breaks on a cancel only through a second signal or a worker's
  death. A pool takes the caller's `cancel_check` and polls it for
  `CANCEL_ARRIVAL_SECONDS`, because the caller's SIGTERM handler can run after the pool
  sees the break (`tests/core/pipeline/test_parallel_run.py`).

## Public types and imports

- The public typed API is in `types/` (`Inputs`, `Result`, `ArtifactSpec`,
  `InputStream`, `DependencyLookup`, the labels sources).
- `__init__.py` binds every public name on access through `mosaic.lazy_exports`,
  which imports the standard library only (`tests/meta/test_pipeline_lazy_exports.py`).
  The media ops and the conversion ops (`convert_ops.py`: `convert-tracks` and
  `convert-labels`, domain `tracking`) register through `ensure_core_ops_registered`
  (`ops.py`, `CORE_OP_MODULES`), which every reader of `OPS` in `ops.py` and
  `mosaic.tracking.register_ops` call.
  `reconcilable_index` imports the modules that register the core indexes.
- `inventory/` imports nothing from `tracking` or `behavior`
  (`tests/core/pipeline/inventory/test_inventory_contributors.py`). Tracking facts
  arrive through the seams the root guide lists, plus the reader registries in
  `tracks_axis.py` (`register_frames_read_reader`, `register_tail_loss_reader`).

## Identifiers

- `compute_run_id` (`run.py`) names a feature run `<version>-<10 hex>` over
  `_params` (`identity_dump()`), `_inputs` and `_frame_range`, plus `_scope_entries`,
  `_tracks`, `_labels`, `_label_revisions` (the sorted `[origin_uuid, key, digest]`
  of each label series revision read, never its number or payload digest),
  `_overlap_frames` and `_definition` (a feature definition's source digest) only
  when present. A new term is omitted when absent, which leaves
  existing identifiers unchanged
  (`tests/core/pipeline/test_feature_run_id_golden.py`). The neighbors an overlapped
  entry read are recorded on its row and left out of the identifier.
- `resolve_labels_variants` takes a pinned `labels_run_id` as given, as a pinned
  `tracks_run_id` is, so a run planned before its labels are converted is named as it
  executes (`tests/core/pipeline/test_labels_identity.py`).
- The tracks variants a run read enter `_tracks`, and the storage directory name
  (`derive_storage_name`) leaves them out. `features/<name>__from__tracks/` stays one
  directory and one index whatever the number of variants in a dataset.
- A storage name over 255 UTF-8 bytes becomes `<slug>__from__<head>~<10 hex>` within
  160 bytes (`derive_storage_name`). Names that fit are unchanged. No slug or literal
  input name contains `~` (`tests/meta/test_storage_name_marker.py`). An encoded claims
  component over 255 bytes becomes a head and a `~` digest of the whole storage name
  (`graph/claims.py`). Cleanup treats a `~` storage of unknown params as reading every
  run (`tests/core/pipeline/graph/test_pipeline_recipe.py`,
  `tests/core/pipeline/graph/test_pipeline_cleanup.py`). The rule lives in
  `storage_names.py`, which imports only the standard library, because the status reads
  reach it through the claims store (`tests/meta/test_status_imports.py`).
- Tracking ops and tracks variants are `<kind>.<version>-<10 hex>` (`op_identity.py`).
  The version is a visible segment and is not hashed. `extract-frames` is carved out
  and frozen, because mosaic-api stores its identifier in annotation paths. Transcode
  folds its version into the digest instead, because its output is a filename with no
  directory level (`tests/core/pipeline/test_op_identity.py`,
  `tests/core/pipeline/test_op_identity_golden.py`).

## Index files and locks

- Every index is written atomically under `index_lock` (`index_lock.py`). Most are
  typed `IndexCSV`s (`index_csv.py`). The media index is written by
  `write_media_index_rows` (`media_index.py`).
- The lock lives on a `<index>.lock` sidecar that is created once and stays, and a
  locked block may `atomic_write` any number of times. Code that enumerates a root
  expects the sidecars (`tests/core/pipeline/test_index_lock_serialization.py`).
- `index_lock` creates the guarded file empty when it is absent. A tracks variant's
  `params.json`, merged under the same lock by `write_tracks_variant`, reads as absent
  while it is empty.
- An unfamiliar errno keeps polling and ends in `IndexLockTimeout`.
- A dry run reads without the lock, through `index_lock_unless_dry_run`, and leaves no
  index or sidecar behind. It still sees one whole version, because every writer
  replaces the file by rename.
- Taking `index_lock` leaves an existing index's modification time unchanged, which
  `InventoryCache.revalidate` relies on
  (`test_acquiring_leaves_an_existing_index_unchanged`).

## The tracks index

- `write_tracks_row` and `write_tracks_rows` (one read-modify-write for a batch of
  `TracksRowDraft`) are the only appenders and `read_tracks_index` the only reader
  (`tracks_index.py`). A writer of many rows uses the batch. Rows written one at a time
  reread the whole index each. Rewrites (`delete_set`, `reconcile_variants`,
  `Dataset.drop_entries`, the backfill passes) take `index_lock` on the same file.
- `index_tracks_variant` writes rows over tables already under `tracks/<run_id>/`
  through `write_tracks_rows`, without converting them. `index_labels_variant` does the
  same for `labels/<kind>/<run_id>/` through `write_labels_rows`, the batch writer of
  the labels index (`LabelsRowDraft`, one kind per batch).
- One row per `(run_id, group, sequence)`, with no camera column. An absent index reads
  as an empty frame with the full schema. An older schema is widened in memory inside
  the write lock (`adopt_legacy_columns`), and a read leaves the file as it is
  (`tests/core/pipeline/test_tracks_index.py`).
- `select_variant_rows` is the only resolver. An unlabeled row (`run_id` empty) loses
  to a labeled one, and two recipes for one entry raise. The caller answers with
  `tracks_run_id=` (`run_id=` on `load_tracks` and `drop_entries`), and `""` names the
  unlabeled tables. It raises `AmbiguousTracksVariant`, and it reduces
  `variant_candidate_rows`, which keeps each recipe of an entry for a caller that
  chooses none (`refuse_mixed_track_schemas`).
- `row_in_scope` is the one filter of resolved rows to a run's entries, and
  `rows_in_scope` applies it to a frame. `None` covers every row, and an empty list
  covers none.
- The writer measures `n_keypoints`, `frame_min`/`frame_max`, `pair_frames` and
  `ids_per_frame` from the parquet (`frame_measures`, one read of the `frame` and `id`
  columns). The producer passes `media_frames`, `frames_read` and `known_tail_loss`, and
  a conversion passes its converter's `frames_read`. A blank cell on every count column
  means unknown. The `TracksIndexRow` docstring defines each column.
- `pair_frames` and `ids_per_frame` are two cells of one measure: the ordered pairs summed
  over the extent's frames and the distinct `(frame, id)` over its length. A table with no
  `id` column writes `none` (`NO_IDS`) in both, which means measured, no ids, and is never
  a bound. `read_pair_frames` and `read_ids_per_frame` are the only readers and read
  `none` and a blank as `None`. `backfill_pair_frames` fills a row whose `pair_frames`
  cell is blank and so leaves a `none` row alone (`mosaic measure-tracks`,
  `Dataset.measure_pair_frames`, `tests/core/pipeline/test_tracks_index.py`).
- `carries_tracklets` is the row's `"true"`, `"false"` or blank for whether `id` follows
  an animal. `read_carries_tracklets` is the only reader. It answers a blank from
  `producer` where the producer settles it and otherwise returns `None`, which no plan
  refuses. A converter declares it (`TrackConverter.carries_tracklets`), SLEAP through
  the user's `tracks_named`. `rows_without_tracklets` lists each entry whose row says
  false with its variant, and `tracklets_by_entry` gives each entry's answer under the
  variants a run reads, which `run_feature` hands to a feature's `set_tracklets`
  (`tests/core/pipeline/test_tracks_index.py`).
- `write_tracks_row` requires `carries_tracklets`. A tracker passes its own answer to
  `publish_tracks_table`. An op that derives a variant from another copies
  `read_carries_tracklets` of its source row, and never passes `None` or a constant
  (`resample-tracks`, `assign-arenas`, `upgrade_trex_tables`). `assign-identities`
  relabels by identity and writes `True`.
- `frame_axis_verdict` is the one rule separating a frame-axis mismatch from a known
  tail shortfall. The `backfill_*` passes behind `mosaic measure-tracks` document their
  rules in their docstrings. The lock discipline of the three cell passes is in
  `_backfill_cell`, and `backfill_frame_extents` takes the lock once for its whole pass
  (`tests/core/pipeline/test_backfill_passes.py`).

## Compositions

- A run identity is params-only. What a run consumed is recorded on its row and
  compared, and `run_id` leaves it out.
- `run_feature` compares each cached entry's `consumed_tracks_composition`
  (`recorded_tracks_reads`, `index.py`) with the tracks row's current composition
  through `cached_entry_disposition` (`composition.py`). A difference, or one known side
  beside a blank one, recomputes the entry. A re-conversion from changed sources
  therefore recomputes (`tests/core/pipeline/test_tracks_source_staleness.py`).
  `tracks_compositions` adds a tracks row's `first_frame` past 0 to its composition, so
  a table published again at a moved first frame recomputes its features.
- A run with a feature input compares instead
  `ResolvedScope.input_tracks_compositions` (`build_manifest`): per entry, its inputs'
  cells joined with `+` in input order, a feature input's cell being its upstream
  row's `consumed_tracks_composition`. An entry therefore recomputes after its
  upstream entry recomputes under the same run id. A run whose only input is `tracks`
  leaves it `None` and records the tracks rows' cell, and a row written without the
  upstream part recomputes once.
- A run with a feature input also compares `consumed_input_composition`
  (`ResolvedScope.input_root_compositions`): its upstreams' declared-root
  compositions and fit states, each upstream row's `consumed_composition` and
  `consumed_input_composition` as `own/inherited`, led by `join=<keys>;` (its
  `consumed_summary_join`) and then `state|` (its `consumed_model_state`) when each is
  not blank: `join=<keys>;state|own/inherited` (`input_part`, `composition.py`),
  joined in input order by `join_input_cells`.
  An entry recomputes after its upstream entry recomputes for a moved declared
  root or a fit written again, at any depth of a chain. Both sides blank is
  absence, served without a warning. The blind tracks warning covers a
  feature input whose upstream chain read a named tracks variant at any depth
  (`TracksVariantWalk`), and names that input's run.
- At `overlap_frames > 0` the row's `consumed_neighbors` (`neighbor_compositions`,
  `run.py`) names the entry keys of the sequences before and after the entry with
  their tracks compositions, and is compared the same way. The cell is then always
  filled. Two divisions of a store are neighbors only where the second's frames
  continue the first's (`recording.adjacent` in `_neighbors`). A gap or a refused
  division leaves none. Declaring a group, adding a sequence to it or converting a
  neighbor again recomputes the entries whose neighbors changed, and a row written
  without the cell recomputes once. It is empty at `overlap_frames=0`
  (`tests/core/pipeline/test_run_feature.py`).
- A step whose inputs may both hold a summary key (`may_share_summary_key`,
  `graph/columns.py`: two inputs whose producers' declarations admit `bin_start`,
  `bin_end` or `arena`, and an undeclared producer admits any) records in
  `consumed_summary_join` the keys that the shared-key join of the merge joined for the
  entry: `summary_join_cell` (`loading.py`) folds `merge_step` over the footers' column
  names, which `entry_summary_join` reads under the run's footer memo, leaving out an
  input that records no rows. The cell is sorted keys joined with `,`, and blank for
  none. `_servable` compares it through `cached_entry_disposition` after the other
  channels, so a row that recorded another cell recomputes once, and a cell blank on
  both sides is served, without a warning. Any other step records `""` and reads no
  input footer. The cell stays out of `run_id`
  (`tests/core/pipeline/test_run_feature.py`).
- `mosaic reconcile` marks the rows of a multi-input run that were merged when a shared
  summary key was kept as two columns (`FeatureReconciler._summary_marks`). It selects a
  run whose marker `summary_join_unmarked` finds below `SUMMARY_JOIN_SCHEME` (`"7"`) and
  whose inputs `may_share_summary_key`, and reads each entry's input footers once. A row
  whose entry the current rule merges differently gets `legacy:<keys>`
  (`LEGACY_SUMMARY_JOIN`) in `consumed_summary_join`, never the current cell, which would
  serve old output as current. A row merged the same way stays blank. The marks are
  written under `--apply`, after an index backup and before the marker, so a crash
  between them leaves the run selected for the next pass. Both the marks and the pending
  note below stay inert until `FEATURE_IDENTITY_SCHEME` reaches `"7"`.
- A run whose inputs cannot be read for its marks is blocked as
  `unresolvable_pre_provenance`, and a run or variant whose new address is taken is
  blocked as `identity_shift_recompute` with no move recorded, in the features and
  variants passes, so that no later run reads an address without a directory.
  `backup_index` gives each backup a unique name, which keeps the earliest copy of an
  index that several writes of one pass touch
  (`tests/core/pipeline/test_reconcile.py`).
- The inventory marks a multi-input run whose marker predates `"7"` in
  `ArtifactRecord.extra["reconcile_pending"]` (`_reconcile_pending`,
  `inventory/scan.py`), and the CLI prints a note for it. It reads the marker and
  `params.json` alone, never a footer or a declaration, so it over-reports a run whose
  inputs cannot share a summary key until reconcile raises the marker.
- A division of a store is a member of its entry's media composition under its
  `imgstore-division/1` uid, so each division has its own composition, and a division
  whose chunks change frame count has a new uid (`tests/core/media/test_media_drift.py`).
- A step that reads a fit records the fit's state in `consumed_model_state`. A fit
  writes `state_digest` into `fit_scope.json` right after `save_state`: a blake2b over
  the files that `save_state` created or changed. The run root is listed before and
  after it (`list_run_root`, which leaves out the engine's files: the claim marker, the
  run's `params.json`, `.identity_scheme` and `fit_scope.json`, and temporary files) and
  digested by `saved_state_digest` (`fit_scope.py`). The cell has one part per fit
  reference (`fit_references`, `fit_scope.py`: a `model`, `templates`, `label_map`,
  `init_from` or `decision_threshold` field that names a pinned run, as an artifact
  reference or a `DefinitionModel`), keyed by field, and a `_state` part for the state
  the run loaded, when its root contains `fit_scope.json`. One part is a bare digest,
  and several are sorted `field=digest` pairs (`fit_state_cell`, `index.py`). The run
  reads each fit's current digest once (`_current_fit_states`) through `fit_state_root`,
  and `_servable` compares the cell with each cached row's. A fit written again under
  its run id then recomputes the rows of the fit and of every step that reads it. A row
  and a fit that both predate the digest are served, with one warning. The digest stays
  out of `run_id`. The contract: an apply reads only the files `save_state` wrote, since
  a file that `fit` leaves in the run root is outside the digest
  (`tests/core/pipeline/test_model_state_drift.py`).
- A fit run records its inputs, and a fit whose inputs changed is fitted again under its
  run id. `fit_scope.json` records `fit_inputs`, the `group`, `sequence`, `roots`,
  `tracks` and `inputs` cells of each entry the fit read (`current_entry_cells`,
  `index.py`, the construction `drifted_entries` shares), the `state_digest` of each fit
  it read on `via`, and the `execution_id` of the attempt that fitted. `params.json`
  records `_fit_references`. None of them is hashed. `records_fit_inputs`
  (`fit_scope.py`) is true for an instance whose `fits` or `scope_dependent` is true,
  and only those runs are compared. A fit pinned by `fit_set_pinned_by` records the
  entries its `fitted_entries` names, a scope-dependent run its resolved scope, and a
  fit from its pinned artifacts alone none, since the digests on `via` record its
  change. `fit_drift` (`index.py`) reports `moved` entries (a cell differs under
  `cached_entry_disposition`), `dropped` training entries missing from the run's inputs,
  for a fit pinned by `fit_set_pinned_by` only, and `refitted` fields (a fit it read has
  a new digest). A cell blank on both sides is undetectable and not drift. A fit saved
  without `fit_inputs` is served with one warning, and its plan step is not marked for a
  refit. A read of the plan, the inventory or the status takes an unreadable record as
  absent and a fit under an unreachable link as blank. The run reads strictly and raises
  `FitStateUnreadable` or `LinkUnreachableError`
  (`tests/core/pipeline/test_fit_input_drift.py`).
- `drifted_entries` (`index.py`) follows `cached_entry_disposition` over four channels,
  given the run's `RunParams`: the declared roots, the tracks cell and the upstreams'
  roots cell, each built as `build_manifest` builds it, and for a run that reads a fit
  the fit-state cell, against the digests `DriftReads.fit_state` reads once per fit
  through `fit_state_root`. A digest is blank under an unreachable link, for an
  unreadable `fit_scope.json`, and for a pattern that matches more than one file of the
  fit's root (`AmbiguousArtifactFile`), since a status read does not raise. One known
  side is drift, and the plan names every entry `run_feature` recomputes. A row whose
  `consumed_summary_join` starts with `legacy:` (`LEGACY_SUMMARY_JOIN`) is drift of its
  entry, read from the row with no footer. A run whose fit drifted (`fit_drift`)
  reports every entry of its rows as drifted, because a refit recomputes every entry it
  is asked for. The inventory status, a plan's drift work and `Pipeline.run` then treat
  the run as due without a second rule. The neighbors cell stays `run_feature`'s
  (`TestDriftedEntries` in
  `tests/core/pipeline/test_tracks_source_staleness.py`). An entry the inputs no longer
  resolve, such as one dropped from the tracks index, is not drift: every input holds it
  for a run that intersects its inputs, and any input for a stacked run, as
  `_scope.stacks_inputs` in `params.json` records (`stacks_inputs`, `loading.py`). A
  file without the key counts any input, so a false drift lasts one run, where every
  input would hide a stacked run's drift. Without params it compares the declared roots
  alone.
- `arenas` is a composition root outside `SourceRoot`: an entry's layout digest
  (`arenas_compositions`, `arena_index.py`), read only where a feature declares it in
  `consumed_roots`. `build_manifest(consumed_roots=...)`, `resolve_feature_identity`
  (for a scope-dependent feature), `drifted_entries` (`DriftReads.arena_lookup`) and
  `provenance._current_compositions` merge it. Each reads through
  `cached_arena_lookup`, one read of the arenas and media indexes per process for
  each state of the two files. `read_entry_compositions` never reads it, so tracks
  conversion, frame extraction and preprocessing read no arenas index
  (`tests/core/pipeline/test_arenas_composition.py`).
- `consumed_media_composition` on tracks rows, media variant rows and tracker run rows
  is compared through `compositions_disagree` (`composition.py`). There a blank side is
  unknown and not drift (`tests/core/pipeline/test_tracks_media_drift.py`).
  `media_drift` also reports a tracks row whose recorded `first_frame` differs from the
  sequence's current one.

## Inputs and claims

- **A track-shaped input records the spaces of the tracks behind it.** For a `Result`
  input of a `_track_input` class, `build_manifest` adds to `ResolvedScope.track_spaces`
  the schema spaces of the tables the upstream run read, following each track-shaped run
  upstream (`upstream_track_spaces`, `track_space.py`). Step preflight
  (`graph/step_preflight.py`) makes the same walk. A run's tables are the labeled
  variants on its tracks edge (`RunParams.consumed_tracks_variants`), then the unlabeled
  tables for the entries none of them holds, because `run_feature` records labeled
  variants only. The walk reads footers (`run_is_track_shaped`) and `params.json`, never
  `FEATURES`, so a derived `Result` adds nothing
  (`tests/core/pipeline/test_track_space.py`).
- `alignment_verdict` (`loading.py`) decides whether inputs merge, and the merge
  raises `MultiInputAlignmentError` from it. Individual-level against pair-level
  output with no shared identity column is refused. `CROSS_JOIN_FEATURES` is the
  closed escape (`tests/core/pipeline/test_multi_input_alignment.py`).
- The merge folds the inputs left to right, and a position takes the first rule that
  applies. Inputs that share an alignment column join on the shared `ALIGN_COLS` plus
  the `SUMMARY_KEYS` (`bin_start`, `bin_end`, `arena`, in `SUMMARY_KEY_ORDER`) that both
  sides have, so a shared summary key is one column and is not renamed
  `<key>__<position>`. A cross-join consumer joins no summary key. Inputs that share no
  alignment column (`cause == "no_keys"`) join through `fallback_join` (`loading.py`),
  a pure function of two column sets that `merge_step` and the plan call: on the
  summary keys both sides have, else by the entry (a cross join, where one side has one
  row) when a side has none of `ID_COLS`. An unidentified side with a summary key the
  other lacks refuses, unless the other is unidentified and has none of `SUMMARY_KEYS`:
  the sides then join by the entry (`FallbackJoin.single`), and the other must have one
  row, which `_merge_by_entry` checks. A join on summary keys refuses a row that the
  broadcast side does not cover. A summary key that is `NaN` on a side refuses before
  any join that uses it, because pandas matches `NaN` keys to each other.
- `expands_individuals` (`loading.py`) is the one gate for widening an individual-level
  input onto the pair rows of a pair-level input 0, as `<col>__id1` and `<col>__id2`:
  `emits` of `as-input` or `pair`, outside `CROSS_JOIN_FEATURES`. Where the pair rows
  and the individual input share an order column the widening joins on it and `id`.
  Where they share none it joins on `id` plus the summary keys both sides have, so a
  binned per-id summary widens onto binned pair summaries, and a repeated key row
  refuses. `run_feature`, `verify.py`, `_load_neighbor` and the declarations
  (`ConsumerDecl.expands_individuals`) all read the gate, and `merge_naming` names the
  columns for the merge and for validation.
- `EntryRowsMissing` (`loading.py`) is the base of the errors for an entry that loses
  the rows its merge needs, and each is recorded as an `entry_error`, in the entry's
  task, or in the parent's load under overlap. `PairRowsMissing` is an entry whose pair
  input holds no rows (`tests/core/pipeline/test_widening_join.py`).
  `BroadcastRowsMissing` is an entry whose input loaded no rows while another input did,
  where the empty input's footer names none of `ID_COLS` and shares none of `ALIGN_COLS`
  with the columns of the inputs that loaded: the rest would be written without its
  columns, and the run's entries would hold mixed schemas
  (`tests/core/pipeline/test_partial_entry_failure.py`). `IndividualRowsMissing` is an
  entry of a widening step whose pair rows loaded while an individual-level input
  loaded none: the pair rows would lack its `<column>__id1` and `<column>__id2`
  columns (`tests/core/pipeline/test_widening_join.py`). A fit's input stream skips each
  of these entries through `iter_manifest`'s `on_rows_missing`, which `skip_rows_missing`
  fills. An input empty at every entry loses every entry, which raises
  `AllEntriesFailed`. `load_values` does not widen.
- Inputs otherwise cover the entries they all hold. A feature that declares
  `stacks_inputs = True` (`stacks_inputs` in `loading.py`, `evaluate-labels`) reads its
  inputs as runs over disjoint entries: `build_manifest(stack=True)` unions them, takes
  each entry from the one input that holds it, and raises `StackedEntryOverlap` for an
  entry two hold (`tests/core/pipeline/test_input_manifest.py`). The planner refuses
  inputs asked for the same entry first (`_input_coverage_problems`,
  `tests/behavior/feature_library/test_evaluate_labels.py`). It sees what each input is
  asked for, not what a scope-free run already holds, so the manifest keeps its check.
- Trackers, `infer-*` and `preprocess` claim each entry through `open_entry`
  (`entry_claim.py`) with an exclusive create, and release and refresh check
  ownership (`tests/core/pipeline/test_run_markers.py`). One-shot ops claim their
  run root through `claim_run_root` in `entry_claim.py`, which raises the
  `run_root_held` refusal (`RunRootHeld`) when a live claim holds it.
- Outside `Pipeline.clean`, a whole feature run is removed only through
  `remove_feature_run` (`feature_run_removal.py`): an intent record with the run's
  `params.json` in the cleanup's run-log, then the index rows, then the directory,
  then a done record (`tests/core/pipeline/test_feature_run_removal.py`).
- A fit runs under `held_run_root` (`entry_claim.py`): `run_feature` claims the run
  root once `load_state` misses, loads the state again under the claim, and a timer
  refreshes the claim until the fit and its save end. A live holder refuses with
  `run_root_held`, which `execute_step` does not count as a step failure. Apply is
  unclaimed (`tests/core/pipeline/test_fit_claim.py`).
- A fit runs again when its record drifts (`_check_fit_inputs`, before the run root
  exists) or under `refit`, once per request: a `refit` loads the fit instead when the
  record in the run root names the invocation's execution id, and a retry of a step then
  fits once. A drift is checked again under the claim, and a record that no longer
  drifts is loaded. A refit offers `load_state` an empty directory outside the dataset
  (`_RefitScratch`, removed after the feature closes) and requires `False` from it. A
  `True` raises `CannotFitAgain` with the cause `loads_elsewhere`, as does a `fit` that
  writes into the directory (`wrote_outside_run_root`). The engine then binds the real
  run root through `bind_run_root`, as it does for a first fit, and a fit pinned by
  `fit_set_pinned_by` streams its training entries whatever the invocation asks for, and
  any other fit streams the invocation's manifest. A drifted fit stored in a linked
  dataset refuses with `linked_fit_drifted` (`LinkedFitDrifted`), and `refit` fits it
  into the reading dataset (`tests/core/pipeline/test_fit_input_drift.py`). `overwrite`
  does not fit. It recomputes outputs from the stored fit.

## Parallel runs

- A `ProcessPoolExecutor` comes from `caller_guarded_process_pool` (`process_pool.py`),
  whose workers wait on the parent sentinel of `multiprocessing.parent_process()` and
  `os._exit` when the caller dies. A worker holds both ends of the executor's pipes, so
  a caller killed by SIGKILL never gives it end of file. The sentinel is a pipe that the
  caller holds under every start method, whereas the worker's own parent is the
  forkserver under `forkserver`. The guard reads no process id and no clock, sets no
  `PR_SET_PDEATHSIG`, which fires on the death of the thread that spawned the worker,
  and leaves signals and process groups alone. The caller's own initializer runs after
  the guard, so an initializer that hangs or raises leaves a worker that ends with the
  caller. The pool is a `CallerGuardedPool`: a watcher thread, started after its first
  `submit`, polls the executor's broken flag and kills the live workers
  (`CallerGuardedPool.kill_workers`) once it is set, because a worker that ignores
  SIGTERM keeps the manager thread's join waiting under the executor's shutdown lock,
  and a kill from any thread that waits for that lock never runs. `EntryPool.drain_one`
  and both settles also call `kill_workers()` before they hand back, raise or return.
  The watcher holds the pool weakly, so that a pool dropped without a shutdown is
  collected, and a pool shut down stays alive until the watcher returns, because a break
  during a shutdown needs the kill. It also wakes the executor's manager thread when a
  started worker has exited while the pool is live, since CPython's `submit` wakes the
  manager before it starts the worker it adds, and a worker that dies alone after the
  last submission would otherwise go unseen and the pool would hang
  (`tests/core/pipeline/test_worker_caller_guard.py`, `test_entry_pool.py`).
- The memory limit comes from `read_memory_limit` (`memory_limit.py`): a container's
  cgroup limit, else the host's memory. The floor, the gate and the run-log's limit
  figures read it, never `psutil.virtual_memory()` alone.
- A run measures the memory the kernel cannot reclaim, which a limit kills a job for:
  the cgroup's `anon + shmem`, plus `kernel` under cgroup v2, where `read_memory_limit`
  returns a cgroup source, else the `/proc` tree sum of `RssAnon + RssShmem`
  (`memory_watch.py`). It fixes its measure at the base sample, non-reclaimable where
  that sample measured it and tree RSS where it did not (macOS). The estimate, the
  worker bases, the peak sample and the budget all use that one measure. A sample or
  worker that lacks the figure in a non-reclaimable run is left out and is never read as
  0 (`tests/core/pipeline/test_run_measures.py`). `peak_rss_bytes` stays tree RSS.
- The per-row estimate is `RowCostEstimate` (`admission.py`). `bytes_per_row`
  is `None` until an observation exceeds the fixed part, and the gate reads
  `None` as no estimate. A process worker samples its own `RssAnon + RssShmem`
  from before its load, or before `apply` when handed a table, to after its
  write (`peak_nonreclaimable_while`).
- Every exit of an entry lowers `EntriesInFlight`: an output written, a load of no rows,
  a failure, a skip without a table, and missing pair rows. An exit that does not would
  stall the gate (`tests/core/pipeline/test_run_measures.py`).
- Each computed entry runs as an `EntryTask` through `run_entry` (`entry_task.py`) in
  every mode. The task's `source` is the table the parent loaded or an `EntryLoad`, which
  the worker loads through `load_entry_data` and `entry_bounds_filters` inside a memo that
  `ParquetFooterMemo.from_records` builds from the parent's `FooterRecord`s, keyed by the
  path as the load spells it. The worker checks the columns, refuses an arena-blind input,
  applies, trims and writes the output with `create_parent=False`, and the parent records
  the row from the report. A load that returns no table reports `written=False`, and the
  entry leaves the in-flight record without a row. A process worker runs it through
  `run_entry_in_process` with the feature its `FeatureWorker` holds, which the worker
  builds at its first task (`_built_feature`) and keeps for the pool's life, and which a
  build that raised leaves unbuilt, so the worker's next task builds again. The pool's
  initializer (`start_feature_worker`) builds nothing. The worker reads its base before
  the load, which for the task that built the feature contains the feature's state, and
  subtracts only a handed table. An input failure (any load error but `EntryRowsMissing`)
  and a trim or write failure return as an `EntryFailure`, and the run raises it.
  `EntryRowsMissing`, the one base of an entry that loses the rows its merge needs, and
  any other entry error raise in the task and are recorded as an `entry_error`. The task
  loads its entry's input, with the load's `expands_individuals` and `pair_filter`, unless
  the run asks for overlap or the feature is the cross join. Those shapes load in the
  parent through `iter_manifest`, which hands the task the table. Both sites call
  `load_entry_data`. A pair-filtered task builds its entry's nearest-neighbor lookup
  through `nn_pair_filter` before the load: a pool thread from the run's dataset, and a
  process worker from the dataset `worker_dataset` opens without its roots
  (`ensure_roots=False`). The fit's input stream builds it in the parent. A neighbor table
  that cannot be read is an input failure, which ends the run with its error and no
  `entry_error`. `entry_task.py` names no index module, the run-log or `job` in its own
  imports (`tests/core/pipeline/test_entry_task.py`, `test_entry_errors.py`,
  `test_pair_filter_load.py`, `tests/meta/test_entry_task_imports.py`).
- `apply_seconds` runs from the dispatch, so it contains the load of an entry whose
  worker loads it and, for the entry whose worker built the feature for it, that build,
  and ends at the end of `apply`. `after_apply` stamps it in serial and thread mode, and
  in process mode it is the moment the outcome was set less the worker's
  `seconds_after_apply`, which runs from the end of `apply` to the worker's return.
- A feature run's process workers start through `start_feature_worker`, which makes
  SIGTERM and SIGINT do nothing in them (`ignore_stop_signals`: a handler, because an
  ignored disposition survives `exec` into a subprocess of `apply`). A stop signal to
  the process group does not stop a worker. With the cancel handler of
  `install_signal_handler` (`mosaic run`, the pipeline driver), either signal cancels
  the run, and the workers finish their entries. Without it SIGINT raises
  `KeyboardInterrupt` in the run's process, which waits for the workers to finish their
  entries, and SIGTERM ends that process by default, so the caller guard
  (`_exit_when_caller_ends`) ends its workers mid-entry. A second signal to a process
  with the handler kills the live workers of every guarded pool
  (`kill_guarded_workers`), and an abandoned settle kills them
  (`EntryPool.kill_workers`), so the run's process can exit. `run_feature` checks for a
  cancel after the pool finishes, so a cancel that arrives while the last entries are in
  flight ends the run `cancelled` in every mode, with every row recorded. The ops'
  pools, the TRex and frame-extraction pools and the worker-base probe keep the default
  dispositions (`tests/core/pipeline/test_run_exits.py`, `test_parallel_run.py`).
- Every exit of `run_feature`'s apply loop, a cancel, an interrupt or an error, flushes
  the index rows, prints one stderr line when entries are in flight, and settles the
  pool with `settle(results_only=True)`: a call that returns has written its output and
  gets its row, and a call that raises is not recorded. It flushes again, records
  `entries_written` once, and raises the exception that ended the loop. A failed flush
  and a settle that raises are each a note on the exception raised. The errors raised
  while recording the outputs in flight, which the settle dropped, are one note:
  `N of the outputs of the entries in flight raised while being recorded, the first: <type>: <message>`.
  In a notebook or the Python API, a second `KeyboardInterrupt` during the settle
  abandons it and is raised instead, with the first exception as its context, after the
  second flush has run. A feature run's pool (`hand_back_on_break`) hands back every
  call finished before a break of a process pool, results first, before it raises; a
  worker's `KeyboardInterrupt` is dropped. The ops' pools (`EntryPool.for_settings`) do
  not hand back. A feature run's process worker that a signal to the process group
  reaches, such as a terminal's Ctrl-C, a notebook's interrupt in process mode or a
  group cancel the run handles, finishes its entry, which the run records
  (`tests/core/pipeline/test_run_exits.py`, `test_entry_pool.py`). A group SIGTERM that
  the run does not handle ends its process and, through the caller guard, its workers
  mid-entry. These leave an output on disk without a row: a parent killed by SIGKILL or
  the out-of-memory killer, for each output its workers wrote whose row it had not yet
  flushed; thread workers still running after a second interrupt; a process worker
  killed after its rename and before it returned. A rerun serves such an output when its
  provenance is blank on both sides and recomputes it otherwise. A process worker killed
  while it writes leaves a hidden temporary file and no output.
- For an entry whose task loads it, the parent checks for a cancel, waits for a free
  worker (`EntryPool.wait_for_worker`), runs the gate (`admission_margin` above 0) and
  then the floor when there are several workers, enters the entry in flight and
  dispatches it, so at most `workers` entries are in flight. For an entry the parent
  loads, the gate and the floor run in `before_load`, before the table is read, and the
  wait for a free worker comes after the load. Both hold back through
  `EntryPool.drain_one` and `wait_for_memory`, and `run_feature` builds its pool with
  `floor_at_submit=False` and its own `MemoryWatch`. Until an entry with input rows is
  written the gate runs each entry alone. Its budget (`admission_budget`) is the
  limit's available memory plus the run's tree in the run's measure, capped at the
  limit, less the floor. `EntryOverMemoryLimit` compares the unmargined prediction with
  the limit less the floor, never with what is available now
  (`tests/core/pipeline/test_parallel_run.py`).
- Entries go largest first (`by_rows`) only with the gate, and only the apply order
  moves. The fit reads `manifest` in its own order.
- `run_feature(phases=...)` applies the entries in `ApplyPhase`s of one attempt, each
  with its own workers, mode, margin and worker-base measure, through one `_apply_entries`
  call and one `_PhaseState` each. The run assigns `_RunCost.phase`, the one attribute the
  sampler reads, after the earlier phase's pool has closed and written its final
  `resources` event and before the next phase's first entry enters. A phase closes its own
  pool on every exit. The cache hit is one claim of the attempt, made after the last phase.
  Consecutive phases that run alike (`_ApplySettings.runs_like`) are one phase, because
  two phases that read as one mode would leave that mode's levels and peak to the later
  phase's last event (`tests/core/pipeline/test_apply_phases.py`).
- A run keeps each written entry's flight, from its dispatch, or from the parent's load
  of an entry the parent loads, to its output written, and `in_flight_levels` of the
  `resources` event is the time, rows and tree CPU at each count of entries in flight
  together. An entry's rows spread evenly over its flight, an entry whose output was not
  written adds neither rows nor time, and level 0 is not written, so `RunCost` rebuilds
  the same levels from the `entry_end` events of a log that wrote none. The peak sample
  sums the covariates of the entries in flight into `peak_in_flight_pair_frames`,
  flagged by `peak_in_flight_pair_frames_bound` when any carries the lower bound of a
  blank cell. An entry's covariates are read once, from the tracks and media indexes
  (`tracks_covariates`), before the first entry loads, so the read falls in no flight
  (`tests/core/pipeline/test_run_measures.py`). The flight of the entry whose
  process worker built the feature for it contains that build, and its
  `entry_end` records the build's seconds as `build_seconds` (`EntryMeasure`,
  `EntryCost`), left out where no build ran.
- `MemoryWatch` keeps the processes below this one and walks the tree
  (`children(recursive=True)`) to find them: in `begin()`, when a read finds the last walk
  `ESTIMATE_SAMPLE_INTERVAL_SECONDS` old, and after an `EntryPool` process submission when
  the calls not yet done (counted before the submit, plus one) outnumber the workers the
  watch has found since the pool was made. A failed walk keeps the members. `sample()` and
  `cpu_seconds()` read the known members between walks, so a process that joined between
  walks, a feature's own subprocess, is missing for at most an interval. A member's CPU
  counts from the walk that first finds it, and a reaped direct child's base is
  subtracted, because the reaped children's time already holds its whole CPU.
  `cpu_seconds()` reads this process's own time under the membership lock after the
  leaves, so a reaped direct member's time is in that read. A grandchild that its parent
  reaps leaves the figure at its reap, and its whole CPU returns when that parent is
  reaped. A run's walks are bounded by its wall time over the interval, plus one per
  process worker, whatever its entry count
  (`tests/core/pipeline/test_per_entry_cost_growth.py`).

## Reading a run

- `load_run_matrix` (`run_matrix.py`) reads one run's parquet files into an array and
  writes nothing. It takes the files from the feature index and never builds a path
  from an entry name, resolves an unpinned `Result` through `current_run_id`, and
  orders entries by `(group, sequence)`.
- It refuses an unfinished run, an output the index lists and disk lacks, and a scope
  naming an entry the run lacks, unless `allow_partial` is set. `load_values` skips a
  missing output with a note.
- The array is sized from the footers (and from the `perspective` column, so the count
  is of the rows returned), checked against available memory, and filled in place a
  batch at a time. A step that concatenates per-entry arrays doubles the peak.
- A batch size does not bound what pyarrow holds: it decodes a whole row group of the
  columns it is asked for, about twice their float64 size, and a feature output is one
  row group up to a million rows. A file that would take more than `BLOCK_BYTES` is read
  a block of columns at a time, and the memory check counts the largest block
  (`tests/core/pipeline/test_run_matrix.py`).

## Verification

- `verify_plan` (`verify.py`, behind `mosaic pipeline verify`) fits each complete fitting
  step again into a scratch directory and compares it with the stored run. It never
  offers a stored bundle to `load_state`, which would answer that the state is cached
  and skip the fit, and it writes nothing under the dataset
  (`tests/core/pipeline/test_verify_plan.py`, `tests/cli/test_cli_read_only.py`). It
  plans under a `load=False` definition scope and raises `load` around the check of a
  `def:` step alone, which loads the version its plan pins
  (`tests/core/pipeline/definitions/test_maintenance.py`).
- A step due a refit is skipped (`PlannedStep.refit`) and counts as unfinished: a fit of
  today's inputs differs from a fit of older inputs without failing to reproduce. A
  fit that verification does run binds its scratch directory as the run root
  (`bind_fit_root`).
- A report that compared nothing is not `ok`. `ok` needs one verified step and no step
  without a complete run or due a refit. Values of different kinds are a mismatch, and a
  value the walker cannot read or traverse is named in the detail and not counted as
  compared (`_Found.compared`, `unchecked`, `mismatch`).
- The fit's frames come from `fit_input_stream` (`run.py`), the one stream `run_feature`
  fits on. A change to what a fit reads changes both.
- A feature is held to equality unless it has an `n_threads` param or a `device` other
  than `cpu`. A new network feature declares one of them, or its verification fails on
  a machine with another thread count.

## Media variants

- `write_media_variant_row` is the one writer of `media/preprocess/index.csv` and
  `read_media_variant_index` the one reader (`preprocess_index.py`).
- `VariantLookup.resolve` (`variant_source.py`) refuses an entry with no variant row,
  a missing file or drifted media
  (`tests/core/media/preprocess/test_variant_source.py`).
- `to_source_space` (`placement.py`) refuses an unclassified numeric column under any
  mapping but the identity (`tests/core/media/preprocess/test_placement_map_back.py`).
  `MediaInputParams` refuses a frame window beside `media`.

## Label series

- The rule a series follows is declared in the registry (`label_series.py`), and a
  path does not decide it.
- The series are `keypoints`, `tags` and the reserved `behavior`. `tags` has one key,
  `dataset`, and a revision is the whole tag table.
- An uploaded label file enters its sequence's composition (`sequences.csv`), and a
  series revision enters none.
- `write_series_revision` (`label_series_index.py`) chooses every filename and number,
  and the caller supplies bytes and an `origin` mapping. A payload whose digest
  matches the latest revision writes nothing. The payload must be deterministic
  (sorted, compact, without a timestamp).
- Numbering happens inside the series index lock, takes the higher of the index and
  the directories on disk, and creates the revision directory exclusively.
- While the dataset indexes no row of the key, the latest revision is the one that a
  reference without an origin resolves through the dataset's links
  (`label_series_rows`), so a copy's first save numbers past the source's latest.
- Only the payload is digested and compared. `revision_export` checks an export's
  digest where it is read. A revision whose payload changed raises
  `LabelSeriesTamperedError` at scan, and `resolve_series_refs` raises it when a
  plan or a run names the revision.
- A series is recognized by its `.mosaic-series` marker (`is_under_label_series`).
  `ensure_series_root` does not count mosaic's bookkeeping files as a squatter.
- A prune verb, when one exists, keeps every consumed revision
  (`tests/core/pipeline/test_label_series.py`).

## Inventory

- The inventory only reads. Truth is the indexes and files on disk, every view is a
  cache, and `InventoryCache.revalidate()` stats the index files to find a stale one.
  A moved feature index rereads the `feature` kind alone, and any other moved index
  every kind. It also stamps the indexes of each linked dataset for the kinds the link
  provides, so a linked index that moves, and a link that becomes reachable or goes
  away, moves the view the same way. `InventoryCache.generation` advances with each
  reread, and `ParquetFooterMemo` (`parquet_reader.py`) keeps the parquet files that
  parsed and the footers a run reads, by path, inode, size and modification time, so a
  scan reads only the outputs that moved. A scan reads each index once, so a per-entry
  lookup is one set or dict built per index, never a pass over its rows per entry
  (`tests/core/pipeline/inventory/test_inventory_scan.py`,
  `tests/core/pipeline/graph/test_pipeline_plan_scale.py`).
- `classify` (`inventory/model.py`) is the one place a status is decided. Files ahead
  of index rows are damage only on a finished run
  (`tests/core/pipeline/inventory/test_inventory_model.py`).
- `ArtifactKind` and `ArtifactStatus` are defined in `artifact_kinds.py`, which
  imports the standard library only, and the inventory re-exports them. A caller
  that declares a field over them loads no pandas.
- Coverage is the set of keys that exist, and the key type differs by kind (the
  `DatasetInventory.coverage` overloads). A kind nobody registered is listed in
  `unavailable_kinds`.
- `read_run_params` is the only reader of a feature run's `params.json`, and tells
  absent from unreadable (`tests/core/pipeline/inventory/test_run_params_reader.py`).
- `ArtifactRecord.refit` is the `FitDrift` of a feature run whose fit is due a refit,
  measured with the run's drift through one `DriftReads` per inventory read, and a
  linked run's record names the link. `RefitWire` (`inventory/wire.py`) projects it for
  the wire as counts, a sample of up to five entries, the fields fitted again, the link
  and `asked`, and its `summary()` gives the line that each command prints
  (`tests/core/pipeline/inventory/test_inventory_scan.py`).
- A feature run's target comes from `feature_run_target` (`inventory/scan.py`): a
  run is measured over the entries its invocations asked for (`_scope.asked`) and
  its index rows. A scope-free run asked for every entry (`_scope.asked_all`) is
  measured over every entry. The scan and the `inventory_wire` projection share it
  (`test_a_scope_dependent_projection_matches_the_scan`,
  `test_a_scope_free_projection_matches_the_scan`).
- `PREPARED_DATA_KINDS` (`models.py`) is the one list of the `models/` kinds that
  contain training data. The trained-model inventory and model deletion both judge a
  kind through `holds_trained_models`.
