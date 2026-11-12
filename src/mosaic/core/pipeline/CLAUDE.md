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
- **`track_universe.current_run_id` is the one rule for an unpinned `Result`**
  (`tests/core/pipeline/test_unpinned_result_rule.py`).
- **A lock the filesystem refuses raises `IndexLockUnsupported`**
  (`core/pipeline/index_lock.py`) and is not degraded to an unlocked write. The lock
  serializes writers on one mount only
  (`tests/core/pipeline/test_index_lock_capability.py`).
- **Entries computed at once go through `EntryPool`** (`entry_pool.py`): `run_feature`
  and each op that sets `parallelizable` (`convert-tracks`, `resample-tracks`), which
  reads its workers from `ctx.parallel`. A worker stages a tracks table under a hidden
  `.stage-*.staged` name beside its final one (`staged_tables.py`) and returns it with
  its index row (`TracksRowDraft`). The calling process, through `OrderedResults`, in
  submission order, publishes the tables in batches (`TableCommitter`: the tables of a
  batch move into place, then one write of the index records their rows, so a run is
  not quadratic in the index; a batch is due after `max_batch` tables, or after the
  longer of 10 s and four times the duration of the last write; a batch with a key
  collision is refused before any table moves), reports the failures and the progress, so the
  outputs do not depend on the worker count. The caller flushes the committer at the
  end of the run and when it stops early. Of two
  files naming one entry the first keeps it, or the last when overwriting. The op
  removes the staged files of an earlier run at its start, and a table with no row is
  made again, not skipped. A process worker reopens the dataset from its manifest path
  (`dataset_ref`, `worker_dataset`). An op that ends on an exception settles the pool
  first and publishes what its workers finished
  (`tests/core/pipeline/test_op_parallelism.py`).

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
  `_tracks`, `_labels` and `_overlap_frames` only when present. A new term is omitted
  when absent, which leaves existing identifiers unchanged
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
- The writer measures `n_keypoints` and `frame_min`/`frame_max` from the parquet. The
  producer passes `media_frames`, `frames_read` and `known_tail_loss`. A blank cell on
  every count column means unknown. The `TracksIndexRow` docstring defines each
  column.
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
  compositions, each upstream row's `consumed_composition` and
  `consumed_input_composition` as `own/inherited`, joined in input order by
  `join_input_cells` (`composition.py`). An entry recomputes after its upstream
  entry recomputes for a moved declared root, at any depth of a chain. Both sides
  blank is absence, served without a warning. The blind tracks warning covers a
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
- A division of a store is a member of its entry's media composition under its
  `imgstore-division/1` uid, so each division has its own composition, and a division
  whose chunks change frame count has a new uid (`tests/core/media/test_media_drift.py`).
- `drifted_entries` (`index.py`) follows `cached_entry_disposition` over three
  channels, given the run's `RunParams`: the declared roots, the tracks cell and the
  upstreams' roots cell, each built as `build_manifest` builds it. One known side is
  drift, and the plan names every entry `run_feature` recomputes. The neighbors cell
  stays `run_feature`'s (`TestDriftedEntries` in
  `tests/core/pipeline/test_tracks_source_staleness.py`). An entry the inputs no
  longer resolve, such as one dropped from the tracks index, is not drift: every input
  holds it for a run that intersects its inputs, and any input for a stacked run, as
  `_scope.stacks_inputs` in `params.json` records (`stacks_inputs`, `loading.py`). A
  file without the key counts any input, so a false drift lasts one run, where every
  input would hide a stacked run's drift. Without params it compares the declared
  roots alone.
- `arenas` is a composition root outside `SourceRoot`: an entry's layout digest
  (`arenas_compositions`, `arena_index.py`), read only where a feature declares it in
  `consumed_roots`. `build_manifest(consumed_roots=...)`, `resolve_feature_identity`
  (for a scope-dependent feature), `drifted_entries` (`DriftReads.arenas`) and
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

- `alignment_verdict` (`loading.py`) decides whether inputs merge, and the merge
  raises `MultiInputAlignmentError` from it. Individual-level against pair-level
  output with no shared identity column is refused. `CROSS_JOIN_FEATURES` is the
  closed escape (`tests/core/pipeline/test_multi_input_alignment.py`).
- `expands_individuals` (`loading.py`) is the one gate for widening an
  individual-level input onto the pair rows of a pair-level input 0, as
  `<col>__id1` and `<col>__id2`: `emits` of `as-input` or `pair`, outside
  `CROSS_JOIN_FEATURES`. `run_feature`, `verify.py`, `_load_neighbor` and the
  declarations (`ConsumerDecl.expands_individuals`) all read it, and `merge_naming`
  names the columns for the merge and for validation. An entry whose pair input holds
  no rows raises `PairRowsMissing`, which the apply loop records as an `entry_error`
  (`tests/core/pipeline/test_widening_join.py`). `load_values` does not widen.
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
  (`tests/core/pipeline/test_verify_plan.py`, `tests/cli/test_cli_read_only.py`).
- A report that compared nothing is not `ok`. `ok` needs one verified step and no step
  without a complete run. Values of different kinds are a mismatch, and a value the
  walker cannot read or reach is named in the detail and never counted as compared
  (`_Found.compared`, `unchecked`, `mismatch`).
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
- An uploaded label file enters its sequence's composition (`sequences.csv`), and a
  series revision enters none.
- `write_series_revision` (`label_series_index.py`) chooses every filename and number,
  and the caller supplies bytes and an `origin` mapping. A payload whose digest
  matches the latest revision writes nothing. The payload must be deterministic
  (sorted, compact, without a timestamp).
- Numbering happens inside the series index lock, takes the higher of the index and
  the directories on disk, and creates the revision directory exclusively.
- Only the payload is digested and compared. `revision_export` checks an export's
  digest where it is read. A revision whose payload changed raises
  `LabelSeriesTamperedError` at scan.
- A series is recognized by its `.mosaic-series` marker (`is_under_label_series`).
  `ensure_series_root` does not count mosaic's bookkeeping files as a squatter.
- A prune verb, when one exists, keeps every consumed revision
  (`tests/core/pipeline/test_label_series.py`).

## Inventory

- The inventory only reads. Truth is the indexes and files on disk, every view is a
  cache, and `InventoryCache.revalidate()` stats the index files to find a stale one.
  A moved feature index rereads the `feature` kind alone, and any other moved index
  every kind. `InventoryCache.generation` advances with each reread, and
  `ParquetFooterMemo` keeps the parquet footers that parsed, by path, size and
  modification time, so a scan reads only the outputs that moved. A scan reads each
  index once, so a per-entry lookup is one set or dict built per index, never a pass
  over its rows per entry (`tests/core/pipeline/inventory/test_inventory_scan.py`,
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
- A feature run's target comes from `feature_run_target` (`inventory/scan.py`): a
  run is measured over the entries its invocations asked for (`_scope.asked`) and
  its index rows. A scope-free run asked for every entry (`_scope.asked_all`) is
  measured over every entry. The scan and the `inventory_wire` projection share it
  (`test_a_scope_dependent_projection_matches_the_scan`,
  `test_a_scope_free_projection_matches_the_scan`).
- `PREPARED_DATA_KINDS` (`models.py`) is the one list of the `models/` kinds that
  contain training data. The trained-model inventory and model deletion both judge a
  kind through `holds_trained_models`.
