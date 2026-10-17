# mosaic.core.pipeline

This package owns feature execution, data loading, output writing, dependency
resolution and the typed indexes of a dataset. Features own computation only.

Rules that span packages are in the root `CLAUDE.md`: layering and the upward imports,
the one parquet writer, `run_feature` as the only path to feature outputs, the run-log
contract, lock refusal, source-space publication and codec refusal. Rules for the
pipeline graph are in `graph/CLAUDE.md`.

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
  (`tests/core/pipeline/test_feature_run_id_golden.py`).
- `resolve_labels_variants` takes a pinned `labels_run_id` as given, as a pinned
  `tracks_run_id` is, so a run planned before its labels are converted is named as it
  executes (`tests/core/pipeline/test_labels_identity.py`).
- The tracks variants a run read enter `_tracks`, and the storage directory name
  (`derive_storage_name`) leaves them out. `features/<name>__from__tracks/` stays one
  directory and one index whatever the number of variants in a dataset.
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

- `write_tracks_row` is the only appender and `read_tracks_index` the only reader
  (`tracks_index.py`). Rewrites (`delete_set`, `reconcile_variants`,
  `Dataset.drop_entries`, the backfill passes) take `index_lock` on the same file.
- One row per `(run_id, group, sequence)`, with no camera column. An absent index reads
  as an empty frame with the full schema. An older schema is widened in memory inside
  the write lock (`adopt_legacy_columns`), and a read leaves the file as it is
  (`tests/core/pipeline/test_tracks_index.py`).
- `select_variant_rows` is the only resolver. An unlabeled row (`run_id` empty) loses
  to a labeled one, and two recipes for one entry raise. The caller answers with
  `tracks_run_id=` (`run_id=` on `load_tracks` and `drop_entries`), and `""` names the
  unlabeled tables.
- The writer measures `n_keypoints` and `frame_min`/`frame_max` from the parquet. The
  producer passes `media_frames`, `frames_read` and `known_tail_loss`. A blank cell on
  every count column means unknown. The `TracksIndexRow` docstring defines each
  column.
- `frame_axis_verdict` is the one rule separating a frame-axis mismatch from a known
  tail shortfall. The `backfill_*` passes behind `mosaic measure-tracks` document their
  rules in their docstrings. The lock discipline of the three cell passes is in
  `_backfill_cell`, and `backfill_frame_extents` takes the lock once for its whole pass
  (`tests/core/pipeline/test_backfill_passes.py`).

## Compositions

- A run identity is params-only. What a run consumed is recorded on its row and
  compared, and `run_id` leaves it out.
- `run_feature` compares each cached entry's `consumed_tracks_composition`
  (`recorded_tracks_composition`, `index.py`) with the tracks row's current
  composition through `cached_entry_disposition` (`run.py`). A difference, or one
  known side beside a blank one, recomputes the entry. A re-conversion from changed
  sources therefore recomputes (`tests/core/pipeline/test_tracks_source_staleness.py`).
- `drifted_entries` (`index.py`) compares the declared-root `consumed_composition`,
  and `consumed_media_composition` on tracks rows, media variant rows and tracker run
  rows is compared the same way, through `compositions_disagree` (`composition.py`).
  There a blank side is unknown and not drift
  (`tests/core/pipeline/test_tracks_media_drift.py`).

## Inputs and claims

- `alignment_verdict` (`loading.py`) decides whether inputs merge, and the merge
  raises `MultiInputAlignmentError` from it. Individual-level against pair-level
  output with no shared identity column is refused. `CROSS_JOIN_FEATURES` is the
  closed escape (`tests/core/pipeline/test_multi_input_alignment.py`).
- Trackers, `infer-*` and `preprocess` claim each entry through `open_entry`
  (`entry_claim.py`) with an exclusive create, and release and refresh check
  ownership (`tests/core/pipeline/test_run_markers.py`). One-shot ops claim their
  run root through `claim_run_root` in `tracking/ops/_common.py`.
- Outside `Pipeline.clean`, a whole feature run is removed only through
  `remove_feature_run` (`feature_run_removal.py`): an intent record with the run's
  `params.json` in the cleanup's run-log, then the index rows, then the directory,
  then a done record (`tests/core/pipeline/test_feature_run_removal.py`).
- `run_feature` runs unclaimed. Two concurrent executions of one fitting identity
  both fit and write into one run root. Do not assume a fit is exclusive.

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
  scope-dependent run is measured over its declared scope. The scan and the
  `inventory_wire` projection share it (`test_a_scope_dependent_projection_matches_the_scan`).
- `PREPARED_DATA_KINDS` (`models.py`) is the one list of the `models/` kinds that
  contain training data. The trained-model inventory and model deletion both judge a
  kind through `holds_trained_models`.
