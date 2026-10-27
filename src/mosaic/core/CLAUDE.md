# mosaic.core

This directory owns the dataset (its roots, manifest and scans), the track converters
and schemas, the parameter vocabulary, and media I/O. The guide in `core/pipeline/`
covers that package. Layering, entry-name validation and the `Params` hashing rule are
in the root guide.

## Imports

- `core/__init__.py` binds every public name on access, and its module-scope imports
  are standard library only. Importing a leaf loads neither pandas nor h5py
  (`tests/meta/test_core_lazy_exports.py`).
- `entry`, `params`, `strict_model` and `individuals` are leaves
  (`tests/meta/test_core_params_is_a_leaf.py`). `manifest.py` and `track_converter.py`
  import nothing from `dataset.py`.
- `ensure_track_converters_registered` imports `track_library` on every call, and
  `ensure_label_converters_registered` returns when `LABEL_CONVERTERS` is non-empty. Do
  not guard the first on emptiness. A caller who registers a converter first would lose
  every built-in (`test_registering_a_converter_first_keeps_the_built_ins` in
  `tests/core/track_library/test_track_converters.py`).
- `core/media/` imports nothing from `behavior` or `tracking`. It does import
  `core.pipeline` modules (`media_index`, `joined_export`, `index_lock`, `provenance`).
- `core/media/__init__.py` and `core/media/preprocess/__init__.py` bind their names
  on access. A module that `Dataset` or the inventory imports defers its import of
  `video_io`, `imgstore_io` or a step module into the function that reads frames
  (`tests/meta/test_media_lazy_exports.py`).
  Frame sampling for annotation is in `tracking/frame_extraction/`.

## Dataset roots

- A root resolves inside the dataset (`validate_root_inside`). A scan source may point
  anywhere and is not created or walked at load time.
- `backfill_roots` (`manifest.py`) fills an absent root and leaves a present one where
  it is. It does not fill `media_raw`, which may be absent. Media accessors therefore
  resolve through `Dataset.resolve_media_root()`.
- Code reads each tracker working root from `TRACKING_ROOTS`
  (`tests/core/pipeline/test_tracking_roots.py`).
- A walk of the dataset for user content skips three things by path, because
  `exclude_patterns` match basenames only: `_tracking/` by path component
  (`is_under_tracking_root`), media variants by resolved path (`media_variant_filter`)
  and label series by their marker (`is_under_label_series`). `_probe_dir_rows` checks
  the first two and `iter_track_files` the first and third. A new walk checks each that
  applies (`tests/core/pipeline/test_tracking_roots.py`,
  `tests/core/media/preprocess/test_preprocess_index.py`).

## Manifest

- `manifest.py` defines the format as pydantic models. A read migrates an older version
  in memory and writes nothing, and a newer version raises `ManifestVersionError`. The
  `MANIFEST_VERSION` docstring says when to bump it. A manifest is written at 4 only
  when a media source declares a `division` or a write raised it for division rows
  (`written_manifest_version`), and every other at `UNDIVIDED_MANIFEST_VERSION`.
  `DatasetManifest` keeps unknown top-level keys in `preserved`, while sources and tags
  refuse unknown keys (`tests/core/test_dataset_manifest.py`).
- A `Dataset` setter checks the manifest it would make before holding it
  (`Dataset._adopt_manifest`). A refused value does not reach `save`
  (`TestARefusedSetterLeavesTheManifest`).
- `write_manifest` is atomic. A writer that may race another goes through
  `Dataset.mutate_manifest`, which re-reads and writes under `index_lock`.
  `declare_scan_source`, `undeclare_scan_source` and `update_tags` are the locked
  forms of the source and tag edits, which the MCP's `manage_dataset` calls. The CLI's
  `sources add` and that tool build a source through `scan_source_from_fields`
  (`source_fields.py`, `tests/core/test_source_fields.py`).
- Dataset tags use the grammar in `typed_attribute.py`, which mosaic-api shares. It
  imports only the standard library and `json_value`. Keep dataset concepts out of it.
- `continuous_groups` refuses an empty or repeated name
  (`test_the_empty_group_cannot_be_continuous`). `Dataset.set_continuous_groups` also
  refuses `__` or `:` in a name, the name of an empty-group sequence, and a group with
  no indexed entry, and a manifest declaring such a name still loads. Each sequence of a
  continuous group resolves to its own media rows. A division's
  `ResolvedMedia.first_frame` is its row's `first_frame` cell. Any other sequence's sums
  the media frames of the sequences before it in name order (`_MemberPlacement` in
  `dataset.py`). A group mixing division rows with other rows places none, and neither
  does an undeclared group holding a division row. Readers map a tracks frame through
  `ResolvedMedia.sequence_frame`.
  A sequence that cannot be placed still resolves for a join or a transcode, and its
  `first_frame` raises (`tests/core/media/test_continuous_group_media.py`).
  `Dataset.sequence_first_frames` reads it for many entries, one per camera, and is
  the one reader for media drift and label conversion. Label conversion refuses dense
  labels for a sequence that starts past frame 0 or cannot be placed, and reads a
  sequence that no member precedes as frame 0 without its media
  (`tests/core/pipeline/test_labels_index.py`).
  `overlap_frames` checks the declaration against recorded frame extents
  (`core/pipeline/manifest.py`).
- `splits` maps a named group to `train`, `test` or `sealed`. It is a top-level key
  and moved no `MANIFEST_VERSION`, so older code preserves it without enforcing it.
  Read it through `core/splits.py` (`role_of`, `groups_with_role`, `held_out_groups`)
  and write it only through `Dataset.set_splits` and `remove_splits`
  (`tests/core/test_splits.py`).

## Scans

- A source kind is bare (`SourceKind`) and the root it feeds ends in `_raw`, joined by
  `_RAW_ROOT_FOR_KIND`. Scanners are named for the kind (`scan_tracks`). A reader name
  ends in `_raw` only where two indexes would share a name (`read_tracks_raw_index`
  beside `read_tracks_index`).
- A scan replaces the rows its sources claim (`ScanClaim` in
  `core/pipeline/scan_claim.py`) and keeps every other row unless
  `prune_unsourced=True`. `overlapping_sources` refuses, at declaration, two sources of
  one kind that could claim one file (`tests/core/test_scan_sources.py`).
- A media scan re-measures every row but keeps the identity cells a caller assigned
  through `MediaIndexScope` (`Dataset._keep_assigned_identity`) unless `reassign=True`,
  and keeps a store's assigned division rows (`_keep_assigned_divisions`).
- A dividing media source and a division `MediaIndexScope` both divide through
  `Dataset._divided_rows`, which plans each recording with
  `media/division_index.py` and raises every refusal before a row is written
  (`tests/core/media/test_divided_store_scan.py`). Its rows replace every row of the
  store, and the recording is then declared continuous. The manifest is raised to 4
  before the index write and the groups are declared after it. A re-division is
  refused while `Dataset.entry_outputs` lists an output of the recording's entries,
  which needs `mosaic.tracking.register_ops()` for the tracking kinds. The undivided
  row of a store is checked on that store's entries alone, also when other stores of
  its camera are divided.
- A labels source with `series` goes to `Dataset.index_label_series` instead of
  `_index_raw`. Series rows are not pruned as unsourced.
- A media write raises `StoreGroupingError` (`core/pipeline/media_index.py`) for
  imgstores it cannot group: a synchronized set of two or more stores with divergent
  base names or a store without `camera_serial`, two recordings that derive one entry,
  and unsynchronized stores of distinct serials under one `MediaIndexScope`. A scan
  checks the serial and entry rules after it restores assigned identity, against the
  kept rows of each entry it writes as well. `camera_serial` is a fact column outside
  every composition (`tests/core/media/test_index_media_imgstore.py`).

## The `group` column

- `group` is required and may be empty. It is a namespace in the entry key, or the name
  of a recording declared in `continuous_groups`. Analysis grouping lives in
  mosaic-api's tags, and an arbitrary subset is `Scope(entries=[...])` (`scope.py`),
  since `groups` and `sequences` combine as a cross product.
- `core/recording.py` is the one rule naming an entry's recording: its continuous group,
  else the entry. Code does not group by the raw `group` column to mean a recording,
  and only `manifest.py`, `recording.py`, the `Dataset` accessors and the echo of the
  declared list in `dataset_summary` read the declaration
  (`tests/core/test_recording.py`).
- A write refuses a new entry whose key equals another entry's, such as `("", "a__b")`
  beside `("a", "b")`, and an empty-group sequence named like a continuous group,
  through `Dataset.refuse_entry_key_collision` (`core/dataset.py`), which combines the
  key rule of `core/entry.py` with the name rule of `core/recording.py`. Rows already in
  an index keep resolving (`tests/core/test_entry_tokens.py`). A replacing media or
  tracks_raw write checks against the entries that survive it and those another index
  still names (`entries_kept_beside`), and a scan checks the entries left after assigned
  identity is restored (`tests/core/media/test_media_index_entry_names.py`).
- Only the sequences of a continuous group have neighbors for `overlap_frames`
  (`build_manifest` in `core/pipeline/manifest.py`), and only their frame extents are
  read, under the tracks variant that the inputs read. A sequence of the empty group or
  of an undeclared group reads only its own frames, and the run notes their count on
  stderr instead of refusing (`tests/core/pipeline/test_run_feature.py`).

## Track converters and schemas

- `TrackConverter.output_schema` is the one place a converter names its schema
  (`TrackingRoot.output_schema` for a tracker). Registration does not check its
  `trex_v1` default. Declare a schema on every new converter, and bump `version` to
  change it.
- A converter writes pixels with `X`/`Y` at the body centre. A centroid-only converter
  writes no `poseX*`/`poseY*` columns and does not copy `X`/`Y` into them. A source
  whose pixel factor is lost declares `mosaic_cm_v1`. TREx specifics are in the
  `track_library/trex.py` module docstring
  (`tests/core/track_library/test_trex_converter_units.py`).
- A converter does not compute a `DERIVED_COLUMNS` quantity. `mosaic_v1` forbids them,
  and `ensure_track_schema` raises `ForbiddenTrackColumnError` whatever `strict` says. A
  tracker that measures one declares a schema with `allows`, as `trex_v2` does.
- `ensure_track_schema` raises `UnknownTrackSchemaError` for an unregistered name and
  accepts any unlisted column (`tests/core/track_library/test_track_schema.py`).
- A new schema joins a family through `extends`. `mosaic_cm_v1` shares
  `STANDARD_COLUMNS` instead, to stay out of the pixel family. `LEGACY_SCHEMA` is what a
  blank recorded schema means. That schema, `trex_v1` (centimeters, `X` at the head),
  stays registered so archived datasets keep resolving. Do not use it for new data.
- **A schema family declares its `space`**, `image_px` or `plane_cm`, and an extension
  inherits it and may not change it. `space` is optional, so a third-party root schema
  registers without one. The overlay and the crops refuse a `plane_cm` table, and draw a
  table of unknown space (no `space`, or a schema this process never registered) as pixels
  with a warning (`require_pixel_positions`,
  `tests/behavior/visualization_library/test_crop_centroid_fallback.py`). They judge the
  table they were handed, from `ResolvedScope.tracks_schemas` or the variant the caller
  pinned. A blank schema is `LEGACY_SCHEMA`, so it is refused.
- A converter receives its entry through `EntryHints`, which is not a `Params` and is
  not hashed (`test_no_converter_can_hash_an_entry_name`). Its `fps` is the measured
  frame rate of the media the tool read, `None` when unmeasured.

## Parameter models

- `Params.identity_dump` drops `HASH_EXCLUDE` fields at the top level only, and a knob
  inside a nested config model is hashed. A field declares its prose with
  `Declared(...)` in its `Annotated`. A `Field(description=...)` overrides it, and
  `tests/meta/test_params_declaration.py` refuses one.

## Media I/O

- Read through `open_frame_reader` or `MultiVideoReader` (`media/video_io.py`). Every
  read names a `ReadTarget`, and `verified_read_facts` (`media/read_target.py`) checks
  injected facts against it. The raw `.h264` rule is in the `video_io` module docstring
  (`tests/core/media/test_media_probe_facade.py`).
- imgstore reads do not import the `imgstore` package, which only test fixtures use to
  write stores (`tests/core/media/test_imgstore_io.py`). A track table's `frame` is
  imgstore's contiguous `frame_index`.
- A store's facts are measured by `measure_store` (`media/imgstore_io.py`): the codec
  in the first chunk's header, or an image store's `format`, and the timing of the
  chunk indexes' frame times. Its row's size and mtime come from `store_stat`, never
  from the directory inode. `verified_read_facts` passes store facts for every target,
  and `NativeStore` gates each video chunk at `"raw"` as it opens it
  (`tests/core/media/test_store_read_gate.py`).
- `media/store_division.py` imports only standard-library leaves. Chunk ids alone
  decide a division's frames, and a store's clock only names it. A frozen division is
  reused or refused, and is not recomputed (`tests/core/media/test_store_division.py`).
- A divided store has one media row per division, sharing the store's `abs_path`.
  The seven trailing division cells (`DIVISION_COLUMNS`) are blank on an undivided
  row. A row is keyed by `media_row_key` (path and chunks): the scan dedupe,
  `_dedupe_scope_rows`, `_keep_assigned_identity`, `_keep_calibration`, the write
  path's prior rows and facts cache, and `MediaRows` key on it. A lookup of a row
  takes `ResolvedMedia.row_keys`. Sites that mean the file keep `abs_path`:
  `_row_claimed`, `reprobe-media` (one `probe_store` per store), prune, path
  rewriting and the summary's video counts (`tests/core/media/test_division_rows.py`).
- A division's `ResolvedMedia.time_offset` and `Dataset.division_facts` come from
  `_recording_divided`, which reads a recording's rows once. A converted table and a
  tool's table of a division are moved by its first frame and that offset. The offset
  continues an adjacent division before it at that row's rate and is taken from the
  store clock after a gap, a store boundary or a missing division
  (`tests/core/media/test_divided_store_end_to_end.py`).
- A reader of a division opens its path on `ResolvedMedia.windows`, and local
  frame `k` is store frame `chunk_first * chunksize + k`. A reader of an entry's
  clips takes `windows` as a required keyword. A window whose chunk is absent or
  unindexed is refused, and other windows read on
  (`tests/core/media/test_store_window_reads.py`).
