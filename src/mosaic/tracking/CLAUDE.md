# mosaic.tracking

This directory owns the integrated trackers, model inference and training ops, frame
extraction, and mosaic's side of the external Ultralytics and POLO environments.

## Cross-package rules

- **A media variant is consumed by name and published in source space.**
  `publish_tracks_table` (`tracking/common/bridge.py`) maps a table through
  `to_source_space` (`core/pipeline/placement.py`,
  `tests/core/media/preprocess/test_placement_map_back.py`).
- **A tool is handed only a file it decodes.** `refuse_undecodable_codec`
  (`tracking/common/tool_input.py`) applies the tool's `ToolDecoder`
  (`core/pipeline/tracking_roots.py`, `tests/tracking/common/test_decoder_probe.py`).
- **Name a model by its run id.** Across datasets, link the dataset that holds it with
  `links:` instead of a path into another dataset's `models/`. The reference string
  enters the inference identity (`resolve_model` in `tracking/model_refs.py`,
  `tests/core/test_dataset_links.py`). A lookup walks `Dataset.lookup_order("models")`
  and a training op asks `linked_training_complete` (`ops/trained_models.py`) whether a
  link finished the run. Neither opens a linked dataset's index by path
  (`tests/meta/test_link_reads.py`).
- **The owner of an annotated image is found by its resolved path.** `video_frames_of`
  (`frame_extraction/frame_runs.py`) gives an image path that resolves outside the image
  root's directory tree, absolute or through `..`, to the dataset whose manifest is
  nearest around it (`Dataset.open_enclosing`). A relative path that stays inside the
  tree belongs to the image root's dataset
  (`tests/tracking/frame_extraction/test_video_frames_of.py`).
- **A revision's content is hashed, and its number is not**
  (`tests/tracking/ops/test_prepare_training_data.py`).


## Imports and registration

- `__init__.py` of `tracking`, `common`, `frame_extraction`, `frame_predictions` and
  `ops` imports no submodule (`tests/meta/test_tracking_lazy_exports.py`). Importing a
  tracking module registers only what that module registers.
- `register_ops` (`registration.py`) imports every module of `REGISTERING_MODULES` and
  registers the inventory contributors. A module that registers something at module
  scope is listed there (`tests/tracking/test_registration.py`). Call `register_ops`
  before a lookup in `OPS`, the reconcilable indexes or the tracks-axis readers.
- `register_inventory_contributors` (`inventory.py`) is the one place a tracking
  inventory kind is registered. Each record builder lives beside the index that it
  reads, in `common/index.py`, `frame_extraction/frame_runs.py`,
  `frame_extraction/selection_index.py`, `frame_predictions/index.py`,
  `ops/prepared_datasets.py` and `ops/trained_models.py`. These modules leave the
  tracker subpackages, the trainers and the extraction code unimported. Of frame
  extraction they import only `frame_runs.py`. The tracker kinds are the `tracker`
  roots of `TRACKING_ROOTS`, read through `TrackerRunRowBase`.

## Wiring a tracker

- `tracking/common/` owns the run loop: `run_tracker` in `common/driver.py`, with scope,
  minting, entry claims, the bridge and the run index beside it. A tracker's own package
  supplies its argv, the settings that define its identity, its phases and its
  converter, plus one `TrackingRoot` row in `core/pipeline/tracking_roots.py`.
- `tests/tracking/common/test_tracker_conformance.py` is parametrized over every root
  and fails by name on a half-wired tracker: registered op, shared params contract,
  `media` parameter, reconcilable index, golden identity, shared model resolver,
  declared decoder.
- Set `output_schema` on the row. Its default is `trex_v1`, the legacy centimeter
  schema, and `test_tracker_conformance.py` does not check it.
- Every tracker and inference op publishes through `publish_tracks_table`
  (`common/bridge.py`) with the entry's `EntryAxis`, its `frames_read` and whether its ids
  follow an animal (`carries_tracklets`). A bridge
  failure goes through `publish_or_record`, and the caller still writes the index row.

## The SLEAP embedding run

- A `sleap` model list whose last model is an embedding model is an embedding run
  (`_embedding_run_shape` in `sleap/dataset_runs.py`). It runs `sleap-nn predict`,
  reads the vectors back, and publishes an appearance table beside each tracks table
  through `publish_tracks_table`. The contract is `core/pipeline/appearance.py`, and
  `assign-identities` (`ops/assign_identities.py`) is its consumer.
- `sleap/version.py` declares the sleap-nn release floor (`SLEAP_NN_EMBEDDING_MIN`,
  `require_sleap_nn_at_least`). train-sleap and the embedding run check it with the
  probe. The version is provenance. No identifier reads it.
- `prepare-embedding-data` (`ops/prepare_embedding.py`) cuts training windows from a
  `sleap` run. Its labels are the only ones that train the `tracklet` scope. It writes
  `labels/{train,val}/shard-NNNNN.slp` from frames decoded once per entry and published
  with one `os.replace`, and `train-sleap` reads that directory (`sleap_label_files`).

## Model references and links

- Every tracker, `infer-*` and `base_model` call site resolves through `resolve_model`
  or `resolve_model_set` (`model_refs.py`). The conformance test refuses a private
  resolver.
- A reference is a path or a registered training run id. `_registered_artifact` asks
  each dataset of `Dataset.lookup_order("models")`, this dataset first, then each linked
  dataset in declaration order, and resolves the cell against the dataset that wrote it.
- Links are one level deep. An unreachable linked dataset is skipped with a warning. A
  reachable one with a different manifest `uuid` raises `LinkMismatchError`.
- A training op reuses a run that this dataset or a linked dataset finished
  (`linked_training_complete` in `ops/trained_models.py`). An offline link that provides
  models, with no finished row in a reachable dataset, raises `LinkUnreachableError`
  rather than training. `training_is_complete` reads one dataset's index, and
  `trained_model_records` keeps it for an own record's `complete` cell.
- `copy_run_root` (`ops/copy_run_root.py`) copies a finished run root into another
  dataset and records `.mosaic-origin.json` in the copy. Code that reconciles or
  classifies model run roots treats a run root that carries the marker as only `ok` or
  `scheme_stale`, and never moves it to another run id.

## Training data and provenance

- A training op's data argument is a path or a preparation run id
  (`resolve_training_data` in `ops/_common.py`). Prefer the run id. The reference
  string is hashed, and a path mints a new identity at every location.
- `prepare-training-data` (`ops/prepare.py`) copies its images and names each through
  `write_split_tree(name_of=...)`, which refuses two images under one name. The
  revision digests it hashes are sorted and measured off disk at plan time.
- `prepare-training-data` writes its `data.yaml` with `portable=True`
  (`make_data_yaml`, `make_polo_data_yaml`), which leaves out the `path` key.
  Ultralytics then roots the tree at the YAML's directory. `convert-points`
  (`ops/convert.py`) calls `make_polo_data_yaml` without it, and its `data.yaml` has an
  absolute `path`. The training fingerprint leaves `path` out. Do not write a relative
  `path`. Ultralytics resolves one against the working directory or its datasets folder.
- A training run root holds `training.json` (`TRAINING_PROVENANCE_FILENAME` in
  `ops/train.py`), never a second `params.json` schema.
  `test_nothing_else_opens_params_json` in
  `tests/core/pipeline/inventory/test_run_params_reader.py` fails on a second reader of
  `params.json`.
- `training_provenance` (`training_provenance.py`) walks from the model row to the
  prepared-data row, the series index row and the revision's `manifest.json` on demand.
  Do not store the chain.

## Ultralytics and POLO environments

- Ultralytics and POLO are AGPL and run only as a separate program.
  `external/ultralytics-env/` and `external/polo-env/` declare environments the user
  builds with `uv sync --python 3.12`. `external/runner/` holds the programs that run in
  them. A JSON request file, a JSON response file and progress lines on stdout cross.
- `tests/meta/test_external_tool_separation.py` fails when a mosaic module outside the
  runner imports the tool, when a mosaic dependency, extra, group or `uv.lock` declares
  it, or when a runner program imports mosaic. `SEPARATED_TOOLS` in
  `tests/helpers/environment.py` lists each tool. It also fails when the runner's
  subcommands differ from `RunnerSubcommand` (`common/ultralytics_env.py`), or when a
  subcommand's request or response is declared outside `ultralytics_protocol.py`.
- POLO ships under the distribution name `ultralytics`. One environment holds
  upstream or POLO, not both. Upstream (`ULTRALYTICS_ENV`) runs the tracker,
  `infer-pose`, `train-pose` and `predict_pose_frames`. POLO (`POLO_ENV`) runs
  `infer-points`, `train-points` and `predict_point_frames`. Both are in
  `common/ultralytics_env.py`, and both frame adapters in
  `pose_training/ultralytics_infer.py`.
- Both ship the same `yolo` script. The `$PATH` rung of the location ladder cannot
  tell them apart. The runner's probe reports `has_locate`. `require_points_model`
  (`pose_training/ultralytics_infer.py`) and `require_points_training_env`
  (`pose_training/ultralytics_train.py`) refuse an upstream build by name.
- `ultralytics_track/run.py` is mosaic's side of the tracker: `probe_ultralytics` once
  per run, then `run_ultralytics_tool` once per entry.
- Both environments are excluded from the uv workspace. The runner directory has its
  own basedpyright execution environment, which resolves its bare
  `ultralytics_protocol` import.

## The lmt-analysis environment

- lmt-analysis is GPL-3.0 and runs only as a separate program.
  `external/runner/lmt_rebuild_runner.py` is the one file that imports `lmtanalysis`
  and its `Rebuild_All_Events` script, through `importlib`, and it imports nothing
  from mosaic. `tests/meta/test_external_tool_separation.py` checks both, and
  `SEPARATED_TOOLS` lists the tool. No lmt-analysis code is copied or reimplemented.
- `external/lmt-env/` declares the environment the user builds with
  `uv sync --python 3.12`, and the user clones lmt-analysis beside it. `LMT_ENV`
  (`common/lmt_env.py`) finds the interpreter through `MOSAIC_LMT_CONDA_ENV`,
  `MOSAIC_LMT_PYTHON` or the directory's `.venv`. `MOSAIC_LMT_ANALYSIS_DIR` names the
  checkout, and `lmt-env/lmt-analysis/` is the default. Neither enters a `run_id`.
  `LMT_README` is the one path of the environment's README, and every environment refusal
  names it.
- A JSON report file, a command line and an exit status cross. Exit 3 is the
  environment, 4 is the input (an unknown builder or a database without a readable
  `FRAME` table) and 5 is `process` raising. In `rebuild` those three raise
  `LmtRebuildError` and fail the entry. Exit 2 and any other status, or a report that
  is missing or does not parse, raise `LmtRunnerDefect`, which fails the run. The
  module docstring of `lmt_env.py` holds the table.
- `rebuild-lmt-events` is a tracking op with its own root and the `rebuild` retention
  class: `sweep-tracking` reclaims a finished copy after `--rebuild-days`. The root
  declares `reads="none"`, so the op reads no media, has no prerequisites and declares
  `reads_media` false. It is outside `TRACKERS`.
- A row of the root's index means a finished copy. The op drops an entry's row before it
  redoes the entry, and `entries_written` counts the entries it rebuilt and the ones it
  found current.
- The converters find a copy by the md5 of the raw file through `rebuilt_copy`
  (`core/track_library/lmt_rebuild.py`), not through the index. A row lost to an
  interrupted write affects only the sweep.
- `tests/helpers/lmt_analysis.py` writes the stand-in checkout that the suite runs
  against, from the names the runner reads and nothing else. No test needs the real
  tool.

## Tool input per entry

- `TrackingRoot.reads` declares `one-file`, `entry-files`, `in-process` or `none`, and
  decides what must exist before a run: a join, a store export, or neither. `none` is
  a root that reads no media, and its op has no prerequisites.
  `plan_prerequisites` (`prerequisites.py`) answers that per entry from the same
  checks (`refuse_unreadable_entry` in `common/tool_input.py`), without a probe. It
  plans the run that each entry lacks: a join, a store export or a media variant.
- The Ultralytics runner (`entry-files`) decodes frames itself and reads the entry's
  files in order on one frame axis (`entry_runner_sources`). Nothing is joined for it.
  A store whose chunks hold the frames (`readable_chunks`) is read chunk by chunk, and
  any other store through its `export-store` video. Each file's facts are gated in
  mosaic before they cross. A division of a store is handed the chunks of its window,
  or its own export, found by its row key (`TrackerWorkItem.row_keys`), and plans its
  own `export-store` run.
  `tests/tracking/ultralytics_track/test_runner_entry_files.py` compares the result
  with the join, pixel for pixel.
- `infer-localizer` (`in-process`) reads through `read_entry_frames`, and a join only
  when `join_to_read` asks for one.

## Frame axis and joins

- One tracks file per `(variant, group, sequence)` (`tracks_table_path`). A tracker or
  `infer-*` op keeps one camera per entry through `one_camera_per_entry`
  (`core/pipeline/consumed_camera.py`), called from `build_work_items`
  (`common/scope.py`) and the inference ops. A dropped camera gets a stderr line and no
  run-log record.
- `frame` counts from the start of the entry, or of its group when the group is
  declared continuous. A tool reads the sequence's media, and `EntryAxis.place` moves
  its table onto the recording's axis by `ResolvedMedia.first_frame`, and its time
  columns by `ResolvedMedia.time_offset`, which `build_work_items` and the inference
  ops pass on. It is 0 outside a continuous
  group. The tracks row records it as `first_frame`. A sequence that cannot be placed
  fails alone as a lost entry, and the run reports `partial`. `media_drift` reports a
  row whose sequence now starts elsewhere, and `reusable_tracks_table`
  (`common/bridge.py`) republishes its table from the tool output, against the first
  frame that `run_tracker` reads once per run (`EntryJob.recorded_first_frame`;
  `tests/tracking/common/test_continuous_group_placement.py`,
  `tests/tracking/common/test_moved_first_frame_reuse.py`). `media_frames` and
  `frame_axis_verdict` stay local to the sequence. The tracker bridge does not check
  repeated `(frame, id)` pairs. The convert sites refuse them (`_placed_conversion` in
  `core/dataset.py`), and so does the inference bridge.
- No tool joins clips. A `one-file` tool is handed the `export-joined` file
  (`core/pipeline/joined_export.py`) through `resolve_tool_inputs`, after
  `refuse_unjoinable` (`common/scope.py`) has refused clips that cannot be one video. A
  join is counted by packets, held to an even timeline, and refused rather than
  published short. Tools read only a join under a current recipe (`current_join`).
  After a version bump, re-join and run `mosaic prune-joined`.
- `publish_or_record` reports a frame-axis mismatch or a known tail loss as a run-log
  event. Keep it a report. The condition is deterministic, and raising would fail the
  entry on every re-run.

## Ultralytics training cancel

- Ultralytics training (`train-pose`, `train-points`) cancels cooperatively.
  `stop_then_kill` (`common/cooperative_cancel.py`) writes the sentinel the runner
  reads between epochs, and kills only after `TRAIN_CANCEL_GRACE_SECONDS`
  (`ops/train.py`). The grace must exceed one epoch and stay under the substrate's own
  kill timer. A local request driver is that substrate for `mosaic pipeline run`, and
  `STEP_CANCEL_GRACE_SECONDS` (`core/pipeline/graph/run.py`) is its timer;
  `tests/tracking/ops/test_train_cancel_grace.py` pins the order. Every other tool, SLEAP and Lightning Pose
  training included, keeps the immediate kill of `run_supervised`.
- The exchange directory is per attempt (`pose_training/ultralytics_train.py`):
  `models/<kind>/<run_id>/.mosaic-train/<execution_id>/`. A run root is reused across
  attempts, and a stale sentinel would cancel the next one.
