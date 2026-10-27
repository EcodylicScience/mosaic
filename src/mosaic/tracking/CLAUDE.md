# mosaic.tracking

This directory owns the integrated trackers, model inference and training ops, frame
extraction, and mosaic's side of the external Ultralytics and POLO environments.

The cross-cutting rules (a model is named by run id, only a revision's content is
hashed, codec refusal, source-space publication) are in the root `CLAUDE.md`.

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
  (`common/bridge.py`) with the entry's `EntryAxis` and its `frames_read`. A bridge
  failure goes through `publish_or_record`, and the caller still writes the index row.

## Model references and libraries

- Every tracker, `infer-*` and `base_model` call site resolves through `resolve_model`
  or `resolve_model_set` (`model_refs.py`). The conformance test refuses a private
  resolver.
- A reference is a path or a registered training run id. `_registered_artifact` asks
  this dataset's index first, then each library from `Dataset.linked_libraries()` in
  declaration order, and resolves the cell against the dataset that wrote it.
- Links are one level deep. An unreachable library is skipped with a warning. A
  reachable one with a different manifest `uuid` raises `LibraryMismatchError`.

## Training data and provenance

- A training op's data argument is a path or a preparation run id
  (`resolve_training_data` in `ops/_common.py`). Prefer the run id. The reference
  string is hashed, and a path mints a new identity at every location.
- `prepare-training-data` (`ops/prepare.py`) copies its images and names each through
  `write_split_tree(name_of=...)`, which refuses two images under one name. The
  revision digests it hashes are sorted and measured off disk at plan time.
- `data.yaml` is written with `portable=True` (`make_data_yaml`, `make_polo_data_yaml`)
  and has no `path` key. Ultralytics then roots the tree at the YAML's directory. Do
  not write a relative `path`. Ultralytics resolves one against the working directory
  or its datasets folder.
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

## Tool input per entry

- `TrackingRoot.reads` declares `one-file`, `entry-files` or `in-process`, and decides
  what must exist before a run: a join, a store export, or neither.
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
