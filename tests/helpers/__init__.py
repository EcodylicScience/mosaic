"""Shared builders for the test suite.

**The only public path.** A test writes ``from tests.helpers import X`` and never
reaches into a submodule, so a helper can move between modules here without
touching a call site. Two spellings used to coexist -- ``from .conftest import``
and ``from tests.conftest import`` -- which read as "neither is the sanctioned
one"; this is the sanctioned one.

What lives where:

- ``annotations`` -- keypoint annotation sets in their saved shape, and the file
  a revision of one is claimed by.
- ``datasets`` -- the `Dataset` a test runs against.
- ``driver_leases``: the lease of a request's driver, live or stale.
- ``agents``: a stand-in for the detached driver that an MCP submission starts.
- ``catalogs``: the declaration catalogs a graph test validates or plans against.
- ``cli`` -- a ``mosaic`` command run for its exit status or its ``--json``
  output.
- ``features`` -- the templates and per-sequence frames the global
  fit-then-apply features are tested on.
- ``latents`` names the frames at the centers of VAME's windows, and checks that
  a vame output table places a finite latent on each.
- ``column_contract`` checks a feature's output against the columns that
  validation predicts from its declaration.
- ``kpms`` contains the params of a kpms feature on three named keypoints.
- ``labels`` writes converted label files of events, and their index rows.
- ``extracted_frames`` writes an entry camera of a frames run: its PNGs, its
  ``run_info.json`` records and its frames-index row, and reads back the files
  of a dataset's one frames row.
- ``frame_predictions`` writes an entry camera's predictions file of a
  predict-frames run and its index row.
- ``frame_selections`` writes an entry's ``selection.parquet`` of a select-frames
  run and its index row.
- ``lightning_action`` stands in for the lightning-action package, with its data
  contract, and records what a feature hands it.
- ``stand_in_features`` -- features that do nothing but declare the source
  roots they consume, for the tests of what a source change reaches.
- ``tracks`` contains track tables, keypoint rows, tracks variants, and raw
  TREx, SLEAP and DeepLabCut exports.
- ``trex``, ``sleap``, ``litpose`` and ``ultralytics`` contain recording
  stand-ins for the tools that those trackers run, installed over each tracker's
  seams, and ``inference`` the same for the environments of ``infer-pose`` and
  ``infer-points`` and for the ``predict-images`` call of a yolo-pose or POLO
  frame adapter. ``ultralytics`` also writes the marked PNGs of a
  ``predict-images`` request. ``sleap`` also stands in for the SLEAP
  environment's probe, for the input writer, ``sleap-nn track`` and the
  readback of a SLEAP frame adapter, and for the labels writer of a sleap
  preparation. ``litpose`` also stands in for the Lightning Pose interpreter
  that runs the version probe and the prediction snippet of a Lightning Pose
  frame adapter.
- ``decode_probe`` contains a fake ``python`` for a tool environment, which stands
  in for the interpreter that runs the codec check's decode probe.
- ``runner_programs`` imports a runner program under its bare name for one test,
  writes fake runners that serve a test's handlers over the runner wire, and
  checks that a runner program answers ``Ready`` and ``shutdown``.
- ``index_locks`` probes from a child process whether an index's lock is held.
- ``processes`` contains a child program that appends to a file, a sleeper
  command, checks of whether a process stopped writing to it or exited, the line
  children of a test, and a cancel fired once a condition is true.
- ``media`` -- media files, media-index rows, transcode derivatives.
- ``variants`` contains media variant files, their index rows and recorded
  recipes, and a count of the index reads that a variant's consumers make.
- ``identity`` -- the digest a feature run is named by, from its parameters.
- ``ops`` -- the smallest params dict that validates for each registered op.
- ``runlog`` -- an attempt's recorded run-log events.
- ``scope`` -- a resolved scope over named entries, for the ops and drivers
  that take their coverage as an argument.
- ``environment`` -- what the surrounding machine provides: the ffmpeg
  toolchain, the variables and ``$PATH`` an external tool is located through,
  the modules each CI job must install, and which files under the
  package root a structural walk should skip -- installed third-party code, and
  mosaic's code that runs in an environment built for an external tool. It
  also names the tools that run in those environments, and the libraries,
  programs and environments of each.
- ``mock_dataset`` -- the duck-typed stand-in, for the pipeline tests that want
  no real roots.
- ``source_scan`` -- reads a module's source as a tree, for the tests that
  assert what a code path reads, calls and imports.
- ``pyproject_requirements``: the requirements a ``pyproject.toml`` declares,
  the distributions they name, and what an extra installs through its
  self-references.
- ``paths`` -- the repository, the source package, the ``tests`` directory and
  the golden-file directory, which a test names from here rather than from its
  own location. It also names the programs of the external tools, relative to
  the package root.
- ``fresh_interpreter``: a program run in a fresh interpreter, and the modules
  that a statement loads there.
- ``golden`` -- reading and rewriting the golden files under ``tests/data/``.
- ``recipes``: the recipes a graph test validates, plans or runs.
- ``trees``: snapshots of a directory tree, and the assertion that a call left
  one unchanged.

Fixtures stay in ``tests/conftest.py``, because pytest collects them only from
there. Their bodies delegate here, so the logic has one home either way.
"""

from __future__ import annotations

from tests.helpers.agents import RecordingStarter
from tests.helpers.annotations import (
    KEYPOINTS_PAYLOAD,
    MOUSE,
    pose_frame,
    pose_object,
    pose_set,
    revision_file,
)
from tests.helpers.catalogs import catalog_without_contract
from tests.helpers.column_contract import assert_writes_declared_columns
from tests.helpers.cli import invoke_json, invoke_ok
from tests.helpers.datasets import make_dataset
from tests.helpers.decode_probe import FakeToolPython, install_fake_tool_python
from tests.helpers.extracted_frames import (
    extract_entry_camera,
    record_frames_row,
    sole_frames_row_files,
    write_frames_run,
)
from tests.helpers.environment import (
    CI_FERAL_MODULES,
    CI_LIGHTNING_ACTION_MODULES,
    CI_IDENTITY_MODULES,
    CI_MCP_MODULES,
    CI_REQUIRED_MODULES,
    FAKE_CONDA,
    FFMPEG_TOOLCHAIN,
    KPMS_TOOL,
    SEPARATED_TOOLS,
    ULTRALYTICS_TOOL,
    VAME_TOOL,
    CiJob,
    EnvironmentMarker,
    SeparatedEnvironment,
    SeparatedTool,
    assert_no_literal_tilde,
    fake_path_lookup,
    inside_a_virtualenv,
    isolate_tool_location,
    missing_ffmpeg_tools,
    require_ffmpeg,
    runs_in_an_external_environment,
    sandbox_home,
)
from tests.helpers.features import (
    make_pair_df,
    make_sequence_df,
    make_templates,
    write_templates,
)
from tests.helpers.frame_predictions import (
    entry_predictions,
    record_entry_predictions,
    record_score_run,
)
from tests.helpers.frame_selections import entry_selection, record_entry_selection
from tests.helpers.kpms import kpms_params
from tests.helpers.latents import assert_latents_at_centers, window_centers
from tests.helpers.fresh_interpreter import modules_loaded_by, run_fresh
from tests.helpers.golden import (
    UPDATE_GOLDEN_ENV,
    golden_path,
    read_golden,
    read_string_golden,
    regenerate_command,
    updating_golden,
    write_golden,
)
from tests.helpers.media import (
    EVEN_STORE_TIMING,
    MOTIF_SYNC_UUID,
    MakeStore,
    MediaClip,
    add_media_sequence,
    add_transcode_derivative,
    clean_facts_cells,
    clip_facts,
    dot_image,
    entry_clips,
    export_entry_store,
    index_media_sequence,
    index_session,
    nameless_store_dataset,
    paint_frame_code,
    point_at_a_store,
    printed_command_words,
    printed_scope_files,
    read_frame_code,
    store_dataset,
    stub_join,
    stub_media_dataset,
    video_store_maker,
    write_h264_mp4,
    write_media_index,
    write_mpeg4_mp4,
    write_painted_entry,
    write_painted_recording,
)
from tests.helpers.driver_leases import write_driver_lease
from tests.helpers.guide_recipes import (
    GUIDE,
    GUIDE_CLASSES,
    guide_blocks,
    guide_commands,
    guide_dataset,
    guide_recipe,
    run_ids,
    step_scopes,
    step_states,
    with_params,
)
from tests.helpers.mock_dataset import MockDataset
from tests.helpers.processes import (
    SLEEPER_COMMAND,
    LineChildren,
    appender_program,
    assert_process_exited,
    assert_stopped_writing,
    cancel_when,
    exited_pid,
    wait_for,
)
from tests.helpers.recipes import (
    CHAIN_RECIPE,
    CONVERT_LABELS_STEP,
    CONVERT_TRACKS_STEP,
    GUPPIES_RECIPE,
    RECIPE_SAVED_BY_0_17,
    SPEED_OF_CONVERTED_STEP,
    SPEED_RECIPE,
    chain_recipe,
    chain_with_sibling_recipe,
    classifier_recipe,
    converting_recipe,
    guppies_recipe,
    named_recipe,
    recipe_saved_by_0_17,
    run_id_of,
    run_speed,
    speed_heading_recipe,
    speed_recipe,
    speed_recipe_stepping,
)
from tests.helpers.models import (
    register_sleap_model,
    register_trained_model,
    rewrite_as_older_prepared_index,
)
from tests.helpers.ops import minimal_op_params
from tests.helpers.paths import (
    GOLDEN_DIR,
    KPMS_PROTOCOL,
    KPMS_SERVER,
    REPO_ROOT,
    SOURCE_ROOT,
    TESTS_ROOT,
    ULTRALYTICS_PROTOCOL,
    VAME_PROTOCOL,
    VAME_RUNNER,
)
from tests.helpers.runlog import (
    entry_error_lines,
    latest_events,
    latest_snapshot,
    record_open_attempt,
    write_reader_params,
)
from tests.helpers.results import model_reference, result_outputs, result_run_root
from tests.helpers.runner_programs import (
    FAKE_HANDLERS,
    RUNNER_LINE_SECONDS,
    EventRecorder,
    RunnerJournal,
    assert_ready_then_shutdown,
    import_runner_program,
    journal_source,
    write_fake_runner,
)
from tests.helpers.scope import resolved_scope, scope_over
from tests.helpers.pyproject_requirements import (
    SELF_REFERENCE,
    distribution_names,
    project_requirements,
    requirements_reaching,
    self_referenced_extras,
)
from tests.helpers.source_scan import (
    ImportedName,
    functions_named,
    imports_in,
    imports_of,
    module_tree,
    names_called_by,
    names_read,
    source_tree,
)
from tests.helpers.documents import dotted_values
from tests.helpers.labels import (
    CALMS21_NAMES,
    LABELS_VARIANT,
    Event,
    add_labels,
    write_sparse_labels,
)
from tests.helpers.lightning_action import (
    LA_CLASS_NAMES,
    LA_CLASSES,
    LA_PAD,
    LA_WINDOW,
    LightningActionRecorder,
    StudyEntry,
    TemplatesStudy,
    fit_lightning_action,
    install_lightning_action_stub,
    lightning_action_params,
    stub_probabilities,
    write_templates_study,
)
from tests.helpers.stand_in_features import (
    CropLike,
    LossyFeature,
    PairClusterFeature,
    ParallelFeature,
    PlainFeature,
    ScopedFeature,
)
from tests.helpers.training import FakeTrainer, healthy_probe, write_data_yaml
from tests.helpers.trees import PathState, assert_tree_unchanged, tree_snapshot
from tests.helpers.tracks import (
    add_track_sequences,
    add_tracks_table,
    add_tracks_variant,
    calms21_dataset,
    keypoint_rows,
    published_table,
    set_tracks_cell,
    track_sequences,
    track_table,
    write_calms21_npy,
    write_dlc_csv,
    write_sleap_analysis_h5,
    write_trex_npz,
)
from tests.helpers.identity import run_id_digest
from tests.helpers.index_locks import index_locked_out
from tests.helpers.inference import (
    FakeInference,
    FakePredictImages,
    fake_tool_probe,
    install_fake_point_inference,
    install_fake_point_probe,
    install_fake_pose_inference,
    install_fake_pose_probe,
    install_fake_predict_images,
    point_image_rows,
    point_predictions,
    pose_image_rows,
    pose_per_frame,
    pose_predictions,
)
from tests.helpers.litpose import (
    FakeLitpose,
    FakeLitposeFrames,
    LitposeSnippetRun,
    install_fake_litpose,
    install_fake_litpose_frames,
    write_litpose_model,
)
from tests.helpers.sleap import (
    FakeSleap,
    FakeSleapFrames,
    SleapInput,
    SleapTrackCall,
    SlpWrite,
    install_fake_sleap,
    install_fake_sleap_frames,
    record_slp_writes,
    sleap_instance,
    write_sleap_model,
)
from tests.helpers.trex import FakeTrex, install_fake_trex, write_pv_header
from tests.helpers.ultralytics import (
    FakeDetections,
    FakeResult,
    FakeUltralytics,
    ULTRALYTICS_KEYPOINTS,
    image_mark,
    install_fake_ultralytics,
    install_ultralytics_config,
    ultralytics_probe_response,
    write_marked_images,
    write_ultralytics_predictions,
)
from tests.helpers.variants import (
    IndexReads,
    add_media_variant,
    count_index_reads,
    finish_media_variant,
    record_variant_recipe,
)

__all__ = [
    "CHAIN_RECIPE",
    "CONVERT_LABELS_STEP",
    "GUIDE",
    "GUIDE_CLASSES",
    "guide_blocks",
    "guide_commands",
    "guide_dataset",
    "guide_recipe",
    "run_ids",
    "step_scopes",
    "step_states",
    "with_params",
    "CONVERT_TRACKS_STEP",
    "CI_FERAL_MODULES",
    "CI_LIGHTNING_ACTION_MODULES",
    "CI_IDENTITY_MODULES",
    "CI_MCP_MODULES",
    "CI_REQUIRED_MODULES",
    "CropLike",
    "EVEN_STORE_TIMING",
    "EnvironmentMarker",
    "Event",
    "EventRecorder",
    "FAKE_CONDA",
    "FAKE_HANDLERS",
    "FFMPEG_TOOLCHAIN",
    "GUPPIES_RECIPE",
    "SLEEPER_COMMAND",
    "SPEED_OF_CONVERTED_STEP",
    "SPEED_RECIPE",
    "FakeDetections",
    "FakeResult",
    "GOLDEN_DIR",
    "ImportedName",
    "KEYPOINTS_PAYLOAD",
    "KPMS_PROTOCOL",
    "KPMS_SERVER",
    "KPMS_TOOL",
    "LA_CLASSES",
    "LA_CLASS_NAMES",
    "LA_PAD",
    "LA_WINDOW",
    "LABELS_VARIANT",
    "LightningActionRecorder",
    "StudyEntry",
    "TemplatesStudy",
    "LineChildren",
    "LitposeSnippetRun",
    "LossyFeature",
    "MOUSE",
    "PairClusterFeature",
    "ParallelFeature",
    "PlainFeature",
    "REPO_ROOT",
    "RUNNER_LINE_SECONDS",
    "RecordingStarter",
    "RunnerJournal",
    "SELF_REFERENCE",
    "SEPARATED_TOOLS",
    "SOURCE_ROOT",
    "CALMS21_NAMES",
    "CiJob",
    "ScopedFeature",
    "SeparatedEnvironment",
    "SeparatedTool",
    "SleapInput",
    "SleapTrackCall",
    "SlpWrite",
    "TESTS_ROOT",
    "ULTRALYTICS_KEYPOINTS",
    "ULTRALYTICS_PROTOCOL",
    "ULTRALYTICS_TOOL",
    "VAME_PROTOCOL",
    "VAME_RUNNER",
    "VAME_TOOL",
    "FakeInference",
    "FakeLitpose",
    "FakeLitposeFrames",
    "FakePredictImages",
    "FakeSleap",
    "FakeSleapFrames",
    "FakeTrainer",
    "FakeTrex",
    "FakeToolPython",
    "FakeUltralytics",
    "IndexReads",
    "MOTIF_SYNC_UUID",
    "MakeStore",
    "MediaClip",
    "MockDataset",
    "PathState",
    "UPDATE_GOLDEN_ENV",
    "add_labels",
    "add_media_sequence",
    "add_media_variant",
    "add_track_sequences",
    "add_tracks_table",
    "add_tracks_variant",
    "add_transcode_derivative",
    "appender_program",
    "calms21_dataset",
    "chain_recipe",
    "chain_with_sibling_recipe",
    "classifier_recipe",
    "converting_recipe",
    "assert_latents_at_centers",
    "assert_no_literal_tilde",
    "assert_process_exited",
    "assert_ready_then_shutdown",
    "assert_stopped_writing",
    "assert_tree_unchanged",
    "assert_writes_declared_columns",
    "cancel_when",
    "catalog_without_contract",
    "fake_path_lookup",
    "golden_path",
    "guppies_recipe",
    "import_runner_program",
    "imports_in",
    "imports_of",
    "healthy_probe",
    "clean_facts_cells",
    "clip_facts",
    "count_index_reads",
    "distribution_names",
    "dot_image",
    "export_entry_store",
    "dotted_values",
    "entry_clips",
    "entry_error_lines",
    "entry_predictions",
    "entry_selection",
    "exited_pid",
    "extract_entry_camera",
    "fake_tool_probe",
    "finish_media_variant",
    "functions_named",
    "image_mark",
    "index_locked_out",
    "index_media_sequence",
    "index_session",
    "inside_a_virtualenv",
    "install_fake_litpose",
    "install_fake_litpose_frames",
    "install_fake_point_inference",
    "install_fake_point_probe",
    "install_fake_pose_inference",
    "install_fake_pose_probe",
    "install_fake_predict_images",
    "install_fake_sleap",
    "install_fake_sleap_frames",
    "install_fake_tool_python",
    "install_fake_trex",
    "fit_lightning_action",
    "install_lightning_action_stub",
    "lightning_action_params",
    "invoke_json",
    "invoke_ok",
    "install_fake_ultralytics",
    "install_ultralytics_config",
    "isolate_tool_location",
    "journal_source",
    "keypoint_rows",
    "kpms_params",
    "latest_events",
    "latest_snapshot",
    "make_dataset",
    "make_pair_df",
    "make_sequence_df",
    "make_templates",
    "minimal_op_params",
    "model_reference",
    "missing_ffmpeg_tools",
    "module_tree",
    "modules_loaded_by",
    "named_recipe",
    "nameless_store_dataset",
    "names_called_by",
    "names_read",
    "paint_frame_code",
    "point_at_a_store",
    "point_image_rows",
    "point_predictions",
    "printed_command_words",
    "printed_scope_files",
    "project_requirements",
    "pose_image_rows",
    "pose_per_frame",
    "pose_predictions",
    "pose_frame",
    "pose_object",
    "pose_set",
    "published_table",
    "read_frame_code",
    "record_entry_predictions",
    "record_entry_selection",
    "record_score_run",
    "record_frames_row",
    "record_slp_writes",
    "read_golden",
    "read_string_golden",
    "record_variant_recipe",
    "regenerate_command",
    "register_sleap_model",
    "register_trained_model",
    "require_ffmpeg",
    "requirements_reaching",
    "resolved_scope",
    "result_outputs",
    "result_run_root",
    "revision_file",
    "rewrite_as_older_prepared_index",
    "run_fresh",
    "run_id_digest",
    "record_open_attempt",
    "run_id_of",
    "run_speed",
    "runs_in_an_external_environment",
    "sandbox_home",
    "scope_over",
    "self_referenced_extras",
    "set_tracks_cell",
    "sleap_instance",
    "sole_frames_row_files",
    "source_tree",
    "speed_heading_recipe",
    "speed_recipe",
    "speed_recipe_stepping",
    "store_dataset",
    "stub_join",
    "stub_media_dataset",
    "stub_probabilities",
    "updating_golden",
    "video_store_maker",
    "wait_for",
    "write_reader_params",
    "window_centers",
    "track_sequences",
    "track_table",
    "tree_snapshot",
    "ultralytics_probe_response",
    "write_data_yaml",
    "write_calms21_npy",
    "write_dlc_csv",
    "write_sparse_labels",
    "write_templates_study",
    "write_driver_lease",
    "write_fake_runner",
    "write_frames_run",
    "write_golden",
    "write_h264_mp4",
    "write_litpose_model",
    "write_media_index",
    "write_marked_images",
    "write_mpeg4_mp4",
    "write_painted_entry",
    "write_painted_recording",
    "write_pv_header",
    "write_sleap_analysis_h5",
    "write_sleap_model",
    "write_templates",
    "write_trex_npz",
    "write_ultralytics_predictions",
    "RECIPE_SAVED_BY_0_17",
    "recipe_saved_by_0_17",
]
