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
- ``example_downloads``: a local server that answers as Hugging Face does for an
  example's repository, with scripted replies and a request log, and the small
  generated archives, ``SHA256SUMS`` body and catalog entry it serves.
- ``agents``: a stand-in for the detached driver that an MCP submission starts.
- ``arenas``: a count of the arenas index reads that a run makes, and a damaged
  arenas index.
- ``cleanup``: the policy a cleanup is planned under, the retired recipe's runs and
  retirement, the reasons a judging gave, and the observers of a judging and of a
  removal.
- ``catalogs``: the declaration catalogs a graph test validates or plans against.
- ``cli`` -- a ``mosaic`` command run for its exit status or its ``--json``
  output.
- ``dropped_frames`` -- a tracks table that dropped frames, with individuals on an
  analytic trajectory, for the tests of the features that read ``time``.
- ``features`` -- the templates and per-sequence frames the global
  fit-then-apply features are tested on.
- ``latents`` names the frames at the centers of VAME's windows, and checks that
  a vame output table places a finite latent on each.
- ``column_contract`` checks a feature's output against the columns that
  validation predicts from its declaration, and a pair table's mirror rows.
- ``kpms`` contains the params of a kpms feature on three named keypoints.
- ``labels`` writes converted label files of events, and their index rows, and
  holds the label converters that the conversion tests register. It also writes
  raw predictions tables and scans them as a labels source.
- ``failures`` -- failure records written into a dataset, a quarantined entry of
  a step.
- ``extracted_frames`` writes an entry camera of a frames run: its PNGs, its
  ``run_info.json`` records and its frames-index row, and reads back the files
  of a dataset's one frames row.
- ``frame_predictions`` writes an entry camera's predictions file of a
  predict-frames run and its index row.
- ``frame_selections`` writes an entry's ``selection.parquet`` of a select-frames
  run and its index row.
- ``linked``: a source dataset holding raw rows of every kind, one whose raw
  tracks are converted, one whose labels are converted, a copy of any of them
  that links it and adopts its raw rows, a copy whose link is offline, and a
  feature run recorded in a dataset.
- ``lightning_action`` stands in for the lightning-action package, with its data
  contract, and records what a feature hands it.
- ``stand_in_features`` -- features that do nothing but declare the source
  roots they consume, for the tests of what a source change reaches, one that
  fits, for the tests of the claim a fit holds, one that fits over its scope,
  for the tests of a fit whose entries change, one that fits a mean, for the
  tests of a fit that a linked dataset holds, and a classifier that records the
  labels it reads.
- ``tracks`` contains track tables, keypoint rows, tracks variants, and raw
  TREx, SLEAP, DeepLabCut, Tracktor, s-DANNCE and zebrafish 3D workflow exports,
  a synthetic track file and a dataset indexing several of them, a window file, a
  CalMS21 dataset and the patch that makes the CalMS21 converter fail one
  sequence, and an object member whose unpickling writes a marker file.
- ``trex``, ``sleap``, ``litpose`` and ``ultralytics`` contain recording
  stand-ins for the tools that those trackers run, installed over each tracker's
  seams, and ``inference`` the same for the environments of ``infer-pose`` and
  ``infer-points`` and for the ``predict-images`` call of a yolo-pose or POLO
  frame adapter. ``ultralytics`` also writes the marked PNGs of a
  ``predict-images`` request. ``sleap`` also stands in for the SLEAP
  environment's probe, for the input writer, ``sleap-nn track`` and the
  readback of a SLEAP frame adapter, and for the labels writer of a sleap
  preparation. It also writes the readback ``.npz`` of tracked instances and a
  stand-in ``sleap_io`` package that a snippet's source runs against.
  ``litpose`` also stands in for the Lightning Pose interpreter that runs the
  version probe and the prediction snippet of a Lightning Pose frame adapter.
- ``decode_probe`` contains a fake ``python`` for a tool environment, which stands
  in for the interpreter that runs the codec check's decode probe.
- ``runner_programs`` imports a runner program under its bare name for one test,
  writes fake runners that serve a test's handlers over the runner wire, and
  checks that a runner program answers ``Ready`` and ``shutdown``.
- ``gpu_cards`` stands in for ``nvidia-smi``, and keeps a test from leasing a
  real GPU card.
- ``index_locks`` probes from a child process whether an index's lock is held.
- ``processes`` contains a child program that appends to a file, a sleeper
  command, checks of whether a process stopped writing to it or exited, a kill of
  the listed processes still running, the line children of a test, a cancel fired
  once a condition is true, the timeout on every wait for a child, a command run
  under it, a child run with its output in files, a child whose process pool loses
  a worker and the calls of that pool, a line read from a child under it, a child
  whose process group is signalled once a condition is true, whether a process is
  running, and a count of the walks of the process tree.
- ``media`` -- media files, media-index rows, transcode derivatives.
- ``variants`` contains media variant files, their index rows and recorded
  recipes, and a count of the index reads that a variant's consumers make.
- ``identity`` -- the digest a feature run is named by, from its parameters.
- ``ops`` -- the smallest params dict that validates for each registered op,
  the payloads that reach each op's validators, a params model narrowed the way
  a consumer narrows it, and the fields a consumer owns for an op kind.
- ``runlog`` -- an attempt's recorded run-log events.
- ``entry_outcomes`` -- an entry task's outcome as a report or a raised error, and
  the output a worker wrote for a task.
- ``entry_points`` -- the installed entry points a test declares in this process,
  entry points that count their loads or fail them, and a child process that sees an
  installed distribution declaring some.
- ``entry_writes`` -- the write of an entry task, wrapped to fault it or hold it.
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
- ``spies`` -- a function with its own signature that calls a watcher first, a
  module function that records the paths it receives, and a property that records
  its reads, for the tests that count what a run calls, and
  a process high-water mark that rises once a feature has applied.
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
- ``state_writes``: the files a feature's ``save_state`` publishes and their order,
  and whether ``load_state`` finds a state after each publish fails.
- ``model_state``: the k-means chain that fits ``global-kmeans`` and applies it
  through ``model``, a refit of it over moved tracks, an entry converted again from
  changed sources, the reasons a run printed for the entries it did not serve, a column
  of a run's index rows, the fit-state digest each row of an apply records, the entries
  a plan reports as drifted, a templates recipe and a notebook pipeline of it, and a
  dataset whose templates fit is due a refit.
- ``feature_definitions``: the sources of feature definitions, a dataset with
  tracks that defines them, an instance of the applying definition for a ``model``
  mapping, and the distance from the origin of each row of an entry's tracks.
- ``trees``: snapshots and file digests of a directory tree, and the assertion
  that a call left one unchanged.
- ``run_summaries``: a feature whose output for each entry is the table a test
  passes for it, a run of it, a run beside entries added after it, and a rewrite of
  a run's feature index.
- ``aggregates``: a run of an aggregate over two entries of labeled tracks, one
  whose label varies inside every group, with what it wrote and lost, and the
  rows of one individual at a steady step, the error text that a missing ``time``
  gives ``duration`` and ``share``, and the values of an ROI column.
- ``tag_tables``: a table of sequence tags in the shape the ``tags`` series saves,
  and its write into a dataset, and the index row of a second origin's copy of it.

Fixtures stay in ``tests/conftest.py``, because pytest collects them only from
there. Their bodies delegate here, so the logic has one home either way.
"""

from __future__ import annotations

from tests.helpers.agents import RecordingStarter
from tests.helpers.arenas import Counted, refuse_index
from tests.helpers.annotations import (
    KEYPOINTS_PAYLOAD,
    MOUSE,
    pose_frame,
    pose_object,
    pose_set,
    revision_file,
)
from tests.helpers.catalogs import catalog_without_contract
from tests.helpers.cleanup import (
    HEADING_STORAGE,
    LOOSE,
    SPEED_STORAGE,
    OldRuns,
    act_after_judging,
    end_attempt,
    kept_reasons,
    managed_dataset,
    observe_removals,
    planned_run_ids,
    reasons_by_run,
    runs_by_id,
)
from tests.helpers.column_contract import assert_writes_declared_columns, mirrored_rows
from tests.helpers.cli import invoke_json, invoke_ok
from tests.helpers.datasets import make_dataset
from tests.helpers.feature_definitions import (
    APPLYING_NAME,
    APPLYING_SECOND_SOURCE,
    APPLYING_SOURCE,
    APPLYING_STORAGE,
    CANARY_NAME,
    CANARY_SOURCE,
    CANARY_VARIABLE,
    GLOBAL_NAME,
    GLOBAL_SOURCE,
    MINIMAL_NAME,
    MINIMAL_SOURCE,
    MISREADING_NAME,
    MISREADING_SOURCE,
    NARROWING_NAME,
    NARROWING_SOURCE,
    PARALLEL_APPLYING_NAME,
    PARALLEL_APPLYING_SOURCE,
    POSE_NAME,
    POSE_SOURCE,
    applying_instance,
    built_by_class,
    defined_dataset,
    definitions_forgotten,
    distances_from_origin,
    fit_over,
    fit_result,
    forget_imported_definitions,
    imported_definitions,
    noting_definition_loads,
    plan_refused,
)
from tests.helpers.decode_probe import FakeToolPython, install_fake_tool_python
from tests.helpers.example_downloads import (
    REVISION,
    ExampleFixture,
    ExampleServer,
    GeneratedArchive,
    RecordedRequest,
    Reply,
    archive_path,
    cdn_path,
    checksums_path,
    device_member,
    example_fixture,
    hardlink_member,
    serve_example,
    serve_example_server,
    sums_text,
    symlink_member,
    tar_gz,
    tar_gz_with,
)
from tests.helpers.failures import quarantine_entry
from tests.helpers.extracted_frames import (
    extract_entry_camera,
    record_frames_row,
    sole_frames_row_files,
    write_frames_run,
)
from tests.helpers.dropped_frames import (
    ANGULAR_VELOCITY,
    GAP_SECONDS,
    SPEED,
    STEP_SECONDS,
    VELOCITY,
    dropped_frame_times,
    dropped_frame_tracks,
    mean_rate,
)
from tests.helpers.entry_outcomes import report_of, written_by_worker
from tests.helpers.entry_points import (
    CountingEntryPoint,
    FailingEntryPoint,
    install_entry_points,
    run_with_distribution,
)
from tests.helpers.entry_writes import WriteOutput, wrapped_write
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
    hide_package,
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
    DIVISION_CREATED_UTC,
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
    write_division_rows,
    write_h264_mp4,
    write_media_index,
    write_mpeg4_mp4,
    write_painted_entry,
    write_painted_recording,
)
from tests.helpers.driver_leases import write_driver_lease
from tests.helpers.guide_recipes import (
    GUIDE_CLASSES,
    GUIDE_PAGES,
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
    CHILD_TIMEOUT_SECONDS,
    CLOCK_TICK_SECONDS,
    EXIT_AFTER_MARK_SECONDS,
    SLEEPER_COMMAND,
    LineChildren,
    PoolCallEnd,
    SignalledGroup,
    appender_program,
    assert_marked_child_ends,
    assert_process_exited,
    assert_stopped_writing,
    cancel_when,
    count_process_tree_walks,
    exited_pid,
    kill_marked,
    kill_running,
    listed_pids,
    pool_call,
    process_alive,
    read_line,
    run_child,
    run_child_to_files,
    signal_group_when,
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
    scored_classifier_recipe,
    speed_heading_recipe,
    speed_recipe,
    speed_recipe_stepping,
)
from tests.helpers.model_state import (
    FITTED_AGAIN,
    KMEANS,
    KMeansChain,
    KMeansFit,
    TEMPLATES_RECIPE,
    drifted,
    drifted_templates,
    index_cells,
    kmeans_apply,
    kmeans_apply_feature,
    kmeans_fit,
    kmeans_fit_and_apply,
    model_cells,
    move_tracks,
    reconvert_entry,
    refit_kmeans,
    refusals,
    run_step,
    templates_pipeline,
    templates_recipe,
)
from tests.helpers.models import (
    register_sleap_model,
    register_trained_model,
    rewrite_as_older_prepared_index,
)
from tests.helpers.ops import (
    CONSUMER_OWNED,
    minimal_op_params,
    narrowed_params,
    rule_reaching_op_params,
)
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
    logged_events,
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
from tests.helpers.run_summaries import (
    LabeledFeature,
    grown_run,
    labeled_run,
    rewrite_index,
)
from tests.helpers.aggregates import (
    MIXED,
    ROI_VALUES,
    STEADY,
    TIME_SENTENCE,
    LabeledTracksRun,
    constant_time_in_mixed,
    record_applied_sequences,
    run_over_edited_tracks,
    run_over_labeled_tracks,
    stepped_individual,
    without_time,
    without_time_in_mixed,
)
from tests.helpers.scope import resolved_scope, scope_over
from tests.helpers.spies import (
    count_property_reads,
    observed,
    raise_high_water_mark_once_applied,
    record_path_calls,
)
from tests.helpers.pyproject_requirements import (
    SELF_REFERENCE,
    distribution_names,
    project_requirements,
    requirements_reaching,
    self_referenced_extras,
)
from tests.helpers.source_scan import (
    ImportedName,
    ReachedImport,
    imports_in,
    imports_of,
    imports_reaching,
    names_called_by,
    package_of,
    source_tree,
)
from tests.helpers.documents import dotted_values
from tests.helpers.labels import (
    CALMS21_NAMES,
    LABELS_VARIANT,
    LISTED_FRAMES,
    PREDICTIONS_VARIANT,
    Event,
    ListedEntries,
    ListedEntriesParams,
    SecondsEntries,
    SharedKeyRefusingEntries,
    TracksAxisEntries,
    add_labels,
    add_predictions,
    scan_predictions_tables,
    write_lineage_run,
    write_listed_entries,
    write_predictions,
    write_sparse_labels,
)
from tests.helpers.linked import (
    CONVERTED_ENTRIES,
    SOURCE_ENTRIES,
    SUPERSEDED_RECIPE,
    add_feature_run,
    converted_source,
    copy_of,
    feature_edge,
    labelled_source,
    link_forms,
    portable_columns,
    raw_source,
    take_link_offline,
    tracks_edge,
    write_run_params,
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
    AlwaysFailsItsBuild,
    CenteringFeature,
    CountsItsBuilds,
    CropLike,
    FailsItsFirstBuild,
    FitControl,
    FittingFeature,
    HOLD_DIRECTORY_VARIABLE,
    HoldsBC,
    HoldsUntilReleased,
    LABELS_CLASSIFIER_STEP,
    LabelsClassifier,
    LossyFeature,
    PairClusterFeature,
    ParallelFeature,
    PlainFeature,
    ScopedFeature,
    ScopedFittingFeature,
    labels_classifier_registered,
)
from tests.helpers.training import FakeTrainer, healthy_probe, write_data_yaml
from tests.helpers.state_writes import (
    load_results_after_each_failed_write,
    record_published_files,
)
from tests.helpers.tag_tables import index_a_second_origin, tag_table, write_tags
from tests.helpers.trees import (
    PathState,
    assert_tree_unchanged,
    file_digests,
    tree_snapshot,
)
from tests.helpers.tracks import (
    add_raw_track_files,
    add_track_sequences,
    add_tracks_table,
    add_tracks_variant,
    add_tracks_variant_in_bulk,
    convert_tracks_from_raw,
    appearance_table,
    calms21_dataset,
    fail_calms21_sequence,
    keypoint_rows,
    published_table,
    set_tracks_cell,
    track_sequences,
    track_table,
    unpickling_member,
    write_calms21_npy,
    write_dlc_csv,
    write_sdannce_mat,
    write_sleap_analysis_h5,
    SYNTHETIC_ARENA_3D,
    SYNTHETIC_DROPOUTS,
    synthetic_dataset,
    write_synthetic_spec,
    write_tracktor_recording,
    write_trex_npz,
    write_window_spec,
    write_zebrafish_3d_csv,
    world_track_table,
    TRACKTOR_STEM,
    TracktorRecording,
    WORLD_POINTS_FORMAT,
    WorldPointsConverter,
    raw_world_dataset,
    registered_world_points,
)
from tests.helpers.identity import run_id_digest
from tests.helpers.index_locks import index_locked_out
from tests.helpers.gpu_cards import (
    TWO_CARDS,
    install_fake_nvidia_smi,
    isolate_card_leases,
)
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
    EMBEDDING_TRACKER_SETTINGS,
    STUB_NODES,
    FakeSleap,
    FakeSleapFrames,
    ReadbackRecord,
    SleapInput,
    SleapTrackCall,
    SlpWrite,
    StubFrame,
    StubInstance,
    StubSavedLabels,
    install_fake_sleap,
    install_fake_sleap_frames,
    read_stub_labels,
    record_slp_writes,
    sleap_instance,
    write_appearance_readback,
    write_sleap_model,
    write_sleap_model_set,
    write_stub_sleap_io,
    write_stub_slp,
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
    "HEADING_STORAGE",
    "LOOSE",
    "SPEED_STORAGE",
    "OldRuns",
    "act_after_judging",
    "end_attempt",
    "kept_reasons",
    "managed_dataset",
    "observe_removals",
    "planned_run_ids",
    "reasons_by_run",
    "runs_by_id",
    "ANGULAR_VELOCITY",
    "GAP_SECONDS",
    "SPEED",
    "STEP_SECONDS",
    "VELOCITY",
    "dropped_frame_times",
    "dropped_frame_tracks",
    "mean_rate",
    "CHAIN_RECIPE",
    "CONVERT_LABELS_STEP",
    "GUIDE_CLASSES",
    "GUIDE_PAGES",
    "guide_blocks",
    "guide_commands",
    "guide_dataset",
    "guide_recipe",
    "run_ids",
    "step_scopes",
    "step_states",
    "with_params",
    "CONSUMER_OWNED",
    "CONVERT_TRACKS_STEP",
    "Counted",
    "APPLYING_NAME",
    "APPLYING_SECOND_SOURCE",
    "APPLYING_SOURCE",
    "APPLYING_STORAGE",
    "CANARY_NAME",
    "CANARY_SOURCE",
    "CANARY_VARIABLE",
    "GLOBAL_NAME",
    "GLOBAL_SOURCE",
    "MINIMAL_NAME",
    "MINIMAL_SOURCE",
    "MISREADING_NAME",
    "MISREADING_SOURCE",
    "NARROWING_NAME",
    "NARROWING_SOURCE",
    "PARALLEL_APPLYING_NAME",
    "PARALLEL_APPLYING_SOURCE",
    "POSE_NAME",
    "POSE_SOURCE",
    "applying_instance",
    "built_by_class",
    "defined_dataset",
    "definitions_forgotten",
    "distances_from_origin",
    "fit_over",
    "fit_result",
    "forget_imported_definitions",
    "imported_definitions",
    "noting_definition_loads",
    "plan_refused",
    "CHILD_TIMEOUT_SECONDS",
    "EXIT_AFTER_MARK_SECONDS",
    "CLOCK_TICK_SECONDS",
    "CI_FERAL_MODULES",
    "CI_LIGHTNING_ACTION_MODULES",
    "CI_IDENTITY_MODULES",
    "CI_MCP_MODULES",
    "CI_REQUIRED_MODULES",
    "AlwaysFailsItsBuild",
    "CenteringFeature",
    "CountsItsBuilds",
    "FailsItsFirstBuild",
    "CropLike",
    "DIVISION_CREATED_UTC",
    "EVEN_STORE_TIMING",
    "EMBEDDING_TRACKER_SETTINGS",
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
    "FitControl",
    "FittingFeature",
    "ScopedFittingFeature",
    "LABELS_CLASSIFIER_STEP",
    "LabelsClassifier",
    "labels_classifier_registered",
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
    "LabeledFeature",
    "LabeledTracksRun",
    "MIXED",
    "ROI_VALUES",
    "STEADY",
    "TIME_SENTENCE",
    "LISTED_FRAMES",
    "PREDICTIONS_VARIANT",
    "LightningActionRecorder",
    "ListedEntries",
    "ListedEntriesParams",
    "StudyEntry",
    "TemplatesStudy",
    "HOLD_DIRECTORY_VARIABLE",
    "HoldsBC",
    "HoldsUntilReleased",
    "LineChildren",
    "PoolCallEnd",
    "LitposeSnippetRun",
    "LossyFeature",
    "MOUSE",
    "PairClusterFeature",
    "ParallelFeature",
    "PlainFeature",
    "REPO_ROOT",
    "REVISION",
    "RUNNER_LINE_SECONDS",
    "ReachedImport",
    "ReadbackRecord",
    "RecordedRequest",
    "RecordingStarter",
    "Reply",
    "RunnerJournal",
    "SELF_REFERENCE",
    "SEPARATED_TOOLS",
    "SOURCE_ROOT",
    "STUB_NODES",
    "SYNTHETIC_ARENA_3D",
    "SYNTHETIC_DROPOUTS",
    "CALMS21_NAMES",
    "CiJob",
    "ScopedFeature",
    "SeparatedEnvironment",
    "SeparatedTool",
    "SecondsEntries",
    "SharedKeyRefusingEntries",
    "SignalledGroup",
    "SleapInput",
    "SleapTrackCall",
    "SlpWrite",
    "StubFrame",
    "StubInstance",
    "StubSavedLabels",
    "TESTS_ROOT",
    "TracksAxisEntries",
    "ULTRALYTICS_KEYPOINTS",
    "ULTRALYTICS_PROTOCOL",
    "ULTRALYTICS_TOOL",
    "VAME_PROTOCOL",
    "VAME_RUNNER",
    "VAME_TOOL",
    "ExampleFixture",
    "ExampleServer",
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
    "GeneratedArchive",
    "IndexReads",
    "MOTIF_SYNC_UUID",
    "MakeStore",
    "MediaClip",
    "MockDataset",
    "PathState",
    "UPDATE_GOLDEN_ENV",
    "add_feature_run",
    "add_labels",
    "add_predictions",
    "scan_predictions_tables",
    "add_media_sequence",
    "add_media_variant",
    "add_raw_track_files",
    "add_track_sequences",
    "add_tracks_table",
    "add_tracks_variant",
    "add_tracks_variant_in_bulk",
    "convert_tracks_from_raw",
    "add_transcode_derivative",
    "appearance_table",
    "appender_program",
    "archive_path",
    "calms21_dataset",
    "cdn_path",
    "chain_recipe",
    "chain_with_sibling_recipe",
    "checksums_path",
    "device_member",
    "classifier_recipe",
    "converting_recipe",
    "assert_latents_at_centers",
    "assert_no_literal_tilde",
    "assert_marked_child_ends",
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
    "imports_reaching",
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
    "report_of",
    "written_by_worker",
    "WriteOutput",
    "wrapped_write",
    "entry_predictions",
    "entry_selection",
    "count_process_tree_walks",
    "example_fixture",
    "hardlink_member",
    "exited_pid",
    "kill_marked",
    "kill_running",
    "listed_pids",
    "extract_entry_camera",
    "fail_calms21_sequence",
    "fake_tool_probe",
    "finish_media_variant",
    "image_mark",
    "index_a_second_origin",
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
    "hide_package",
    "install_lightning_action_stub",
    "lightning_action_params",
    "invoke_json",
    "invoke_ok",
    "grown_run",
    "install_fake_ultralytics",
    "install_ultralytics_config",
    "isolate_tool_location",
    "journal_source",
    "keypoint_rows",
    "kpms_params",
    "labeled_run",
    "load_results_after_each_failed_write",
    "latest_events",
    "latest_snapshot",
    "logged_events",
    "make_dataset",
    "make_pair_df",
    "make_sequence_df",
    "make_templates",
    "minimal_op_params",
    "mirrored_rows",
    "model_reference",
    "FITTED_AGAIN",
    "KMEANS",
    "KMeansChain",
    "KMeansFit",
    "TEMPLATES_RECIPE",
    "drifted",
    "drifted_templates",
    "index_cells",
    "kmeans_apply",
    "kmeans_apply_feature",
    "kmeans_fit",
    "kmeans_fit_and_apply",
    "model_cells",
    "move_tracks",
    "reconvert_entry",
    "refit_kmeans",
    "refusals",
    "run_step",
    "templates_pipeline",
    "templates_recipe",
    "missing_ffmpeg_tools",
    "modules_loaded_by",
    "named_recipe",
    "narrowed_params",
    "nameless_store_dataset",
    "names_called_by",
    "package_of",
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
    "quarantine_entry",
    "read_frame_code",
    "record_entry_predictions",
    "record_entry_selection",
    "record_score_run",
    "record_frames_row",
    "record_slp_writes",
    "read_golden",
    "read_string_golden",
    "read_stub_labels",
    "record_variant_recipe",
    "regenerate_command",
    "register_sleap_model",
    "register_trained_model",
    "require_ffmpeg",
    "requirements_reaching",
    "count_property_reads",
    "observed",
    "raise_high_water_mark_once_applied",
    "record_path_calls",
    "resolved_scope",
    "result_outputs",
    "result_run_root",
    "record_published_files",
    "revision_file",
    "rewrite_as_older_prepared_index",
    "rewrite_index",
    "rule_reaching_op_params",
    "pool_call",
    "process_alive",
    "read_line",
    "refuse_index",
    "run_child",
    "run_child_to_files",
    "run_over_edited_tracks",
    "record_applied_sequences",
    "constant_time_in_mixed",
    "without_time",
    "without_time_in_mixed",
    "run_over_labeled_tracks",
    "stepped_individual",
    "run_fresh",
    "run_id_digest",
    "record_open_attempt",
    "run_id_of",
    "run_speed",
    "runs_in_an_external_environment",
    "sandbox_home",
    "scope_over",
    "scored_classifier_recipe",
    "self_referenced_extras",
    "serve_example",
    "serve_example_server",
    "set_tracks_cell",
    "signal_group_when",
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
    "sums_text",
    "symlink_member",
    "tar_gz",
    "tar_gz_with",
    "updating_golden",
    "video_store_maker",
    "wait_for",
    "TWO_CARDS",
    "install_fake_nvidia_smi",
    "CountingEntryPoint",
    "FailingEntryPoint",
    "install_entry_points",
    "run_with_distribution",
    "isolate_card_leases",
    "write_reader_params",
    "window_centers",
    "world_track_table",
    "WORLD_POINTS_FORMAT",
    "WorldPointsConverter",
    "raw_world_dataset",
    "registered_world_points",
    "track_sequences",
    "synthetic_dataset",
    "track_table",
    "tag_table",
    "CONVERTED_ENTRIES",
    "SOURCE_ENTRIES",
    "SUPERSEDED_RECIPE",
    "converted_source",
    "copy_of",
    "feature_edge",
    "file_digests",
    "labelled_source",
    "link_forms",
    "portable_columns",
    "raw_source",
    "take_link_offline",
    "tracks_edge",
    "write_run_params",
    "tree_snapshot",
    "ultralytics_probe_response",
    "unpickling_member",
    "write_data_yaml",
    "write_tags",
    "write_appearance_readback",
    "write_calms21_npy",
    "write_dlc_csv",
    "write_lineage_run",
    "write_listed_entries",
    "write_predictions",
    "write_sparse_labels",
    "write_templates_study",
    "write_driver_lease",
    "write_fake_runner",
    "write_frames_run",
    "write_golden",
    "write_division_rows",
    "write_h264_mp4",
    "write_litpose_model",
    "write_media_index",
    "write_marked_images",
    "write_mpeg4_mp4",
    "write_painted_entry",
    "write_painted_recording",
    "write_pv_header",
    "write_sdannce_mat",
    "write_sleap_analysis_h5",
    "write_sleap_model",
    "write_sleap_model_set",
    "write_stub_sleap_io",
    "write_stub_slp",
    "write_synthetic_spec",
    "write_templates",
    "write_tracktor_recording",
    "write_trex_npz",
    "write_window_spec",
    "write_zebrafish_3d_csv",
    "TRACKTOR_STEM",
    "TracktorRecording",
    "write_ultralytics_predictions",
    "RECIPE_SAVED_BY_0_17",
    "recipe_saved_by_0_17",
]
