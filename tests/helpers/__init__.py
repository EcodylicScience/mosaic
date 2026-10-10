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
- ``catalogs``: the declaration catalogs a graph test validates or plans against.
- ``cli`` -- a ``mosaic`` command run for its ``--json`` output.
- ``features`` -- the templates and per-sequence frames the global
  fit-then-apply features are tested on.
- ``latents`` names the frames at the centers of VAME's windows, and checks that
  a vame output table places a finite latent on each.
- ``column_contract`` checks a feature's output against the columns that
  validation predicts from its declaration.
- ``kpms`` contains the params of a kpms feature on three named keypoints.
- ``stand_in_features`` -- features that do nothing but declare the source
  roots they consume, for the tests of what a source change reaches.
- ``tracks`` contains track tables, keypoint rows, tracks variants, and raw
  TREx, SLEAP and DeepLabCut exports.
- ``trex``, ``sleap``, ``litpose`` and ``ultralytics`` contain recording
  stand-ins for the tools that those trackers run, installed over each tracker's
  seams, and ``inference`` the same for the environments of ``infer-pose`` and
  ``infer-points``.
- ``decode_probe`` contains a fake ``python`` for a tool environment, which stands
  in for the interpreter that runs the codec check's decode probe.
- ``runner_programs`` imports a runner program under its bare name for one test,
  writes fake runners that serve a test's handlers over the runner wire, and
  checks that a runner program answers ``Ready`` and ``shutdown``.
- ``processes`` contains a child program that appends to a file, checks of
  whether a process stopped writing to it or exited, the line children of a
  test, and a cancel fired once a condition is true.
- ``media`` -- media files, media-index rows, transcode derivatives.
- ``variants`` -- media variant files and their index rows, and a count of the
  index reads a variant's consumers make.
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
- ``golden`` -- reading and rewriting the golden files under ``tests/data/``.
- ``recipes``: the recipes a graph test validates, plans or runs.

Fixtures stay in ``tests/conftest.py``, because pytest collects them only from
there. Their bodies delegate here, so the logic has one home either way.
"""

from __future__ import annotations

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
from tests.helpers.cli import invoke_json
from tests.helpers.datasets import make_dataset
from tests.helpers.decode_probe import FakeToolPython, install_fake_tool_python
from tests.helpers.environment import (
    CI_FERAL_MODULES,
    CI_IDENTITY_MODULES,
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
from tests.helpers.kpms import kpms_params
from tests.helpers.latents import assert_latents_at_centers, window_centers
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
    MOTIF_SYNC_UUID,
    MakeStore,
    MediaClip,
    add_media_sequence,
    add_transcode_derivative,
    clean_facts_cells,
    clip_facts,
    dot_image,
    index_media_sequence,
    index_session,
    paint_frame_code,
    point_at_a_store,
    read_frame_code,
    store_dataset,
    stub_join,
    stub_media_dataset,
    video_store_maker,
    write_h264_mp4,
    write_media_index,
    write_mpeg4_mp4,
    write_painted_entry,
)
from tests.helpers.mock_dataset import MockDataset
from tests.helpers.processes import (
    LineChildren,
    appender_program,
    assert_process_exited,
    assert_stopped_writing,
    cancel_when,
    wait_for,
)
from tests.helpers.recipes import GUPPIES_RECIPE, guppies_recipe
from tests.helpers.models import register_trained_model
from tests.helpers.ops import minimal_op_params
from tests.helpers.paths import (
    GOLDEN_DIR,
    KPMS_PROTOCOL,
    KPMS_SERVER,
    REPO_ROOT,
    SOURCE_ROOT,
    TESTS_ROOT,
    VAME_PROTOCOL,
    VAME_RUNNER,
)
from tests.helpers.runlog import entry_error_lines, latest_events, latest_snapshot
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
from tests.helpers.stand_in_features import CropLike, PlainFeature
from tests.helpers.training import FakeTrainer, healthy_probe, write_data_yaml
from tests.helpers.tracks import (
    add_track_sequences,
    add_tracks_table,
    add_tracks_variant,
    keypoint_rows,
    published_table,
    set_tracks_cell,
    track_sequences,
    track_table,
    write_dlc_csv,
    write_sleap_analysis_h5,
    write_trex_npz,
)
from tests.helpers.identity import run_id_digest
from tests.helpers.inference import (
    FakeInference,
    install_fake_point_inference,
    install_fake_point_probe,
    install_fake_pose_inference,
    install_fake_pose_probe,
    point_predictions,
    pose_per_frame,
    pose_predictions,
)
from tests.helpers.litpose import FakeLitpose, install_fake_litpose, write_litpose_model
from tests.helpers.sleap import FakeSleap, install_fake_sleap, write_sleap_model
from tests.helpers.trex import FakeTrex, install_fake_trex, write_pv_header
from tests.helpers.ultralytics import (
    FakeDetections,
    FakeResult,
    FakeUltralytics,
    ULTRALYTICS_KEYPOINTS,
    install_fake_ultralytics,
    ultralytics_probe_response,
    write_ultralytics_predictions,
)
from tests.helpers.variants import (
    IndexReads,
    add_media_variant,
    count_index_reads,
    finish_media_variant,
)

__all__ = [
    "CI_FERAL_MODULES",
    "CI_IDENTITY_MODULES",
    "CI_REQUIRED_MODULES",
    "CropLike",
    "EnvironmentMarker",
    "EventRecorder",
    "FAKE_CONDA",
    "FAKE_HANDLERS",
    "FFMPEG_TOOLCHAIN",
    "GUPPIES_RECIPE",
    "FakeDetections",
    "FakeResult",
    "GOLDEN_DIR",
    "ImportedName",
    "KEYPOINTS_PAYLOAD",
    "KPMS_PROTOCOL",
    "KPMS_SERVER",
    "KPMS_TOOL",
    "LineChildren",
    "MOUSE",
    "PlainFeature",
    "REPO_ROOT",
    "RUNNER_LINE_SECONDS",
    "RunnerJournal",
    "SELF_REFERENCE",
    "SEPARATED_TOOLS",
    "SOURCE_ROOT",
    "CiJob",
    "SeparatedEnvironment",
    "SeparatedTool",
    "TESTS_ROOT",
    "ULTRALYTICS_KEYPOINTS",
    "ULTRALYTICS_TOOL",
    "VAME_PROTOCOL",
    "VAME_RUNNER",
    "VAME_TOOL",
    "FakeInference",
    "FakeLitpose",
    "FakeSleap",
    "FakeTrainer",
    "FakeTrex",
    "FakeToolPython",
    "FakeUltralytics",
    "IndexReads",
    "MOTIF_SYNC_UUID",
    "MakeStore",
    "MediaClip",
    "MockDataset",
    "UPDATE_GOLDEN_ENV",
    "add_media_sequence",
    "add_media_variant",
    "add_track_sequences",
    "add_tracks_table",
    "add_tracks_variant",
    "add_transcode_derivative",
    "appender_program",
    "assert_latents_at_centers",
    "assert_no_literal_tilde",
    "assert_process_exited",
    "assert_ready_then_shutdown",
    "assert_stopped_writing",
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
    "dotted_values",
    "entry_error_lines",
    "finish_media_variant",
    "functions_named",
    "index_media_sequence",
    "index_session",
    "inside_a_virtualenv",
    "install_fake_litpose",
    "install_fake_point_inference",
    "install_fake_point_probe",
    "install_fake_pose_inference",
    "install_fake_pose_probe",
    "install_fake_sleap",
    "install_fake_tool_python",
    "install_fake_trex",
    "invoke_json",
    "install_fake_ultralytics",
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
    "names_called_by",
    "names_read",
    "paint_frame_code",
    "point_at_a_store",
    "point_predictions",
    "project_requirements",
    "pose_per_frame",
    "pose_predictions",
    "pose_frame",
    "pose_object",
    "pose_set",
    "published_table",
    "read_frame_code",
    "read_golden",
    "read_string_golden",
    "regenerate_command",
    "register_trained_model",
    "require_ffmpeg",
    "requirements_reaching",
    "resolved_scope",
    "result_outputs",
    "result_run_root",
    "revision_file",
    "run_id_digest",
    "runs_in_an_external_environment",
    "sandbox_home",
    "scope_over",
    "self_referenced_extras",
    "set_tracks_cell",
    "source_tree",
    "store_dataset",
    "stub_join",
    "stub_media_dataset",
    "updating_golden",
    "video_store_maker",
    "wait_for",
    "window_centers",
    "track_sequences",
    "track_table",
    "ultralytics_probe_response",
    "write_data_yaml",
    "write_dlc_csv",
    "write_fake_runner",
    "write_golden",
    "write_h264_mp4",
    "write_litpose_model",
    "write_media_index",
    "write_mpeg4_mp4",
    "write_painted_entry",
    "write_pv_header",
    "write_sleap_analysis_h5",
    "write_sleap_model",
    "write_templates",
    "write_trex_npz",
    "write_ultralytics_predictions",
]
