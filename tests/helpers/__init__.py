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
- ``features`` -- the templates and per-sequence frames the global
  fit-then-apply features are tested on.
- ``stand_in_features`` -- features that do nothing but declare the source
  roots they consume, for the tests of what a source change reaches.
- ``tracks`` -- track tables, tracks variants, raw TREx, SLEAP and DeepLabCut
  exports.
- ``trex``, ``sleap``, ``litpose`` and ``ultralytics`` contain recording
  stand-ins for the tools that those trackers run, installed over each tracker's
  seams, and ``inference`` the same for the environments of ``infer-pose`` and
  ``infer-points``.
- ``decode_probe`` contains a fake ``python`` for a tool environment, which stands
  in for the interpreter that runs the codec check's decode probe.
- ``media`` -- media files, media-index rows, transcode derivatives.
- ``variants`` -- media variant files and their index rows, and a count of the
  index reads a variant's consumers make.
- ``ops`` -- the smallest params dict that validates for each registered op.
- ``runlog`` -- an attempt's recorded run-log events.
- ``scope`` -- a resolved scope over named entries, for the ops and drivers
  that take their coverage as an argument.
- ``environment`` -- what the surrounding machine provides: the ffmpeg
  toolchain, the modules each CI job must install, and which files under the
  package root a structural walk should skip -- installed third-party code, and
  mosaic's own code that runs in an environment built for an external tool.
- ``mock_dataset`` -- the duck-typed stand-in, for the pipeline tests that want
  no real roots.
- ``source_scan`` -- reads a module's source as a tree, for the tests that
  assert what a code path reads and what it calls.
- ``paths`` -- the repository, the ``tests`` directory and the golden-file
  directory, which a test names from here rather than from its own location.
- ``golden`` -- reading and rewriting the golden files under ``tests/data/``.

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
from tests.helpers.datasets import make_dataset
from tests.helpers.decode_probe import FakeToolPython, install_fake_tool_python
from tests.helpers.environment import (
    CI_FERAL_MODULES,
    CI_IDENTITY_MODULES,
    CI_REQUIRED_MODULES,
    FFMPEG_TOOLCHAIN,
    assert_no_literal_tilde,
    inside_a_virtualenv,
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
    paint_frame_code,
    point_at_a_store,
    read_frame_code,
    store_dataset,
    stub_join,
    video_store_maker,
    write_h264_mp4,
    write_media_index,
    write_mpeg4_mp4,
    write_painted_entry,
)
from tests.helpers.mock_dataset import MockDataset
from tests.helpers.models import register_trained_model
from tests.helpers.ops import minimal_op_params
from tests.helpers.paths import (
    GOLDEN_DIR,
    REPO_ROOT,
    TESTS_ROOT,
)
from tests.helpers.runlog import entry_error_lines, latest_events, latest_snapshot
from tests.helpers.scope import resolved_scope, scope_over
from tests.helpers.source_scan import (
    functions_named,
    module_tree,
    names_called_by,
    names_read,
    source_tree,
)
from tests.helpers.documents import dotted_values, is_section
from tests.helpers.stand_in_features import (
    CropLike,
    PlainFeature,
)
from tests.helpers.training import (
    FakeTrainer,
    healthy_probe,
    write_data_yaml,
)
from tests.helpers.tracks import (
    add_track_sequences,
    add_tracks_variant,
    published_table,
    set_tracks_cell,
    track_sequences,
    write_dlc_csv,
    write_sleap_analysis_h5,
    write_trex_npz,
)
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
from tests.helpers.litpose import (
    FakeLitpose,
    install_fake_litpose,
    write_litpose_model,
)
from tests.helpers.sleap import FakeSleap, install_fake_sleap, write_sleap_model
from tests.helpers.trex import (
    FakeTrex,
    install_fake_trex,
    write_pv_header,
)
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
    "FFMPEG_TOOLCHAIN",
    "FakeDetections",
    "FakeResult",
    "GOLDEN_DIR",
    "KEYPOINTS_PAYLOAD",
    "MOUSE",
    "PlainFeature",
    "REPO_ROOT",
    "TESTS_ROOT",
    "ULTRALYTICS_KEYPOINTS",
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
    "add_tracks_variant",
    "add_transcode_derivative",
    "assert_no_literal_tilde",
    "golden_path",
    "healthy_probe",
    "clean_facts_cells",
    "clip_facts",
    "count_index_reads",
    "dot_image",
    "dotted_values",
    "entry_error_lines",
    "finish_media_variant",
    "functions_named",
    "index_media_sequence",
    "inside_a_virtualenv",
    "install_fake_litpose",
    "install_fake_point_inference",
    "install_fake_point_probe",
    "install_fake_pose_inference",
    "install_fake_pose_probe",
    "install_fake_sleap",
    "install_fake_tool_python",
    "install_fake_trex",
    "install_fake_ultralytics",
    "is_section",
    "latest_events",
    "latest_snapshot",
    "make_dataset",
    "make_pair_df",
    "make_sequence_df",
    "make_templates",
    "minimal_op_params",
    "missing_ffmpeg_tools",
    "module_tree",
    "names_called_by",
    "names_read",
    "paint_frame_code",
    "point_at_a_store",
    "point_predictions",
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
    "resolved_scope",
    "revision_file",
    "runs_in_an_external_environment",
    "sandbox_home",
    "scope_over",
    "set_tracks_cell",
    "source_tree",
    "store_dataset",
    "stub_join",
    "updating_golden",
    "video_store_maker",
    "track_sequences",
    "ultralytics_probe_response",
    "write_data_yaml",
    "write_dlc_csv",
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
