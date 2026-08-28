"""Shared builders for the test suite.

**The only public path.** A test writes ``from tests.helpers import X`` and never
reaches into a submodule, so a helper can move between modules here without
touching a call site. Two spellings used to coexist -- ``from .conftest import``
and ``from tests.conftest import`` -- which read as "neither is the sanctioned
one"; this is the sanctioned one.

What lives where:

- ``datasets`` -- the `Dataset` a test runs against.
- ``features`` -- the templates and per-sequence frames the global
  fit-then-apply features are tested on.
- ``tracks`` -- track tables, tracks variants, raw TREx exports.
- ``media`` -- media files, media-index rows, transcode derivatives.
- ``ops`` -- the smallest params dict that validates for each registered op.
- ``scope`` -- a resolved scope over named entries, for the ops and drivers
  that take their coverage as an argument.
- ``environment`` -- what the surrounding machine provides: the ffmpeg
  toolchain, and which files under the package root a structural walk should
  skip -- installed third-party code, and mosaic's own code that runs in an
  environment built for an external tool.
- ``mock_dataset`` -- the duck-typed stand-in, for the pipeline tests that want
  no real roots.
- ``source_scan`` -- reads a module's source as a tree, for the tests that
  assert what a code path reads and what it calls.

Fixtures stay in ``tests/conftest.py``, because pytest collects them only from
there. Their bodies delegate here, so the logic has one home either way.
"""

from __future__ import annotations

from tests.helpers.datasets import make_dataset
from tests.helpers.environment import (
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
from tests.helpers.media import (
    MediaClip,
    add_media_sequence,
    add_transcode_derivative,
    clean_facts_cells,
    write_media_index,
    write_mpeg4_mp4,
)
from tests.helpers.mock_dataset import MockDataset
from tests.helpers.ops import minimal_op_params
from tests.helpers.scope import resolved_scope, scope_over
from tests.helpers.source_scan import (
    functions_named,
    module_tree,
    names_called_by,
    names_read,
    source_tree,
)
from tests.helpers.training import FakeTrainer, healthy_probe
from tests.helpers.tracks import (
    add_track_sequences,
    add_tracks_variant,
    track_sequences,
    write_trex_npz,
)

__all__ = [
    "FFMPEG_TOOLCHAIN",
    "FakeTrainer",
    "MediaClip",
    "MockDataset",
    "add_media_sequence",
    "add_track_sequences",
    "add_tracks_variant",
    "add_transcode_derivative",
    "assert_no_literal_tilde",
    "healthy_probe",
    "clean_facts_cells",
    "functions_named",
    "inside_a_virtualenv",
    "make_dataset",
    "make_pair_df",
    "make_sequence_df",
    "make_templates",
    "minimal_op_params",
    "missing_ffmpeg_tools",
    "module_tree",
    "names_called_by",
    "names_read",
    "require_ffmpeg",
    "resolved_scope",
    "runs_in_an_external_environment",
    "sandbox_home",
    "scope_over",
    "source_tree",
    "track_sequences",
    "write_media_index",
    "write_mpeg4_mp4",
    "write_templates",
    "write_trex_npz",
]
