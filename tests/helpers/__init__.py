"""Shared builders for the test suite.

**The only public path.** A test writes ``from tests.helpers import X`` and never
reaches into a submodule, so a helper can move between modules here without
touching a call site. Two spellings used to coexist -- ``from .conftest import``
and ``from tests.conftest import`` -- which read as "neither is the sanctioned
one"; this is the sanctioned one.

What lives where:

- ``datasets`` -- the `Dataset` a test runs against.
- ``tracks`` -- track tables, tracks variants, raw TREx exports.
- ``media`` -- media files, media-index rows, transcode derivatives.
- ``environment`` -- what the surrounding machine provides (the ffmpeg toolchain).
- ``mock_dataset`` -- the duck-typed stand-in, for the pipeline tests that want
  no real roots.

Fixtures stay in ``tests/conftest.py``, because pytest collects them only from
there. Their bodies delegate here, so the logic has one home either way.
"""

from __future__ import annotations

from tests.helpers.datasets import make_dataset
from tests.helpers.environment import (
    FFMPEG_TOOLCHAIN,
    missing_ffmpeg_tools,
    require_ffmpeg,
)
from tests.helpers.media import (
    add_media_sequence,
    add_transcode_derivative,
    clean_facts_cells,
    write_media_index,
)
from tests.helpers.mock_dataset import MockDataset
from tests.helpers.tracks import (
    add_track_sequences,
    add_tracks_variant,
    track_sequences,
    write_trex_npz,
)

__all__ = [
    "FFMPEG_TOOLCHAIN",
    "MockDataset",
    "add_media_sequence",
    "add_track_sequences",
    "add_tracks_variant",
    "add_transcode_derivative",
    "clean_facts_cells",
    "make_dataset",
    "missing_ffmpeg_tools",
    "require_ffmpeg",
    "track_sequences",
    "write_media_index",
    "write_trex_npz",
]
