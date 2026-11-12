"""The graph orders a label conversion after the op kinds it lists as writing tracks.

The list is a constant so that ordering a recipe reads no registry. It has to name
exactly the registered ops that declare they write a tracks variant.
"""

from __future__ import annotations

from mosaic.core.pipeline.graph.resolve import op_class_for_kind, op_writes_tracks
from mosaic.core.pipeline.graph.topo import TRACK_WRITING_OPS
from mosaic.core.pipeline.ops import OPS


def test_the_listed_kinds_are_the_registered_ops_that_write_tracks() -> None:
    _ = op_class_for_kind("convert-tracks")

    registered = {kind for kind, op in OPS.items() if op_writes_tracks(kind, op)}

    assert TRACK_WRITING_OPS == registered
