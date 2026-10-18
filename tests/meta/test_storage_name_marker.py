"""No unshortened storage name contains the shortened-name marker.

A storage name is built from feature slugs and the ``tracks`` literal. Cleanup
reads a ``~`` in a name of unknown params as an abbreviated lineage. That
reading is correct while neither source can contain one.
"""

from __future__ import annotations

from typing import get_args

from mosaic.behavior.feature_library.registry import FEATURES
from mosaic.core.pipeline.storage_names import SHORTENED_MARKER
from mosaic.core.pipeline.types import TrackInput


def test_no_feature_slug_contains_the_marker() -> None:
    assert [slug for slug in FEATURES if SHORTENED_MARKER in slug] == []


def test_no_literal_input_name_contains_the_marker() -> None:
    literals = get_args(TrackInput)
    assert literals == ("tracks",)
    assert [name for name in literals if SHORTENED_MARKER in name] == []
