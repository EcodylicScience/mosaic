"""Fixtures of the definition tests."""

from __future__ import annotations

import sys
from collections.abc import Iterator

import pytest

from mosaic.core.pipeline.definitions.scope import clear_definition_caches


@pytest.fixture(autouse=True)
def forget_loaded_definitions() -> Iterator[None]:
    """Remove the definition modules a test imported, and the caches it filled.

    Two datasets that define one source share a module name. A module left in
    ``sys.modules`` answers a later test's load without reading its files.
    """
    before = set(sys.modules)
    clear_definition_caches()
    yield
    clear_definition_caches()
    for name in set(sys.modules) - before:
        if name.startswith("mosaic_definitions."):
            _ = sys.modules.pop(name)
