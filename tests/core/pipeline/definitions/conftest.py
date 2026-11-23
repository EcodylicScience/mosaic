"""Fixtures of the definition tests."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from tests.helpers import definitions_forgotten


@pytest.fixture(autouse=True)
def forget_loaded_definitions() -> Iterator[None]:
    """Remove the definition modules a test imported, and the caches it filled."""
    with definitions_forgotten():
        yield
