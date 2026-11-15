"""Fixtures the CLI suites share."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.definitions.scope import clear_definition_caches
from tests.helpers import (
    add_track_sequences,
    forget_imported_definitions,
    make_dataset,
)


@pytest.fixture
def dataset(tmp_path: Path) -> tuple[Path, Dataset]:
    """A real Dataset with two synthetic tracks (columns speed-angvel needs).

    Returned with its manifest, which is what a command is pointed at.
    """
    ds = make_dataset(tmp_path)
    add_track_sequences(ds, ("g", "s1"), ("g", "s2"), n_rows=12)
    manifest = ds.manifest_path
    return manifest, ds


@pytest.fixture
def fresh_definitions() -> Iterator[None]:
    """Start and end a test with no definition imported and no definition cached.

    A definition module that an earlier test left in ``sys.modules`` answers a
    later load without running its top level. A test that asserts nothing imported
    a definition then passes without testing the claim.
    """
    clear_definition_caches()
    forget_imported_definitions()
    yield
    clear_definition_caches()
    forget_imported_definitions()
