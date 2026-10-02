"""Fixtures the CLI suites share."""

from __future__ import annotations

from pathlib import Path

import pytest

from mosaic.core.dataset import Dataset
from tests.helpers import add_track_sequences, make_dataset


@pytest.fixture
def dataset(tmp_path: Path) -> tuple[Path, Dataset]:
    """A real Dataset with two synthetic tracks (columns speed-angvel needs).

    Returned with its manifest, which is what a command is pointed at.
    """
    ds = make_dataset(tmp_path)
    add_track_sequences(ds, ("g", "s1"), ("g", "s2"), n_rows=12)
    manifest = ds.manifest_path
    return manifest, ds
