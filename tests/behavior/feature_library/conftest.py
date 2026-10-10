"""Fixtures shared by the feature library's tests."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def scratch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Return the temporary directory of this process during the test.

    A feature that creates its exchange directory and its runner's log under
    the system's temporary directory creates them here, where a test checks
    that they were removed.
    """
    directory = tmp_path / "scratch"
    directory.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(directory))
    return directory
