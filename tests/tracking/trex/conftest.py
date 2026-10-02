"""Fixtures the TREx suites share."""

from __future__ import annotations

from pathlib import Path

import pytest

from mosaic.core.dataset import Dataset
from tests.helpers import FakeTrex, install_fake_trex, stub_media_dataset


@pytest.fixture
def ds(tmp_path: Path) -> Dataset:
    """A dataset with one sequence, ``vid1``, backed by ``vid1.mp4``."""
    return stub_media_dataset(tmp_path, ["vid1"])


@pytest.fixture
def trex(monkeypatch: pytest.MonkeyPatch) -> FakeTrex:
    """A recording stand-in for TREx, installed over both of its phases."""
    return install_fake_trex(monkeypatch)
