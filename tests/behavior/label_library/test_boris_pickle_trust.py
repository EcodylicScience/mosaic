"""The BORIS pickle label converter reads a pickle only when pickles are trusted.

The suite trusts them by default (`tests/conftest.py`); each test here sets the
variable itself.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
import pytest

from mosaic.behavior.label_library.boris_pandas_pickle import (
    BorisPandasPickleConverter,
    BorisPandasPickleParams,
)
from mosaic.core.track_library.safe_load import (
    TRUST_VARIABLE,
    PickledFormatNotTrustedError,
)
from tests.helpers import unpickling_member


def _convert(path: Path) -> int:
    entries = BorisPandasPickleConverter().convert(
        path, BorisPandasPickleParams(), {"group": ""}
    )
    return len(entries)


def test_an_untrusted_pickle_is_refused_unread_and_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(TRUST_VARIABLE, raising=False)
    marker = tmp_path / "marker"
    path = tmp_path / "boris.pkl"
    _ = path.write_bytes(pickle.dumps(unpickling_member(marker)))

    with pytest.raises(PickledFormatNotTrustedError, match=r"boris\.pkl") as caught:
        _ = _convert(path)

    assert "aggregated CSV" in str(caught.value)
    assert not marker.exists(), "the converter unpickled the file"


def test_a_trusted_pickle_converts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(TRUST_VARIABLE, "1")
    path = tmp_path / "boris.pkl"
    pd.DataFrame(
        {
            "Observation id": ["obs"],
            "Subject": ["ant"],
            "Behavior": ["approach"],
            "Start (s)": [0.0],
            "Stop (s)": [0.5],
            "FPS": [2.0],
        }
    ).to_pickle(path)

    assert _convert(path) == 1
