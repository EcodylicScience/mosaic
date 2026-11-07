"""A CalMS21 `.npy` is a pickled dict, so it loads only when pickles are trusted.

The suite trusts them by default (`tests/conftest.py`); each test here removes that.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mosaic.core.track_converter import EntryHints
from mosaic.core.track_library.calms21 import Calms21Converter, Calms21Params
from mosaic.core.track_library.helpers import load_calms21
from mosaic.core.track_library.safe_load import (
    TRUST_VARIABLE,
    PickledFormatNotTrustedError,
    UnsafeRawFileError,
)
from tests.helpers import unpickling_member, write_calms21_npy


@pytest.fixture
def untrusted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(TRUST_VARIABLE, raising=False)


def test_an_untrusted_npy_is_refused_by_name(tmp_path: Path, untrusted: None) -> None:
    npy = tmp_path / "calms21.npy"
    write_calms21_npy(npy, {"train": {"s0": 4}})

    with pytest.raises(PickledFormatNotTrustedError, match="calms21.npy") as caught:
        _ = load_calms21(npy)

    assert caught.value.path == npy
    assert TRUST_VARIABLE in str(caught.value)
    assert ".json" in str(caught.value)


def test_an_untrusted_npy_is_not_unpickled(tmp_path: Path, untrusted: None) -> None:
    marker = tmp_path / "ran"
    npy = tmp_path / "hostile.npy"
    np.save(npy, unpickling_member(marker), allow_pickle=True)

    with pytest.raises(PickledFormatNotTrustedError):
        _ = load_calms21(npy)

    assert not marker.exists()


def test_a_trusted_npy_loads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(TRUST_VARIABLE, "1")
    npy = tmp_path / "calms21.npy"
    write_calms21_npy(npy, {"train": {"s0": 4}})

    assert list(load_calms21(npy)["train"]) == ["s0"]


def test_a_flag_set_to_zero_does_not_trust(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(TRUST_VARIABLE, "0")
    npy = tmp_path / "calms21.npy"
    write_calms21_npy(npy, {"train": {"s0": 4}})

    with pytest.raises(PickledFormatNotTrustedError):
        _ = load_calms21(npy)


def test_json_loads_without_the_flag(tmp_path: Path, untrusted: None) -> None:
    path = tmp_path / "calms21.json"
    path.write_text(json.dumps({"train": {"s0": {"keypoints": []}}}))

    assert load_calms21(path) == {"train": {"s0": {"keypoints": []}}}


@pytest.mark.parametrize("operation", ["convert", "enumerate"])
def test_the_track_converter_refuses_an_untrusted_npy_and_names_it(
    operation: str, tmp_path: Path, untrusted: None
) -> None:
    npy = tmp_path / "calms21.npy"
    write_calms21_npy(npy, {"train": {"s0": 4}})
    converter = Calms21Converter()

    with pytest.raises(UnsafeRawFileError, match="calms21.npy"):
        if operation == "convert":
            _ = converter.convert(npy, Calms21Params(), EntryHints())
        else:
            _ = converter.enumerate_sequences(npy)


@pytest.mark.parametrize("pickled", [[1, 2], [3], "text"], ids=["pair", "one", "str"])
def test_a_trusted_npy_that_is_not_a_dict_is_refused_by_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pickled: object
) -> None:
    monkeypatch.setenv(TRUST_VARIABLE, "1")
    npy = tmp_path / "not_a_dict.npy"
    np.save(npy, np.array(pickled, dtype=object), allow_pickle=True)

    with pytest.raises(ValueError, match=r"not_a_dict\.npy"):
        _ = load_calms21(npy)
