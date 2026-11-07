"""The CalMS21 label converter reads a pickled `.npy` only when pickles are trusted.

The suite trusts them by default (`tests/conftest.py`); each test here removes that.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mosaic.behavior.label_library.calms21_behavior import (
    CalMS21BehaviorConverter,
    CalMS21BehaviorParams,
)
from mosaic.core.track_library.safe_load import (
    TRUST_VARIABLE,
    PickledFormatNotTrustedError,
)
from tests.helpers import make_dataset, write_calms21_npy


@pytest.fixture
def untrusted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(TRUST_VARIABLE, raising=False)


def test_the_converter_refuses_an_untrusted_npy_and_names_it(
    tmp_path: Path, untrusted: None
) -> None:
    npy = tmp_path / "labels.npy"
    write_calms21_npy(npy, {"annotator": {"s0": 4}})

    with pytest.raises(PickledFormatNotTrustedError, match="labels.npy"):
        _ = CalMS21BehaviorConverter().convert(
            npy, CalMS21BehaviorParams(), {"group": ""}
        )


def test_a_refused_file_fails_the_whole_label_conversion(
    tmp_path: Path, untrusted: None
) -> None:
    """Label conversion has no per-file catch, so a refusal ends the run, named."""
    ds = make_dataset(tmp_path / "ds", save=False, ensure_roots=False)
    uploads = tmp_path / "uploads"
    write_calms21_npy(uploads / "a_first.npy", {"g": {"s0": 4}})
    write_calms21_npy(uploads / "b_second.npy", {"g": {"s1": 4}})
    ds.index_labels_raw([uploads], patterns=["*.npy"], src_format="calms21_npy")

    with pytest.raises(PickledFormatNotTrustedError, match=r"a_first\.npy"):
        _ = ds.convert_all_labels(kind="behavior", source_format="calms21_npy")

    assert not list(ds.get_root("labels").rglob("*.npz"))
