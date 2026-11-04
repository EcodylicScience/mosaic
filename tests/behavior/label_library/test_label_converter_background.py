"""Each label converter records what a frame without an event means.

A reader used to assume class 0. In CalMS21 class 0 is ``attack``, and a BORIS
background sorted among the behaviors took an alphabetical id, so the frames nobody
annotated were read as a behavior. The converters now put the background at a known
id and write it into the file as ``background_id``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mosaic.behavior.label_library.boris_aggregated_csv import (
    BorisAggregatedCSVConverter,
    BorisAggregatedCSVParams,
)
from mosaic.behavior.label_library.boris_pandas_pickle import (
    BorisPandasPickleConverter,
    BorisPandasPickleParams,
)
from mosaic.behavior.label_library.calms21_behavior import (
    CalMS21BehaviorConverter,
    CalMS21BehaviorParams,
)
from mosaic.core.helpers import UNLABELED
from mosaic.core.label_converter import LabelEntry


def _boris_events() -> pd.DataFrame:
    """Two behaviors that sort on either side of the background, ``none``."""
    return pd.DataFrame(
        {
            "Observation id": ["obs", "obs", "obs"],
            "Subject": ["ant", "ant", "ant"],
            "Behavior": ["approach", "none", "zigzag"],
            "Start (s)": [0.0, 1.0, 2.0],
            "Stop (s)": [0.5, 1.5, 2.5],
            "FPS": [2.0, 2.0, 2.0],
        }
    )


def _convert_boris(kind: str, tmp_path: Path) -> list[LabelEntry]:
    if kind == "csv":
        path = tmp_path / "boris.tsv"
        _boris_events().to_csv(path, sep="\t", index=False)
        return BorisAggregatedCSVConverter().convert(
            path, BorisAggregatedCSVParams(), {"group": ""}
        )
    path = tmp_path / "boris.pkl"
    _boris_events().to_pickle(path)
    return BorisPandasPickleConverter().convert(
        path, BorisPandasPickleParams(), {"group": ""}
    )


def test_calms21_labels_declare_no_background(tmp_path: Path) -> None:
    """Every CalMS21 class is a behavior, so an unmatched row is unlabeled."""
    npy = tmp_path / "calms21.npy"
    # The CalMS21 file is a pickled dict, which numpy's stubs do not admit.
    np.save(
        npy,
        {  # pyright: ignore[reportArgumentType]
            "annotator-id_0": {
                "task1/train/m1": {
                    "keypoints": np.zeros((3, 2, 2, 7), dtype=float),
                    "annotations": np.array([0, 3, 1]),
                }
            }
        },
        allow_pickle=True,
    )

    [entry] = CalMS21BehaviorConverter().convert(
        npy, CalMS21BehaviorParams(), {"group": ""}
    )

    assert int(np.asarray(entry.payload["background_id"])) == UNLABELED
    assert CalMS21BehaviorConverter.version == "0.3"


@pytest.mark.parametrize("kind", ["csv", "pickle"])
def test_boris_background_is_id_zero_even_when_it_occurs_in_the_data(
    kind: str, tmp_path: Path
) -> None:
    """Sorted with the behaviors, ``none`` took id 1 here, after ``approach``."""
    [entry] = _convert_boris(kind, tmp_path)

    names = [str(name) for name in np.asarray(entry.payload["label_names"])]

    assert names == ["none", "approach", "zigzag"]
    assert entry.label_names == ("none", "approach", "zigzag")


@pytest.mark.parametrize("kind", ["csv", "pickle"])
def test_boris_writes_its_background_id(kind: str, tmp_path: Path) -> None:
    [entry] = _convert_boris(kind, tmp_path)

    assert int(np.asarray(entry.payload["background_id"])) == 0
    assert BorisAggregatedCSVConverter.version == "0.2"
    assert BorisPandasPickleConverter.version == "0.2"
