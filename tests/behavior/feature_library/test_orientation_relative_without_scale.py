"""``orientation-rel`` runs without a body-scale run, as its ``scale`` description says."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mosaic.behavior.feature_library.heading import HeadingFeature
from mosaic.behavior.feature_library.orientation_relative import (
    OrientationRelativeFeature,
)
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.run import run_feature
from mosaic.core.pipeline.types import BodyScaleResult, Result
from tests.helpers import add_tracks_variant, make_dataset

VARIANT = "v1"


@pytest.fixture
def tracked(tmp_path: Path) -> Dataset:
    """Two tracked sequences with a heading run and no body-scale run."""
    ds = make_dataset(tmp_path / "tracked", name="tracked")
    add_tracks_variant(ds, VARIANT, "seq_a", "seq_b", std_format="mosaic_v1")
    return ds


def _orientation(ds: Dataset, **params: object) -> OrientationRelativeFeature:
    """The feature reading the tracks beside a heading run of *ds*."""
    heading = run_feature(ds, HeadingFeature(), track=False)
    return OrientationRelativeFeature(
        inputs=OrientationRelativeFeature.Inputs(
            ("tracks", Result(feature=heading.feature, run_id=heading.run_id))
        ),
        params=params,
    )


def test_a_run_without_a_body_scale_run_falls_back_to_each_rows_own_spread(
    tracked: Dataset,
) -> None:
    result = run_feature(tracked, _orientation(tracked), track=False)

    run_root = tracked.get_root("features") / result.feature / str(result.run_id)
    outputs = [pd.read_parquet(path) for path in sorted(run_root.glob("*.parquet"))]
    assert len(outputs) == 2
    assert all(not output.empty for output in outputs)


def test_a_scale_reference_to_a_run_that_is_absent_still_fails(
    tracked: Dataset,
) -> None:
    feature = _orientation(tracked, scale=BodyScaleResult(run_id="0.1-missing"))

    with pytest.raises(FileNotFoundError):
        _ = run_feature(tracked, feature, track=False)
