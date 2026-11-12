"""Chains that read ``time`` plan, because each step before them keeps it.

``pair-wavelet`` and ``temporal-stack`` measure their frame rate from ``time``, so the
per-frame feature before them must write it. The CalMS21 template chains
``pair-posedistance-pca`` into ``pair-wavelet``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.json_value import JsonValue
from mosaic.core.pipeline.graph import plan_pipeline, read_recipe
from mosaic.core.pipeline.run import run_feature
from mosaic.behavior.feature_library.pairposedistancepca import PairPoseDistancePCA
from tests.helpers import add_tracks_variant, make_dataset

VARIANT = "convert-sleap.0.1-2222222222"


def _dataset(tmp_path: Path) -> Dataset:
    ds = make_dataset(tmp_path / "d")
    add_tracks_variant(
        ds, VARIANT, "seq_a", "seq_b", std_format="mosaic_v1", keypoints=7
    )
    return ds


CHAINS: list[tuple[str, dict[str, JsonValue]]] = [
    ("pair-wavelet", {}),
    ("temporal-stack", {}),
]


@pytest.mark.parametrize(
    ("second", "params"), CHAINS, ids=["pair-wavelet", "temporal-stack-defaults"]
)
def test_pair_posedistance_pca_plans_into_a_step_that_reads_time(
    tmp_path: Path, second: str, params: dict[str, JsonValue]
) -> None:
    ds = _dataset(tmp_path)
    pca: dict[str, JsonValue] = {
        "id": "pca",
        "type": "feature",
        "feature": "pair-posedistance-pca",
        "params": {},
    }
    after: dict[str, JsonValue] = {
        "id": "next",
        "type": "feature",
        "feature": second,
        "params": params,
        "inputs": [{"step": "pca"}],
    }
    recipe = read_recipe({"steps": [pca, after]})

    plan = plan_pipeline(ds, recipe)

    assert plan.problems == ()


def test_pair_posedistance_pca_writes_the_time_of_each_frame(tmp_path: Path) -> None:
    ds = _dataset(tmp_path)

    result = run_feature(ds, PairPoseDistancePCA(), track=False)

    run_root = ds.get_root("features") / result.feature / str(result.run_id)
    for path in sorted(run_root.glob("*.parquet")):
        table = pd.read_parquet(path)
        assert "time" in table.columns
        assert table["time"].notna().all()
