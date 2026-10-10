"""Read the run root and the outputs of a ``run_feature`` attempt from its Result."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mosaic.core.pipeline.index import DatasetRoots, feature_run_root
from mosaic.core.pipeline.types import Result


def result_run_root(ds: DatasetRoots, result: Result) -> Path:
    """Return the run root of the feature run that *result* names."""
    assert result.run_id is not None
    return feature_run_root(ds, result.feature, result.run_id)


def result_outputs(ds: DatasetRoots, result: Result) -> dict[str, pd.DataFrame]:
    """Return each output table of *result*'s run by its file's stem."""
    return {
        path.stem: pd.read_parquet(path)
        for path in result_run_root(ds, result).glob("*.parquet")
    }


def model_reference(result: Result) -> dict[str, object]:
    """Return the ``model`` parameter that names the model fitted by *result*'s run."""
    return {"feature": result.feature, "run_id": result.run_id}
