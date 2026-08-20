"""`lightning-action` must carry a pair row's identity onto its predictions.

The same defect `xgboost` carried until commit 250051e, in a feature that declares
`emits = "as-input"` and is routinely handed pair-level input: the passthrough list
stopped at `[frame, time, id, group, sequence]`, so the predictions came back with
two rows per `(group, sequence, frame)` and nothing to tell them apart. Its comment
said the output "matches XGBoost format"; once xgboost was fixed, it did not.

No test covered this feature's `apply` at all, because the extra it needs does not
resolve on a machine without CUDA -- `lightning-action` requires `nvidia-dali-cuda110`
with no environment marker. So the model is stubbed rather than skipped: what is
under test is which columns mosaic copies onto the result, and that is mosaic's code
either way. A skip here would be a test that never runs anywhere.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from mosaic.behavior.feature_library.lightning_action_feature import (
    LightningActionFeature,
)
from mosaic.core.pipeline.types import Result
from tests.helpers import make_pair_df

_CLASSES = [1, 2, 3]
_FEATURES = ["feat_0", "feat_1"]


class _StubModel:
    """Stands in for `lightning_action.api.model.Model`.

    `predict` writes the one CSV the feature reads back, one row per input row and
    one column per class, so the probabilities are deterministic and the assertions
    are about column bookkeeping rather than about a model.
    """

    def __init__(self, n_rows: int) -> None:
        self._n_rows = n_rows

    def predict(self, data_path: str, input_dir: str, output_dir: str) -> None:
        probs = np.tile(np.array([0.2, 0.3, 0.5]), (self._n_rows, 1))
        frame = pd.DataFrame(probs, columns=[f"p{c}" for c in _CLASSES])
        frame.to_csv(Path(output_dir) / "seq.csv", index=False)


@pytest.fixture(autouse=True)
def _stub_lightning_action(monkeypatch: pytest.MonkeyPatch) -> None:
    """Satisfy the unconditional `from lightning_action.api.model import Model`."""
    package = types.ModuleType("lightning_action")
    api = types.ModuleType("lightning_action.api")
    model_module = types.ModuleType("lightning_action.api.model")
    model_module.Model = _StubModel  # pyright: ignore[reportAttributeAccessIssue]
    monkeypatch.setitem(sys.modules, "lightning_action", package)
    monkeypatch.setitem(sys.modules, "lightning_action.api", api)
    monkeypatch.setitem(sys.modules, "lightning_action.api.model", model_module)


def _fitted(n_rows: int) -> LightningActionFeature:
    feature = LightningActionFeature(
        LightningActionFeature.Inputs((Result(feature="temporal-stack"),)),
        params={"default_class": 3, "model": {"feature": "lightning-action"}},
    )
    state: Any = feature
    state._feature_columns = list(_FEATURES)
    state._classes = list(_CLASSES)
    state._la_model = _StubModel(n_rows)
    return feature


def test_apply_keeps_pair_identity() -> None:
    df = make_pair_df(6, len(_FEATURES))

    out = _fitted(len(df)).apply(df)

    key = ["frame", "id1", "id2", "perspective"]
    assert set(key) <= set(out.columns)
    assert not out.duplicated(subset=key).any()
    assert {"prob_1", "prob_2", "prob_3", "predicted_label"} <= set(out.columns)


def test_apply_keeps_individual_identity() -> None:
    """The individual case must not lose what it already carried."""
    n = 5
    df = pd.DataFrame(
        {
            "frame": np.arange(n),
            "time": np.arange(n, dtype=float) / 30.0,
            "id": np.zeros(n, dtype=int),
            "group": ["g"] * n,
            "sequence": ["s"] * n,
            "feat_0": np.zeros(n),
            "feat_1": np.ones(n),
        }
    )

    out = _fitted(n).apply(df)

    assert {"frame", "time", "id", "group", "sequence"} <= set(out.columns)
    assert "id1" not in out.columns


def test_predictions_are_labels_not_indices() -> None:
    """The stub's argmax is class 3, which is a label and not an index into it."""
    df = make_pair_df(4, len(_FEATURES))

    out = _fitted(len(df)).apply(df)

    assert set(out["predicted_label"]) == {3}
