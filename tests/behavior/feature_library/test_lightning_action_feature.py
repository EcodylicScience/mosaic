"""`lightning-action` must carry a pair row's identity onto its predictions.

The same defect `xgboost` carried until commit 250051e, in a feature that declares
`emits = "as-input"` and is routinely handed pair-level input: the passthrough list
stopped at `[frame, time, id, group, sequence]`, so the predictions came back with
two rows per `(group, sequence, frame)` and nothing to tell them apart. Its comment
said the output "matches XGBoost format"; once xgboost was fixed, it did not.

The extra it needs does not resolve on a machine without CUDA -- `lightning-action`
requires `nvidia-dali-cuda110` with no environment marker -- so here the model is
stubbed rather than skipped, and these tests run in every job. The stub keeps the
package's CSV contract: it reads its input with `index_col=0` and writes the row
index first, one column per class name, and NaN past the last whole window. A stub
that wrote CSVs its own way once hid a fit that raised on every real call.
`test_lightning_action_real.py` fits the real package on CPU in CI's
`lightning-action` job.
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
_PROBS = np.array([0.2, 0.3, 0.5])
_WINDOW = 4


class _StubModel:
    """Stands in for `lightning_action.api.model.Model`.

    `predict` reads each input CSV and writes its predictions as the package does
    (`lightning_action/data/utils.py` `load_feature_csv`, `api/model.py` `predict`):
    the input's first column is the index, and the output carries the row index
    first, one column per class name and NaN rows past the last whole window. The
    probabilities are fixed, so the assertions are about bookkeeping rather than
    about a model.
    """

    def __init__(self) -> None:
        self.inputs: list[pd.DataFrame] = []

    def predict(self, data_path: str, input_dir: str, output_dir: str) -> None:
        for path in sorted((Path(data_path) / input_dir).glob("*.csv")):
            features = pd.read_csv(path, index_col=0)
            self.inputs.append(features)
            n = len(features)
            probs = np.tile(_PROBS, (n, 1))
            probs[n - n % _WINDOW :] = np.nan
            frame = pd.DataFrame(probs, columns=[str(c) for c in _CLASSES])
            frame.to_csv(Path(output_dir) / f"{path.stem}_predictions.csv")


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


def _fitted() -> LightningActionFeature:
    feature = LightningActionFeature(
        LightningActionFeature.Inputs((Result(feature="temporal-stack"),)),
        params={"default_class": 3, "model": {"feature": "lightning-action"}},
    )
    state: Any = feature
    state._feature_columns = list(_FEATURES)
    state._classes = list(_CLASSES)
    state._class_names = [str(c) for c in _CLASSES]
    state._la_model = _StubModel()
    return feature


def test_apply_keeps_pair_identity() -> None:
    df = make_pair_df(6, len(_FEATURES))

    out = _fitted().apply(df)

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

    out = _fitted().apply(df)

    assert {"frame", "time", "id", "group", "sequence"} <= set(out.columns)
    assert "id1" not in out.columns


def test_predictions_are_labels_not_indices() -> None:
    """The stub's argmax is class 3, which is a label and not an index into it."""
    df = make_pair_df(4, len(_FEATURES))

    out = _fitted().apply(df)

    assert set(out["predicted_label"]) == {3}


def test_each_probability_column_holds_its_own_class() -> None:
    """Read by position, the package's index column became `prob_<first class>`."""
    n = 2 * _WINDOW + 1
    df = make_pair_df(n, len(_FEATURES)).iloc[:n]

    out = _fitted().apply(df)

    whole = out.iloc[: 2 * _WINDOW]
    for cls, prob in zip(_CLASSES, _PROBS, strict=True):
        np.testing.assert_array_equal(whole[f"prob_{cls}"], prob)
    assert (
        out.iloc[2 * _WINDOW :][[f"prob_{c}" for c in _CLASSES]].isna().all(axis=None)
    )
    assert out["predicted_label"].iloc[-1] == 3  # default_class past the last window


def test_predict_reads_every_feature_column() -> None:
    feature = _fitted()
    df = make_pair_df(_WINDOW, len(_FEATURES))

    feature.apply(df)

    stub: Any = feature
    [seen] = stub._la_model.inputs
    assert list(seen.columns) == _FEATURES
    np.testing.assert_allclose(seen.to_numpy(), df[_FEATURES].to_numpy())


def test_training_csvs_read_back_with_the_package_reader(tmp_path: Path) -> None:
    """The package reads both files with `index_col=0`; no column may be lost."""
    feature = _fitted()
    labels = [1, 1, 2, 3, 3, 2]
    templates = pd.DataFrame(
        {
            "feat_0": np.arange(6, dtype=float),
            "feat_1": np.arange(6, dtype=float) * 10,
            "label": labels,
            "split": ["train"] * 4 + ["test"] * 2,
        }
    )

    feature._write_training_csvs(templates, list(_FEATURES), tmp_path)  # pyright: ignore[reportPrivateUsage]

    for split, rows in (("train", slice(0, 4)), ("test", slice(4, 6))):
        features = pd.read_csv(tmp_path / "features" / f"{split}.csv", index_col=0)
        one_hot = pd.read_csv(tmp_path / "labels" / f"{split}.csv", index_col=0)
        assert list(features.columns) == _FEATURES
        np.testing.assert_array_equal(
            features.to_numpy(), templates[_FEATURES].to_numpy()[rows]
        )
        assert list(one_hot.columns) == [str(c) for c in _CLASSES]
        hot = np.argmax(one_hot.to_numpy(dtype=np.int64), axis=1)
        decoded = [_CLASSES[int(i)] for i in hot]
        assert decoded == labels[rows]
