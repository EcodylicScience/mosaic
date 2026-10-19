"""`lightning-action` reads an individual table, and refuses a pair table.

`apply` writes an entry's rows to the model as one sequence in table order. A pair
table holds two rows per frame, one per perspective, so that sequence would
interleave the two animals' frames, and a window of the temporal model would read
one animal's frame after the other's. `apply` refuses such a table until it reads
each pair as a sequence of its own. `fit` reads the templates and not its input
stream, so it cannot tell a pair input apart, and the refusal comes at the first
entry.

The identity columns an individual table carries still reach the predictions. The
passthrough list once stopped at `[frame, time, id, group, sequence]` while the
feature was handed pair tables, so predictions came back with two rows per
`(group, sequence, frame)` and nothing to tell them apart.

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
from mosaic.core.pipeline.types import InputStream, Result
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


def _individual_df(n: int) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.standard_normal((n, len(_FEATURES))), columns=_FEATURES)
    df.insert(0, "frame", np.arange(n))
    df.insert(1, "id", 0)
    return df


def test_apply_refuses_a_pair_table() -> None:
    df = make_pair_df(6, len(_FEATURES))

    with pytest.raises(ValueError, match=r"^\[lightning-action\] .*pair table"):
        _fitted().apply(df)


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
    df = make_pair_df(4, len(_FEATURES)).query("perspective == 0")
    df = df.drop(columns=["id1", "id2", "perspective"]).assign(id=0)

    out = _fitted().apply(df)

    assert set(out["predicted_label"]) == {3}


class _RecordingModel:
    """Records the headers of the CSVs that `fit` hands to lightning-action."""

    config: dict[str, Any] = {}
    headers: dict[str, list[str]] = {}

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> _RecordingModel:
        cls.config = config
        return cls()

    @classmethod
    def from_dir(cls, path: str) -> _RecordingModel:
        return cls()

    def train(self, output_dir: str, post_inference: bool) -> None:
        data = Path(self.config["data"]["data_path"])
        # The package reads each file with `index_col=0`.
        type(self).headers = {
            str(path.relative_to(data)): list(pd.read_csv(path, index_col=0).columns)
            for path in sorted(data.rglob("*.csv"))
        }
        Path(output_dir).mkdir(parents=True, exist_ok=True)


def _templates_table(*, with_identity: bool) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 12
    table = pd.DataFrame(rng.standard_normal((n, 2)), columns=_FEATURES)
    table["label"] = np.tile(_CLASSES, n // len(_CLASSES))
    table["split"] = ["train"] * 8 + ["test"] * 4
    if with_identity:
        table["group"] = ["g", ""] * (n // 2)
        table["sequence"] = ["s1", "12", "s3"] * (n // 3)
    return table


@pytest.mark.parametrize("with_identity", [False, True], ids=["0.5", "0.6"])
def test_the_templates_identity_columns_reach_neither_the_features_nor_the_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, with_identity: bool
) -> None:
    """`extract-labeled-templates` 0.6 writes `group` and `sequence` beside `split`.

    `lightning-action` reads the same columns of a 0.6 table as of a 0.5 one: the
    feature columns come from `feature_columns`, and the CSVs it writes hold those and
    the one-hot labels only.
    """
    monkeypatch.setattr(
        sys.modules["lightning_action.api.model"], "Model", _RecordingModel
    )
    path = tmp_path / "templates.parquet"
    _templates_table(with_identity=with_identity).to_parquet(path, index=False)
    feature = LightningActionFeature(
        LightningActionFeature.Inputs((Result(feature="temporal-stack"),)),
        params={
            "templates": {"feature": "extract-labeled-templates"},
            "default_class": 3,
            "num_epochs": 1,
            "sequence_length": 10,
        },
    )

    assert feature.load_state(tmp_path / "run", {"templates": path}, {}) is False
    state: Any = feature
    assert state._feature_columns == _FEATURES
    feature.fit(InputStream(lambda: iter([]), n_entries=0))

    assert _RecordingModel.headers == {
        "features/test.csv": _FEATURES,
        "features/train.csv": _FEATURES,
        "labels/test.csv": [str(c) for c in _CLASSES],
        "labels/train.csv": [str(c) for c in _CLASSES],
    }


def test_each_probability_column_holds_its_own_class() -> None:
    """Read by position, the package's index column became `prob_<first class>`."""
    df = _individual_df(2 * _WINDOW + 1)

    out = _fitted().apply(df)

    whole = out.iloc[: 2 * _WINDOW]
    for cls, prob in zip(_CLASSES, _PROBS, strict=True):
        np.testing.assert_array_equal(whole[f"prob_{cls}"], prob)
    tail = out.iloc[2 * _WINDOW :][[f"prob_{c}" for c in _CLASSES]]
    assert tail.isna().all(axis=None)
    assert out["predicted_label"].iloc[-1] == 3  # default_class past the last window


def test_predict_reads_every_feature_column() -> None:
    feature = _fitted()
    df = _individual_df(_WINDOW)

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
