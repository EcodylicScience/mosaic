"""`lightning-action` fits and applies with the real package, on CPU.

The stubbed tests in `test_lightning_action_feature.py` mirror the package's CSV
contract, but only a real fit proves it: through 0.3 the feature wrote its CSVs
without the index column the package reads with `index_col=0`, so the package
dropped the first feature column and every fit raised a channel-count error.

The `lightning-action` extra does not install without CUDA, because the package
requires a DALI build that only builds with CUDA. CI's `lightning-action` job
installs the package with `--no-deps` beside its pure-Python requirements and sets
MOSAIC_CI_LIGHTNING_ACTION, which makes a missing package an error rather than a
skip.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mosaic.behavior.feature_library.lightning_action_feature import (
    LightningActionFeature,
)
from mosaic.core.pipeline.types import InputStream, Result

_HAS_LIGHTNING_ACTION = importlib.util.find_spec("lightning_action") is not None

_CLASSES = [0, 1, 2]
_FEATURES = [f"feat_{i}" for i in range(5)]


def _templates(n_rows: int, seed: int) -> pd.DataFrame:
    """Labels in runs of ten frames, each class shifting its own feature.

    `sequence_length` is 50 rather than the minimum of 10: the package cuts its
    windows wrongly when the dtcn's padding is longer than a window.
    """
    rng = np.random.default_rng(seed)
    labels = np.repeat(rng.integers(0, len(_CLASSES), size=n_rows // 10 + 1), 10)
    labels = labels[:n_rows]
    x = rng.standard_normal((n_rows, len(_FEATURES)))
    for cls in _CLASSES:
        x[labels == cls, cls] += 2.0
    frame = pd.DataFrame(x, columns=_FEATURES)
    frame["label"] = labels
    frame["split"] = "train"
    return frame


def _feature() -> LightningActionFeature:
    return LightningActionFeature(
        LightningActionFeature.Inputs((Result(feature="temporal-stack"),)),
        params={
            "default_class": 0,
            "model": {"feature": "lightning-action"},
            "device": "cpu",
            "num_epochs": 1,
            "sequence_length": 50,
            "batch_size": 4,
            "num_hid_units": 8,
        },
    )


@pytest.mark.lightning_action
@pytest.mark.skipif(
    not _HAS_LIGHTNING_ACTION, reason="requires the lightning-action package"
)
# Lightning's advice for a tiny CPU run, and deprecations inside the packages.
@pytest.mark.filterwarnings("ignore:GPU available but not used:UserWarning")
@pytest.mark.filterwarnings("ignore:.*does not have many workers:UserWarning")
@pytest.mark.filterwarnings("ignore:The number of training batches:UserWarning")
@pytest.mark.filterwarnings("ignore:To copy construct from a tensor:UserWarning")
@pytest.mark.filterwarnings("ignore:`isinstance\\(treespec, LeafSpec\\)`:FutureWarning")
def test_fit_save_load_apply(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from lightning_action.data import DataModule

    # The package's data loaders default to four persistent worker processes. On
    # macOS they are spawned, and under pytest's output capture they hang. Its
    # config passes `num_workers` and no `persistent_workers`, which then refuses
    # zero workers, so both are set on the data module itself.
    init = DataModule.__init__

    def in_process(self: DataModule, **kwargs: object) -> None:
        kwargs.update(num_workers=0, persistent_workers=False, pin_memory=False)
        init(self, **kwargs)  # pyright: ignore[reportArgumentType]

    monkeypatch.setattr(DataModule, "__init__", in_process)
    feature = _feature()
    templates = _templates(1000, seed=0)
    feature._templates = templates  # pyright: ignore[reportPrivateUsage]
    feature._feature_columns = list(_FEATURES)  # pyright: ignore[reportPrivateUsage]

    feature.fit(InputStream(lambda: iter(()), n_entries=0))
    feature.save_state(tmp_path)

    reloaded = _feature()
    assert reloaded.load_state(tmp_path, {}, {})
    df = _templates(120, seed=1)[_FEATURES]
    df.insert(0, "frame", np.arange(len(df)))
    out = reloaded.apply(df)

    prob_cols = [f"prob_{c}" for c in _CLASSES]
    assert list(out["frame"]) == list(df["frame"])
    # The package predicts whole windows only and pads the rest with NaN.
    whole = out[prob_cols].dropna()
    assert len(whole) == 100
    np.testing.assert_allclose(whole.sum(axis=1), 1.0, rtol=1e-4)
    assert set(out["predicted_label"]) <= set(_CLASSES)
