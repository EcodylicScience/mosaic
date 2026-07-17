"""Feature-level tests for the AR-HMM feature (fit/apply, backends, caching)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mosaic.behavior.feature_library.arhmm import ArHmmFeature
from mosaic.core.pipeline.types import Result

_INPUTS = ArHmmFeature.Inputs((Result(feature="speed-angvel"),))


def _make_sequence_df(
    sequence: str = "s1",
    n: int = 400,
    seed: int = 0,
    n_features: int = 3,
) -> pd.DataFrame:
    """Two separable motion regimes so discrete states are recoverable."""
    rng = np.random.default_rng(seed)
    half = n // 2
    a = np.cumsum(rng.standard_normal((half, n_features)) * 0.1, axis=0)
    b = np.cumsum(rng.standard_normal((n - half, n_features)) * 0.1, axis=0) + 5.0
    X = np.vstack([a, b])
    data: dict[str, object] = {
        "frame": np.arange(n),
        "time": np.arange(n, dtype=float) / 30.0,
        "id": np.zeros(n, dtype=int),
        "group": "g",
        "sequence": sequence,
    }
    for i in range(n_features):
        data[f"f{i}"] = X[:, i]
    return pd.DataFrame(data)


def _make_feature(backend: str = "auto", **params: object) -> ArHmmFeature:
    p: dict[str, object] = {
        "n_states": 6,
        "n_lags": 1,
        "n_iter": 30,
        "backend": backend,
    }
    p.update(params)
    return ArHmmFeature(inputs=_INPUTS, params=p)


def _fit(feat: ArHmmFeature, dfs: list[pd.DataFrame]) -> ArHmmFeature:
    stream = [(f"g__{df['sequence'].iloc[0]}", df) for df in dfs]
    feat.fit(lambda: iter(stream))
    return feat


# --------------------------------------------------------------------------
# Fit + apply
# --------------------------------------------------------------------------


class TestFitApply:
    def test_apply_output_schema(self) -> None:
        feat = _fit(_make_feature("numba"), [_make_sequence_df("s1", seed=1)])
        df = _make_sequence_df("s1", seed=1)
        result = feat.apply(df)

        assert list(result.columns) == ["frame", "syllable", "id"]
        assert result["syllable"].dtype == np.int32
        assert len(result) == len(df)
        pd.testing.assert_series_equal(result["frame"], df["frame"], check_names=False)

    def test_apply_raises_before_fit(self) -> None:
        feat = _make_feature("numba")
        with pytest.raises(RuntimeError, match="not fitted"):
            feat.apply(_make_sequence_df())

    def test_fit_raises_without_sequences(self) -> None:
        feat = _make_feature("numba")
        with pytest.raises(RuntimeError, match="No valid sequences"):
            feat.fit(lambda: iter([]))

    def test_jax_backend_not_implemented(self) -> None:
        feat = _make_feature("jax")
        with pytest.raises(NotImplementedError, match="jax"):
            feat.fit(lambda: iter([("g__s1", _make_sequence_df())]))


# --------------------------------------------------------------------------
# Backend equivalence
# --------------------------------------------------------------------------


class TestBackends:
    def test_numba_matches_numpy(self) -> None:
        dfs = [_make_sequence_df("s1", seed=1), _make_sequence_df("s2", seed=2)]
        f_nb = _fit(
            _make_feature("numba", prune_threshold=0.0), [d.copy() for d in dfs]
        )
        f_np = _fit(
            _make_feature("numpy", prune_threshold=0.0), [d.copy() for d in dfs]
        )

        held = _make_sequence_df("s1", seed=1)
        agree = (
            f_nb.apply(held)["syllable"].values == f_np.apply(held)["syllable"].values
        ).mean()
        assert agree >= 0.99

    def test_auto_selects_numba(self) -> None:
        assert _make_feature("auto")._select_backend() == "numba"
        assert _make_feature("numpy")._select_backend() == "numpy"


# --------------------------------------------------------------------------
# Serialization + caching
# --------------------------------------------------------------------------


class TestSerialization:
    def test_save_load_round_trip(self, tmp_path: Path) -> None:
        feat = _fit(_make_feature("numba"), [_make_sequence_df("s1", seed=1)])
        feat.save_state(tmp_path)
        assert (tmp_path / "arhmm_model.joblib").exists()

        df = _make_sequence_df("s1", seed=1)
        expected = feat.apply(df)

        feat2 = _make_feature("numba")
        assert feat2.load_state(tmp_path, {}, {}) is True
        pd.testing.assert_frame_equal(feat2.apply(df), expected)

    def test_load_state_false_when_missing(self, tmp_path: Path) -> None:
        assert _make_feature().load_state(tmp_path, {}, {}) is False

    def test_numba_model_loads_and_applies_with_numpy(self, tmp_path: Path) -> None:
        """A numba-fit model must decode identically via the numpy backend."""
        feat = _fit(_make_feature("numba"), [_make_sequence_df("s1", seed=3)])
        feat.save_state(tmp_path)

        df = _make_sequence_df("s1", seed=3)
        via_numba = feat.apply(df)

        feat_np = _make_feature("numpy")
        feat_np.load_state(tmp_path, {}, {})
        pd.testing.assert_frame_equal(feat_np.apply(df), via_numba)


class TestCacheKey:
    def test_backend_excluded_from_run_id_hash(self) -> None:
        """`backend` is HASH_EXCLUDE: numpy and numba runs share a cache entry."""
        dump_np = _make_feature("numpy").params.identity_dump()
        dump_nb = _make_feature("numba").params.identity_dump()
        assert dump_np == dump_nb
        assert "backend" not in dump_np
        # ...but it still round-trips through the full model dump.
        assert "backend" in _make_feature("numba").params.model_dump()
