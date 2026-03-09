"""Tests for typed LoadSpec dispatch in helpers.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mosaic.behavior.feature_library.helpers import (
    _load_array_from_spec,
)
from mosaic.behavior.feature_library.params import (
    ArtifactSpec,
    JoblibLoadSpec,
    NpzLoadSpec,
    ParquetLoadSpec,
    Result,
)


# --- Fixtures ---


@pytest.fixture()
def npz_path(tmp_path: Path) -> Path:
    p = tmp_path / "data.npz"
    np.savez(p, features=np.arange(12, dtype=np.float32).reshape(3, 4))
    return p


@pytest.fixture()
def parquet_path(tmp_path: Path) -> Path:
    p = tmp_path / "data.parquet"
    df = pd.DataFrame(
        {
            "frame": [0, 1, 2],
            "id": [1, 1, 1],
            "feat_a": [1.0, 2.0, 3.0],
            "feat_b": [4.0, 5.0, 6.0],
        }
    )
    df.to_parquet(p, index=False)
    return p


# --- _load_array_from_spec: NpzLoadSpec ---


class TestNpzLoadSpec:
    def test_basic_load(self, npz_path: Path) -> None:
        spec = NpzLoadSpec(key="features")
        arr, frames = _load_array_from_spec(npz_path, spec)
        assert arr is not None
        assert arr.shape == (3, 4)
        assert arr.dtype == np.float32
        assert frames is None

    def test_missing_key_returns_none(self, npz_path: Path) -> None:
        spec = NpzLoadSpec(key="nonexistent")
        arr, frames = _load_array_from_spec(npz_path, spec)
        assert arr is None
        assert frames is None

    def test_transpose(self, npz_path: Path) -> None:
        spec = NpzLoadSpec(key="features", transpose=True)
        arr, _ = _load_array_from_spec(npz_path, spec)
        assert arr is not None
        assert arr.shape == (4, 3)

    def test_1d_array_becomes_2d(self, tmp_path: Path) -> None:
        p = tmp_path / "vec.npz"
        np.savez(p, vec=np.array([1.0, 2.0, 3.0]))
        spec = NpzLoadSpec(key="vec")
        arr, _ = _load_array_from_spec(p, spec)
        assert arr is not None
        assert arr.ndim == 2
        assert arr.shape == (1, 3)


# --- _load_array_from_spec: ParquetLoadSpec ---


class TestParquetLoadSpec:
    def test_numeric_only_default(self, parquet_path: Path) -> None:
        spec = ParquetLoadSpec()
        arr, frames = _load_array_from_spec(parquet_path, spec)
        assert arr is not None
        assert arr.shape == (3, 2)
        assert arr.dtype == np.float32
        assert frames is None

    def test_explicit_columns(self, parquet_path: Path) -> None:
        spec = ParquetLoadSpec(columns=["feat_a"])
        arr, _ = _load_array_from_spec(parquet_path, spec)
        assert arr is not None
        assert arr.shape == (3, 1)

    def test_drop_columns(self, parquet_path: Path) -> None:
        spec = ParquetLoadSpec(drop_columns=["feat_b"])
        arr, _ = _load_array_from_spec(parquet_path, spec)
        assert arr is not None
        assert arr.shape == (3, 1)

    def test_frame_column_extraction(self, parquet_path: Path) -> None:
        spec = ParquetLoadSpec(frame_column="frame")
        arr, frames = _load_array_from_spec(parquet_path, spec)
        assert arr is not None
        assert frames is not None
        np.testing.assert_array_equal(frames, [0, 1, 2])

    def test_extract_frame_col_param(self, parquet_path: Path) -> None:
        spec = ParquetLoadSpec()
        arr, frames = _load_array_from_spec(
            parquet_path, spec, extract_frame_col="frame"
        )
        assert frames is not None
        np.testing.assert_array_equal(frames, [0, 1, 2])

    def test_transpose(self, parquet_path: Path) -> None:
        spec = ParquetLoadSpec(transpose=True)
        arr, _ = _load_array_from_spec(parquet_path, spec)
        assert arr is not None
        assert arr.shape == (2, 3)

    def test_numeric_only_false(self, parquet_path: Path) -> None:
        spec = ParquetLoadSpec(numeric_only=False)
        arr, _ = _load_array_from_spec(parquet_path, spec)
        assert arr is not None
        # All 4 columns coerced to numeric
        assert arr.shape[0] == 3

    def test_df_filter(self, parquet_path: Path) -> None:
        spec = ParquetLoadSpec()
        arr, _ = _load_array_from_spec(
            parquet_path,
            spec,
            df_filter=lambda df: df[df["frame"] >= 1],
        )
        assert arr is not None
        assert arr.shape[0] == 2

    def test_df_filter_empty_returns_none(self, parquet_path: Path) -> None:
        spec = ParquetLoadSpec()
        arr, frames = _load_array_from_spec(
            parquet_path,
            spec,
            df_filter=lambda df: df[df["frame"] > 100],
        )
        assert arr is None
        assert frames is None

    def test_preloaded_df(self, parquet_path: Path) -> None:
        df = pd.read_parquet(parquet_path)
        spec = ParquetLoadSpec()
        arr, _ = _load_array_from_spec(parquet_path, spec, df=df)
        assert arr is not None
        assert arr.shape == (3, 2)


# --- _load_array_from_spec: unsupported type ---


def test_unsupported_load_spec_raises() -> None:
    spec = JoblibLoadSpec()
    with pytest.raises(ValueError, match="Unsupported load spec type"):
        _load_array_from_spec(Path("dummy.joblib"), spec)


# --- _load_identity_from_spec ---


class TestLoadIdentityFromSpec:
    def test_non_parquet_returns_global(self) -> None:
        from mosaic.behavior.feature_library.helpers import StreamingFeatureHelper

        helper = StreamingFeatureHelper(None, "test")
        spec = NpzLoadSpec(key="x")
        id1, id2, level = helper._load_identity_from_spec(Path("dummy"), spec)
        assert id1 is None
        assert id2 is None
        assert level == "global"

    def test_parquet_with_ids(self, tmp_path: Path) -> None:
        from mosaic.behavior.feature_library.helpers import StreamingFeatureHelper

        p = tmp_path / "pairs.parquet"
        df = pd.DataFrame(
            {"id1": [1, 1], "id2": [2, 2], "feat": [0.5, 0.6]}
        )
        df.to_parquet(p, index=False)

        helper = StreamingFeatureHelper(None, "test")
        spec = ParquetLoadSpec()
        id1, id2, level = helper._load_identity_from_spec(p, spec)
        assert level == "pair"
        assert id1 is not None
        assert id2 is not None


# --- _load_joblib_artifact: type narrowing ---


def test_load_joblib_artifact_rejects_non_joblib() -> None:
    from mosaic.behavior.feature_library.helpers import _load_joblib_artifact

    artifact = ArtifactSpec(feature="test", load=ParquetLoadSpec())
    with pytest.raises(ValueError, match="requires JoblibLoadSpec"):
        _load_joblib_artifact(None, artifact)


# --- temporal_stacking bind_dataset produces ArtifactSpec ---


def test_temporal_stacking_bind_produces_artifact_specs() -> None:
    from unittest.mock import MagicMock

    from mosaic.behavior.feature_library.temporal_stacking import (
        TemporalStackingFeature,
    )

    inputs = TemporalStackingFeature.Inputs(
        (Result(feature="speed-angvel"), Result(feature="pair-wavelet"))
    )
    ts = TemporalStackingFeature(inputs=inputs)
    ts.bind_dataset(MagicMock())

    assert len(ts._inputs) == 2
    for spec in ts._inputs:
        assert isinstance(spec, ArtifactSpec)
        assert isinstance(spec.load, ParquetLoadSpec)
        assert spec.load.numeric_only is True
        assert spec.pattern == "*.parquet"

    assert ts._inputs[0].feature == "speed-angvel"
    assert ts._inputs[1].feature == "pair-wavelet"


# --- _extract_key ---


class TestExtractKey:
    def test_seq_pattern(self) -> None:
        from mosaic.behavior.feature_library.helpers import StreamingFeatureHelper

        helper = StreamingFeatureHelper(None, "test")
        path = Path("/tmp/global_tsne_coords_seq=my_sequence.npz")
        assert helper._extract_key(path, {}) == "my_sequence"

    def test_index_lookup(self) -> None:
        from mosaic.behavior.feature_library.helpers import StreamingFeatureHelper

        helper = StreamingFeatureHelper(None, "test")
        path = Path("/tmp/some_file.parquet")
        seq_map = {path.resolve(): "indexed_seq"}
        assert helper._extract_key(path, seq_map) == "indexed_seq"

    def test_unindexed_returns_none(self) -> None:
        from mosaic.behavior.feature_library.helpers import StreamingFeatureHelper

        helper = StreamingFeatureHelper(None, "test")
        path = Path("/tmp/cluster_sizes.parquet")
        assert helper._extract_key(path, {}) is None


# --- build_manifest skips global artifacts ---


class TestBuildManifestSkipsGlobalArtifacts:
    def test_global_artifacts_excluded(self, tmp_path: Path) -> None:
        """build_manifest should skip files not in the index and without seq= naming."""
        from unittest.mock import patch

        from mosaic.behavior.feature_library.helpers import StreamingFeatureHelper

        run_root = tmp_path / "features" / "global-kmeans" / "run_001"
        run_root.mkdir(parents=True)

        # Per-sequence file (seq= naming)
        seq_file = run_root / "seq=my_sequence.parquet"
        pd.DataFrame({"feat": [1.0]}).to_parquet(seq_file)

        # Per-sequence file (index-tracked, no seq= naming)
        indexed_file = run_root / "my_group__my_sequence.parquet"
        pd.DataFrame({"feat": [2.0]}).to_parquet(indexed_file)

        # Global artifact (should be excluded)
        global_file = run_root / "cluster_sizes.parquet"
        pd.DataFrame({"cluster": [0], "count": [10]}).to_parquet(global_file)

        # Marker file (should be excluded)
        marker_file = run_root / "__global__.parquet"
        pd.DataFrame({"run_marker": [True]}).to_parquet(marker_file)

        helper = StreamingFeatureHelper(None, "test")

        spec = ArtifactSpec(
            feature="global-kmeans", load=ParquetLoadSpec()
        )

        seq_map = {indexed_file.resolve(): "my_sequence_2"}

        with (
            patch.object(helper, "_get_seq_map", return_value=seq_map),
            patch(
                "mosaic.core.dataset._latest_feature_run_root",
                return_value=("run_001", run_root),
            ),
        ):
            manifest = helper.build_manifest([spec])

        assert "my_sequence" in manifest
        assert "my_sequence_2" in manifest
        assert "cluster_sizes" not in manifest
        assert "__global__" not in manifest
        assert len(manifest) == 2
