"""Tests for typed LoadSpec dispatch in helpers.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mosaic.behavior.feature_library.helpers import (
    _load_array_from_spec,
)
from mosaic.behavior.feature_library.spec import (
    ArtifactSpec,
    JoblibLoadSpec,
    NpzLoadSpec,
    ParquetLoadSpec,
    Result,
)
from mosaic.core.pipeline._utils import Scope
from mosaic.core.pipeline.loading import (
    _concat_into,
    _load_artifact_matrix,
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
    def test_returns_bare_stem(self) -> None:
        from mosaic.behavior.feature_library.helpers import StreamingFeatureHelper

        helper = StreamingFeatureHelper(None, "test")
        path = Path("/tmp/calms21_task1_test__task1%2Ftest%2Fmouse075.parquet")
        assert (
            helper._extract_key(path) == "calms21_task1_test__task1%2Ftest%2Fmouse075"
        )

    def test_entry_key_format(self) -> None:
        from mosaic.behavior.feature_library.helpers import StreamingFeatureHelper

        helper = StreamingFeatureHelper(None, "test")
        path = Path("/tmp/my_group__my_sequence.parquet")
        assert helper._extract_key(path) == "my_group__my_sequence"


# --- build_manifest skips global artifacts ---


class TestBuildManifestSkipsGlobalArtifacts:
    def test_global_artifacts_excluded(self, tmp_path: Path) -> None:
        """build_manifest should skip files not in the index."""
        from unittest.mock import patch

        from mosaic.behavior.feature_library.helpers import StreamingFeatureHelper

        run_root = tmp_path / "features" / "global-kmeans" / "run_001"
        run_root.mkdir(parents=True)

        # Per-sequence file (index-tracked)
        indexed_file = run_root / "my_group__my_sequence.parquet"
        pd.DataFrame({"feat": [2.0]}).to_parquet(indexed_file)

        # Global artifact (should be excluded)
        global_file = run_root / "cluster_sizes.parquet"
        pd.DataFrame({"cluster": [0], "count": [10]}).to_parquet(global_file)

        # Marker file (should be excluded)
        marker_file = run_root / "__global__.parquet"
        pd.DataFrame({"run_marker": [True]}).to_parquet(marker_file)

        helper = StreamingFeatureHelper(None, "test")

        spec = ArtifactSpec(feature="global-kmeans", load=ParquetLoadSpec())

        # Scope restricts to known entries only
        scope = Scope(entries={("my_group", "my_sequence")})

        with patch(
            "mosaic.core.pipeline.index.latest_feature_run_root",
            return_value=("run_001", run_root),
        ):
            manifest = helper.build_manifest([spec], scope=scope)

        assert "my_group__my_sequence" in manifest
        assert "cluster_sizes" not in manifest
        assert "__global__" not in manifest
        assert len(manifest) == 1


class TestLoadEntryDataUnified:
    """Tests for the unified load_entry_data returning EntryData."""

    @pytest.fixture()
    def aligned_parquets(self, tmp_path: Path) -> tuple[Path, Path]:
        """Two parquet files with same frames but different feature columns."""
        p1 = tmp_path / "feat1.parquet"
        p2 = tmp_path / "feat2.parquet"
        df1 = pd.DataFrame(
            {
                "frame": [0, 1, 2, 3],
                "feat_a": [1.0, 2.0, 3.0, 4.0],
            }
        )
        df2 = pd.DataFrame(
            {
                "frame": [0, 1, 2, 3],
                "feat_b": [10.0, 20.0, 30.0, 40.0],
            }
        )
        df1.to_parquet(p1, index=False)
        df2.to_parquet(p2, index=False)
        return p1, p2

    @pytest.fixture()
    def misaligned_parquets(self, tmp_path: Path) -> tuple[Path, Path]:
        """Two parquets with overlapping but different frame ranges."""
        p1 = tmp_path / "feat1.parquet"
        p2 = tmp_path / "feat2.parquet"
        df1 = pd.DataFrame(
            {
                "frame": [0, 1, 2, 3],
                "feat_a": [1.0, 2.0, 3.0, 4.0],
            }
        )
        df2 = pd.DataFrame(
            {
                "frame": [2, 3, 4, 5],
                "feat_b": [30.0, 40.0, 50.0, 60.0],
            }
        )
        df1.to_parquet(p1, index=False)
        df2.to_parquet(p2, index=False)
        return p1, p2

    @pytest.fixture()
    def identity_parquets(self, tmp_path: Path) -> tuple[Path, Path]:
        """Two parquets with identity columns (pair feature)."""
        p1 = tmp_path / "feat1.parquet"
        p2 = tmp_path / "feat2.parquet"
        df1 = pd.DataFrame(
            {
                "frame": [0, 0, 1, 1],
                "id1": [1, 2, 1, 2],
                "id2": [2, 1, 2, 1],
                "feat_a": [1.0, 2.0, 3.0, 4.0],
            }
        )
        df2 = pd.DataFrame(
            {
                "frame": [0, 0, 1, 1],
                "id1": [1, 2, 1, 2],
                "id2": [2, 1, 2, 1],
                "feat_b": [10.0, 20.0, 30.0, 40.0],
            }
        )
        df1.to_parquet(p1, index=False)
        df2.to_parquet(p2, index=False)
        return p1, p2

    def test_aligned_inputs_merged(self, aligned_parquets: tuple[Path, Path]) -> None:
        from mosaic.behavior.feature_library.helpers import (
            EntryData,
            StreamingFeatureHelper,
        )
        from mosaic.behavior.feature_library.spec import ParquetLoadSpec

        p1, p2 = aligned_parquets
        helper = StreamingFeatureHelper(None, "test")
        spec = ParquetLoadSpec()
        result = helper.load_entry_data([(p1, spec), (p2, spec)])
        assert isinstance(result, EntryData)
        assert result.features.shape == (4, 2)
        np.testing.assert_array_equal(result.frames, [0, 1, 2, 3])

    def test_misaligned_inputs_inner_join(
        self, misaligned_parquets: tuple[Path, Path]
    ) -> None:
        """Overlapping frames only -- the core alignment fix."""
        from mosaic.behavior.feature_library.helpers import (
            EntryData,
            StreamingFeatureHelper,
        )
        from mosaic.behavior.feature_library.spec import ParquetLoadSpec

        p1, p2 = misaligned_parquets
        helper = StreamingFeatureHelper(None, "test")
        spec = ParquetLoadSpec()
        result = helper.load_entry_data([(p1, spec), (p2, spec)])
        assert isinstance(result, EntryData)
        assert result.features.shape == (2, 2)
        np.testing.assert_array_equal(result.frames, [2, 3])
        np.testing.assert_array_almost_equal(result.features[:, 0], [3.0, 4.0])
        np.testing.assert_array_almost_equal(result.features[:, 1], [30.0, 40.0])

    def test_identity_columns_extracted(
        self, identity_parquets: tuple[Path, Path]
    ) -> None:
        from mosaic.behavior.feature_library.helpers import (
            EntryData,
            StreamingFeatureHelper,
        )
        from mosaic.behavior.feature_library.spec import ParquetLoadSpec

        p1, p2 = identity_parquets
        helper = StreamingFeatureHelper(None, "test")
        spec = ParquetLoadSpec()
        result = helper.load_entry_data([(p1, spec), (p2, spec)])
        assert isinstance(result, EntryData)
        assert result.entity_level == "pair"
        assert result.id1 is not None
        assert result.id2 is not None
        assert result.features.shape == (4, 2)
        np.testing.assert_array_equal(result.id1, [1, 2, 1, 2])
        np.testing.assert_array_equal(result.id2, [2, 1, 2, 1])

    def test_single_input(self, parquet_path: Path) -> None:
        """Single input should work without merge."""
        from mosaic.behavior.feature_library.helpers import (
            EntryData,
            StreamingFeatureHelper,
        )
        from mosaic.behavior.feature_library.spec import ParquetLoadSpec

        helper = StreamingFeatureHelper(None, "test")
        spec = ParquetLoadSpec()
        result = helper.load_entry_data([(parquet_path, spec)])
        assert isinstance(result, EntryData)
        assert result.features.shape == (3, 2)
        np.testing.assert_array_equal(result.frames, [0, 1, 2])
        assert result.entity_level == "individual"

    def test_no_overlap_returns_none(self, tmp_path: Path) -> None:
        """Two inputs with no shared frames should return None."""
        from mosaic.behavior.feature_library.helpers import StreamingFeatureHelper
        from mosaic.behavior.feature_library.spec import ParquetLoadSpec

        p1 = tmp_path / "a.parquet"
        p2 = tmp_path / "b.parquet"
        pd.DataFrame({"frame": [0, 1], "f": [1.0, 2.0]}).to_parquet(p1)
        pd.DataFrame({"frame": [5, 6], "f": [3.0, 4.0]}).to_parquet(p2)

        helper = StreamingFeatureHelper(None, "test")
        spec = ParquetLoadSpec()
        result = helper.load_entry_data([(p1, spec), (p2, spec)])
        assert result is None

    def test_global_entity_level_when_no_id(self, tmp_path: Path) -> None:
        from mosaic.behavior.feature_library.helpers import (
            EntryData,
            StreamingFeatureHelper,
        )
        from mosaic.behavior.feature_library.spec import ParquetLoadSpec

        p = tmp_path / "no_id.parquet"
        pd.DataFrame({"frame": [0, 1], "feat": [1.0, 2.0]}).to_parquet(p)

        helper = StreamingFeatureHelper(None, "test")
        result = helper.load_entry_data([(p, ParquetLoadSpec())])
        assert isinstance(result, EntryData)
        assert result.entity_level == "global"
        assert result.id1 is None
        assert result.id2 is None

    def test_duplicate_column_names_across_inputs(self, tmp_path: Path) -> None:
        """Two inputs with the same feature column names must keep all features."""
        from mosaic.behavior.feature_library.helpers import (
            EntryData,
            StreamingFeatureHelper,
        )
        from mosaic.behavior.feature_library.spec import ParquetLoadSpec

        p1 = tmp_path / "a.parquet"
        p2 = tmp_path / "b.parquet"
        pd.DataFrame(
            {"frame": [0, 1, 2], "feat_0": [1.0, 2.0, 3.0], "feat_1": [4.0, 5.0, 6.0]}
        ).to_parquet(p1, index=False)
        pd.DataFrame(
            {
                "frame": [0, 1, 2],
                "feat_0": [10.0, 20.0, 30.0],
                "feat_1": [40.0, 50.0, 60.0],
            }
        ).to_parquet(p2, index=False)

        helper = StreamingFeatureHelper(None, "test")
        spec = ParquetLoadSpec()
        result = helper.load_entry_data([(p1, spec), (p2, spec)])
        assert isinstance(result, EntryData)
        # Must have 4 feature columns (2 from each input), not 2
        assert result.features.shape == (3, 4)
        # Input 1 features first, then input 2 features
        np.testing.assert_array_almost_equal(result.features[:, 0], [1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(result.features[:, 1], [4.0, 5.0, 6.0])
        np.testing.assert_array_almost_equal(result.features[:, 2], [10.0, 20.0, 30.0])
        np.testing.assert_array_almost_equal(result.features[:, 3], [40.0, 50.0, 60.0])

    def test_aligned_values_correct(self, aligned_parquets: tuple[Path, Path]) -> None:
        """Verify feature values, not just shape, for aligned inputs."""
        from mosaic.behavior.feature_library.helpers import (
            EntryData,
            StreamingFeatureHelper,
        )
        from mosaic.behavior.feature_library.spec import ParquetLoadSpec

        p1, p2 = aligned_parquets
        helper = StreamingFeatureHelper(None, "test")
        spec = ParquetLoadSpec()
        result = helper.load_entry_data([(p1, spec), (p2, spec)])
        assert isinstance(result, EntryData)
        # col 0 = feat_a from input 1, col 1 = feat_b from input 2
        np.testing.assert_array_almost_equal(
            result.features[:, 0], [1.0, 2.0, 3.0, 4.0]
        )
        np.testing.assert_array_almost_equal(
            result.features[:, 1], [10.0, 20.0, 30.0, 40.0]
        )

    def test_meta_columns_excluded_from_features(self, tmp_path: Path) -> None:
        """Numeric metadata columns (time, id) must not appear in features."""
        from mosaic.behavior.feature_library.helpers import (
            EntryData,
            StreamingFeatureHelper,
        )
        from mosaic.behavior.feature_library.spec import ParquetLoadSpec

        p = tmp_path / "with_meta.parquet"
        pd.DataFrame(
            {
                "frame": [0, 1, 2],
                "time": [0.0, 0.033, 0.066],
                "id": [1, 1, 1],
                "id1": [1, 1, 1],
                "id2": [2, 2, 2],
                "feat_x": [10.0, 20.0, 30.0],
                "feat_y": [40.0, 50.0, 60.0],
            }
        ).to_parquet(p, index=False)

        helper = StreamingFeatureHelper(None, "test")
        result = helper.load_entry_data([(p, ParquetLoadSpec())])
        assert isinstance(result, EntryData)
        # Only feat_x, feat_y should be features -- not frame/time/id/id1/id2
        assert result.features.shape == (3, 2)
        np.testing.assert_array_almost_equal(result.features[:, 0], [10.0, 20.0, 30.0])
        np.testing.assert_array_almost_equal(result.features[:, 1], [40.0, 50.0, 60.0])

    def test_non_numeric_columns_excluded(self, tmp_path: Path) -> None:
        """String columns must not leak into the feature matrix."""
        from mosaic.behavior.feature_library.helpers import (
            EntryData,
            StreamingFeatureHelper,
        )
        from mosaic.behavior.feature_library.spec import ParquetLoadSpec

        p = tmp_path / "with_strings.parquet"
        pd.DataFrame(
            {
                "frame": [0, 1],
                "label": ["walk", "run"],
                "category": ["A", "B"],
                "feat": [1.0, 2.0],
            }
        ).to_parquet(p, index=False)

        helper = StreamingFeatureHelper(None, "test")
        result = helper.load_entry_data([(p, ParquetLoadSpec())])
        assert isinstance(result, EntryData)
        assert result.features.shape == (2, 1)
        np.testing.assert_array_almost_equal(result.features[:, 0], [1.0, 2.0])

    def test_misaligned_row_alignment_correct(self, tmp_path: Path) -> None:
        """Verify row values are correctly aligned after merge, not just shape."""
        from mosaic.behavior.feature_library.helpers import (
            EntryData,
            StreamingFeatureHelper,
        )
        from mosaic.behavior.feature_library.spec import ParquetLoadSpec

        p1 = tmp_path / "a.parquet"
        p2 = tmp_path / "b.parquet"
        # Input 1: frames 0-3, input 2: frames 2-5 (reversed order)
        pd.DataFrame(
            {"frame": [0, 1, 2, 3], "feat_a": [100.0, 200.0, 300.0, 400.0]}
        ).to_parquet(p1, index=False)
        pd.DataFrame(
            {"frame": [5, 4, 3, 2], "feat_b": [50.0, 40.0, 30.0, 20.0]}
        ).to_parquet(p2, index=False)

        helper = StreamingFeatureHelper(None, "test")
        spec = ParquetLoadSpec()
        result = helper.load_entry_data([(p1, spec), (p2, spec)])
        assert isinstance(result, EntryData)
        # Overlap is frames 2,3. Values must be correctly paired by frame.
        np.testing.assert_array_equal(result.frames, [2, 3])
        np.testing.assert_array_almost_equal(result.features[:, 0], [300.0, 400.0])
        np.testing.assert_array_almost_equal(result.features[:, 1], [20.0, 30.0])


class TestLoadParquetDataFrame:
    """Tests for _load_parquet_dataframe."""

    def test_basic_load(self, parquet_path: Path) -> None:
        from mosaic.behavior.feature_library.helpers import _load_parquet_dataframe

        spec = ParquetLoadSpec()
        df = _load_parquet_dataframe(parquet_path, spec)
        assert df is not None
        assert "feat_a" in df.columns
        assert "feat_b" in df.columns
        assert "frame" in df.columns

    def test_with_df_filter(self, parquet_path: Path) -> None:
        from mosaic.behavior.feature_library.helpers import _load_parquet_dataframe

        spec = ParquetLoadSpec()
        df = _load_parquet_dataframe(
            parquet_path, spec, df_filter=lambda d: d[d["frame"] >= 1]
        )
        assert df is not None
        assert len(df) == 2

    def test_empty_after_filter_returns_none(self, parquet_path: Path) -> None:
        from mosaic.behavior.feature_library.helpers import _load_parquet_dataframe

        spec = ParquetLoadSpec()
        df = _load_parquet_dataframe(
            parquet_path, spec, df_filter=lambda d: d[d["frame"] > 100]
        )
        assert df is None

    def test_non_parquet_returns_none(self, npz_path: Path) -> None:
        from mosaic.behavior.feature_library.helpers import _load_parquet_dataframe

        spec = NpzLoadSpec(key="features")
        df = _load_parquet_dataframe(npz_path, spec)
        assert df is None


# --- _concat_into ---


class TestConcatInto:
    def test_vstack(self) -> None:
        a = np.array([[1, 2], [3, 4]], dtype=np.float32)
        b = np.array([[5, 6]], dtype=np.float32)
        result = _concat_into([a, b], (3, 2))
        assert result.shape == (3, 2)
        assert result.dtype == np.float32
        np.testing.assert_array_equal(result, [[1, 2], [3, 4], [5, 6]])

    def test_hstack(self) -> None:
        a = np.array([[1], [2], [3]], dtype=np.float32)
        b = np.array([[10, 20], [30, 40], [50, 60]], dtype=np.float32)
        result = _concat_into([a, b], (3, 3), axis=1)
        assert result.shape == (3, 3)
        np.testing.assert_array_equal(result, [[1, 10, 20], [2, 30, 40], [3, 50, 60]])

    def test_generator_input(self) -> None:
        """Generators should work (single-pass, no intermediate list)."""
        arrays = (np.full((2, 3), fill_value=i, dtype=np.float32) for i in range(3))
        result = _concat_into(arrays, (6, 3))
        assert result.shape == (6, 3)
        np.testing.assert_array_equal(result[0:2], 0.0)
        np.testing.assert_array_equal(result[2:4], 1.0)
        np.testing.assert_array_equal(result[4:6], 2.0)

    def test_single_array(self) -> None:
        a = np.array([[1, 2, 3]], dtype=np.float32)
        result = _concat_into([a], (1, 3))
        np.testing.assert_array_equal(result, a)


# --- _load_artifact_matrix (multi-file) ---


class _MockDatasetForArtifacts:
    """Minimal dataset that returns a features root directory."""

    def __init__(self, root: Path) -> None:
        self._root = root
        (root / "features").mkdir(parents=True, exist_ok=True)

    def get_root(self, key: str) -> Path:
        return self._root / key


class TestLoadArtifactMatrixNpz:
    def test_single_file(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "features" / "my-feat" / "run1"
        run_dir.mkdir(parents=True)
        np.savez(run_dir / "s1.npz", X=np.arange(6).reshape(2, 3))

        ds = _MockDatasetForArtifacts(tmp_path)
        artifact = ArtifactSpec(
            feature="my-feat",
            run_id="run1",
            load=NpzLoadSpec(key="X"),
            pattern="*.npz",
        )
        result = _load_artifact_matrix(ds, artifact)
        assert result.shape == (2, 3)
        assert result.dtype == np.float32

    def test_multiple_files_vstacked(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "features" / "my-feat" / "run1"
        run_dir.mkdir(parents=True)
        np.savez(run_dir / "s1.npz", X=np.array([[1, 2], [3, 4]]))
        np.savez(run_dir / "s2.npz", X=np.array([[5, 6]]))

        ds = _MockDatasetForArtifacts(tmp_path)
        artifact = ArtifactSpec(
            feature="my-feat",
            run_id="run1",
            load=NpzLoadSpec(key="X"),
            pattern="*.npz",
        )
        result = _load_artifact_matrix(ds, artifact)
        assert result.shape == (3, 2)
        np.testing.assert_array_equal(result, [[1, 2], [3, 4], [5, 6]])

    def test_transpose(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "features" / "my-feat" / "run1"
        run_dir.mkdir(parents=True)
        np.savez(run_dir / "s1.npz", X=np.arange(6).reshape(2, 3))

        ds = _MockDatasetForArtifacts(tmp_path)
        artifact = ArtifactSpec(
            feature="my-feat",
            run_id="run1",
            load=NpzLoadSpec(key="X", transpose=True),
            pattern="*.npz",
        )
        result = _load_artifact_matrix(ds, artifact)
        assert result.shape == (3, 2)

    def test_missing_key_raises(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "features" / "my-feat" / "run1"
        run_dir.mkdir(parents=True)
        np.savez(run_dir / "s1.npz", other=np.array([1]))

        ds = _MockDatasetForArtifacts(tmp_path)
        artifact = ArtifactSpec(
            feature="my-feat",
            run_id="run1",
            load=NpzLoadSpec(key="X"),
            pattern="*.npz",
        )
        with pytest.raises(FileNotFoundError, match="No NPZ containing key"):
            _load_artifact_matrix(ds, artifact)


class TestLoadArtifactMatrixParquet:
    def _make_feature_parquet(
        self,
        path: Path,
        n_rows: int = 3,
        extra_cols: dict | None = None,
    ) -> None:
        data: dict = {
            "frame": list(range(n_rows)),
            "id": [0] * n_rows,
            "feat_a": np.arange(n_rows, dtype=np.float64),
            "feat_b": np.arange(100, 100 + n_rows, dtype=np.float64),
        }
        if extra_cols:
            data.update(extra_cols)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(data).to_parquet(path, index=False)

    def test_single_file_numeric_only(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "features" / "my-feat" / "run1"
        self._make_feature_parquet(run_dir / "s1.parquet")

        ds = _MockDatasetForArtifacts(tmp_path)
        artifact = ArtifactSpec(
            feature="my-feat",
            run_id="run1",
            load=ParquetLoadSpec(),
            pattern="*.parquet",
        )
        result = _load_artifact_matrix(ds, artifact)
        # numeric_only=True drops frame/id, keeps feat_a + feat_b
        assert result.shape == (3, 2)
        assert result.dtype == np.float32
        np.testing.assert_array_almost_equal(result[:, 0], [0, 1, 2])
        np.testing.assert_array_almost_equal(result[:, 1], [100, 101, 102])

    def test_multiple_files_vstacked(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "features" / "my-feat" / "run1"
        self._make_feature_parquet(run_dir / "s1.parquet", n_rows=2)
        self._make_feature_parquet(run_dir / "s2.parquet", n_rows=3)

        ds = _MockDatasetForArtifacts(tmp_path)
        artifact = ArtifactSpec(
            feature="my-feat",
            run_id="run1",
            load=ParquetLoadSpec(),
            pattern="*.parquet",
        )
        result = _load_artifact_matrix(ds, artifact)
        assert result.shape == (5, 2)
        # s1: feat_a=[0,1], s2: feat_a=[0,1,2] -> vstacked
        np.testing.assert_array_almost_equal(result[:, 0], [0, 1, 0, 1, 2])

    def test_explicit_columns(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "features" / "my-feat" / "run1"
        self._make_feature_parquet(run_dir / "s1.parquet")

        ds = _MockDatasetForArtifacts(tmp_path)
        artifact = ArtifactSpec(
            feature="my-feat",
            run_id="run1",
            load=ParquetLoadSpec(columns=["feat_a"]),
            pattern="*.parquet",
        )
        result = _load_artifact_matrix(ds, artifact)
        assert result.shape == (3, 1)
        np.testing.assert_array_almost_equal(result[:, 0], [0, 1, 2])

    def test_meta_columns_dropped(self, tmp_path: Path) -> None:
        """frame, id, id1, id2 etc. must be excluded from the feature matrix."""
        run_dir = tmp_path / "features" / "my-feat" / "run1"
        path = run_dir / "s1.parquet"
        path.parent.mkdir(parents=True)
        pd.DataFrame(
            {
                "frame": [0, 1],
                "time": [0.0, 0.033],
                "id": [1, 1],
                "id1": [1, 1],
                "id2": [2, 2],
                "id_a": [1, 1],
                "id_b": [2, 2],
                "feat_x": [10.0, 20.0],
            }
        ).to_parquet(path, index=False)

        ds = _MockDatasetForArtifacts(tmp_path)
        artifact = ArtifactSpec(
            feature="my-feat",
            run_id="run1",
            load=ParquetLoadSpec(),
            pattern="*.parquet",
        )
        result = _load_artifact_matrix(ds, artifact)
        # Only feat_x survives
        assert result.shape == (2, 1)
        np.testing.assert_array_almost_equal(result[:, 0], [10.0, 20.0])

    def test_transpose(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "features" / "my-feat" / "run1"
        self._make_feature_parquet(run_dir / "s1.parquet", n_rows=3)

        ds = _MockDatasetForArtifacts(tmp_path)
        artifact = ArtifactSpec(
            feature="my-feat",
            run_id="run1",
            load=ParquetLoadSpec(transpose=True),
            pattern="*.parquet",
        )
        result = _load_artifact_matrix(ds, artifact)
        assert result.shape == (2, 3)

    def test_multiple_files_values_correct(self, tmp_path: Path) -> None:
        """Verify values are preserved across multi-file vstack."""
        run_dir = tmp_path / "features" / "my-feat" / "run1"
        p1 = run_dir / "s1.parquet"
        p2 = run_dir / "s2.parquet"
        p1.parent.mkdir(parents=True)
        pd.DataFrame({"feat": [1.0, 2.0]}).to_parquet(p1, index=False)
        pd.DataFrame({"feat": [3.0, 4.0, 5.0]}).to_parquet(p2, index=False)

        ds = _MockDatasetForArtifacts(tmp_path)
        artifact = ArtifactSpec(
            feature="my-feat",
            run_id="run1",
            load=ParquetLoadSpec(numeric_only=False),
            pattern="*.parquet",
        )
        result = _load_artifact_matrix(ds, artifact)
        assert result.shape == (5, 1)
        np.testing.assert_array_almost_equal(
            result[:, 0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
        )
