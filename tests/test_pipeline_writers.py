from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mosaic.core.pipeline._utils import ChunkedPayload, DataPayload, FeatureMeta
from mosaic.core.pipeline.run import build_feature_meta, build_output_path
from mosaic.core.pipeline.writers import trim_feature_output, write_output


@pytest.fixture
def meta(tmp_path):
    return FeatureMeta(
        group="g1",
        sequence="s1",
        out_path=tmp_path / "g1__s1.parquet",
    )


class TestTrimFeatureOutput:
    def test_trim_dataframe(self):
        df = pd.DataFrame({"a": range(10)})
        trimmed = trim_feature_output(df, 2, 8)
        assert len(trimmed) == 6
        assert trimmed.iloc[0]["a"] == 2

    def test_trim_noop_when_full_range(self):
        df = pd.DataFrame({"a": range(5)})
        trimmed = trim_feature_output(df, 0, 5)
        assert len(trimmed) == 5

    def test_trim_none_returns_none(self):
        assert trim_feature_output(None, 0, 5) is None

    def test_trim_chunked_payload(self):
        data = np.arange(20).reshape(10, 2).astype(float)
        cp = ChunkedPayload(parquet_data=data, columns=["x", "y"])
        trimmed = trim_feature_output(cp, 2, 7)
        assert isinstance(trimmed, ChunkedPayload)
        assert trimmed.parquet_data.shape[0] == 5

    def test_trim_data_payload(self):
        data = np.arange(20).reshape(10, 2).astype(float)
        dp = DataPayload(data=data, columns=["x", "y"])
        trimmed = trim_feature_output(dp, 3, 8)
        assert isinstance(trimmed, DataPayload)
        assert trimmed.data.shape[0] == 5


class TestWriteOutput:
    def test_write_dataframe(self, meta):
        df = pd.DataFrame({"frame": [0, 1], "val": [1.0, 2.0], "sequence": ["s1", "s1"]})
        n_rows = write_output(meta, df)
        assert meta.out_path.exists()
        assert n_rows == 2
        result = pd.read_parquet(meta.out_path)
        assert len(result) == 2

    def test_write_data_payload(self, meta):
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        dp = DataPayload(data=data, columns=["x", "y"], sequence="s1", group="g1")
        n_rows = write_output(meta, dp)
        assert meta.out_path.exists()
        assert n_rows == 2

    def test_write_chunked_payload(self, meta):
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        cp = ChunkedPayload(parquet_data=data, columns=["x", "y"], sequence="s1", group="g1", chunk_size=2)
        n_rows = write_output(meta, cp)
        assert meta.out_path.exists()
        assert n_rows == 3

    def test_write_none_creates_empty(self, meta):
        n_rows = write_output(meta, None)
        assert meta.out_path.exists()
        assert n_rows == 0


class TestBuildFeatureMeta:
    def test_basic(self, tmp_path):
        meta = build_feature_meta("group1", "seq1", tmp_path)
        assert meta.group == "group1"
        assert meta.sequence == "seq1"
        assert meta.out_path.parent == tmp_path

    def test_empty_group(self, tmp_path):
        meta = build_feature_meta("", "seq1", tmp_path)
        assert "__" not in meta.out_path.name  # no double-underscore prefix


class TestBuildOutputPath:
    def test_with_group(self, tmp_path):
        path = build_output_path("g1", "s1", tmp_path)
        assert path.name.endswith(".parquet")
        assert path.parent == tmp_path

    def test_without_group(self, tmp_path):
        path = build_output_path("", "s1", tmp_path)
        assert "__" not in path.name or path.name.startswith("__") is False
