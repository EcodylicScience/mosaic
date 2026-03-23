from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import pandas as pd
import pytest

from mosaic.core.pipeline.frames import FramesIndexRow, frames_index
from mosaic.core.pipeline.index import FeatureIndexRow, feature_index
from mosaic.core.pipeline.index_csv import IndexCSV
from mosaic.core.pipeline.models import ModelIndexRow, model_index


@pytest.fixture
def tmp_csv(tmp_path):
    return tmp_path / "index.csv"


# --- Schema ---


@dataclass(frozen=True, slots=True)
class SampleRow:
    name: str
    status: str = ""
    value: int = 0


class TestEnsure:
    def test_creates_file_with_columns(self, tmp_csv):
        idx: IndexCSV[SampleRow] = IndexCSV(tmp_csv, SampleRow)
        idx.ensure()
        assert tmp_csv.exists()
        df = pd.read_csv(tmp_csv)
        assert list(df.columns) == ["name", "status", "value"]
        assert len(df) == 0

    def test_idempotent(self, tmp_csv):
        idx: IndexCSV[SampleRow] = IndexCSV(tmp_csv, SampleRow)
        idx.ensure()
        idx.ensure()
        df = pd.read_csv(tmp_csv)
        assert len(df) == 0

    def test_creates_parent_dirs(self, tmp_path):
        deep = tmp_path / "a" / "b" / "index.csv"
        idx: IndexCSV[SampleRow] = IndexCSV(deep, SampleRow)
        idx.ensure()
        assert deep.exists()


class TestAppend:
    def test_append_creates_if_missing(self, tmp_csv):
        idx: IndexCSV[SampleRow] = IndexCSV(tmp_csv, SampleRow)
        idx.append([SampleRow(name="foo", value=1, status="ok")])
        df = pd.read_csv(tmp_csv)
        assert len(df) == 1
        assert df.iloc[0]["name"] == "foo"

    def test_append_adds_rows(self, tmp_csv):
        idx: IndexCSV[SampleRow] = IndexCSV(tmp_csv, SampleRow)
        idx.ensure()
        idx.append([SampleRow(name="a", value=1, status="ok")])
        idx.append([SampleRow(name="b", value=2, status="ok")])
        df = pd.read_csv(tmp_csv)
        assert len(df) == 2

    def test_append_fills_missing_keys(self, tmp_csv):
        idx: IndexCSV[SampleRow] = IndexCSV(tmp_csv, SampleRow)
        idx.append([SampleRow(name="x")])
        df = pd.read_csv(tmp_csv)
        assert df.iloc[0]["name"] == "x"
        # missing string columns get empty string, int columns get default 0
        assert df.iloc[0]["status"] == "" or pd.isna(df.iloc[0]["status"])


class TestDedup:
    def test_dedup_by_keys(self, tmp_csv):
        idx: IndexCSV[SampleRow] = IndexCSV(tmp_csv, SampleRow, dedup_keys=["name"])
        idx.append([SampleRow(name="a", value=1, status="v1")])
        idx.append([SampleRow(name="a", value=2, status="v2")])
        df = pd.read_csv(tmp_csv)
        assert len(df) == 1
        assert df.iloc[0]["value"] == 2

    def test_dedup_composite_key(self, tmp_csv):
        idx: IndexCSV[SampleRow] = IndexCSV(
            tmp_csv, SampleRow, dedup_keys=["name", "status"]
        )
        idx.append([SampleRow(name="a", value=1, status="v1")])
        idx.append([SampleRow(name="a", value=2, status="v2")])
        df = pd.read_csv(tmp_csv)
        assert len(df) == 2  # different composite key, both kept

    def test_dedup_no_keys_means_no_dedup(self, tmp_csv):
        idx: IndexCSV[SampleRow] = IndexCSV(tmp_csv, SampleRow)
        idx.append([SampleRow(name="a", value=1, status="ok")])
        idx.append([SampleRow(name="a", value=2, status="ok")])
        df = pd.read_csv(tmp_csv)
        assert len(df) == 2


class TestRead:
    def test_read_returns_dataframe(self, tmp_csv):
        idx: IndexCSV[SampleRow] = IndexCSV(tmp_csv, SampleRow)
        idx.append([SampleRow(name="a", value=1, status="ok")])
        df = idx.read()
        assert len(df) == 1

    def test_read_empty(self, tmp_csv):
        idx: IndexCSV[SampleRow] = IndexCSV(tmp_csv, SampleRow)
        idx.ensure()
        df = idx.read()
        assert len(df) == 0

    def test_read_missing_file_raises(self, tmp_csv):
        idx: IndexCSV[SampleRow] = IndexCSV(tmp_csv, SampleRow)
        with pytest.raises(FileNotFoundError):
            idx.read()


# --- Feature Index ---


class TestFeatureIndex:
    def test_schema_has_required_columns(self):
        names = {f.name for f in dataclasses.fields(FeatureIndexRow)}
        assert "feature" in names
        assert "run_id" in names
        assert "n_rows" in names
        assert "finished_at" in names

    def test_factory_returns_index_csv(self, tmp_path):
        idx = feature_index(tmp_path / "index.csv")
        assert isinstance(idx, IndexCSV)

    def test_ensure_creates_with_all_columns(self, tmp_path):
        idx = feature_index(tmp_path / "index.csv")
        idx.ensure()
        df = pd.read_csv(tmp_path / "index.csv")
        assert "feature" in df.columns

    def test_dedup_by_run_group_sequence(self, tmp_path):
        idx = feature_index(tmp_path / "index.csv")
        p = tmp_path / "data.parquet"
        p.touch()
        row = FeatureIndexRow(
            run_id="v1-abc",
            feature="speed",
            version="v1",
            group="a",
            sequence="s1",
            abs_path=str(p),
            params_hash="h",
            n_rows=10,
        )
        idx.append([row])
        row2 = FeatureIndexRow(
            run_id="v1-abc",
            feature="speed",
            version="v1",
            group="a",
            sequence="s1",
            abs_path=str(p),
            params_hash="h",
            n_rows=20,
        )
        idx.append([row2])
        df = idx.read()
        assert len(df) == 1
        assert df.iloc[0]["n_rows"] == 20


class TestFeatureIndexRow:
    def test_fields_match_schema(self, tmp_path):
        p = tmp_path / "G1__S1.parquet"
        p.touch()
        row = FeatureIndexRow(
            run_id="abc123",
            feature="speed",
            version="0.1",
            group="G1",
            sequence="S1",
            abs_path=str(p),
            n_rows=100,
            params_hash="deadbeef",
        )
        df = pd.DataFrame([row])
        assert set(df.columns) == {f.name for f in dataclasses.fields(FeatureIndexRow)}
        assert df.iloc[0]["feature"] == "speed"
        assert df.iloc[0]["n_rows"] == 100
        assert df.iloc[0]["finished_at"] == ""

    def test_finished_at_default(self, tmp_path):
        p = tmp_path / "data.parquet"
        p.touch()
        row = FeatureIndexRow(
            run_id="r",
            feature="f",
            version="v",
            group="",
            sequence="s",
            abs_path=str(p),
            n_rows=0,
            params_hash="h",
        )
        assert row.finished_at == ""

    def test_started_at_auto_populated(self, tmp_path):
        p = tmp_path / "data.parquet"
        p.touch()
        row = FeatureIndexRow(
            run_id="r",
            feature="f",
            version="v",
            group="",
            sequence="s",
            abs_path=str(p),
            params_hash="h",
        )
        assert row.started_at != ""
        assert "T" in row.started_at  # ISO format

    def test_empty_abs_path_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            FeatureIndexRow(
                run_id="r",
                feature="f",
                version="v",
                group="",
                sequence="s",
                abs_path="",
                params_hash="h",
            )

    def test_nonexistent_abs_path_raises(self):
        with pytest.raises(FileNotFoundError, match="does not exist"):
            FeatureIndexRow(
                run_id="r",
                feature="f",
                version="v",
                group="",
                sequence="s",
                abs_path="/no/such/file.parquet",
                params_hash="h",
            )

    def test_appendable_to_feature_index(self, tmp_path):
        idx = feature_index(tmp_path / "index.csv")
        p = tmp_path / "data.parquet"
        p.touch()
        row = FeatureIndexRow(
            run_id="abc",
            feature="speed",
            version="0.1",
            group="G1",
            sequence="S1",
            abs_path=str(p),
            n_rows=10,
            params_hash="hash",
        )
        idx.append([row])
        df = idx.read()
        assert len(df) == 1
        assert df.iloc[0]["feature"] == "speed"


# --- Frames Index ---


class TestFramesIndex:
    def test_schema_has_required_columns(self):
        names = {f.name for f in dataclasses.fields(FramesIndexRow)}
        assert "method" in names
        assert "n_frames_extracted" in names

    def test_factory_returns_index_csv(self, tmp_path):
        idx = frames_index(tmp_path / "index.csv")
        assert isinstance(idx, IndexCSV)

    def test_ensure_creates(self, tmp_path):
        idx = frames_index(tmp_path / "index.csv")
        idx.ensure()
        df = pd.read_csv(tmp_path / "index.csv")
        assert "method" in df.columns

    def test_dedup(self, tmp_path):
        idx = frames_index(tmp_path / "index.csv")
        row = FramesIndexRow(
            run_id="r1",
            method="uniform",
            group="g",
            sequence="s",
            abs_path="p",
            video_abs_path="v",
            params_hash="h",
            n_frames_extracted=10,
        )
        idx.append([row])
        row2 = FramesIndexRow(
            run_id="r1",
            method="uniform",
            group="g",
            sequence="s",
            abs_path="p",
            video_abs_path="v",
            params_hash="h",
            n_frames_extracted=20,
        )
        idx.append([row2])
        df = idx.read()
        assert len(df) == 1


class TestFramesIndexRow:
    def test_fields_match_schema(self):
        row = FramesIndexRow(
            run_id="r1",
            method="uniform",
            group="G1",
            sequence="S1",
            abs_path="frames/uniform/r1/G1__S1",
            n_frames_extracted=50,
            n_frames_requested=50,
            video_abs_path="/videos/v1.mp4",
            params_hash="abc",
        )
        df = pd.DataFrame([row])
        assert set(df.columns) == {f.name for f in dataclasses.fields(FramesIndexRow)}
        assert df.iloc[0]["method"] == "uniform"
        assert df.iloc[0]["n_frames_extracted"] == 50

    def test_finished_at_default(self):
        row = FramesIndexRow(
            run_id="r",
            method="m",
            group="",
            sequence="s",
            abs_path="p",
            n_frames_extracted=0,
            n_frames_requested=0,
            video_abs_path="v",
            params_hash="h",
        )
        assert row.finished_at == ""

    def test_appendable_to_frames_index(self, tmp_path):
        idx = frames_index(tmp_path / "index.csv")
        row = FramesIndexRow(
            run_id="r1",
            method="uniform",
            group="G1",
            sequence="S1",
            abs_path="path",
            n_frames_extracted=10,
            n_frames_requested=50,
            video_abs_path="/v.mp4",
            params_hash="h",
        )
        idx.append([row])
        df = idx.read()
        assert len(df) == 1
        assert df.iloc[0]["method"] == "uniform"


# --- Model Index ---


class TestModelIndex:
    def test_schema_has_required_columns(self):
        names = {f.name for f in dataclasses.fields(ModelIndexRow)}
        assert "model" in names
        assert "config_hash" in names

    def test_factory_returns_index_csv(self, tmp_path):
        idx = model_index(tmp_path / "index.csv")
        assert isinstance(idx, IndexCSV)

    def test_ensure_creates(self, tmp_path):
        idx = model_index(tmp_path / "index.csv")
        idx.ensure()
        df = pd.read_csv(tmp_path / "index.csv")
        assert "model" in df.columns

    def test_append_no_dedup(self, tmp_path):
        idx = model_index(tmp_path / "index.csv")
        row = ModelIndexRow(
            run_id="r1",
            model="m1",
            version="v1",
            config_path="c",
            config_hash="h",
            metrics_path="",
            status="ok",
            notes="",
        )
        idx.append([row])
        idx.append([row])
        df = idx.read()
        assert len(df) == 2  # model index is append-only, no dedup


class TestModelIndexRow:
    def test_fields_match_schema(self):
        row = ModelIndexRow(
            run_id="0.1-abc",
            model="xgb",
            version="0.1",
            config_path="/path/config.json",
            config_hash="abc",
            metrics_path="/path/metrics.json",
            status="success",
            notes="",
        )
        df = pd.DataFrame([row])
        assert set(df.columns) == {f.name for f in dataclasses.fields(ModelIndexRow)}
        assert df.iloc[0]["model"] == "xgb"
        assert df.iloc[0]["status"] == "success"

    def test_appendable_to_model_index(self, tmp_path):
        idx = model_index(tmp_path / "index.csv")
        row = ModelIndexRow(
            run_id="r1",
            model="xgb",
            version="0.1",
            config_path="c",
            config_hash="h",
            metrics_path="",
            status="success",
            notes="",
        )
        idx.append([row])
        df = idx.read()
        assert len(df) == 1
        assert df.iloc[0]["model"] == "xgb"


# --- latest_run_id ---


@dataclass(frozen=True, slots=True)
class RunRow:
    run_id: str
    started_at: str
    finished_at: str = ""


class TestLatestRunId:
    def test_returns_latest_finished(self, tmp_path):
        idx_path = tmp_path / "index.csv"
        idx = IndexCSV(idx_path, RunRow)
        idx.append(
            [
                RunRow(
                    run_id="old",
                    started_at="2025-01-01T00:00:00",
                    finished_at="2025-01-01T01:00:00",
                ),
                RunRow(
                    run_id="new",
                    started_at="2025-01-02T00:00:00",
                    finished_at="2025-01-02T01:00:00",
                ),
            ]
        )
        assert idx.latest_run_id() == "new"

    def test_prefers_finished_over_unfinished(self, tmp_path):
        idx_path = tmp_path / "index.csv"
        idx = IndexCSV(idx_path, RunRow)
        idx.append(
            [
                RunRow(
                    run_id="finished",
                    started_at="2025-01-01T00:00:00",
                    finished_at="2025-01-01T01:00:00",
                ),
                RunRow(
                    run_id="in_progress",
                    started_at="2025-06-01T00:00:00",
                    finished_at="",
                ),
            ]
        )
        assert idx.latest_run_id() == "finished"

    def test_falls_back_to_started_at(self, tmp_path):
        idx_path = tmp_path / "index.csv"
        idx = IndexCSV(idx_path, RunRow)
        idx.append(
            [
                RunRow(
                    run_id="old",
                    started_at="2025-01-01T00:00:00",
                    finished_at="",
                ),
                RunRow(
                    run_id="new",
                    started_at="2025-06-01T00:00:00",
                    finished_at="",
                ),
            ]
        )
        assert idx.latest_run_id() == "new"

    def test_empty_raises(self, tmp_path):
        idx_path = tmp_path / "index.csv"
        idx = IndexCSV(idx_path, RunRow)
        idx.ensure()
        with pytest.raises(ValueError, match="No runs found"):
            idx.latest_run_id()
