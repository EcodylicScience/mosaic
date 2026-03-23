"""Integration tests for run_feature()."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from mosaic.core.pipeline.index import feature_index, feature_index_path
from mosaic.core.pipeline.run import run_feature
from mosaic.core.pipeline.types import (
    InputRequire,
    Inputs,
    Params,
    Result,
    TrackInput,
)


# --- Mock dataset ---


class _MockDataset:
    def __init__(self, root: Path):
        self._root = root
        for directory in ("tracks", "features"):
            (root / directory).mkdir(parents=True, exist_ok=True)

    def get_root(self, key: str) -> Path:
        return self._root / key

    def resolve_path(self, stored_path: object, anchor: object = None) -> Path:
        path = Path(str(stored_path))
        return path if path.is_absolute() else self._root / path

    @property
    def meta(self) -> dict[str, object]:
        return {"fps_default": 30.0}


# --- Helpers ---


def _write_tracks_index(ds: _MockDataset, entries: list[tuple[str, str, Path]]) -> None:
    idx_path = ds.get_root("tracks") / "index.csv"
    rows = [{"group": g, "sequence": s, "abs_path": str(p)} for g, s, p in entries]
    pd.DataFrame(rows).to_csv(idx_path, index=False)


def _make_parquet(path: Path, n_rows: int = 10, start_frame: int = 0) -> None:
    df = pd.DataFrame(
        {
            "frame": range(start_frame, start_frame + n_rows),
            "time": [f / 30.0 for f in range(start_frame, start_frame + n_rows)],
            "id": [0] * n_rows,
            "feat_a": np.random.randn(n_rows),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


def _setup_tracks(
    ds: _MockDataset, pairs: list[tuple[str, str]], n_rows: int = 10
) -> list[tuple[str, str, Path]]:
    """Create track parquets and write the tracks index. Returns entries."""
    entries: list[tuple[str, str, Path]] = []
    for group, sequence in pairs:
        path = ds.get_root("tracks") / f"{group}__{sequence}.parquet"
        _make_parquet(path, n_rows=n_rows)
        entries.append((group, sequence, path))
    _write_tracks_index(ds, entries)
    return entries


# --- Mock features ---


class _StatelessFeature:
    name = "test-stateless"
    version = "0.1"
    parallelizable = True
    scope_dependent = False

    class Inputs(Inputs[TrackInput]):
        pass

    class Params(Params):
        pass

    def __init__(
        self,
        inputs: _StatelessFeature.Inputs | None = None,
        params: dict[str, object] | None = None,
    ):
        self._inputs = inputs or self.Inputs(("tracks",))
        self._params = self.Params.from_overrides(params)

    @property
    def inputs(self) -> _StatelessFeature.Inputs:
        return self._inputs

    @property
    def params(self) -> _StatelessFeature.Params:
        return self._params

    def load_state(
        self,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_indices: dict[str, pd.DataFrame],
    ) -> bool:
        return True

    def fit(self, inputs: Callable[[], Iterator[tuple[str, pd.DataFrame]]]) -> None:
        pass

    def save_state(self, run_root: Path) -> None:
        pass

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"frame": df["frame"], "value": df["feat_a"] * 2})


class _StatefulFeature:
    name = "test-stateful"
    version = "0.1"
    parallelizable = True
    scope_dependent = True

    class Inputs(Inputs[TrackInput]):
        pass

    class Params(Params):
        pass

    def __init__(
        self,
        inputs: _StatefulFeature.Inputs | None = None,
        params: dict[str, object] | None = None,
    ):
        self._inputs = inputs or self.Inputs(("tracks",))
        self._params = self.Params.from_overrides(params)
        self._fitted = False
        self._mean = 0.0

    @property
    def inputs(self) -> _StatefulFeature.Inputs:
        return self._inputs

    @property
    def params(self) -> _StatefulFeature.Params:
        return self._params

    def load_state(
        self,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_indices: dict[str, pd.DataFrame],
    ) -> bool:
        state_path = run_root / "state.json"
        if state_path.exists():
            data = json.loads(state_path.read_text())
            self._fitted = True
            self._mean = data["mean"]
            return True
        return False

    def fit(self, factory: Callable[[], Iterator[tuple[str, pd.DataFrame]]]) -> None:
        all_values: list[float] = []
        for _key, df in factory():
            all_values.extend(df["feat_a"].tolist())
        self._mean = float(np.mean(all_values))
        self._fitted = True

    def save_state(self, run_root: Path) -> None:
        state_path = run_root / "state.json"
        state_path.write_text(json.dumps({"mean": self._mean}))

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {"frame": df["frame"], "centered": df["feat_a"] - self._mean}
        )


class _EmptyInputFeature:
    name = "test-global"
    version = "0.1"
    parallelizable = False
    scope_dependent = False

    class Inputs(Inputs[Result]):
        _require: ClassVar[InputRequire] = "empty"

    class Params(Params):
        pass

    def __init__(
        self,
        inputs: _EmptyInputFeature.Inputs | None = None,
        params: dict[str, object] | None = None,
    ):
        self._inputs = inputs or self.Inputs(())
        self._params = self.Params.from_overrides(params)

    @property
    def inputs(self) -> _EmptyInputFeature.Inputs:
        return self._inputs

    @property
    def params(self) -> _EmptyInputFeature.Params:
        return self._params

    def load_state(
        self,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_indices: dict[str, pd.DataFrame],
    ) -> bool:
        return True

    def fit(self, inputs: Callable[[], Iterator[tuple[str, pd.DataFrame]]]) -> None:
        pass

    def save_state(self, run_root: Path) -> None:
        pass

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame()


# --- Tests ---


def test_stateless_basic(tmp_path: Path) -> None:
    ds = _MockDataset(tmp_path)
    _setup_tracks(ds, [("g", "s1"), ("g", "s2")])

    feature = _StatelessFeature()
    result = run_feature(ds, feature)

    assert isinstance(result, Result)
    assert result.feature == "test-stateless__from__tracks"
    assert result.run_id is not None and len(result.run_id) > 0

    # Output parquets exist
    from mosaic.core.pipeline.index import feature_run_root

    run_root = feature_run_root(ds, result.feature, result.run_id)
    parquets = sorted(run_root.glob("*.parquet"))
    assert len(parquets) == 2

    # Index CSV has 2 finished rows
    idx = feature_index(feature_index_path(ds, result.feature))
    idx_df = idx.read(run_id=result.run_id)
    assert len(idx_df) == 2
    assert all(idx_df["finished_at"] != "")


def test_stateful_fit_and_apply(tmp_path: Path) -> None:
    ds = _MockDataset(tmp_path)
    _setup_tracks(ds, [("g", "s1"), ("g", "s2")])

    feature = _StatefulFeature()
    result = run_feature(ds, feature)

    # State file written
    from mosaic.core.pipeline.index import feature_run_root

    run_root = feature_run_root(ds, result.feature, result.run_id)
    state_path = run_root / "state.json"
    assert state_path.exists()

    # Output parquets exist
    parquets = sorted(run_root.glob("*.parquet"))
    assert len(parquets) == 2

    # Verify centered values: read all input feat_a to compute expected mean
    all_feat_a: list[float] = []
    for pq_path in sorted(ds.get_root("tracks").glob("*.parquet")):
        track_df = pd.read_parquet(pq_path)
        all_feat_a.extend(track_df["feat_a"].tolist())
    expected_mean = float(np.mean(all_feat_a))

    # Check each output
    for pq_path in parquets:
        out_df = pd.read_parquet(pq_path)
        assert "centered" in out_df.columns
        # Read corresponding input to verify
        track_name = pq_path.name
        track_path = ds.get_root("tracks") / track_name
        track_df = pd.read_parquet(track_path)
        expected_centered = track_df["feat_a"].values - expected_mean
        np.testing.assert_allclose(
            out_df["centered"].values, expected_centered, atol=1e-10
        )


def test_overwrite_false_skips(tmp_path: Path) -> None:
    ds = _MockDataset(tmp_path)
    _setup_tracks(ds, [("g", "s1")])

    feature = _StatelessFeature()

    # First run
    result1 = run_feature(ds, feature)
    from mosaic.core.pipeline.index import feature_run_root

    run_root = feature_run_root(ds, result1.feature, result1.run_id)
    parquets = sorted(run_root.glob("*.parquet"))
    assert len(parquets) == 1
    mtime_before = os.path.getmtime(parquets[0])

    # Second run with overwrite=False
    result2 = run_feature(ds, feature, overwrite=False)
    assert result2.run_id == result1.run_id

    parquets_after = sorted(run_root.glob("*.parquet"))
    mtime_after = os.path.getmtime(parquets_after[0])
    assert mtime_after == mtime_before


def test_global_marker(tmp_path: Path) -> None:
    ds = _MockDataset(tmp_path)
    # Tracks exist but feature has empty inputs
    _setup_tracks(ds, [("g", "s1")])

    feature = _EmptyInputFeature()
    result = run_feature(ds, feature)

    assert result.feature == "test-global"

    # No parquet files in run root
    from mosaic.core.pipeline.index import feature_run_root

    run_root = feature_run_root(ds, result.feature, result.run_id)
    parquets = list(run_root.glob("*.parquet"))
    assert len(parquets) == 0

    # Index CSV has a global marker row
    idx = feature_index(feature_index_path(ds, result.feature))
    idx_df = idx.read(run_id=result.run_id)
    assert len(idx_df) == 1
    assert idx_df.iloc[0]["sequence"] == "__global__"
    assert idx_df.iloc[0]["group"] == ""


def test_frame_filter(tmp_path: Path) -> None:
    ds = _MockDataset(tmp_path)
    _setup_tracks(ds, [("g", "s1")], n_rows=20)

    feature = _StatelessFeature()
    result = run_feature(ds, feature, filter_start_frame=5, filter_end_frame=15)

    from mosaic.core.pipeline.index import feature_run_root

    run_root = feature_run_root(ds, result.feature, result.run_id)
    parquets = sorted(run_root.glob("*.parquet"))
    assert len(parquets) == 1

    out_df = pd.read_parquet(parquets[0])
    assert list(out_df["frame"]) == list(range(5, 15))


def test_scope_dependent_hashing(tmp_path: Path) -> None:
    ds = _MockDataset(tmp_path)
    _setup_tracks(ds, [("g", "s1"), ("g", "s2"), ("g", "s3")])

    feature1 = _StatefulFeature()
    result1 = run_feature(ds, feature1, sequences=["s1", "s2"])

    feature2 = _StatefulFeature()
    result2 = run_feature(ds, feature2, sequences=["s1", "s3"])

    assert result1.run_id != result2.run_id


def test_overlap_frame_filter_mutual_exclusion(tmp_path: Path) -> None:
    ds = _MockDataset(tmp_path)
    _setup_tracks(ds, [("g", "s1")])

    feature = _StatelessFeature()
    with pytest.raises(ValueError, match="mutually exclusive"):
        run_feature(ds, feature, overlap_frames=5, filter_start_frame=0)


def test_result_type(tmp_path: Path) -> None:
    ds = _MockDataset(tmp_path)
    _setup_tracks(ds, [("g", "s1")])

    feature = _StatelessFeature()
    result = run_feature(ds, feature)

    assert isinstance(result, Result)
    assert isinstance(result.feature, str)
    assert isinstance(result.run_id, str)
