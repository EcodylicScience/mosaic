"""Integration tests for run_feature()."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated, ClassVar

import numpy as np
import pandas as pd
import pytest

from mosaic.core.pipeline.index import feature_index, feature_index_path
from mosaic.core.pipeline.run import run_feature
from mosaic.core.pipeline.types import (
    HASH_EXCLUDE,
    InputRequire,
    Inputs,
    InputStream,
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

    def relative_to_root(self, path: object) -> str:
        try:
            return str(Path(str(path)).resolve().relative_to(self._root.resolve()))
        except ValueError:
            return str(path)

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
        dependency_lookups: dict[str, dict[tuple[str, str], Path]],
    ) -> bool:
        return True

    def fit(self, inputs: InputStream) -> None:
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
        dependency_lookups: dict[str, dict[tuple[str, str], Path]],
    ) -> bool:
        state_path = run_root / "state.json"
        if state_path.exists():
            data = json.loads(state_path.read_text())
            self._fitted = True
            self._mean = data["mean"]
            return True
        return False

    def fit(self, factory: InputStream) -> None:
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
        dependency_lookups: dict[str, dict[tuple[str, str], Path]],
    ) -> bool:
        return True

    def fit(self, inputs: InputStream) -> None:
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


def test_entries_selects_arbitrary_subset(tmp_path: Path) -> None:
    """run_feature(entries=...) runs exactly the requested (group, sequence) pairs.

    Sequence name "s1" repeats across groups g1 and g2, so a bare sequences=
    filter would be ambiguous; entries= selects (g1, s1) and (g2, s1) while
    excluding (g1, s2) -- the arbitrary, tag-resolvable subset case.
    """
    ds = _MockDataset(tmp_path)
    _setup_tracks(ds, [("g1", "s1"), ("g1", "s2"), ("g2", "s1")])

    feature = _StatelessFeature()
    result = run_feature(ds, feature, entries=[("g1", "s1"), ("g2", "s1")])

    from mosaic.core.pipeline.index import feature_run_root

    run_root = feature_run_root(ds, result.feature, result.run_id)
    parquets = sorted(p.name for p in run_root.glob("*.parquet"))
    assert parquets == ["g1__s1.parquet", "g2__s1.parquet"]

    idx = feature_index(feature_index_path(ds, result.feature))
    idx_df = idx.read(run_id=result.run_id)
    produced = set(zip(idx_df["group"], idx_df["sequence"]))
    assert produced == {("g1", "s1"), ("g2", "s1")}


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


# --- check_output / cache-hit pre-pass ---


class _CountingStateless(_StatelessFeature):
    """Stateless feature that records how many times apply() runs."""

    name = "test-counting"

    def __init__(
        self,
        inputs: _StatelessFeature.Inputs | None = None,
        params: dict[str, object] | None = None,
    ):
        super().__init__(inputs, params)
        self.apply_calls = 0

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        self.apply_calls += 1
        return super().apply(df)


class _SidecarFeature(_StatelessFeature):
    """Feature whose output is only valid if a side-car file also exists.

    Models non-idempotent features (e.g. the crop pipeline writing .mp4s)
    via a custom check_output() override.
    """

    name = "test-sidecar"

    def load_state(
        self,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_lookups: dict[str, dict[tuple[str, str], Path]],
    ) -> bool:
        self._run_root = run_root
        return True

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        (self._run_root / "sidecar.flag").write_text("x")
        return super().apply(df)

    def check_output(self, meta, run_root) -> bool:
        return (run_root / "sidecar.flag").exists()


class _RaisingCheckFeature(_CountingStateless):
    """Feature whose custom check_output raises (a buggy validator)."""

    name = "test-raising-check"

    def check_output(self, meta, run_root) -> bool:
        raise RuntimeError("validator boom")


class _NonCallableCheckFeature(_CountingStateless):
    """Feature that mistakenly defines check_output as a non-callable flag."""

    name = "test-noncallable-check"
    check_output = True  # type: ignore[assignment]


def test_cache_hit_skips_without_loading_input(tmp_path: Path) -> None:
    ds = _MockDataset(tmp_path)
    _setup_tracks(ds, [("g", "s1"), ("g", "s2")])

    f1 = _CountingStateless()
    run_feature(ds, f1)
    assert f1.apply_calls == 2  # first run computes both entries

    f2 = _CountingStateless()
    run_feature(ds, f2)
    assert f2.apply_calls == 0  # all cache hits -> nothing recomputed

    # Corrupt the INPUT parquet but leave it on disk (stays in the manifest).
    # A cache hit must NOT deserialize it; loading would raise on the garbage.
    (ds.get_root("tracks") / "g__s1.parquet").write_bytes(b"not a parquet")

    f3 = _CountingStateless()
    run_feature(ds, f3)  # must not raise: cached input is never loaded
    assert f3.apply_calls == 0


def test_check_output_recomputes_corrupt_parquet(tmp_path: Path) -> None:
    ds = _MockDataset(tmp_path)
    _setup_tracks(ds, [("g", "s1"), ("g", "s2")])
    from mosaic.core.pipeline.index import feature_run_root

    r1 = run_feature(ds, _StatelessFeature())
    run_root = feature_run_root(ds, r1.feature, r1.run_id)
    s1_out = run_root / "g__s1.parquet"
    s2_mtime = os.path.getmtime(run_root / "g__s2.parquet")

    # Corrupt s1's OUTPUT so a deep read fails (footer + data unreadable).
    s1_out.write_bytes(b"PAR1 not a valid parquet PAR1")

    # check_output=True: deep validation fails for s1 -> recompute; s2 stays cached.
    r2 = run_feature(ds, _StatelessFeature(), check_output=True)
    assert r2.run_id == r1.run_id

    out = pd.read_parquet(s1_out)  # readable again
    assert "value" in out.columns
    assert len(out) == 10
    # s2 was a valid cache hit and must not have been rewritten.
    assert os.path.getmtime(run_root / "g__s2.parquet") == s2_mtime


def test_corrupt_output_recomputes_on_default_path(tmp_path: Path) -> None:
    # check_output=False (the default) must not crash on an unreadable cached
    # output: a truncated/corrupt footer falls through to recompute that entry.
    ds = _MockDataset(tmp_path)
    _setup_tracks(ds, [("g", "s1"), ("g", "s2")])
    from mosaic.core.pipeline.index import feature_run_root

    f1 = _CountingStateless()
    r1 = run_feature(ds, f1)
    assert f1.apply_calls == 2
    run_root = feature_run_root(ds, r1.feature, r1.run_id)
    s1_out = run_root / "g__s1.parquet"
    s2_mtime = os.path.getmtime(run_root / "g__s2.parquet")

    # Truncate s1's output so the footer is unreadable (output_n_rows would raise).
    s1_out.write_bytes(b"PAR1 truncated garbage")

    f2 = _CountingStateless()
    run_feature(ds, f2)  # must not raise
    assert f2.apply_calls == 1  # only s1 recomputed; s2 stayed a cache hit
    out = pd.read_parquet(s1_out)  # readable again
    assert "value" in out.columns
    assert len(out) == 10
    assert os.path.getmtime(run_root / "g__s2.parquet") == s2_mtime


def test_custom_check_output_triggers_recompute(tmp_path: Path) -> None:
    ds = _MockDataset(tmp_path)
    _setup_tracks(ds, [("g", "s1")])
    from mosaic.core.pipeline.index import feature_run_root

    r1 = run_feature(ds, _SidecarFeature())
    run_root = feature_run_root(ds, r1.feature, r1.run_id)
    flag = run_root / "sidecar.flag"
    assert flag.exists()
    flag.unlink()

    # Default fast path: cache hit on existence -> skipped, side-car NOT restored.
    run_feature(ds, _SidecarFeature(), check_output=False)
    assert not flag.exists()

    # check_output=True: custom validator fails -> recompute restores the side-car.
    run_feature(ds, _SidecarFeature(), check_output=True)
    assert flag.exists()


def test_raising_custom_check_recomputes_instead_of_crashing(tmp_path: Path) -> None:
    ds = _MockDataset(tmp_path)
    _setup_tracks(ds, [("g", "s1")])

    f1 = _RaisingCheckFeature()
    run_feature(ds, f1)
    assert f1.apply_calls == 1  # first run computes

    # check_output=True: the custom validator raises -> caught -> recompute.
    f2 = _RaisingCheckFeature()
    run_feature(ds, f2, check_output=True)  # must not raise
    assert f2.apply_calls == 1  # recomputed, not skipped, not crashed


def test_noncallable_check_output_falls_back_to_default(tmp_path: Path) -> None:
    ds = _MockDataset(tmp_path)
    _setup_tracks(ds, [("g", "s1")])

    f1 = _NonCallableCheckFeature()
    run_feature(ds, f1)
    assert f1.apply_calls == 1

    # A non-callable check_output attribute must not cause a TypeError; the
    # default validator is used instead, so a valid output is a cache hit.
    f2 = _NonCallableCheckFeature()
    run_feature(ds, f2, check_output=True)  # must not raise
    assert f2.apply_calls == 0


# --- HASH_EXCLUDE: throughput-only params don't bust the cache ---


class _ThroughputFeature(_StatelessFeature):
    """Like _StatelessFeature but with a hash-excluded throughput knob."""

    name = "test-throughput"

    class Params(Params):
        real: int = 0
        batch_size: Annotated[int, HASH_EXCLUDE] = 4


def test_hash_excluded_param_reuses_cache(tmp_path: Path) -> None:
    from mosaic.core.pipeline.index import feature_run_root

    ds = _MockDataset(tmp_path)
    _setup_tracks(ds, [("g", "s1")])

    # First run at batch_size=4.
    r1 = run_feature(ds, _ThroughputFeature(params={"batch_size": 4}))
    run_root = feature_run_root(ds, r1.feature, r1.run_id)
    parquet = sorted(run_root.glob("*.parquet"))[0]
    mtime_before = os.path.getmtime(parquet)

    # Re-run at batch_size=8: same run_id, output untouched (cache reused).
    r2 = run_feature(ds, _ThroughputFeature(params={"batch_size": 8}), overwrite=False)
    assert r2.run_id == r1.run_id
    assert os.path.getmtime(parquet) == mtime_before


def test_real_param_change_busts_cache(tmp_path: Path) -> None:
    ds = _MockDataset(tmp_path)
    _setup_tracks(ds, [("g", "s1")])

    r1 = run_feature(ds, _ThroughputFeature(params={"real": 0}))
    r2 = run_feature(ds, _ThroughputFeature(params={"real": 1}))
    assert r2.run_id != r1.run_id
