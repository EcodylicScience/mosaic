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
from mosaic.core.pipeline.manifest import build_manifest
from mosaic.core.pipeline.run import run_feature
from mosaic.core.pipeline.tracks_index import backfill_frame_extents
from mosaic.core.pipeline.types.feature import EmitsLevel
from mosaic.core.pipeline.types import (
    InputRequire,
    Inputs,
    InputStream,
    Result,
    TrackInput,
    TrackInputs,
)
from mosaic.core.params import (
    HASH_EXCLUDE,
    Params,
)
from mosaic.core.scope import Scope
from mosaic.core.dataset import Dataset
from tests.helpers import (
    MockDataset,
    add_track_sequences,
    add_tracks_table,
    make_dataset,
    result_run_root,
    set_tracks_cell,
    track_table,
)

# --- Mock features ---


class _StatelessFeature:
    name = "test-stateless"
    version = "0.1"
    parallelizable = True
    scope_dependent = False
    consumed_roots: tuple[str, ...] = ()

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
    consumed_roots: tuple[str, ...] = ()

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
    consumed_roots: tuple[str, ...] = ()

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
    ds = MockDataset(tmp_path)
    add_track_sequences(ds, ("g", "s1"), ("g", "s2"))

    feature = _StatelessFeature()
    result = run_feature(ds, feature)

    assert isinstance(result, Result)
    assert result.feature == "test-stateless__from__tracks"
    assert result.run_id is not None and len(result.run_id) > 0

    # Output parquets exist
    run_root = result_run_root(ds, result)
    parquets = sorted(run_root.glob("*.parquet"))
    assert len(parquets) == 2

    # Index CSV has 2 finished rows
    idx = feature_index(feature_index_path(ds, result.feature))
    idx_df = idx.read(run_id=result.run_id)
    assert len(idx_df) == 2
    assert all(idx_df["finished_at"] != "")


def test_entries_selects_arbitrary_subset(tmp_path: Path) -> None:
    """An entries scope runs exactly the requested (group, sequence) pairs.

    Sequence name "s1" repeats across groups g1 and g2. A bare sequences
    selector would be ambiguous there. An entries one selects (g1, s1) and
    (g2, s1) while excluding (g1, s2). That is the arbitrary, tag-resolvable
    subset case.
    """
    ds = MockDataset(tmp_path)
    add_track_sequences(ds, ("g1", "s1"), ("g1", "s2"), ("g2", "s1"))

    feature = _StatelessFeature()
    result = run_feature(ds, feature, scope=Scope(entries=[("g1", "s1"), ("g2", "s1")]))

    run_root = result_run_root(ds, result)
    parquets = sorted(p.name for p in run_root.glob("*.parquet"))
    assert parquets == ["g1__s1.parquet", "g2__s1.parquet"]

    idx = feature_index(feature_index_path(ds, result.feature))
    idx_df = idx.read(run_id=result.run_id)
    produced = set(zip(idx_df["group"], idx_df["sequence"]))
    assert produced == {("g1", "s1"), ("g2", "s1")}


def test_stateful_fit_and_apply(tmp_path: Path) -> None:
    ds = MockDataset(tmp_path)
    add_track_sequences(ds, ("g", "s1"), ("g", "s2"))

    feature = _StatefulFeature()
    result = run_feature(ds, feature)

    # State file written
    run_root = result_run_root(ds, result)
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
    ds = MockDataset(tmp_path)
    add_track_sequences(ds, ("g", "s1"))

    feature = _StatelessFeature()

    # First run
    result1 = run_feature(ds, feature)
    run_root = result_run_root(ds, result1)
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
    ds = MockDataset(tmp_path)
    # Tracks exist but feature has empty inputs
    add_track_sequences(ds, ("g", "s1"))

    feature = _EmptyInputFeature()
    result = run_feature(ds, feature)

    assert result.feature == "test-global"

    # No parquet files in run root
    run_root = result_run_root(ds, result)
    parquets = list(run_root.glob("*.parquet"))
    assert len(parquets) == 0

    # Index CSV has a global marker row
    idx = feature_index(feature_index_path(ds, result.feature))
    idx_df = idx.read(run_id=result.run_id)
    assert len(idx_df) == 1
    assert idx_df.iloc[0]["sequence"] == "__global__"
    assert idx_df.iloc[0]["group"] == ""


def test_frame_filter(tmp_path: Path) -> None:
    ds = MockDataset(tmp_path)
    add_track_sequences(ds, ("g", "s1"), n_rows=20)

    feature = _StatelessFeature()
    result = run_feature(ds, feature, filter_start_frame=5, filter_end_frame=15)

    run_root = result_run_root(ds, result)
    parquets = sorted(run_root.glob("*.parquet"))
    assert len(parquets) == 1

    out_df = pd.read_parquet(parquets[0])
    assert list(out_df["frame"]) == list(range(5, 15))


def test_scope_dependent_hashing(tmp_path: Path) -> None:
    ds = MockDataset(tmp_path)
    add_track_sequences(ds, ("g", "s1"), ("g", "s2"), ("g", "s3"))

    feature1 = _StatefulFeature()
    result1 = run_feature(ds, feature1, scope=Scope(sequences=["s1", "s2"]))

    feature2 = _StatefulFeature()
    result2 = run_feature(ds, feature2, scope=Scope(sequences=["s1", "s3"]))

    assert result1.run_id != result2.run_id


type _RunRecord = tuple[str | None, dict[tuple[str, str], str]]


def _run_record(ds: Dataset, result: Result) -> _RunRecord:
    """A run's id, and the stored output path of each entry it wrote."""
    index = feature_index(feature_index_path(ds, result.feature)).read()
    rows = index[index["run_id"] == result.run_id]
    return result.run_id, {
        (str(row["group"]), str(row["sequence"])): str(row["abs_path"])
        for _, row in rows.iterrows()
    }


def test_declaring_a_continuous_group_moves_no_key_run_id_or_path(
    tmp_path: Path,
) -> None:
    """A dataset of undivided entries keeps its identities when a group is declared.

    The empty-group and namespace entries keep their run ids and output paths,
    and so do the declared group's own sequences, under a scope-free and a
    scope-dependent feature.
    """
    ds = make_dataset(tmp_path / "ds")
    add_track_sequences(ds, ("", "a"), ("ns", "x"), ("rec", "s1"), ("rec", "s2"))
    undivided = Scope(entries=[("", "a"), ("ns", "x")])

    def runs(*, overwrite: bool) -> list[_RunRecord]:
        return [
            _run_record(ds, run_feature(ds, _StatelessFeature(), overwrite=overwrite)),
            _run_record(
                ds,
                run_feature(
                    ds, _StatefulFeature(), scope=undivided, overwrite=overwrite
                ),
            ),
        ]

    before = runs(overwrite=False)
    ds.set_continuous_groups(["rec"])
    after = runs(overwrite=True)

    assert after == before
    assert set(before[0][1]) == {("", "a"), ("ns", "x"), ("rec", "s1"), ("rec", "s2")}


def test_overlap_frame_filter_mutual_exclusion(tmp_path: Path) -> None:
    ds = MockDataset(tmp_path)
    add_track_sequences(ds, ("g", "s1"))

    feature = _StatelessFeature()
    with pytest.raises(ValueError, match="mutually exclusive"):
        run_feature(ds, feature, overlap_frames=5, filter_start_frame=0)


# --- overlap: what it refuses, and what it computes ---


def _setup_continuous(
    ds: MockDataset, n_frames: int = 10, *, contiguous: bool = True
) -> None:
    """Three sequences of one group, with recorded extents, as a real index has."""
    for index, name in enumerate(("s1", "s2", "s3")):
        start = index * n_frames if contiguous else 0
        add_track_sequences(ds, ("g", name), n_rows=n_frames, start_frame=start)
    _ = backfill_frame_extents(ds)


def _written_by_entry(
    ds: MockDataset, result: Result
) -> dict[tuple[str, str], pd.DataFrame]:
    """Each entry's written output for *result*, read through the feature index."""
    index = feature_index(feature_index_path(ds, result.feature)).read()
    rows = index[index["run_id"] == result.run_id]
    return {
        (str(row["group"]), str(row["sequence"])): pd.read_parquet(
            ds.resolve_path(row["abs_path"])
        )
        for _, row in rows.iterrows()
    }


def test_overlap_reads_no_context_outside_a_continuous_group(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Independent sequences are not neighbors, however their frames are numbered.

    No measurement separates "two divisions of one recording" from "two
    recordings numbered consecutively". Without the declaration each entry reads
    only its own frames, writes the rows an overlap-0 run writes, and the run
    says how many entries read no context.
    """
    ds = MockDataset(tmp_path)
    _setup_continuous(ds)
    _OverlapAwareFeature.seen = []

    with_overlap = run_feature(ds, _OverlapAwareFeature(), overlap_frames=3)
    note = capsys.readouterr().err

    assert sorted(_OverlapAwareFeature.seen) == [
        list(range(0, 10)),
        list(range(10, 20)),
        list(range(20, 30)),
    ]
    assert "3 entries" in note
    assert "ds.set_continuous_groups" in note

    without = run_feature(ds, _OverlapAwareFeature(), overlap_frames=0)
    written_with = _written_by_entry(ds, with_overlap)
    written_without = _written_by_entry(ds, without)
    assert written_with.keys() == written_without.keys()
    for entry, frame in written_with.items():
        pd.testing.assert_frame_equal(frame, written_without[entry])


def test_overlap_note_counts_one_entry_in_the_singular(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """One entry outside a continuous group is named as one entry."""
    ds = MockDataset(tmp_path)
    add_track_sequences(ds, ("g", "s1"), n_rows=10)
    _ = backfill_frame_extents(ds)

    _ = run_feature(ds, _OverlapAwareFeature(), overlap_frames=3)
    note = capsys.readouterr().err

    assert "1 entry outside a continuous group reads no context" in note


def _setup_mixed(ds: MockDataset, *, contiguous: bool = True) -> None:
    """A continuous group ``g`` beside namespace and empty-group sequences.

    ``g`` and the namespace group ``ns`` record frame extents. ``ns`` restarts
    its numbering at 0 in each sequence. The empty-group entries are added after
    the extents are measured, so they record none.
    """
    for index, name in enumerate(("s1", "s2", "s3")):
        start = index * 10 if contiguous else 0
        add_track_sequences(ds, ("g", name), n_rows=10, start_frame=start)
    add_track_sequences(ds, ("ns", "x"), ("ns", "y"), n_rows=10)
    _ = backfill_frame_extents(ds)
    add_track_sequences(ds, ("", "a"), ("", "b"), n_rows=10)


def test_overlap_reads_context_only_inside_the_continuous_group_of_a_mixed_dataset(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A dataset-wide overlap run succeeds beside sequences that are not divisions.

    The declared group's middle sequence sees its neighbors' frames. Every other
    entry sees only its own frames, whether its group is empty or a namespace,
    and whether or not it records a frame extent.
    """
    ds = MockDataset(tmp_path, continuous_groups=("g",))
    _setup_mixed(ds)
    _OverlapAwareFeature.seen = []

    result = run_feature(ds, _OverlapAwareFeature(), overlap_frames=3)
    note = capsys.readouterr().err

    seen = _OverlapAwareFeature.seen
    assert list(range(0, 13)) in seen
    assert list(range(7, 23)) in seen
    assert list(range(17, 30)) in seen
    assert seen.count(list(range(0, 10))) == 4
    assert "4 entries" in note

    written = _written_by_entry(ds, result)
    assert sorted(written[("g", "s2")]["frame"].unique()) == list(range(10, 20))
    for entry in (("", "a"), ("", "b"), ("ns", "x"), ("ns", "y")):
        assert sorted(written[entry]["frame"].unique()) == list(range(0, 10))


def test_overlap_refuses_a_continuous_group_that_restarts_its_numbering_in_a_mixed_dataset(
    tmp_path: Path,
) -> None:
    """Undivided entries beside a declared group leave its checks in force."""
    ds = MockDataset(tmp_path, continuous_groups=("g",))
    _setup_mixed(ds, contiguous=False)

    with pytest.raises(ValueError, match="one frame axis"):
        run_feature(ds, _OverlapAwareFeature(), overlap_frames=3)


def test_overlap_refuses_sequences_that_restart_their_frame_numbering(
    tmp_path: Path,
) -> None:
    """Segments sharing frame numbers are not one axis, declaration or not.

    This is what every mosaic converter writes today, so it is the case that has
    to fail loudly: concatenating three segments that all start at frame 0 hands
    the feature three rows for frame 7, and every sort or groupby on frame then
    interleaves or merges three recordings.
    """
    ds = MockDataset(tmp_path, continuous_groups=("g",))
    _setup_continuous(ds, contiguous=False)

    with pytest.raises(ValueError, match="one frame axis"):
        run_feature(ds, _OverlapAwareFeature(), overlap_frames=3)


def test_overlap_refuses_a_feature_that_has_not_declared_it(tmp_path: Path) -> None:
    """An undeclared feature is refused rather than assumed safe."""
    ds = MockDataset(tmp_path, continuous_groups=("g",))
    _setup_continuous(ds)

    feature = _StatelessFeature()
    assert not hasattr(type(feature), "accepts_overlap")
    with pytest.raises(AttributeError, match="accepts_overlap"):
        run_feature(ds, feature, overlap_frames=3)


class _OverlapAwareFeature(_StatelessFeature):
    """Records the frames it was handed, so the context is observable.

    Deliberately does *not* file what it saw under ``df["sequence"].iloc[0]``.
    Under overlap that row belongs to the previous sequence, so the record would
    be filed under the neighbour's name -- the very defect ``accepts_overlap``
    exists to keep out of features that cannot handle it, and it would make this
    test lie about what it observed.
    """

    name = "overlap-aware"
    accepts_overlap = True
    emits: EmitsLevel = "individual"
    seen: ClassVar[list[list[int]]] = []

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        _OverlapAwareFeature.seen.append(sorted({int(f) for f in df["frame"]}))
        out = df[["frame", "time", "id", "group", "sequence"]].copy()
        out["doubled"] = df["feat_a"].to_numpy() * 2.0
        return out


def test_overlap_gives_context_and_trims_it_back_off(tmp_path: Path) -> None:
    """The run sees the neighbours and writes only its own rows.

    Both halves matter. Without the first there is no point to the parameter;
    without the second the neighbour's rows land in this entry's parquet, which
    is what the positional trim did whenever a feature reordered or dropped rows.
    """
    ds = MockDataset(tmp_path, continuous_groups=("g",))
    _setup_continuous(ds)
    _OverlapAwareFeature.seen = []

    result = run_feature(ds, _OverlapAwareFeature(), overlap_frames=3)

    # The middle sequence covers frames 10-19 and is handed 3 on either side.
    assert list(range(7, 23)) in _OverlapAwareFeature.seen

    index = feature_index(feature_index_path(ds, result.feature)).read()
    row = index[index["sequence"] == "s2"].iloc[0]
    written = pd.read_parquet(ds.resolve_path(row["abs_path"]))
    assert sorted(written["frame"].unique()) == list(range(10, 20))
    assert set(written["sequence"]) == {"s2"}


def test_overlap_changes_the_identifier_so_it_cannot_reuse_the_cache(
    tmp_path: Path,
) -> None:
    """Two context widths are two computations, addressed apart.

    They write the same rows under the same keys with different numbers near the
    boundaries, so a shared identifier means the second run silently receives the
    first's output.
    """
    ds = MockDataset(tmp_path, continuous_groups=("g",))
    _setup_continuous(ds)
    _OverlapAwareFeature.seen = []

    without = run_feature(ds, _OverlapAwareFeature(), overlap_frames=0)
    with_context = run_feature(ds, _OverlapAwareFeature(), overlap_frames=3)

    assert without.run_id != with_context.run_id
    assert not with_context.cache_hit


# --- overlap: a cached entry whose neighbors change ---


def _neighbor_cells(ds: Dataset | MockDataset, result: Result) -> dict[str, str]:
    """Each written entry's ``consumed_neighbors`` cell for *result*, by sequence."""
    index = feature_index(feature_index_path(ds, result.feature)).read()
    rows = index[index["run_id"] == result.run_id]
    return {
        str(row["sequence"]): str(row["consumed_neighbors"])
        for _, row in rows.iterrows()
    }


def test_declaring_a_group_recomputes_the_entries_of_an_overlapped_run(
    tmp_path: Path,
) -> None:
    """The declaration gives each sequence neighbors, and each is recomputed.

    The run id is the same before and after the declaration, and the entries that
    read their new neighbors are recomputed under it.
    """
    ds = make_dataset(tmp_path / "ds")
    for index, name in enumerate(("s1", "s2", "s3")):
        add_track_sequences(ds, ("rec", name), n_rows=10, start_frame=index * 10)
    _ = backfill_frame_extents(ds)
    _OverlapAwareFeature.seen = []
    before = run_feature(ds, _OverlapAwareFeature(), overlap_frames=3)
    _OverlapAwareFeature.seen = []

    ds.set_continuous_groups(["rec"])
    after = run_feature(ds, _OverlapAwareFeature(), overlap_frames=3)

    assert after.run_id == before.run_id
    assert not after.cache_hit
    assert sorted(_OverlapAwareFeature.seen) == [
        list(range(0, 13)),
        list(range(7, 23)),
        list(range(17, 30)),
    ]
    assert "rec__s1" in _neighbor_cells(ds, after)["s2"]
    assert "rec__s3" in _neighbor_cells(ds, after)["s2"]


def test_adding_a_sequence_recomputes_only_the_sequence_before_it(
    tmp_path: Path,
) -> None:
    """A growing recording's previous last sequence reads its new successor."""
    ds = MockDataset(tmp_path, continuous_groups=("g",))
    _setup_continuous(ds)
    _ = run_feature(ds, _OverlapAwareFeature(), overlap_frames=3)
    _OverlapAwareFeature.seen = []

    add_track_sequences(ds, ("g", "s4"), n_rows=10, start_frame=30)
    _ = backfill_frame_extents(ds)
    _ = run_feature(ds, _OverlapAwareFeature(), overlap_frames=3)

    assert sorted(_OverlapAwareFeature.seen) == [
        list(range(17, 33)),
        list(range(27, 40)),
    ]


def test_an_unchanged_neighborhood_serves_every_entry(tmp_path: Path) -> None:
    """Neighbors that are as they were recompute nothing."""
    ds = MockDataset(tmp_path, continuous_groups=("g",))
    _setup_continuous(ds)
    first = run_feature(ds, _OverlapAwareFeature(), overlap_frames=3)
    _OverlapAwareFeature.seen = []

    second = run_feature(ds, _OverlapAwareFeature(), overlap_frames=3)

    assert second.run_id == first.run_id
    assert second.cache_hit
    assert _OverlapAwareFeature.seen == []


def test_a_run_without_overlap_records_no_neighbors(tmp_path: Path) -> None:
    ds = MockDataset(tmp_path, continuous_groups=("g",))
    _setup_continuous(ds)

    result = run_feature(ds, _OverlapAwareFeature(), overlap_frames=0)

    assert _neighbor_cells(ds, result) == {"s1": "", "s2": "", "s3": ""}


def test_a_reconverted_neighbor_recomputes_the_sequences_that_read_it(
    tmp_path: Path,
) -> None:
    """``s1`` is converted again from changed sources, under the same variant.

    ``s1`` recomputes for its table, and ``s2`` for the neighbor it reads.
    ``s3`` reads only ``s2`` and is served.
    """
    ds = make_dataset(tmp_path / "ds")
    for index, name in enumerate(("s1", "s2", "s3")):
        add_tracks_table(
            ds,
            "convert.v1",
            track_table("rec", name, n_rows=10, start_frame=index * 10),
        )
    ds.set_continuous_groups(["rec"])
    set_tracks_cell(ds, "consumed_composition", "sources-1")
    _ = run_feature(ds, _OverlapAwareFeature(), overlap_frames=3)
    _OverlapAwareFeature.seen = []

    set_tracks_cell(ds, "consumed_composition", "sources-2", entry=("rec", "s1"))
    _ = run_feature(ds, _OverlapAwareFeature(), overlap_frames=3)

    assert sorted(_OverlapAwareFeature.seen) == [
        list(range(0, 13)),
        list(range(7, 23)),
    ]


def test_a_neighbor_republished_at_a_moved_first_frame_recomputes_its_readers(
    tmp_path: Path,
) -> None:
    """``s3`` is published again at another first frame, as a tracker republishes it.

    ``s3`` recomputes for its moved table, and ``s2`` for the neighbor it reads.
    ``s1`` reads only ``s2`` and is served.
    """
    ds = make_dataset(tmp_path / "ds")
    for index, name in enumerate(("s1", "s2", "s3")):
        add_tracks_table(
            ds,
            "convert.v1",
            track_table("rec", name, n_rows=10, start_frame=index * 10),
        )
    ds.set_continuous_groups(["rec"])
    set_tracks_cell(ds, "consumed_composition", "sources-1")
    _ = run_feature(ds, _OverlapAwareFeature(), overlap_frames=3)
    _OverlapAwareFeature.seen = []

    set_tracks_cell(ds, "first_frame", "20", entry=("rec", "s3"))
    _ = run_feature(ds, _OverlapAwareFeature(), overlap_frames=3)

    assert sorted(_OverlapAwareFeature.seen) == [
        list(range(7, 23)),
        list(range(17, 30)),
    ]


def test_an_overlapped_entry_recorded_without_its_neighbors_recomputes_once(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A row written before the cell existed reads blank beside a known value."""
    ds = MockDataset(tmp_path, continuous_groups=("g",))
    _setup_continuous(ds)
    result = run_feature(ds, _OverlapAwareFeature(), overlap_frames=3)
    path = feature_index_path(ds, result.feature)
    rows = pd.read_csv(path, dtype=str, keep_default_na=False)
    rows.drop(columns=["consumed_neighbors"]).to_csv(path, index=False)
    _OverlapAwareFeature.seen = []
    _ = capsys.readouterr()

    recomputed = run_feature(ds, _OverlapAwareFeature(), overlap_frames=3)
    note = capsys.readouterr().err
    served = run_feature(ds, _OverlapAwareFeature(), overlap_frames=3)

    assert not recomputed.cache_hit
    assert len(_OverlapAwareFeature.seen) == 3
    assert "nothing records which neighbors it read" in note
    assert served.cache_hit


def test_reindexing_features_keeps_the_neighbors_an_entry_read(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds")
    for index, name in enumerate(("s1", "s2")):
        add_track_sequences(ds, ("rec", name), n_rows=10, start_frame=index * 10)
    _ = backfill_frame_extents(ds)
    ds.set_continuous_groups(["rec"])
    result = run_feature(ds, _OverlapAwareFeature(), overlap_frames=3)
    cells = _neighbor_cells(ds, result)
    index = feature_index(feature_index_path(ds, result.feature)).read()
    first = index[index["sequence"] == "s1"].iloc[0]
    ds.resolve_path(first["abs_path"]).unlink()

    _ = ds.reindex_features(result.feature, dry_run=False)

    assert _neighbor_cells(ds, result) == {"s2": cells["s2"]}


def test_result_type(tmp_path: Path) -> None:
    ds = MockDataset(tmp_path)
    add_track_sequences(ds, ("g", "s1"))

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
    ds = MockDataset(tmp_path)
    add_track_sequences(ds, ("g", "s1"), ("g", "s2"))

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
    ds = MockDataset(tmp_path)
    add_track_sequences(ds, ("g", "s1"), ("g", "s2"), n_rows=10)
    r1 = run_feature(ds, _StatelessFeature())
    run_root = result_run_root(ds, r1)
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
    ds = MockDataset(tmp_path)
    add_track_sequences(ds, ("g", "s1"), ("g", "s2"), n_rows=10)
    f1 = _CountingStateless()
    r1 = run_feature(ds, f1)
    assert f1.apply_calls == 2
    run_root = result_run_root(ds, r1)
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
    ds = MockDataset(tmp_path)
    add_track_sequences(ds, ("g", "s1"))
    r1 = run_feature(ds, _SidecarFeature())
    run_root = result_run_root(ds, r1)
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
    ds = MockDataset(tmp_path)
    add_track_sequences(ds, ("g", "s1"))

    f1 = _RaisingCheckFeature()
    run_feature(ds, f1)
    assert f1.apply_calls == 1  # first run computes

    # check_output=True: the custom validator raises -> caught -> recompute.
    f2 = _RaisingCheckFeature()
    run_feature(ds, f2, check_output=True)  # must not raise
    assert f2.apply_calls == 1  # recomputed, not skipped, not crashed


def test_noncallable_check_output_falls_back_to_default(tmp_path: Path) -> None:
    ds = MockDataset(tmp_path)
    add_track_sequences(ds, ("g", "s1"))

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
    ds = MockDataset(tmp_path)
    add_track_sequences(ds, ("g", "s1"))

    # First run at batch_size=4.
    r1 = run_feature(ds, _ThroughputFeature(params={"batch_size": 4}))
    run_root = result_run_root(ds, r1)
    parquet = sorted(run_root.glob("*.parquet"))[0]
    mtime_before = os.path.getmtime(parquet)

    # Re-run at batch_size=8: same run_id, output untouched (cache reused).
    r2 = run_feature(ds, _ThroughputFeature(params={"batch_size": 8}), overwrite=False)
    assert r2.run_id == r1.run_id
    assert os.path.getmtime(parquet) == mtime_before


def test_real_param_change_busts_cache(tmp_path: Path) -> None:
    ds = MockDataset(tmp_path)
    add_track_sequences(ds, ("g", "s1"))

    r1 = run_feature(ds, _ThroughputFeature(params={"real": 0}))
    r2 = run_feature(ds, _ThroughputFeature(params={"real": 1}))
    assert r2.run_id != r1.run_id


# --- frame extents of a feature input over two tracks variants ---------------


class _FrameEcho:
    """Writes its input's track columns back, so its output feeds another run."""

    name = "frame-echo"
    version = "0.1"
    parallelizable = True
    scope_dependent = False
    accepts_overlap = True
    emits: EmitsLevel = "individual"
    consumed_roots: tuple[str, ...] = ()

    class Inputs(TrackInputs):
        pass

    class Params(Params):
        pass

    def __init__(self, inputs: _FrameEcho.Inputs | None = None) -> None:
        self._inputs = inputs or self.Inputs(("tracks",))
        self._params = self.Params()

    @property
    def inputs(self) -> _FrameEcho.Inputs:
        return self._inputs

    @property
    def params(self) -> _FrameEcho.Params:
        return self._params

    def load_state(
        self,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_lookups: dict[str, dict[tuple[str, str], Path]],
    ) -> bool:
        _ = run_root, artifact_paths, dependency_lookups
        return True

    def fit(self, inputs: InputStream) -> None:
        _ = inputs

    def save_state(self, run_root: Path) -> None:
        _ = run_root

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[["frame", "time", "id", "group", "sequence", "X", "Y"]].copy()


def _two_variants(tmp_path: Path, *, declared: bool) -> Dataset:
    """``rec/a``, ``rec/b`` and ``solo``, each in the variants ``convert.v1`` and ``.v2``.

    ``convert.v1`` numbers ``rec`` across the recording. ``convert.v2`` restarts
    ``rec/b`` at frame 0, so its extents would refuse an overlapped read.
    """
    ds = make_dataset(tmp_path / "ds")
    for run_id, b_start in (("convert.v1", 10), ("convert.v2", 0)):
        add_tracks_table(ds, run_id, track_table("rec", "a", n_rows=10))
        add_tracks_table(
            ds, run_id, track_table("rec", "b", n_rows=10, start_frame=b_start)
        )
        add_tracks_table(ds, run_id, track_table("", "solo", n_rows=10))
    if declared:
        ds.set_continuous_groups(["rec"])
    return ds


def _frames_written(ds: Dataset, result: Result) -> dict[tuple[str, str], list[int]]:
    """Each entry's distinct output frames for *result*, from the feature index."""
    index = feature_index(feature_index_path(ds, result.feature)).read()
    rows = index[index["run_id"] == result.run_id]
    return {
        (str(row["group"]), str(row["sequence"])): sorted(
            int(frame)
            for frame in pd.read_parquet(ds.resolve_path(row["abs_path"]))[
                "frame"
            ].unique()
        )
        for _, row in rows.iterrows()
    }


@pytest.mark.parametrize("declared", [True, False], ids=["declared", "undeclared"])
@pytest.mark.parametrize("overlap", [0, 3])
def test_a_feature_input_reads_frame_extents_under_its_upstream_variant(
    tmp_path: Path, declared: bool, overlap: int
) -> None:
    """A feature-only run needs no pinned tracks variant beside two variants.

    The upstream read ``convert.v1``, so the extents of the continuous group come
    from that variant. ``convert.v2``'s extents would refuse the overlapped run.
    """
    ds = _two_variants(tmp_path, declared=declared)
    upstream = run_feature(
        ds, _FrameEcho(), overlap_frames=overlap, tracks_run_id="convert.v1"
    )

    result = run_feature(
        ds, _FrameEcho(_FrameEcho.Inputs((upstream,))), overlap_frames=overlap
    )

    frames = _frames_written(ds, result)
    assert sorted(frames) == [("", "solo"), ("rec", "a"), ("rec", "b")]
    assert frames[("rec", "b")] == list(range(10, 20))


def test_an_upstream_of_unknown_variant_refuses_a_member_with_two_variants(
    tmp_path: Path,
) -> None:
    """Without a known variant there is no defensible choice between the two."""
    ds = _two_variants(tmp_path, declared=True)
    upstream = run_feature(ds, _FrameEcho(), tracks_run_id="convert.v1")
    (result_run_root(ds, upstream) / "params.json").unlink()
    inputs = _FrameEcho.Inputs((upstream,))

    with pytest.raises(ValueError, match="2 variants"):
        _ = build_manifest(ds, inputs)

    _ = build_manifest(ds, inputs, tracks_run_id="convert.v1")
