"""Tests for the unified manifest builder."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mosaic.core.pipeline.index import FeatureIndexRow, feature_index
from mosaic.core.pipeline.manifest import (
    ManifestEntry,
    build_manifest,
    iter_manifest,
)
from mosaic.core.pipeline.types import Inputs, ParquetLoadSpec, Result, TrackInputs


from tests.mock_dataset import MockDataset as _MockDataset


def _make_parquet(path: Path, n_rows: int = 10, *, track_shaped: bool = False) -> None:
    data: dict[str, object] = {
        "frame": range(n_rows),
        "time": [f / 30.0 for f in range(n_rows)],
        "id": [0] * n_rows,
        "feat_a": np.random.randn(n_rows),
    }
    if track_shaped:
        # A track-producing feature passes the track frame through, keeping X/Y.
        data["X"] = np.linspace(0.0, 5.0, n_rows)
        data["Y"] = np.linspace(0.0, 2.0, n_rows)
    df = pd.DataFrame(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


def _write_tracks_index(ds, entries):
    idx_path = ds.get_root("tracks") / "index.csv"
    rows = [{"group": g, "sequence": s, "abs_path": str(p)} for g, s, p in entries]
    pd.DataFrame(rows).to_csv(idx_path, index=False)


def _setup_feature(ds, feat_name, pairs, run_id="v1-abc", *, track_shaped=False):
    feat_dir = ds.get_root("features") / feat_name
    run_dir = feat_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    idx = feature_index(feat_dir / "index.csv")
    idx.ensure()
    rows = []
    for g, s in pairs:
        p = run_dir / f"{g}__{s}.parquet"
        _make_parquet(p, track_shaped=track_shaped)
        rows.append(
            FeatureIndexRow(
                run_id=run_id,
                feature=feat_name,
                version="0.1",
                group=g,
                sequence=s,
                abs_path=str(p),
                n_rows=10,
                params_hash="abc",
            )
        )
    idx.append(rows)
    idx.mark_finished(run_id)
    return run_id


def test_build_manifest_tracks_only(tmp_path):
    ds = _MockDataset(tmp_path)
    entries = []
    for g, s in [("g1", "s1"), ("g1", "s2")]:
        p = tmp_path / "tracks" / f"{g}__{s}.parquet"
        _make_parquet(p)
        entries.append((g, s, p))
    _write_tracks_index(ds, entries)

    inputs = Inputs(("tracks",))
    manifest, scope = build_manifest(ds, inputs)
    assert len(manifest) == 2
    assert scope.entries == {("g1", "s1"), ("g1", "s2")}
    # Each entry has one (path, ParquetLoadSpec) tuple
    for key, entry in manifest.items():
        assert isinstance(entry, ManifestEntry)
        assert len(entry.file_specs) == 1
        path, load_spec = entry.file_specs[0]
        assert path.exists()


def test_build_manifest_feature_result(tmp_path):
    ds = _MockDataset(tmp_path)
    run_id = _setup_feature(ds, "speed-angvel", [("g1", "s1"), ("g1", "s2")])
    inputs = Inputs((Result(feature="speed-angvel", run_id=run_id),))

    manifest, scope = build_manifest(ds, inputs)
    assert len(manifest) == 2
    assert scope.entries == {("g1", "s1"), ("g1", "s2")}


def test_track_inputs_accepts_result_type():
    """The widened track-input type accepts a feature Result (the CLI-facing fix).

    Previously ``Inputs[TrackInput]`` rejected any ``{"feature": ...}`` reference
    (``Input should be 'tracks'``), so ``mosaic run --feature speed-angvel --inputs
    '[{"feature":"trajectory-smooth__from__tracks"}]'`` could not be expressed.
    """
    inp = TrackInputs.model_validate([{"feature": "trajectory-smooth__from__tracks"}])
    assert inp.feature_inputs[0].feature == "trajectory-smooth__from__tracks"
    assert TrackInputs.model_validate(["tracks"]).has_tracks
    assert getattr(TrackInputs, "_track_input", False) is True


def test_build_manifest_track_result_ok(tmp_path):
    """A Result from a track-producing feature (output keeps X/Y) is a valid track input."""
    ds = _MockDataset(tmp_path)
    run_id = _setup_feature(
        ds,
        "trajectory-smooth__from__tracks",
        [("g1", "s1"), ("g1", "s2")],
        track_shaped=True,
    )
    inputs = TrackInputs(
        (Result(feature="trajectory-smooth__from__tracks", run_id=run_id),)
    )
    manifest, scope = build_manifest(ds, inputs)
    assert scope.entries == {("g1", "s1"), ("g1", "s2")}
    assert len(manifest) == 2


def test_build_manifest_derived_result_rejected(tmp_path):
    """A Result from a derived feature (no X/Y) is refused as a track input.

    speed-angvel's output drops X/Y, so chaining speed-angvel -> a track feature is
    a type-valid Result but not a track-shaped one; resolution must reject it with a
    clear error rather than silently KeyError at apply time.
    """
    ds = _MockDataset(tmp_path)
    run_id = _setup_feature(
        ds,
        "speed-angvel__from__tracks",
        [("g1", "s1")],
        track_shaped=False,
    )
    inputs = TrackInputs((Result(feature="speed-angvel__from__tracks", run_id=run_id),))
    with pytest.raises(ValueError, match="track-shaped|track-producing"):
        build_manifest(ds, inputs)


def test_build_manifest_mixed_intersects(tmp_path):
    ds = _MockDataset(tmp_path)
    # Tracks have s1, s2, s3
    entries = []
    for s in ("s1", "s2", "s3"):
        p = tmp_path / "tracks" / f"g1__{s}.parquet"
        _make_parquet(p)
        entries.append(("g1", s, p))
    _write_tracks_index(ds, entries)
    # Feature has only s1, s2 -- no run_id, exercises latest-run resolution
    _setup_feature(ds, "nn", [("g1", "s1"), ("g1", "s2")])

    inputs = Inputs(("tracks", Result(feature="nn")))
    manifest, scope = build_manifest(ds, inputs)
    # Intersection: only s1 and s2
    assert scope.entries == {("g1", "s1"), ("g1", "s2")}
    assert len(manifest) == 2
    # Each entry has 2 file specs (track + feature)
    for key, entry in manifest.items():
        assert len(entry.file_specs) == 2


def test_build_manifest_group_filter(tmp_path):
    ds = _MockDataset(tmp_path)
    entries = []
    for g, s in [("g1", "s1"), ("g2", "s2")]:
        p = tmp_path / "tracks" / f"{g}__{s}.parquet"
        _make_parquet(p)
        entries.append((g, s, p))
    _write_tracks_index(ds, entries)

    inputs = Inputs(("tracks",))
    manifest, scope = build_manifest(ds, inputs, groups={"g1"})
    assert scope.entries == {("g1", "s1")}


def test_build_manifest_entries_filter(tmp_path):
    """entries= selects an arbitrary (group, sequence) subset.

    Critically, the sequence name "s1" is reused across groups g1 and g2, so a
    bare sequences= filter would be ambiguous; entries= picks exactly the pairs
    requested and excludes (g1, s2).
    """
    ds = _MockDataset(tmp_path)
    entries = []
    for g, s in [("g1", "s1"), ("g1", "s2"), ("g2", "s1")]:
        p = tmp_path / "tracks" / f"{g}__{s}.parquet"
        _make_parquet(p)
        entries.append((g, s, p))
    _write_tracks_index(ds, entries)

    inputs = Inputs(("tracks",))
    manifest, scope = build_manifest(ds, inputs, entries={("g1", "s1"), ("g2", "s1")})
    assert scope.entries == {("g1", "s1"), ("g2", "s1")}
    assert set(manifest.keys()) == {"g1__s1", "g2__s1"}


def test_build_manifest_entries_intersects_groups(tmp_path):
    """entries= intersects with groups= when both are given."""
    ds = _MockDataset(tmp_path)
    entries = []
    for g, s in [("g1", "s1"), ("g1", "s2"), ("g2", "s1")]:
        p = tmp_path / "tracks" / f"{g}__{s}.parquet"
        _make_parquet(p)
        entries.append((g, s, p))
    _write_tracks_index(ds, entries)

    inputs = Inputs(("tracks",))
    # entries asks for two pairs; groups restricts to g1 -> only (g1, s1) survives
    manifest, scope = build_manifest(
        ds, inputs, groups={"g1"}, entries={("g1", "s1"), ("g2", "s1")}
    )
    assert scope.entries == {("g1", "s1")}


def test_build_manifest_entries_feature_input(tmp_path):
    """entries= also scopes a feature-result input (via IndexCSV.read)."""
    ds = _MockDataset(tmp_path)
    run_id = _setup_feature(
        ds, "speed-angvel", [("g1", "s1"), ("g1", "s2"), ("g2", "s1")]
    )
    inputs = Inputs((Result(feature="speed-angvel", run_id=run_id),))

    manifest, scope = build_manifest(ds, inputs, entries={("g1", "s2"), ("g2", "s1")})
    assert scope.entries == {("g1", "s2"), ("g2", "s1")}
    assert set(manifest.keys()) == {"g1__s2", "g2__s1"}


def test_build_manifest_adjacency(tmp_path):
    """Verify prev/next adjacency pointers are set correctly."""
    ds = _MockDataset(tmp_path)
    entries = []
    for s in ("s1", "s2", "s3"):
        p = tmp_path / "tracks" / f"g1__{s}.parquet"
        _make_parquet(p)
        entries.append(("g1", s, p))
    _write_tracks_index(ds, entries)

    inputs = Inputs(("tracks",))
    manifest, scope = build_manifest(ds, inputs)
    assert len(manifest) == 3

    entry_s1 = manifest["g1__s1"]
    assert entry_s1.prev_entry_key is None
    assert entry_s1.prev_file_specs is None
    assert entry_s1.next_entry_key == "g1__s2"
    assert entry_s1.next_file_specs is not None

    entry_s2 = manifest["g1__s2"]
    assert entry_s2.prev_entry_key == "g1__s1"
    assert entry_s2.prev_file_specs is not None
    assert entry_s2.next_entry_key == "g1__s3"
    assert entry_s2.next_file_specs is not None

    entry_s3 = manifest["g1__s3"]
    assert entry_s3.prev_entry_key == "g1__s2"
    assert entry_s3.prev_file_specs is not None
    assert entry_s3.next_entry_key is None
    assert entry_s3.next_file_specs is None


def test_build_manifest_adjacency_cross_group(tmp_path):
    """Adjacency does not cross group boundaries."""
    ds = _MockDataset(tmp_path)
    entries = []
    for g, s in [("g1", "s1"), ("g1", "s2"), ("g2", "s3")]:
        p = tmp_path / "tracks" / f"{g}__{s}.parquet"
        _make_parquet(p)
        entries.append((g, s, p))
    _write_tracks_index(ds, entries)

    inputs = Inputs(("tracks",))
    manifest, _ = build_manifest(ds, inputs)

    # g1__s2 next should be None (s3 is in g2)
    assert manifest["g1__s2"].next_entry_key is None
    assert manifest["g1__s2"].next_file_specs is None

    # g2__s3 prev should be None (s2 is in g1)
    assert manifest["g2__s3"].prev_entry_key is None
    assert manifest["g2__s3"].prev_file_specs is None


def test_iter_manifest_yields_keydata(tmp_path):
    ds = _MockDataset(tmp_path)
    entries = []
    for g, s in [("g1", "s1"), ("g1", "s2")]:
        p = tmp_path / "tracks" / f"{g}__{s}.parquet"
        _make_parquet(p, n_rows=10)
        entries.append((g, s, p))
    _write_tracks_index(ds, entries)

    inputs = Inputs(("tracks",))
    manifest, _ = build_manifest(ds, inputs)

    results = list(iter_manifest(manifest))
    assert len(results) == 2
    for entry_key, df in results:
        assert isinstance(entry_key, str)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 10


def test_iter_manifest_mixed_inner_join(tmp_path):
    ds = _MockDataset(tmp_path)

    # Tracks: frames 0-9
    track_path = tmp_path / "tracks" / "g1__s1.parquet"
    track_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "frame": range(10),
            "time": [f / 30.0 for f in range(10)],
            "id": [0] * 10,
            "feat_a": np.random.randn(10),
        }
    ).to_parquet(track_path)
    _write_tracks_index(ds, [("g1", "s1", track_path)])

    # Feature: frames 2-7 only
    run_id = "v1-abc"
    feat_dir = ds.get_root("features") / "narrowfeat"
    run_dir = feat_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    feat_path = run_dir / "g1__s1.parquet"
    pd.DataFrame(
        {
            "frame": range(2, 8),
            "time": [f / 30.0 for f in range(2, 8)],
            "id": [0] * 6,
            "feat_b": np.random.randn(6),
        }
    ).to_parquet(feat_path)

    idx = feature_index(feat_dir / "index.csv")
    idx.ensure()
    idx.append(
        [
            FeatureIndexRow(
                run_id=run_id,
                feature="narrowfeat",
                version="0.1",
                group="g1",
                sequence="s1",
                abs_path=str(feat_path),
                n_rows=6,
                params_hash="abc",
            )
        ]
    )
    idx.mark_finished(run_id)

    inputs = Inputs(("tracks", Result(feature="narrowfeat", run_id=run_id)))
    manifest, _ = build_manifest(ds, inputs)

    results = list(iter_manifest(manifest))
    assert len(results) == 1
    entry_key, df = results[0]
    # Inner join on frames 2-7 -> 6 rows
    assert len(df) == 6


# --- Helpers for overlap / filter_factory tests ---


def _make_simple_parquet(
    path: Path,
    n_frames: int,
    start_frame: int = 0,
    n_ids: int = 1,
    sequence: str = "",
    group: str = "g1",
) -> None:
    """Write a frame-major parquet carrying identity, as a real tracks table does.

    ``n_ids`` and the identity columns are not decoration. An overlap defect only
    shows itself where a frame holds several individuals -- with one id per frame
    a row offset and a frame number happen to agree, which is why the suite could
    not see any of this before.
    """
    frames = np.repeat(np.arange(start_frame, start_frame + n_frames), n_ids)
    ids = np.tile(np.arange(n_ids), n_frames)
    df = pd.DataFrame(
        {
            "frame": frames,
            "time": frames / 30.0,
            "id": ids,
            "group": group,
            "sequence": sequence,
            "feat_a": np.arange(len(frames), dtype=float),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


def _build_three_seq_manifest(
    tmp_path: Path,
    *,
    n_ids: int = 1,
    contiguous: bool = True,
    continuous: bool = True,
    n_frames: int = 10,
) -> dict[str, ManifestEntry]:
    """A 3-sequence group with adjacency, as the resolvers would have built it.

    ``contiguous`` chooses the frame axis: numbered across the whole group, as a
    continuous recording is, or restarted at zero for each sequence, which is
    what every mosaic converter writes today and what overlap has to refuse.
    """
    names = ("s1", "s2", "s3")
    paths: dict[str, Path] = {}
    extents: dict[str, tuple[int, int]] = {}
    for index, name in enumerate(names):
        start = index * n_frames if contiguous else 0
        path = tmp_path / f"g1__{name}.parquet"
        _make_simple_parquet(
            path, n_frames=n_frames, start_frame=start, n_ids=n_ids, sequence=name
        )
        paths[name] = path
        extents[name] = (start, start + n_frames - 1)

    def specs(name: str) -> list[tuple[Path, ParquetLoadSpec]]:
        return [(paths[name], ParquetLoadSpec())]

    def entry(index: int) -> ManifestEntry:
        name = names[index]
        prev_name = names[index - 1] if index > 0 else None
        next_name = names[index + 1] if index < len(names) - 1 else None
        return ManifestEntry(
            file_specs=specs(name),
            prev_file_specs=None if prev_name is None else specs(prev_name),
            prev_entry_key=None if prev_name is None else f"g1__{prev_name}",
            next_file_specs=None if next_name is None else specs(next_name),
            next_entry_key=None if next_name is None else f"g1__{next_name}",
            entry_key=f"g1__{name}",
            group="g1",
            sequence=name,
            continuous=continuous,
            core_extent=extents[name],
            prev_extent=None if prev_name is None else extents[prev_name],
            next_extent=None if next_name is None else extents[next_name],
        )

    return {f"g1__{name}": entry(index) for index, name in enumerate(names)}


# --- overlap_frames tests ---


def test_iter_manifest_overlap_frames_zero(tmp_path):
    """overlap_frames=0 loads no neighbour and selects the whole segment."""
    manifest = _build_three_seq_manifest(tmp_path)
    results = list(iter_manifest(manifest, overlap_frames=0))
    assert len(results) == 3
    for index, (_key, df, selector) in enumerate(results):
        assert len(df) == 10
        assert (selector.first, selector.last) == (index * 10, index * 10 + 9)


def test_iter_manifest_overlap_frames_positive(tmp_path):
    """overlap_frames > 0 loads and concatenates neighbour data."""
    manifest = _build_three_seq_manifest(tmp_path)
    results = list(iter_manifest(manifest, overlap_frames=3))
    assert len(results) == 3

    # s1: no prev, has next -> 10 core + 3 next = 13
    key_s1, df_s1, sel_s1 = results[0]
    assert key_s1 == "g1__s1"
    assert (sel_s1.first, sel_s1.last) == (0, 9)
    assert len(df_s1) == 13

    # s2: has prev and next -> 3 prev + 10 core + 3 next = 16
    key_s2, df_s2, sel_s2 = results[1]
    assert key_s2 == "g1__s2"
    assert (sel_s2.first, sel_s2.last) == (10, 19)
    assert len(df_s2) == 16

    # s3: has prev, no next -> 3 prev + 10 core = 13
    key_s3, df_s3, sel_s3 = results[2]
    assert key_s3 == "g1__s3"
    assert (sel_s3.first, sel_s3.last) == (20, 29)
    assert len(df_s3) == 13


def test_overlap_context_is_counted_in_frames_not_rows(tmp_path):
    """The window is N frames wide however many individuals a frame holds.

    ``.iloc[-N:]`` took N rows, so three individuals turned a request for three
    frames of context into one. The row count then varied with the population and
    nothing said so.
    """
    manifest = _build_three_seq_manifest(tmp_path, n_ids=3)
    _key, df, selector = list(iter_manifest(manifest, overlap_frames=3))[1]

    before = sorted(df.loc[df["frame"] < selector.first, "frame"].unique())
    after = sorted(df.loc[df["frame"] > selector.last, "frame"].unique())
    assert before == [7, 8, 9]
    assert after == [20, 21, 22]
    assert len(df) == 3 * (3 + 10 + 3)


def test_iter_manifest_overlap_frames_exceeds_neighbor(tmp_path):
    """A window wider than the neighbour takes all of it, with no special case."""
    manifest = _build_three_seq_manifest(tmp_path)
    results = list(iter_manifest(manifest, overlap_frames=50))
    assert len(results) == 3

    key_s2, df_s2, sel_s2 = results[1]
    assert key_s2 == "g1__s2"
    assert (sel_s2.first, sel_s2.last) == (10, 19)
    assert len(df_s2) == 30


def test_iter_manifest_no_overlap_yields_two_tuples(tmp_path):
    """overlap_frames=None (default) yields 2-tuples."""
    manifest = _build_three_seq_manifest(tmp_path)
    results = list(iter_manifest(manifest))
    assert len(results) == 3
    for item in results:
        assert len(item) == 2


# --- filter_factory tests ---


def test_iter_manifest_filter_factory(tmp_path):
    """filter_factory filters are applied to the loaded data."""
    manifest = _build_three_seq_manifest(tmp_path)

    def factory(entry_key: str) -> Iterable[Callable[[pd.DataFrame], pd.DataFrame]]:
        # Keep only first 5 rows
        return [lambda df: df.iloc[:5]]

    results = list(iter_manifest(manifest, filter_factory=factory))
    assert len(results) == 3
    for entry_key, df in results:
        assert len(df) == 5


def test_iter_manifest_filter_factory_with_overlap(tmp_path):
    """filter_factory is applied to both core and neighbor segments."""
    manifest = _build_three_seq_manifest(tmp_path)

    def factory(entry_key: str) -> Iterable[Callable[[pd.DataFrame], pd.DataFrame]]:
        # Keep only first 5 rows of each segment
        return [lambda df: df.iloc[:5]]

    results = list(iter_manifest(manifest, filter_factory=factory, overlap_frames=3))
    assert len(results) == 3

    # Each segment is filtered to its first 5 frames, then the neighbours are
    # windowed to the 3 frames nearest the boundary of what survived.
    key_s2, df_s2, sel_s2 = results[1]
    assert key_s2 == "g1__s2"
    assert (sel_s2.first, sel_s2.last) == (10, 14)
    assert sorted(df_s2["frame"].unique()) == [2, 3, 4, 10, 11, 12, 13, 14, 20, 21, 22]


def test_iter_manifest_filter_factory_empty_skips(tmp_path):
    """Entries filtered to empty are skipped."""
    manifest = _build_three_seq_manifest(tmp_path)

    def factory(entry_key: str) -> Iterable[Callable[[pd.DataFrame], pd.DataFrame]]:
        # Return empty DataFrame
        return [lambda df: df.iloc[0:0]]

    results = list(iter_manifest(manifest, filter_factory=factory))
    assert len(results) == 0


# --------------------------------------------------------------------------- #
# Feature-chaining contract for the social/interaction features
# --------------------------------------------------------------------------- #

_CHAINED_FEATURES = [
    "ffgroups-metrics",
    "social-motion-summary",
    "nn-delta-response",
    "nn-delta-bins",
    "collective-motion-metrics",
    "local-order-metrics",
]


@pytest.mark.parametrize("feature_name", _CHAINED_FEATURES)
def test_social_features_accept_derived_results(tmp_path, feature_name):
    """These four are wired downstream of nearest-neighbor / speed-angvel / ffgroups.

    Those producers join back only ``meta_set()`` and so drop X/Y, which the
    track-input contract refuses. The features therefore declare the permissive
    ``Inputs[TrackInput | Result]`` rather than ``TrackInputs``: they take their
    positions from a track-shaped input (trajectory-smooth) and merely *merge*
    the derived columns. Reverting any of them to ``TrackInputs`` breaks the
    whole downstream half of a realistic pipeline, and it breaks it at manifest
    time with an error naming only one of the inputs -- hence this test.
    """
    from mosaic.behavior.feature_library import FEATURES

    cls = next(c for c in FEATURES.values() if getattr(c, "name", None) == feature_name)
    assert getattr(cls.Inputs, "_track_input", False) is False, (
        f"{feature_name} declares TrackInputs; it cannot be fed a derived Result"
    )

    ds = _MockDataset(tmp_path)
    smooth_run = _setup_feature(
        ds, "trajectory-smooth__from__tracks", [("g1", "s1")], track_shaped=True
    )
    derived_run = _setup_feature(
        ds, "speed-angvel__from__tracks", [("g1", "s1")], track_shaped=False
    )
    inputs = cls.Inputs(
        (
            Result(feature="trajectory-smooth__from__tracks", run_id=smooth_run),
            Result(feature="speed-angvel__from__tracks", run_id=derived_run),
        )
    )
    manifest, scope = build_manifest(ds, inputs)
    assert scope.entries == {("g1", "s1")}
    assert len(manifest) == 1


@pytest.mark.parametrize("feature_name", _CHAINED_FEATURES)
def test_widening_inputs_did_not_move_the_identifier(feature_name):
    """The permissive form is a *class* change, so the hashed payload is untouched.

    ``run_id`` hashes ``feature.inputs.model_dump()``, and ``Inputs`` is a
    RootModel -- the dump is the tuple of items, not the class. If this ever
    stops holding, every cached run of these features silently orphans.
    """
    from mosaic.behavior.feature_library import FEATURES

    cls = next(c for c in FEATURES.values() if getattr(c, "name", None) == feature_name)
    assert cls.Inputs(("tracks",)).model_dump() == ("tracks",)
