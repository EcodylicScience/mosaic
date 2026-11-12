"""The ``arenas`` composition: each entry's layout digest, read where a run declared it."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

import mosaic.core.pipeline.arena_index as arena_index_module
from mosaic.core.arenas import Arena, ArenaKey, ArenaLayout, Rect
from mosaic.core.dataset import Dataset
from mosaic.core.entry import Entry
from mosaic.core.pipeline.arena_index import (
    ARENA_LOOKUP_CACHE,
    IndexedLayout,
    arena_index_path,
    arenas_compositions,
    cached_arena_lookup,
)
from mosaic.core.pipeline.index import (
    drifted_entries,
    feature_index_path,
    feature_run_root,
)
from mosaic.core.pipeline.inventory import FeatureRunRef, inventory
from mosaic.core.pipeline.inventory.params import read_run_params
from mosaic.core.pipeline.provenance import reached_by
from mosaic.core.pipeline.run import resolve_feature_identity, run_feature
from mosaic.core.pipeline.sequence_index import (
    media_compositions_for,
    read_entry_compositions,
)
from mosaic.core.pipeline.types import Result
from mosaic.core.pipeline.types.feature import EmitsLevel
from mosaic.core.scope import Scope
from tests.helpers import (
    MediaClip,
    PlainFeature,
    add_track_sequences,
    convert_tracks_from_raw,
    make_dataset,
    write_media_index,
)

A: Entry = ("", "seq_a")
B: Entry = ("", "seq_b")
C: Entry = ("", "seq_c")

REFUSED = "cannot be read"
"""What the refusal of an unreadable arenas index says."""


class Reader(PlainFeature):
    """A per-entry stand-in that declares no root and records what it computed."""

    name = "arena-free"
    accepts_overlap = False
    emits: EmitsLevel = "as-input"

    def __init__(self) -> None:
        super().__init__()
        self.applied: list[str] = []

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if not df.empty:
            self.applied.append(str(df["sequence"].iloc[0]))
        return df


class ArenaReader(Reader):
    """A :class:`Reader` that declares ``arenas``."""

    name = "arena-reader"
    consumed_roots: tuple[str, ...] = ("arenas",)


class ScopedArenaReader(ArenaReader):
    """An :class:`ArenaReader` whose scope is hashed into its run id."""

    name = "arena-scoped"
    scope_dependent = True


def _layout(width: int = 200) -> ArenaLayout:
    return ArenaLayout(
        width=width,
        height=100,
        arenas=(Arena(name="all", region=Rect(x0=0, y0=0, x1=width, y1=100)),),
    )


def _set(ds: Dataset, entry: Entry, layout: ArenaLayout, camera: str = "") -> str:
    """Declare *layout* for the undivided recording *entry*, returning its digest."""
    _ = ds.set_arena_layout(layout, group=entry[0], sequence=entry[1], camera=camera)
    return layout.digest()


def _refuse_index(ds: Dataset) -> None:
    """Make the arenas index unreadable, as a hand edit can."""
    path = arena_index_path(ds)
    rows = pd.read_csv(path, dtype=str, keep_default_na=False)
    rows["n_arenas"] = "many"
    rows.to_csv(path, index=False)


@pytest.fixture
def dataset(tmp_path: Path) -> Dataset:
    """Three undivided recordings, ``seq_a`` and ``seq_b`` with layouts."""
    ds = make_dataset(tmp_path / "ds")
    add_track_sequences(ds, "seq_a", "seq_b", "seq_c")
    _ = _set(ds, A, _layout(200))
    _ = _set(ds, B, _layout(300))
    return ds


class _Counted:
    """Counts the calls of the two index reads the lookup makes."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.rows = 0
        self.cameras = 0
        read_rows = arena_index_module.read_arena_index_rows
        read_cameras = arena_index_module.media_cameras

        def rows(ds: Dataset) -> dict[ArenaKey, IndexedLayout]:
            self.rows += 1
            return read_rows(ds)

        def cameras(ds: Dataset) -> dict[Entry, set[str]]:
            self.cameras += 1
            return read_cameras(ds)

        monkeypatch.setattr(arena_index_module, "read_arena_index_rows", rows)
        monkeypatch.setattr(arena_index_module, "media_cameras", cameras)


@pytest.fixture
def counted(monkeypatch: pytest.MonkeyPatch) -> _Counted:
    return _Counted(monkeypatch)


def _cached_keys(ds: Dataset) -> list[object]:
    """The cache entries that name *ds*'s arenas index."""
    path = arena_index_path(ds)
    return [key for key in ARENA_LOOKUP_CACHE if key[0] == path]


class TestArenasCompositions:
    def test_each_entry_has_its_row_s_digest(self, dataset: Dataset) -> None:
        found = arenas_compositions(dataset, [A, B, C])

        assert found == {A: _layout(200).digest(), B: _layout(300).digest()}

    def test_a_dataset_without_an_arenas_index_has_none(self, tmp_path: Path) -> None:
        ds = make_dataset(tmp_path / "bare")
        add_track_sequences(ds, "seq_a")

        assert arenas_compositions(ds, [A]) == {}

    def test_an_entry_whose_cameras_conflict_has_none(self, dataset: Dataset) -> None:
        write_media_index(
            dataset,
            [
                MediaClip(sequence="seq_c", filename="c_top.mp4", camera="top"),
                MediaClip(sequence="seq_c", filename="c_side.mp4", camera="side"),
            ],
        )
        _ = _set(dataset, C, _layout(200), camera="top")
        _ = _set(dataset, C, _layout(300), camera="side")

        found = arenas_compositions(dataset, [A, C])

        assert found == {A: _layout(200).digest()}

    def test_a_refused_index_leaves_every_entry_without_one_and_says_so_once(
        self, dataset: Dataset, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _refuse_index(dataset)

        first = arenas_compositions(dataset, [A, B])
        second = arenas_compositions(dataset, [A, B])

        assert first == second == {}
        assert capsys.readouterr().err.count(REFUSED) == 1


class TestCachedArenaLookup:
    def test_repeated_calls_read_each_index_once(
        self, dataset: Dataset, counted: _Counted
    ) -> None:
        for _ in range(3):
            _ = cached_arena_lookup(dataset)

        assert (counted.rows, counted.cameras) == (1, 1)

    def test_a_layout_write_is_read_again(
        self, dataset: Dataset, counted: _Counted
    ) -> None:
        before = cached_arena_lookup(dataset)
        digest = _set(dataset, C, _layout(400))

        after = cached_arena_lookup(dataset)

        assert counted.rows == 2
        assert after.row(dataset, C) is not None
        assert before.rows != after.rows
        assert arenas_compositions(dataset, [C]) == {C: digest}
        assert len(_cached_keys(dataset)) == 1

    def test_a_media_index_write_is_read_again(
        self, dataset: Dataset, counted: _Counted
    ) -> None:
        _ = cached_arena_lookup(dataset)
        write_media_index(dataset, ["seq_a"])

        lookup = cached_arena_lookup(dataset)

        assert counted.cameras == 2
        assert A in lookup.cameras
        assert len(_cached_keys(dataset)) == 1

    def test_two_datasets_are_cached_apart(
        self, dataset: Dataset, tmp_path: Path
    ) -> None:
        other = make_dataset(tmp_path / "other")
        add_track_sequences(other, "seq_a")
        digest = _set(other, A, _layout(500))

        assert arenas_compositions(dataset, [A]) == {A: _layout(200).digest()}
        assert arenas_compositions(other, [A]) == {A: digest}


class TestRunIdentity:
    def test_a_scope_dependent_run_id_moves_with_a_layout_edit(
        self, dataset: Dataset
    ) -> None:
        scope = Scope(entries=[A, B])
        before = run_feature(dataset, ScopedArenaReader(), scope=scope)
        predicted, _ = resolve_feature_identity(
            dataset, ScopedArenaReader(), scope=scope
        )
        _ = _set(dataset, A, _layout(250))

        after = run_feature(dataset, ScopedArenaReader(), scope=scope)

        assert predicted == before.run_id
        assert after.run_id != before.run_id

    def test_a_scope_free_identity_reads_neither_index(
        self, dataset: Dataset, counted: _Counted
    ) -> None:
        _ = resolve_feature_identity(dataset, ArenaReader(), scope=Scope(entries=[A]))

        assert (counted.rows, counted.cameras) == (0, 0)


def _run(ds: Dataset, feature: ArenaReader | None = None) -> tuple[Result, list[str]]:
    reader = ArenaReader() if feature is None else feature
    result = run_feature(ds, reader)
    return result, sorted(reader.applied)


def _drifted(ds: Dataset, result: Result) -> tuple[Entry, ...]:
    assert result.run_id is not None
    params = read_run_params(feature_run_root(ds, result.feature, result.run_id))
    return drifted_entries(ds, result.feature, result.run_id, params.params)


class TestCache:
    def test_a_layout_edit_recomputes_that_recording_only(
        self, dataset: Dataset
    ) -> None:
        first, computed = _run(dataset)
        assert computed == ["seq_a", "seq_b", "seq_c"]
        served, computed = _run(dataset)
        assert computed == []
        _ = _set(dataset, A, _layout(250))

        edited, computed = _run(dataset)

        assert computed == ["seq_a"]
        assert first.run_id == served.run_id == edited.run_id

    def test_an_entry_that_gains_a_layout_recomputes_once(
        self, dataset: Dataset
    ) -> None:
        _ = _run(dataset)
        _ = _set(dataset, C, _layout(400))

        _, gained = _run(dataset)
        _, again = _run(dataset)

        assert (gained, again) == (["seq_c"], [])

    def test_serving_an_entry_without_a_layout_prints_no_blind_roots_warning(
        self, dataset: Dataset, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _ = _run(dataset)
        _ = capsys.readouterr()

        _, computed = _run(dataset)

        assert computed == []
        err = capsys.readouterr().err
        assert "unnoticed" not in err
        assert "mosaic scan" not in err


class TestDrift:
    def test_drift_names_the_edited_recording(self, dataset: Dataset) -> None:
        result, _ = _run(dataset)
        assert _drifted(dataset, result) == ()

        _ = _set(dataset, B, _layout(250))

        assert _drifted(dataset, result) == (B,)

    def test_an_inventory_scan_reads_the_run_current_then_drifted(
        self, dataset: Dataset
    ) -> None:
        result, _ = _run(dataset)
        assert result.run_id is not None
        ref = FeatureRunRef(name=result.feature, run_id=result.run_id)
        current = inventory(dataset, kinds=["feature"]).record(ref)

        _ = _set(dataset, B, _layout(250))
        drifted = inventory(dataset, kinds=["feature"]).record(ref)

        assert current is not None and current.status == "complete"
        assert drifted is not None and drifted.status == "complete-but-drifted"
        assert drifted.drift == (B,)

    def test_a_refused_index_is_noted_once_per_scan(
        self, dataset: Dataset, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _ = _run(dataset)
        _ = run_feature(dataset, ScopedArenaReader())
        _refuse_index(dataset)
        _ = capsys.readouterr()

        found = inventory(dataset, kinds=["feature"])

        assert [record.drift for record in found.records] == [(A, B), (A, B)]
        assert capsys.readouterr().err.count(REFUSED) == 1


def test_provenance_names_the_current_digest(dataset: Dataset) -> None:
    _ = _run(dataset)
    digest = _set(dataset, A, _layout(250))

    reached = reached_by(dataset, [A], "arenas")

    assert list(reached["current"]) == [digest]
    assert list(reached["verdict"]) == ["drifted"]


def test_a_feature_without_arenas_records_no_arenas_digest(dataset: Dataset) -> None:
    result = run_feature(dataset, Reader())
    assert result.run_id is not None

    rows = pd.read_csv(
        feature_index_path(dataset, result.feature), dtype=str, keep_default_na=False
    )
    params = json.loads(
        (
            feature_run_root(dataset, result.feature, result.run_id) / "params.json"
        ).read_text()
    )

    assert set(rows["consumed_composition"]) == {""}
    assert "arenas" not in json.dumps(params["_scope"]["compositions"])


def _refusing(name: str) -> Callable[..., object]:
    def refuse(*_args: object, **_kwargs: object) -> object:
        raise AssertionError(f"{name} was read")

    return refuse


def test_source_reads_run_as_before_on_a_refused_index(
    dataset: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Conversion, the source compositions and a run without ``arenas`` read no
    arenas index."""
    _refuse_index(dataset)
    for name in ("read_arena_index_rows", "media_cameras", "cached_arena_lookup"):
        monkeypatch.setattr(arena_index_module, name, _refusing(name))
    write_media_index(dataset, ["seq_a"])

    convert_tracks_from_raw(dataset, "conv.0.1-aaaaaaaaaa", "seq_a")

    assert set(read_entry_compositions(dataset, [A])[A]) == {"tracks_raw"}
    assert set(media_compositions_for(dataset, [A])) <= {A}
    assert run_feature(dataset, Reader()).failed_entries == ()
