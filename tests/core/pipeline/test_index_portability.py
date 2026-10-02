"""Index paths that survive a dataset moving, and the feature index's reindex.

Every path column of every index is relativized by ``make_portable``, remapped
by ``rewrite_index_paths``, and still names its file once the dataset moves:
one contract, run over an index of each kind that carries a path beyond
``abs_path``.

The rest guards the fix that makes feature indexes store dataset-root-*relative*
paths and the resolve-then-skip behavior in ``manifest._resolve_feature``:

- relocated-but-present outputs resolve under a different root (no false-fail),
- an all-missing run raises a loud, actionable error (dataset moved),
- a partially-missing run skips the gone entries (they recompute upstream),
- ``Dataset.reindex_features`` prunes only genuinely-missing rows and reconciles
  the SQLite registry mirror,
- a real ``run_feature`` writes relative paths (regression guard).
"""

from __future__ import annotations

import shutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import pytest
import yaml

from mosaic.behavior.feature_library import SpeedAngvel
from mosaic.core.dataset import Dataset, new_dataset_manifest
from mosaic.core.pipeline import FeatureStep, Pipeline
from mosaic.core.pipeline.index import (
    FeatureIndexRow,
    feature_index,
    feature_index_path,
)
from mosaic.core.pipeline.manifest import _resolve_feature
from mosaic.core.pipeline.models import model_index_path, model_run_root
from mosaic.core.pipeline.tracks_index import tracks_index_path, write_tracks_row
from mosaic.tracking.trex.dataset_runs import TRexIndexRow, trex_index, trex_index_path
from tests.helpers import (
    MockDataset,
    add_track_sequences,
    make_dataset,
    register_trained_model,
    write_litpose_model,
)


# --- Helpers ---


def _make_feat_parquet(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "frame": range(5),
            "time": [f / 30.0 for f in range(5)],
            "id": [0] * 5,
            "feat": np.arange(5.0),
        }
    ).to_parquet(path)


def _write_relative_feature_index(
    root: Path,
    feat: str,
    run_id: str,
    pairs: list[tuple[str, str]],
) -> None:
    """Write a feature index under *root* whose abs_path values are RELATIVE."""
    idx = feature_index(feature_index_path(MockDataset(root), feat))
    idx.ensure()
    rows: list[FeatureIndexRow] = []
    for g, s in pairs:
        rel = f"features/{feat}/{run_id}/{g}__{s}.parquet"
        _make_feat_parquet(root / rel)
        rows.append(
            FeatureIndexRow(
                run_id=run_id,
                feature=feat,
                version="0.1",
                group=g,
                sequence=s,
                abs_path=rel,
                n_rows=5,
                params_hash="h",
            )
        )
    idx.append(rows)
    idx.mark_finished(run_id)


# --- _resolve_feature portability behavior ---


def test_relative_index_resolves_under_a_different_root(tmp_path: Path) -> None:
    """A relative index built under root A resolves after a copy to root B."""
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    root_a.mkdir()
    _write_relative_feature_index(root_a, "feat", "0.1-abc", [("g", "s1"), ("g", "s2")])
    shutil.copytree(root_a, root_b)

    ds_b = MockDataset(root_b)
    result = _resolve_feature(ds_b, "feat", "0.1-abc")
    assert result.entries == {("g", "s1"), ("g", "s2")}
    assert result.full_order == [("g", "s1"), ("g", "s2")]
    # Every resolved path lives under root_b and exists.
    for resolved, _spec in result.path_map_all.values():
        assert resolved.exists()
        assert root_b in resolved.parents


def test_all_missing_run_raises_actionable_error(tmp_path: Path) -> None:
    root = tmp_path / "a"
    root.mkdir()
    _write_relative_feature_index(root, "feat", "0.1-abc", [("g", "s1"), ("g", "s2")])
    # Remove every output for the run -> "dataset moved" signal.
    shutil.rmtree(root / "features" / "feat" / "0.1-abc")

    ds = MockDataset(root)
    with pytest.raises(FileNotFoundError, match="output file"):
        _resolve_feature(ds, "feat", "0.1-abc")


def test_partial_missing_run_skips(tmp_path: Path) -> None:
    root = tmp_path / "a"
    root.mkdir()
    _write_relative_feature_index(root, "feat", "0.1-abc", [("g", "s1"), ("g", "s2")])
    (root / "features" / "feat" / "0.1-abc" / "g__s1.parquet").unlink()

    ds = MockDataset(root)
    result = _resolve_feature(ds, "feat", "0.1-abc")
    # The surviving entry is kept; the missing one is dropped (recomputed upstream).
    assert result.entries == {("g", "s2")}
    assert result.full_order == [("g", "s2")]


# --- Dataset.reindex_features ---


def _dataset_with_manual_feature(tmp_path: Path) -> tuple[Dataset, str, str]:
    ds = make_dataset(tmp_path)
    feat, run_id = "feat", "0.1-abc"
    _write_relative_feature_index(
        ds.get_root("features").parent, feat, run_id, [("g", "s1"), ("g", "s2")]
    )
    return ds, feat, run_id


def test_reindex_dry_run_reports_without_writing(tmp_path: Path) -> None:
    ds, feat, run_id = _dataset_with_manual_feature(tmp_path)
    (ds.get_root("features") / feat / run_id / "g__s1.parquet").unlink()

    report = ds.reindex_features(dry_run=True)
    assert sum(report.values()) == 1
    # Nothing rewritten on disk.
    idx = feature_index(feature_index_path(ds, feat))
    assert len(idx.read(run_id=run_id)) == 2


def test_reindex_drops_missing_keeps_present(tmp_path: Path) -> None:
    ds, feat, run_id = _dataset_with_manual_feature(tmp_path)
    # index.csv starts with 2 entries.
    idx = feature_index(feature_index_path(ds, feat))
    assert len(idx.read(run_id=run_id)) == 2

    (ds.get_root("features") / feat / run_id / "g__s1.parquet").unlink()

    report = ds.reindex_features(dry_run=False)
    assert sum(report.values()) == 1

    # index.csv is the source of truth: the missing entry is pruned, s2 kept.
    remaining = feature_index(feature_index_path(ds, feat)).read(run_id=run_id)
    assert len(remaining) == 1
    assert (remaining.iloc[0]["group"], remaining.iloc[0]["sequence"]) == ("g", "s2")


def test_reindex_keeps_relocated_present_rows(tmp_path: Path) -> None:
    """Relative paths that resolve to existing files are never pruned."""
    ds, feat, run_id = _dataset_with_manual_feature(tmp_path)
    report = ds.reindex_features(dry_run=False)
    assert report == {}  # all present -> nothing dropped
    idx = feature_index(feature_index_path(ds, feat))
    assert len(idx.read(run_id=run_id)) == 2


# --- run_feature writes relative paths (regression guard for the root cause) ---


def test_run_feature_writes_relative_paths(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path)
    add_track_sequences(ds, ("g", "s1"), ("g", "s2"), n_rows=12)

    pipe = Pipeline()
    pipe.add(FeatureStep("speed", SpeedAngvel, {"step_size": 1}))
    results = pipe.run(ds)
    feat, run_id = results["speed"].feature, results["speed"].run_id

    df_idx = feature_index(feature_index_path(ds, feat)).read(run_id=run_id)
    assert len(df_idx) == 2
    for stored in df_idx["abs_path"]:
        assert not Path(stored).is_absolute(), f"index path is absolute: {stored}"
        # And it still resolves to a real file under the dataset root.
        assert ds.resolve_path(stored).exists()


# --- the manifest across a load and a save -----------------------------------


def test_manifest_identity_survives_load_save(tmp_path: Path) -> None:
    """uuid and created_at survive a load -> save round-trip.

    They identify the dataset, not its most recent edit, so a save must carry
    them through. ``save()`` once wrote a fixed key list that omitted them, which
    is why callers needing them stable had to avoid ``save()`` entirely.
    """
    manifest = new_dataset_manifest(name="ds", base_dir=tmp_path)
    ds = Dataset(manifest_path=manifest).load()
    seeded = (ds.uuid, ds.created_at)
    assert all(seeded), f"manifest identity not loaded: {seeded}"

    ds.save()
    reloaded = Dataset(manifest_path=manifest).load()
    assert (reloaded.uuid, reloaded.created_at) == seeded


def test_a_key_the_current_format_does_not_model_survives_a_save(
    tmp_path: Path,
) -> None:
    """Retiring a field must not delete it from anybody's file.

    ``index_format``, ``dataset_type`` and three others stopped being modeled
    when the manifest reached version 2. They are not deleted: an unmodeled
    top-level key is carried through the load-and-save round trip untouched, so a
    dataset written by an older mosaic keeps everything it held.
    """
    manifest = new_dataset_manifest(name="ds", base_dir=tmp_path)
    text = manifest.read_text(encoding="utf-8")
    manifest.write_text(
        text + "index_format: group/sequence\ndataset_type: continuous\n",
        encoding="utf-8",
    )

    ds = Dataset(manifest_path=manifest).load()
    assert ds.manifest.preserved["dataset_type"] == "continuous"

    ds.save()
    written = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    assert written["index_format"] == "group/sequence"
    assert written["dataset_type"] == "continuous"


# --- every path column of every index -------------------------------------
#
# Both path passes read raw CSVs and rewrite the columns declared for the index's
# root. A column missing from that declaration silently stops being portable,
# and an index the enumeration does not reach is not rewritten at all. Each case
# writes one row through its producer's own writer, so every cell starts
# relative and names a file that exists.


@dataclass(frozen=True)
class _IndexedRow:
    """One index row, and the columns of it that hold a path."""

    dataset: Dataset
    index: Path
    columns: tuple[str, ...]


def _tracks_row(base: Path) -> _IndexedRow:
    """A converted table, and the upload it came from in ``source_abs_path``."""
    ds = make_dataset(base)
    source = ds.get_root("tracks_raw") / "raw.npz"
    _ = source.write_bytes(b"x")
    out = ds.get_root("tracks") / "s.parquet"
    pd.DataFrame({"frame": [0], "id": [0]}).to_parquet(out)
    write_tracks_row(
        ds,
        run_id="convert-x.0.1-aaaaaaaaaa",
        group="",
        sequence="s",
        out_path=out,
        producer="convert-x",
        std_format="trex_v1",
        n_rows=1,
        source=source,
    )
    return _IndexedRow(ds, tracks_index_path(ds), ("abs_path", "source_abs_path"))


def _model_row(base: Path) -> _IndexedRow:
    """A directory-shaped trained model, its checkpoint and its directory."""
    ds = make_dataset(base)
    run_id = "train-litpose.0.1-abcdef0123"
    directory = write_litpose_model(model_run_root(ds, "train-litpose", run_id))
    weights = next(directory.rglob("best.ckpt"))
    register_trained_model(ds, "train-litpose", run_id, weights, directory=directory)
    columns = ("abs_path", "best_model_path", "artifact_path")
    return _IndexedRow(ds, model_index_path(ds, "train-litpose"), columns)


def _trex_row(base: Path) -> _IndexedRow:
    """A tracker run, the video it read and the ``.pv`` it converted to.

    The tracker root is a subdirectory of the ``_tracking`` root, whose own
    ``index.csv`` the passes once never visited.
    """
    ds = make_dataset(base)
    video = ds.get_root(ds.resolve_media_root()) / "vid1.mp4"
    video.parent.mkdir(parents=True, exist_ok=True)
    _ = video.write_bytes(b"fake")
    seq_dir = ds.get_root("trex") / "trex-abc" / "vid1"
    seq_dir.mkdir(parents=True)
    pv_path = seq_dir / "vid1.pv"
    _ = pv_path.write_bytes(b"fake")
    index = trex_index(trex_index_path(ds))
    index.ensure()
    index.append(
        [
            TRexIndexRow(
                run_id="trex-abc",
                group="",
                sequence="vid1",
                abs_path=Path(ds.relative_to_root(seq_dir)),
                video_abs_path=ds.relative_to_root(video),
                params_hash="abc",
                n_ids=1,
                pv_path=ds.relative_to_root(pv_path),
            )
        ]
    )
    columns = ("abs_path", "video_abs_path", "pv_path")
    return _IndexedRow(ds, trex_index_path(ds), columns)


_ROWS: Final[Mapping[str, Callable[[Path], _IndexedRow]]] = {
    "models": _model_row,
    "trex": _trex_row,
    "tracks": _tracks_row,
}


def _cells(index: Path, columns: tuple[str, ...]) -> dict[str, str]:
    """The path cells of the one row *index* holds."""
    frame = pd.read_csv(index, keep_default_na=False)
    assert len(frame) == 1
    return {column: str(frame.loc[0, column]) for column in columns}


def _set_cells(index: Path, cells: Mapping[str, str]) -> None:
    """Overwrite cells of the one row, as an index written earlier holds them."""
    frame = pd.read_csv(index, keep_default_na=False)
    for column, value in cells.items():
        frame.loc[0, column] = value
    frame.to_csv(index, index=False)


@pytest.mark.parametrize("kind", sorted(_ROWS))
def test_make_portable_relativizes_every_path_column(tmp_path: Path, kind: str) -> None:
    """A legacy row of absolutes is repaired in every column, not only ``abs_path``."""
    row = _ROWS[kind](tmp_path / "ds")
    relative = _cells(row.index, row.columns)
    ds = row.dataset
    _set_cells(row.index, {c: str(ds.resolve_path(v)) for c, v in relative.items()})

    changed = ds.make_portable()

    assert _cells(row.index, row.columns) == relative
    visited = {Path(key).resolve() for key in changed}
    assert row.index.resolve() in visited, f"the index was not visited: {changed}"


@pytest.mark.parametrize("kind", sorted(_ROWS))
def test_rewrite_index_paths_remaps_every_path_column(
    tmp_path: Path, kind: str
) -> None:
    """Absolutes naming another machine are remapped, and the files left alone."""
    row = _ROWS[kind](tmp_path / "ds")
    relative = _cells(row.index, row.columns)
    ds = row.dataset
    _set_cells(row.index, {c: f"/old/machine/{v}" for c, v in relative.items()})

    _ = ds.rewrite_index_paths({"/old/machine": str(ds.base_dir)})

    for column, value in _cells(row.index, row.columns).items():
        assert "/old/machine" not in value, column
        assert ds.resolve_path(value) == ds.resolve_path(relative[column]), column
        assert ds.resolve_path(value).exists(), column


@pytest.mark.parametrize("kind", sorted(_ROWS))
def test_a_relative_row_resolves_after_the_dataset_moves(
    tmp_path: Path, kind: str
) -> None:
    """The point of storing relative: every cell still names its file elsewhere."""
    row = _ROWS[kind](tmp_path / "ds")
    index = row.index.relative_to(row.dataset.base_dir)
    moved = tmp_path / "moved"

    _ = shutil.move(str(row.dataset.base_dir), str(moved))

    reloaded = Dataset(manifest_path=moved / "dataset.yaml").load()
    for column, value in _cells(moved / index, row.columns).items():
        resolved = reloaded.resolve_path(value)
        assert resolved.is_relative_to(moved) and resolved.exists(), column
