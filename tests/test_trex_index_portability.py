"""The tracker index's path columns survive a move.

``TRexIndexRow`` carries three paths, not one: the working directory, the
source video, and the ``.pv``. Only ``abs_path`` was ever made portable, and
neither path pass reached the tracker root at all -- its default location,
``_tracking/trex``, is a *subdirectory* of the ``_tracking`` root, whose own
``index.csv`` the loops never visited.

That was invisible until the tracker's reuse guard began comparing the stored
source video against the freshly resolved one: a stored absolute never matches
after a move or a sync between machines, so the guard would invert into a
permanent full recompute of every sequence.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mosaic.core.dataset import Dataset, new_dataset_manifest
from mosaic.tracking.trex.dataset_runs import (
    TRexIndexRow,
    trex_index,
    trex_index_path,
)


def dataset_with_trex_row(tmp_path: Path, *, absolute: bool) -> tuple[Dataset, Path]:
    """A dataset holding one tracker row, written with absolute or relative paths."""
    manifest = new_dataset_manifest("t", base_dir=tmp_path)
    ds = Dataset(manifest_path=manifest).load(ensure_roots=True)

    video = ds.get_root(ds.resolve_media_root()) / "vid1.mp4"
    video.parent.mkdir(parents=True, exist_ok=True)
    _ = video.write_bytes(b"fake")

    seq_dir = ds.get_root("trex") / "trex-abc" / "vid1"
    seq_dir.mkdir(parents=True, exist_ok=True)
    pv_path = seq_dir / "vid1.pv"
    _ = pv_path.write_bytes(b"fake")

    def store(path: Path) -> str:
        return str(path) if absolute else ds.relative_to_root(path)

    index = trex_index(trex_index_path(ds))
    index.ensure()
    index.append(
        [
            TRexIndexRow(
                run_id="trex-abc",
                group="",
                sequence="vid1",
                abs_path=Path(ds.relative_to_root(seq_dir)),
                video_abs_path=store(video),
                params_hash="abc",
                n_individuals=1,
                pv_path=store(pv_path),
            )
        ]
    )
    return ds, video


def read_row(ds: Dataset) -> dict[str, str]:
    df = pd.read_csv(trex_index_path(ds))
    row = df.iloc[0]
    return {str(col): str(row[col]) for col in list(df.columns)}


def test_make_portable_converts_every_tracker_path_column(tmp_path: Path) -> None:
    """A legacy index full of absolutes is repaired, not just its abs_path."""
    ds, _ = dataset_with_trex_row(tmp_path, absolute=True)

    before = read_row(ds)
    assert Path(before["video_abs_path"]).is_absolute()
    assert Path(before["pv_path"]).is_absolute()

    changed = ds.make_portable()

    after = read_row(ds)
    assert not Path(after["video_abs_path"]).is_absolute()
    assert not Path(after["pv_path"]).is_absolute()
    assert any("trex" in key for key in changed), (
        f"the tracker index was not visited: {sorted(changed)}"
    )


def test_rewrite_index_paths_remaps_the_source_video(tmp_path: Path) -> None:
    """A relocated dataset remaps the video column, not only abs_path."""
    ds, video = dataset_with_trex_row(tmp_path, absolute=True)
    moved_root = tmp_path.parent / "moved"

    counts = ds.rewrite_index_paths({str(tmp_path): str(moved_root)})

    after = read_row(ds)
    assert after["video_abs_path"].startswith(str(moved_root))
    assert after["pv_path"].startswith(str(moved_root))
    assert sum(counts.values()) >= 2, f"expected both columns remapped: {counts}"
    assert video.exists(), "rewriting an index must not touch the files it names"


def test_a_relative_index_resolves_after_the_dataset_moves(tmp_path: Path) -> None:
    """The point of storing relative: the row still names the right file elsewhere."""
    _ = dataset_with_trex_row(tmp_path, absolute=False)

    moved = tmp_path.parent / "moved"
    moved.mkdir()
    for child in tmp_path.iterdir():
        _ = child.rename(moved / child.name)

    reloaded = Dataset(manifest_path=moved / "dataset.yaml").load()
    row = read_row(reloaded)

    resolved = reloaded.resolve_path(row["video_abs_path"])
    assert resolved.exists()
    assert resolved.parent.parent == moved
