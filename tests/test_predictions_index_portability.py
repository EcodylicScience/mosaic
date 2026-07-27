"""An inference index's path columns survive a move -- and get visited at all.

Same defect as the models index, one step worse: neither path-repair pass
mentioned the ``predictions`` root under any spelling, so
``predictions/<kind>/index.csv`` was unreachable by both. ``InferenceIndexRow``
carries ``video_abs_path``, and a stored absolute recorded on another machine is
precisely what inverted the tracker's reuse guard into a permanent recompute --
the failure ``test_trex_index_portability`` was written for.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mosaic.core.dataset import Dataset, new_dataset_manifest
from mosaic.tracking.ops.infer import (
    InferenceIndexRow,
    inference_index,
    prediction_index_path,
)

KIND = "pose"
RUN_ID = "0.1-abcdef0123"


def dataset_with_prediction_row(
    tmp_path: Path, *, absolute: bool
) -> tuple[Dataset, Path]:
    """A dataset holding one inference row, written with absolute or relative paths.

    Nested under ``tmp_path/orig`` so the move test relocates it inside this
    test's own directory: ``tmp_path.parent`` is shared across the pytest session.
    """
    base = tmp_path / "orig"
    base.mkdir(parents=True, exist_ok=True)
    ds = Dataset(manifest_path=new_dataset_manifest("p", base_dir=base)).load(
        ensure_roots=True
    )

    video = ds.get_root(ds.resolve_media_root()) / "vid1.mp4"
    video.parent.mkdir(parents=True, exist_ok=True)
    _ = video.write_bytes(b"fake")

    run_dir = ds.get_root("predictions") / KIND / RUN_ID
    run_dir.mkdir(parents=True, exist_ok=True)

    index = inference_index(prediction_index_path(ds, KIND))
    index.ensure()
    index.append(
        [
            InferenceIndexRow(
                run_id=RUN_ID,
                model_run_id="0.1-fedcba9876",
                group="",
                sequence="vid1",
                video_abs_path=(str(video) if absolute else ds.relative_to_root(video)),
                start_frame=0,
                end_frame=10,
                n_rows=10,
                abs_path=Path(ds.relative_to_root(run_dir)),
            )
        ]
    )
    return ds, video


def read_row(ds: Dataset) -> dict[str, str]:
    df = pd.read_csv(prediction_index_path(ds, KIND))
    row = df.iloc[0]
    return {str(col): str(row[col]) for col in list(df.columns)}


def test_make_portable_converts_the_prediction_video_column(tmp_path: Path) -> None:
    """A legacy index full of absolutes is repaired, not just its abs_path."""
    ds, _ = dataset_with_prediction_row(tmp_path, absolute=True)

    assert Path(read_row(ds)["video_abs_path"]).is_absolute()

    changed = ds.make_portable()

    assert not Path(read_row(ds)["video_abs_path"]).is_absolute()
    assert any("predictions" in key for key in changed), (
        f"the predictions index was not visited: {sorted(changed)}"
    )


def test_rewrite_index_paths_remaps_the_prediction_video(tmp_path: Path) -> None:
    """A relocated dataset remaps the video column, not only abs_path."""
    ds, video = dataset_with_prediction_row(tmp_path, absolute=True)
    moved_root = tmp_path / "elsewhere"

    counts = ds.rewrite_index_paths({str(tmp_path / "orig"): str(moved_root)})

    assert read_row(ds)["video_abs_path"].startswith(str(moved_root))
    assert any("predictions" in key for key in counts), (
        f"the predictions index was not visited: {sorted(counts)}"
    )
    assert video.exists(), "rewriting an index must not touch the files it names"


def test_a_relative_index_resolves_after_the_dataset_moves(tmp_path: Path) -> None:
    """The point of storing relative: the row still names the right file elsewhere."""
    _ = dataset_with_prediction_row(tmp_path, absolute=False)

    moved = tmp_path / "moved"
    _ = (tmp_path / "orig").rename(moved)

    reloaded = Dataset(manifest_path=moved / "dataset.yaml").load()
    resolved = reloaded.resolve_path(read_row(reloaded)["video_abs_path"])

    assert resolved.exists()
    assert resolved.is_relative_to(moved)
