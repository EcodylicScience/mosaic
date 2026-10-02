"""The trained-model index's paths survive a move, and are reached at all.

Two defects, one story. ``models`` was registered in ``_ROOT_SHAPES`` as a
``root``-shaped index, meaning ``models/index.csv`` -- a file nothing has ever
written. Every real index is ``models/<kind>/index.csv``, so the whole root was
invisible to ``make_portable``, ``rewrite_index_paths`` and ``reconcile``. And
even once visible, ``_INDEX_PATH_COLUMNS`` named no columns for it, so its three
path-bearing columns would have been skipped anyway.

A registered model survived a move only because ``resolve_path`` re-anchors a
relative string; an absolute one, which is what an older row holds, did not.
That is the ``detect_model=<run_id>`` handoff breaking on a dataset synced
between machines.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.dataset_indexes import iter_dataset_indexes
from mosaic.core.pipeline.models import model_index_path
from mosaic.tracking.model_refs import resolve_model
from mosaic.tracking.ops.train import TrainedModelIndexRow, trained_model_index
from tests.helpers import make_dataset

RUN_ID = "train-litpose.0.1-abcdef0123"


def _litpose_artifact(run_root: Path) -> Path:
    checkpoints = run_root / "tb_logs" / "run" / "version_0" / "checkpoints"
    checkpoints.mkdir(parents=True)
    _ = (run_root / "config.yaml").write_text("model:\n  model_type: heatmap\n")
    _ = (checkpoints / "best.ckpt").write_bytes(b"weights")
    return checkpoints / "best.ckpt"


def _register(ds: Dataset, *, absolute: bool) -> Path:
    """One directory-shaped model row, written with absolute or relative paths."""
    run_root = ds.get_root("models") / "train-litpose" / RUN_ID
    weights = _litpose_artifact(run_root)

    def store(path: Path) -> str:
        return str(path) if absolute else ds.relative_to_root(path)

    index = trained_model_index(model_index_path(ds, "train-litpose"))
    index.ensure()
    index.append(
        [
            TrainedModelIndexRow(
                run_id=RUN_ID,
                kind="train-litpose",
                base_model="",
                base_run_id="",
                best_model_path=store(weights),
                metrics_path="",
                n_epochs=1,
                status="finished",
                artifact_shape="directory",
                artifact_path=store(run_root),
                model_type="heatmap",
                abs_path=Path(ds.relative_to_root(run_root)),
            )
        ]
    )
    return run_root


def test_the_model_indexes_are_enumerated(tmp_path: Path) -> None:
    """The shape fix, stated directly: ``models/<kind>/index.csv`` is now visited."""
    ds = make_dataset(tmp_path, name="m", save=False)
    _ = _register(ds, absolute=False)

    visited = [index.path for index in iter_dataset_indexes(ds)]
    assert model_index_path(ds, "train-litpose") in visited


def test_a_directory_shaped_model_resolves_by_its_run(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path, name="m", save=False)
    run_root = _register(ds, absolute=False)

    resolved = resolve_model(ds, RUN_ID, "train-litpose")
    assert resolved.path == run_root, "the directory, not the checkpoint inside it"
    assert resolved.model_id == RUN_ID, "a registered model is named by its run"
    assert resolved.model_type == "heatmap"


def test_make_portable_relativizes_the_model_paths(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path, name="m", save=False)
    _ = _register(ds, absolute=True)
    index_path = model_index_path(ds, "train-litpose")

    before = pd.read_csv(index_path, keep_default_na=False)
    assert Path(str(before.loc[0, "artifact_path"])).is_absolute()

    _ = ds.make_portable()

    after = pd.read_csv(index_path, keep_default_na=False)
    for column in ("abs_path", "best_model_path", "artifact_path"):
        assert not Path(str(after.loc[0, column])).is_absolute(), column


def test_a_registered_model_survives_the_dataset_moving(tmp_path: Path) -> None:
    """The point of the two fixes, end to end."""
    origin = tmp_path / "origin"
    origin.mkdir()
    ds = make_dataset(origin, name="m", save=False)
    _ = _register(ds, absolute=True)
    _ = ds.make_portable()

    destination = tmp_path / "moved"
    shutil.move(str(origin), str(destination))

    moved = Dataset(manifest_path=destination / "dataset.yaml").load()
    resolved = resolve_model(moved, RUN_ID, "train-litpose")
    assert resolved.path.is_relative_to(destination)
    assert resolved.model_id == RUN_ID


def test_a_row_without_an_artifact_path_still_resolves(tmp_path: Path) -> None:
    """Every row written before a model could be a directory names only the file.

    An empty cell also reads back from pandas as ``NaN``, which is truthy -- so
    the fallback has to test the value, not merely its presence.
    """
    ds = make_dataset(tmp_path, name="m", save=False)
    weights = ds.get_root("models") / "train-pose" / "r1" / "best.pt"
    weights.parent.mkdir(parents=True)
    _ = weights.write_bytes(b"yolo weights")

    index = trained_model_index(model_index_path(ds, "train-pose"))
    index.ensure()
    index.append(
        [
            TrainedModelIndexRow(
                run_id="train-pose.0.1-legacy0000",
                kind="train-pose",
                base_model="",
                base_run_id="",
                best_model_path=ds.relative_to_root(weights),
                metrics_path="",
                n_epochs=1,
                status="finished",
                abs_path=Path(ds.relative_to_root(weights.parent)),
            )
        ]
    )

    resolved = resolve_model(ds, "train-pose.0.1-legacy0000", "train-pose")
    assert resolved.path == weights
    assert "nan" not in str(resolved.path)
