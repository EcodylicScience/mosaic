"""A trained-model index's path columns survive a move -- and get visited at all.

``model_index_path`` writes ``models/<kind>/index.csv``, and nothing has ever
written a root-level ``models/index.csv``. Both path-repair passes listed
``models`` among the *root-index-only* keys, so each one opened a file that does
not exist, found nothing, and reported zero changes. A dataset whose model index
was full of absolutes therefore came back from ``make_portable`` looking repaired.

The cost is not theoretical. ``tracking.model_refs.resolve_model`` reads
``best_model_path`` out of that index and hands it to ``resolve_path``; a stored
absolute recorded on another machine resolves to nothing, so a model referenced
by run_id becomes unloadable after a sync.

The rows also outlive their writers: the legacy ``ModelIndexRow`` (removed in
f7ae561) spelled its paths ``config_path`` and ``metrics_path``, and datasets
written before that removal still carry those columns. Hence one shared column
tuple covering every writer's schema, current and historical.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mosaic.core.dataset import Dataset, new_dataset_manifest
from mosaic.core.pipeline.models import model_index_path
from mosaic.tracking.model_refs import resolve_model
from mosaic.tracking.ops.convert import (
    ConvertedDatasetIndexRow,
    converted_dataset_index,
)
from mosaic.tracking.ops.train import TrainedModelIndexRow, trained_model_index

KIND = "pose"
RUN_ID = "0.1-abcdef0123"


def _new_dataset(tmp_path: Path) -> Dataset:
    """A dataset under ``tmp_path/orig``.

    Nested rather than sitting at ``tmp_path`` so the move tests below can
    relocate it to a sibling that is still inside this test's own directory.
    ``tmp_path.parent`` is shared across the whole pytest session, so two tests
    both claiming a ``moved/`` there collide on whichever runs second.
    """
    base = tmp_path / "orig"
    base.mkdir(parents=True, exist_ok=True)
    manifest = new_dataset_manifest("m", base_dir=base)
    return Dataset(manifest_path=manifest).load(ensure_roots=True)


def _move_dataset(tmp_path: Path) -> Dataset:
    """Relocate ``tmp_path/orig`` to ``tmp_path/moved`` and reload it from there."""
    origin = tmp_path / "orig"
    moved = tmp_path / "moved"
    _ = origin.rename(moved)
    return Dataset(manifest_path=moved / "dataset.yaml").load()


def _run_dir(ds: Dataset, kind: str = KIND) -> Path:
    run_dir = ds.get_root("models") / kind / RUN_ID
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def dataset_with_model_row(tmp_path: Path, *, absolute: bool) -> tuple[Dataset, Path]:
    """A dataset holding one trained-model row, with absolute or relative paths."""
    ds = _new_dataset(tmp_path)
    run_dir = _run_dir(ds)
    checkpoint = run_dir / "best.pt"
    metrics = run_dir / "metrics.json"
    _ = checkpoint.write_bytes(b"fake")
    _ = metrics.write_text("{}")

    def store(path: Path) -> str:
        return str(path) if absolute else ds.relative_to_root(path)

    index = trained_model_index(model_index_path(ds, KIND))
    index.ensure()
    index.append(
        [
            TrainedModelIndexRow(
                run_id=RUN_ID,
                kind=KIND,
                base_model="yolo11n-pose",
                base_run_id="",
                best_model_path=store(checkpoint),
                metrics_path=store(metrics),
                n_epochs=3,
                status="finished",
                abs_path=Path(ds.relative_to_root(run_dir)),
            )
        ]
    )
    return ds, checkpoint


def dataset_with_legacy_model_row(tmp_path: Path) -> Dataset:
    """One row in the pre-f7ae561 schema, written raw.

    Deliberately not built from a row class: ``ModelIndexRow`` no longer exists,
    and the thing under test is a CSV that predates its removal.
    """
    ds = _new_dataset(tmp_path)
    run_dir = _run_dir(ds, "feral")
    config = run_dir / "config.json"
    metrics = run_dir / "metrics.json"
    _ = config.write_text("{}")
    _ = metrics.write_text("{}")

    index_path = model_index_path(ds, "feral")
    index_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "abs_path": ds.relative_to_root(run_dir),
                "run_id": RUN_ID,
                "started_at": "2026-01-01T00:00:00+00:00",
                "finished_at": "",
                "model": "feral",
                "version": "0.1",
                "config_path": str(config),
                "config_hash": "abcdef0123",
                "metrics_path": str(metrics),
                "status": "success",
                "notes": "",
            }
        ]
    ).to_csv(index_path, index=False)
    return ds


def dataset_with_converted_row(tmp_path: Path) -> Dataset:
    """One ``convert-points`` row whose ``data_yaml`` is stored absolute."""
    ds = _new_dataset(tmp_path)
    run_dir = _run_dir(ds, "convert-points")
    data_yaml = run_dir / "data.yaml"
    _ = data_yaml.write_text("names: [bee]\n")

    index = converted_dataset_index(model_index_path(ds, "convert-points"))
    index.ensure()
    index.append(
        [
            ConvertedDatasetIndexRow(
                run_id=RUN_ID,
                kind="convert-points",
                source_format="cvat_points",
                data_yaml=str(data_yaml),
                class_names="bee",
                n_train=1,
                n_valid=1,
                n_test=1,
                status="finished",
                abs_path=Path(ds.relative_to_root(run_dir)),
            )
        ]
    )
    return ds


def read_row(ds: Dataset, kind: str = KIND) -> dict[str, str]:
    df = pd.read_csv(model_index_path(ds, kind))
    row = df.iloc[0]
    return {str(col): str(row[col]) for col in list(df.columns)}


def test_make_portable_converts_every_model_path_column(tmp_path: Path) -> None:
    """A legacy index full of absolutes is repaired, not just its abs_path."""
    ds, _ = dataset_with_model_row(tmp_path, absolute=True)

    before = read_row(ds)
    assert Path(before["best_model_path"]).is_absolute()
    assert Path(before["metrics_path"]).is_absolute()

    changed = ds.make_portable()

    after = read_row(ds)
    assert not Path(after["best_model_path"]).is_absolute()
    assert not Path(after["metrics_path"]).is_absolute()
    assert any("models" in key for key in changed), (
        f"the models index was not visited: {sorted(changed)}"
    )


def test_rewrite_index_paths_remaps_the_model_checkpoint(tmp_path: Path) -> None:
    """A relocated dataset remaps the checkpoint column, not only abs_path."""
    ds, checkpoint = dataset_with_model_row(tmp_path, absolute=True)
    moved_root = tmp_path / "elsewhere"

    counts = ds.rewrite_index_paths({str(tmp_path / "orig"): str(moved_root)})

    after = read_row(ds)
    assert after["best_model_path"].startswith(str(moved_root))
    assert after["metrics_path"].startswith(str(moved_root))
    assert sum(counts.values()) >= 2, f"expected both columns remapped: {counts}"
    assert checkpoint.exists(), "rewriting an index must not touch the files it names"


def test_the_legacy_schema_is_repaired_too(tmp_path: Path) -> None:
    """``config_path`` / ``metrics_path`` -- the spelling still on disk in old datasets."""
    ds = dataset_with_legacy_model_row(tmp_path)

    changed = ds.make_portable()

    after = read_row(ds, "feral")
    assert not Path(after["config_path"]).is_absolute()
    assert not Path(after["metrics_path"]).is_absolute()
    assert any("models" in key for key in changed), (
        f"the legacy models index was not visited: {sorted(changed)}"
    )


def test_the_converted_dataset_yaml_column_is_repaired(tmp_path: Path) -> None:
    """``data_yaml`` is a path column too, under a different model kind."""
    ds = dataset_with_converted_row(tmp_path)

    changed = ds.make_portable()

    after = read_row(ds, "convert-points")
    assert not Path(after["data_yaml"]).is_absolute()
    assert any("convert-points" in key for key in changed), (
        f"the converted-dataset index was not visited: {sorted(changed)}"
    )


def test_a_relative_index_resolves_after_the_dataset_moves(tmp_path: Path) -> None:
    """The point of storing relative: the row still names the right file elsewhere."""
    _ = dataset_with_model_row(tmp_path, absolute=False)

    reloaded = _move_dataset(tmp_path)
    row = read_row(reloaded)

    resolved = reloaded.resolve_path(row["best_model_path"])
    assert resolved.exists()
    assert resolved.is_relative_to(tmp_path / "moved")


def test_resolve_model_survives_a_move(tmp_path: Path) -> None:
    """End-to-end through the consumer that made this worth fixing."""
    _ = dataset_with_model_row(tmp_path, absolute=False)

    reloaded = _move_dataset(tmp_path)
    weights, run_id = resolve_model(reloaded, RUN_ID, KIND)

    assert run_id == RUN_ID
    assert weights.exists(), f"resolve_model returned a dead path: {weights}"


def test_make_portable_alone_does_not_repair_a_foreign_absolute(tmp_path: Path) -> None:
    """``make_portable`` is not a substitute for ``rewrite_index_paths``.

    ``_make_rel`` does ``p.resolve().relative_to(root)``. A path recorded on a
    *different* machine is not under this dataset's root, so ``relative_to``
    raises and the value is deliberately left absolute. Repairing a moved dataset
    is therefore two passes in order -- remap the old prefix first, relativize
    second -- and pinning that here stops a future reader assuming one is enough.
    """
    ds = _new_dataset(tmp_path)
    run_dir = _run_dir(ds)
    foreign = "/media/otherbox/T9/mosaic_datasets/m/models/pose/0.1-abcdef0123/best.pt"

    index = trained_model_index(model_index_path(ds, KIND))
    index.ensure()
    index.append(
        [
            TrainedModelIndexRow(
                run_id=RUN_ID,
                kind=KIND,
                base_model="yolo11n-pose",
                base_run_id="",
                best_model_path=foreign,
                metrics_path="",
                n_epochs=1,
                status="finished",
                abs_path=Path(ds.relative_to_root(run_dir)),
            )
        ]
    )

    assert ds.make_portable() == {}, "a foreign absolute is not relativizable"
    assert read_row(ds)["best_model_path"] == foreign

    counts = ds.rewrite_index_paths(
        {"/media/otherbox/T9/mosaic_datasets/m": str(tmp_path)}
    )
    assert any("models" in key for key in counts), (
        f"rewrite_index_paths must reach the models index: {sorted(counts)}"
    )
    assert read_row(ds)["best_model_path"].startswith(str(tmp_path))
