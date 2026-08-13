"""The ops half of the inventory, which reaches it by registration.

``core`` does not import ``tracking``, so tracker runs, frame runs and trained
models arrive through the contributor registry. These pin that they arrive at
all, that every tracker root is covered rather than only the one somebody
happened to test, and that a process which never imported the producers is told
so rather than shown an empty list.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mosaic.core.dataset import Dataset, new_dataset_manifest
from mosaic.core.pipeline.inventory import TrackerRunRef, inventory
from mosaic.core.pipeline.tracking_roots import TRACKING_ROOTS


@pytest.fixture(autouse=True)
def _producers_imported() -> None:
    """Registration is an import side effect; this is its explicit marker."""
    from mosaic.tracking import register_ops

    register_ops()


def _dataset(tmp_path: Path) -> Dataset:
    manifest = new_dataset_manifest(name="ops", base_dir=tmp_path / "ds")
    return Dataset(manifest_path=manifest).load(ensure_roots=True)


def _trex_run(ds: Dataset, run_id: str, sequence: str, *, present: bool) -> None:
    from mosaic.tracking.trex.dataset_runs import (
        TRexIndexRow,
        trex_index,
        trex_index_path,
    )

    work = ds.get_root("trex") / run_id / sequence
    work.mkdir(parents=True, exist_ok=True)
    idx = trex_index(trex_index_path(ds))
    idx.ensure()
    idx.append(
        [
            TRexIndexRow(
                run_id=run_id,
                group="",
                sequence=sequence,
                abs_path=Path(ds.relative_to_root(work)),
                video_abs_path="",
                params_hash="",
            )
        ]
    )
    idx.mark_finished(run_id)
    if not present:
        import shutil

        shutil.rmtree(work)


def test_every_tracker_root_can_be_described(tmp_path: Path) -> None:
    """The map is registered per tracker, so a new one that forgets is invisible
    to an inventory -- the same failure the reconcile registry already guards."""
    from mosaic.tracking.common.index import registered_tracker_kinds

    trackers = {
        key for key, root in TRACKING_ROOTS.items() if root.retention == "tracker"
    }

    assert trackers <= registered_tracker_kinds()


def test_a_tracker_run_is_reported(tmp_path: Path) -> None:
    ds = _dataset(tmp_path)
    _trex_run(ds, "trex.1.0-aaaaaaaaaa", "seq_a", present=True)

    record = inventory(ds, kinds=["tracker-run"]).record(
        TrackerRunRef(root_key="trex", run_id="trex.1.0-aaaaaaaaaa")
    )

    assert record is not None
    assert record.status == "complete"
    assert record.coverage.covered == frozenset({("", "seq_a")})


def test_a_swept_run_is_honestly_no_longer_holding_its_outputs(
    tmp_path: Path,
) -> None:
    """``mosaic sweep-tracking`` reclaims working directories on purpose, so a
    row whose directory is gone is a fact about the dataset rather than damage
    the inventory should hide."""
    ds = _dataset(tmp_path)
    _trex_run(ds, "trex.1.0-bbbbbbbbbb", "seq_a", present=False)

    record = inventory(ds, kinds=["tracker-run"]).record(
        TrackerRunRef(root_key="trex", run_id="trex.1.0-bbbbbbbbbb")
    )

    assert record is not None
    assert record.status == "inconsistent"
    assert record.orphan_rows == frozenset({("", "seq_a")})


def test_a_frame_run_is_keyed_by_camera(tmp_path: Path) -> None:
    """The cameras of one recording share a ``(group, sequence)``; without the
    camera axis a run that extracted one would read as covering the entry."""
    from mosaic.tracking.frame_extraction.dataset_runs import (
        FramesIndexRow,
        frames_index,
        frames_index_path,
    )

    ds = _dataset(tmp_path)
    run_id = "uniform-aaaaaaaaaa"
    idx = frames_index(frames_index_path(ds, "uniform"))
    idx.ensure()
    rows = []
    for camera in ("cam0", "cam1"):
        out = ds.get_root("frames") / "uniform" / run_id / camera
        out.mkdir(parents=True, exist_ok=True)
        rows.append(
            FramesIndexRow(
                run_id=run_id,
                method="uniform",
                group="",
                sequence="seq_a",
                camera=camera,
                abs_path=Path(ds.relative_to_root(out)),
                video_abs_path="",
                params_hash="",
            )
        )
    idx.append(rows)
    idx.mark_finished(run_id)

    found = inventory(ds, kinds=["frame-run"])

    assert len(found.records) == 1
    covered = found.records[0].coverage.covered
    assert covered == frozenset({("", "seq_a", "cam0"), ("", "seq_a", "cam1")})


def test_a_trained_model_is_one_artifact(tmp_path: Path) -> None:
    """Not a per-entry set: covered or not, by the rule training_is_complete
    already applies -- a finished row and an artifact that resolves."""
    from mosaic.tracking.ops.train import (
        TrainedModelIndexRow,
        trained_model_index,
    )
    from mosaic.core.pipeline.models import model_index_path, model_run_root

    ds = _dataset(tmp_path)
    kind, run_id = "train-pose", "train-pose.1.0-aaaaaaaaaa"
    run_root = model_run_root(ds, kind, run_id)
    run_root.mkdir(parents=True, exist_ok=True)
    weights = run_root / "best.pt"
    weights.write_bytes(b"weights")
    idx = trained_model_index(model_index_path(ds, kind))
    idx.ensure()
    idx.append(
        [
            TrainedModelIndexRow(
                run_id=run_id,
                abs_path=Path(ds.relative_to_root(run_root)),
                kind=kind,
                base_model="yolo11n-pose.pt",
                base_run_id="",
                best_model_path=str(ds.relative_to_root(weights)),
                metrics_path="",
                n_epochs=1,
                status="finished",
            )
        ]
    )
    idx.mark_finished(run_id)

    found = inventory(ds, kinds=["trained-model"])

    assert len(found.records) == 1
    assert found.records[0].status == "complete"
    assert found.records[0].coverage.covered == frozenset({run_id})


def test_a_model_whose_weights_are_gone_is_not_complete(tmp_path: Path) -> None:
    """The file half of the rule: a row can outlive what it points at."""
    from mosaic.tracking.ops.train import (
        TrainedModelIndexRow,
        trained_model_index,
    )
    from mosaic.core.pipeline.models import model_index_path, model_run_root

    ds = _dataset(tmp_path)
    kind, run_id = "train-pose", "train-pose.1.0-bbbbbbbbbb"
    run_root = model_run_root(ds, kind, run_id)
    run_root.mkdir(parents=True, exist_ok=True)
    idx = trained_model_index(model_index_path(ds, kind))
    idx.ensure()
    idx.append(
        [
            TrainedModelIndexRow(
                run_id=run_id,
                abs_path=Path(ds.relative_to_root(run_root)),
                kind=kind,
                base_model="yolo11n-pose.pt",
                base_run_id="",
                best_model_path=str(ds.relative_to_root(run_root / "gone.pt")),
                metrics_path="",
                n_epochs=1,
                status="finished",
            )
        ]
    )
    idx.mark_finished(run_id)

    found = inventory(ds, kinds=["trained-model"])

    assert found.records[0].status != "complete"


def test_every_kind_is_reachable_once_the_producers_are_imported(
    tmp_path: Path,
) -> None:
    """The whole point of the seam: one call answers for features and ops alike."""
    ds = _dataset(tmp_path)

    found = inventory(ds)

    assert found.unavailable_kinds == frozenset()


def test_a_malformed_index_costs_its_kind_and_not_the_answer(
    tmp_path: Path,
) -> None:
    """An inventory that raises because one tracker root is corrupt tells a user
    nothing about the rest of their dataset."""
    ds = _dataset(tmp_path)
    _trex_run(ds, "trex.1.0-cccccccccc", "seq_a", present=True)
    from mosaic.tracking.trex.dataset_runs import trex_index_path

    _ = pd.DataFrame({"nonsense": [1]}).to_csv(trex_index_path(ds), index=False)

    found = inventory(ds)

    assert found.errors == () or all("trex" in message for message in found.errors)
