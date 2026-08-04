"""The Lightning Pose training op, with the trainer faked.

Lightning Pose is not installed on the machine this was written on, so unlike
the SLEAP side none of it has been run against the real thing. What is asserted
is the part mosaic owns: what it hands over, what it records, and that the model
type on the row comes from the artifact rather than the request.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from mosaic.core.dataset import Dataset, new_dataset_manifest
from mosaic.core.pipeline.models import model_index_path
from mosaic.core.pipeline.ops import run_op
from mosaic.tracking import register_ops
from mosaic.tracking.litpose import training as training_module
from mosaic.tracking.ops.train import trained_model_index

register_ops()


def _dataset(tmp_path: Path) -> Dataset:
    manifest = new_dataset_manifest("t", base_dir=tmp_path)
    return Dataset(manifest_path=manifest).load(ensure_roots=True)


def _base_config(tmp_path: Path) -> Path:
    """Stand in for Lightning Pose's own ``config_default.yaml``.

    Supplied by the caller rather than generated, because Lightning Pose merges
    no defaults of its own and ships no template.
    """
    base = tmp_path / "litpose_default.yaml"
    _ = base.write_text("training:\n  num_gpus: 0\n")
    return base


def _project(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    project.mkdir()
    _ = (project / "config.yaml").write_text("data:\n  num_keypoints: 3\n")
    _ = (project / "CollectedData.csv").write_text("scorer\nbodyparts\ncoords\n")
    return project


def _fake_trainer(
    monkeypatch: pytest.MonkeyPatch, model_type: str = "heatmap"
) -> list[list[str]]:
    seen: list[list[str]] = []

    def run(argv: Sequence[str], **kw: object) -> tuple[str, str, int]:
        seen.append(list(argv))
        out = Path(argv[argv.index("-c") + 4])
        checkpoints = out / "tb_logs" / "run" / "version_0" / "checkpoints"
        checkpoints.mkdir(parents=True, exist_ok=True)
        _ = (out / "config.yaml").write_text(f"model:\n  model_type: {model_type}\n")
        _ = (checkpoints / "best.ckpt").write_bytes(b"weights")
        return ("done", "", 0)

    monkeypatch.setattr(training_module, "run_supervised", run)
    return seen


def _point_at_litpose(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "bin").mkdir(exist_ok=True)
    for name in ("litpose", "python"):
        _ = (tmp_path / "bin" / name).write_text("")
    monkeypatch.setenv("MOSAIC_LITPOSE_BIN", str(tmp_path / "bin" / "litpose"))


def test_it_registers_a_directory_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = _dataset(tmp_path)
    _point_at_litpose(tmp_path, monkeypatch)
    _ = _fake_trainer(monkeypatch)

    run_id = run_op(
        ds,
        "train-litpose",
        {
            "project": str(_project(tmp_path)),
            "base_config": str(_base_config(tmp_path)),
            "max_epochs": 2,
        },
    )
    assert run_id.startswith("train-litpose.")

    row = trained_model_index(model_index_path(ds, "train-litpose")).read().iloc[0]
    assert row["status"] == "finished"
    assert row["artifact_shape"] == "directory"
    assert str(row["best_model_path"]).endswith("best.ckpt")


def test_the_recorded_model_type_comes_from_the_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fake writes heatmap_mhcrnn whatever it is asked for."""
    ds = _dataset(tmp_path)
    _point_at_litpose(tmp_path, monkeypatch)
    _ = _fake_trainer(monkeypatch, model_type="heatmap_mhcrnn")

    _ = run_op(
        ds,
        "train-litpose",
        {
            "project": str(_project(tmp_path)),
            "base_config": str(_base_config(tmp_path)),
            "model_type": "regression",
        },
    )
    row = trained_model_index(model_index_path(ds, "train-litpose")).read().iloc[0]
    assert row["model_type"] == "heatmap_mhcrnn", "read back, not echoed"


def test_the_head_and_backbone_reach_the_trainer_as_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Lightning Pose is configured by Hydra assignment, so these must be argv."""
    ds = _dataset(tmp_path)
    _point_at_litpose(tmp_path, monkeypatch)
    seen = _fake_trainer(monkeypatch)

    _ = run_op(
        ds,
        "train-litpose",
        {
            "project": str(_project(tmp_path)),
            "base_config": str(_base_config(tmp_path)),
            "model_type": "heatmap_mhcrnn",
            "backbone": "vitb_sam",
            "max_epochs": 5,
        },
    )
    argv = seen[0]
    assert "model.model_type=heatmap_mhcrnn" in argv
    assert "model.backbone=vitb_sam" in argv
    assert "training.max_epochs=5" in argv


def test_the_trained_model_resolves_back_as_a_litpose_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mosaic.tracking.model_refs import resolve_model

    ds = _dataset(tmp_path)
    _point_at_litpose(tmp_path, monkeypatch)
    _ = _fake_trainer(monkeypatch)

    run_id = run_op(
        ds,
        "train-litpose",
        {
            "project": str(_project(tmp_path)),
            "base_config": str(_base_config(tmp_path)),
        },
    )
    resolved = resolve_model(ds, run_id, "train-litpose")

    assert resolved.model_id == run_id
    assert resolved.path.is_dir()
    assert [p.name for p in resolved.significant_files] == ["config.yaml", "best.ckpt"]


def test_a_directory_that_is_not_a_project_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No config.yaml means Lightning Pose has nothing to read."""
    ds = _dataset(tmp_path)
    _point_at_litpose(tmp_path, monkeypatch)
    _ = _fake_trainer(monkeypatch)
    bare = tmp_path / "bare"
    bare.mkdir()

    with pytest.raises(FileNotFoundError, match="no config.yaml"):
        _ = run_op(
            ds,
            "train-litpose",
            {"project": str(bare), "base_config": str(_base_config(tmp_path))},
        )
