"""The SLEAP training op, with the trainer itself faked.

What matters here is everything around the subprocess: that the config mosaic
writes says what the parameters said, that the row records a *directory*, and
that the head is read back off the artifact rather than echoed from the request.
The last one is the point -- a row claiming a model type the directory does not
have is a row that lies about what was trained.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest
import yaml

from mosaic.core.dataset import Dataset, new_dataset_manifest
from mosaic.core.pipeline.models import model_index_path
from mosaic.core.pipeline.ops import run_op
from mosaic.tracking import register_ops
from mosaic.tracking.ops.train import trained_model_index
from mosaic.tracking.sleap import training as training_module
from mosaic.tracking.sleap.training import sleap_train_config

register_ops()


def _dataset(tmp_path: Path) -> Dataset:
    manifest = new_dataset_manifest("t", base_dir=tmp_path)
    return Dataset(manifest_path=manifest).load(ensure_roots=True)


def _fake_trainer(monkeypatch: pytest.MonkeyPatch, head: str = "centered_instance"):
    """Stand in for sleap-nn-train, writing the directory it would have."""
    seen: list[list[str]] = []

    def run(argv: Sequence[str], **kw: object) -> tuple[str, str, int]:
        seen.append(list(argv))
        config_dir = Path(argv[argv.index("--config-dir") + 1])
        config = yaml.safe_load((config_dir / "config.yaml").read_text())
        produced = config_dir / config["trainer_config"]["run_name"]
        produced.mkdir(parents=True, exist_ok=True)
        _ = (produced / "best.ckpt").write_bytes(b"weights")
        _ = (produced / "training_config.yaml").write_text(
            f"head_configs:\n  {head}: {{}}\n"
        )
        return ("done", "", 0)

    monkeypatch.setattr(training_module, "run_supervised", run)
    return seen


def _point_at_sleap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "bin").mkdir(exist_ok=True)
    _ = (tmp_path / "bin" / "sleap-nn-train").write_text("")
    monkeypatch.setenv("MOSAIC_SLEAP_BIN", str(tmp_path / "bin" / "sleap-nn-train"))


# --- the config -------------------------------------------------------------


def test_the_config_says_what_the_parameters_said() -> None:
    """Buildable without running anything, so it is assertable on its own."""
    config = sleap_train_config(
        Path("session.slp"),
        Path("/runs/r1"),
        head="centroid",
        backbone="convnext",
        max_epochs=30,
        seed=7,
        validation_fraction=0.25,
        run_name="model",
    )
    assert config["data_config"]["train_labels_path"] == ["session.slp"]
    assert config["data_config"]["validation_fraction"] == 0.25
    assert config["model_config"]["backbone_config"] == {"convnext": {}}
    assert config["model_config"]["head_configs"] == {"centroid": {"confmaps": {}}}
    assert config["trainer_config"]["max_epochs"] == 30
    assert config["trainer_config"]["seed"] == 7
    assert config["trainer_config"]["save_ckpt"] is True, "otherwise nothing is written"


def test_the_config_states_the_preprocessing_sleap_nn_reads_unmerged() -> None:
    """sleap-nn completes the config, then reads two keys off the version it did not.

    ``run_training`` passes the document through ``verify_training_cfg`` and keeps
    the completed result on ``trainer.config``, but its post-training evaluation
    reads ``ensure_rgb`` / ``ensure_grayscale`` from the raw ``config`` it was
    handed. Omitting them trains to completion, writes the checkpoint, and then
    fails the evaluation pass with ``Key 'preprocessing' is not in struct`` -- an
    error after the model is already on disk. Stated here at sleap-nn's own
    defaults, so nothing about the run changes.
    """
    config = sleap_train_config(
        Path("session.slp"),
        Path("/runs/r1"),
        head="centered_instance",
        backbone="unet",
        max_epochs=1,
        seed=1,
        validation_fraction=0.1,
        run_name="model",
    )
    assert config["data_config"]["preprocessing"] == {
        "ensure_rgb": False,
        "ensure_grayscale": False,
    }


@pytest.mark.parametrize(
    ("head", "sections"),
    [
        ("single_instance", {"confmaps"}),
        ("centroid", {"confmaps"}),
        ("centered_instance", {"confmaps"}),
        ("bottomup", {"confmaps", "pafs"}),
        ("multi_class_bottomup", {"confmaps", "class_maps"}),
        ("multi_class_topdown", {"confmaps", "class_vectors"}),
    ],
)
def test_every_head_names_its_output_sections(head: str, sections: set[str]) -> None:
    """An empty head block is not a defaulted one, and sleap-nn cannot start from it.

    sleap-nn merges this config over its own structured one, where each section
    defaults to ``None``, then walks the head's sections filling ``part_names``
    and ``edges`` from the labels. A section left ``None`` has no keys to walk, so
    ``{head: {}}`` raises ``AttributeError: 'NoneType' object has no attribute
    'keys'`` in ``model_trainer._setup_head_config`` before the first epoch.
    Naming each section instantiates it at its own defaults.
    """
    config = sleap_train_config(
        Path("session.slp"),
        Path("/runs/r1"),
        head=head,  # pyright: ignore[reportArgumentType]
        backbone="unet",
        max_epochs=1,
        seed=1,
        validation_fraction=0.1,
        run_name="model",
    )
    written = config["model_config"]["head_configs"]
    assert set(written) == {head}
    assert set(written[head]) == sections
    assert all(value == {} for value in written[head].values())  # pyright: ignore[reportAttributeAccessIssue]


# --- the op -----------------------------------------------------------------


def test_it_registers_a_directory_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reason the model reference had to stop being a single file."""
    ds = _dataset(tmp_path)
    _point_at_sleap(tmp_path, monkeypatch)
    _ = _fake_trainer(monkeypatch)
    labels = tmp_path / "session.slp"
    _ = labels.write_bytes(b"slp")

    run_id = run_op(
        ds, "train-sleap", {"labels": str(labels), "max_epochs": 3, "head": "centroid"}
    )
    assert run_id.startswith("train-sleap.")

    row = trained_model_index(model_index_path(ds, "train-sleap")).read().iloc[0]
    assert row["run_id"] == run_id
    assert row["status"] == "finished"
    assert row["artifact_shape"] == "directory"
    assert row["artifact_path"], "a directory artifact records where it is"
    assert str(row["best_model_path"]).endswith("best.ckpt")


def test_the_recorded_head_comes_from_the_artifact_not_the_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A row must describe what was produced, not what was asked for.

    The fake writes a centroid config whatever it is told, so a row echoing the
    request would say ``bottomup`` here.
    """
    ds = _dataset(tmp_path)
    _point_at_sleap(tmp_path, monkeypatch)
    _ = _fake_trainer(monkeypatch, head="centroid")
    labels = tmp_path / "session.slp"
    _ = labels.write_bytes(b"slp")

    _ = run_op(
        ds, "train-sleap", {"labels": str(labels), "head": "bottomup", "max_epochs": 1}
    )
    row = trained_model_index(model_index_path(ds, "train-sleap")).read().iloc[0]
    assert row["model_type"] == "centroid", "read back off the artifact"


def test_the_trained_model_resolves_back_as_a_sleap_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The handoff the whole branch exists for: train here, track with it there."""
    from mosaic.tracking.model_refs import resolve_model

    ds = _dataset(tmp_path)
    _point_at_sleap(tmp_path, monkeypatch)
    _ = _fake_trainer(monkeypatch)
    labels = tmp_path / "session.slp"
    _ = labels.write_bytes(b"slp")

    run_id = run_op(ds, "train-sleap", {"labels": str(labels), "max_epochs": 1})
    resolved = resolve_model(ds, run_id, "train-sleap")

    assert resolved.model_id == run_id, "named by its run, not a digest"
    assert resolved.path.is_dir(), "a directory, which is what sleap-track wants"
    assert [p.name for p in resolved.significant_files] == ["best.ckpt"]


def test_different_labels_are_a_different_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = _dataset(tmp_path)
    _point_at_sleap(tmp_path, monkeypatch)
    _ = _fake_trainer(monkeypatch)
    first = tmp_path / "a.slp"
    _ = first.write_bytes(b"one")
    second = tmp_path / "b.slp"
    _ = second.write_bytes(b"two different bytes")

    a = run_op(ds, "train-sleap", {"labels": str(first), "max_epochs": 1})
    b = run_op(ds, "train-sleap", {"labels": str(second), "max_epochs": 1})
    assert a != b


def test_absent_labels_abort_before_anything_is_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A recorded run naming labels that were never there describes nothing."""
    ds = _dataset(tmp_path)
    _point_at_sleap(tmp_path, monkeypatch)
    _ = _fake_trainer(monkeypatch)

    with pytest.raises(FileNotFoundError, match="labels file does not exist"):
        _ = run_op(ds, "train-sleap", {"labels": str(tmp_path / "nope.slp")})
