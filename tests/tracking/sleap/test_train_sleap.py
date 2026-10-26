"""The SLEAP training op, with the trainer itself faked.

What matters here is everything around the subprocess: that the config mosaic
writes says what the parameters said, that the row records a *directory*, and
that the head is read back off the artifact rather than echoed from the request.
The last one is the point -- a row claiming a model type the directory does not
have is a row that lies about what was trained.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from mosaic.core.dataset import Dataset
from mosaic.core.narrowing import is_str_mapping
from mosaic.core.pipeline._utils import hash_params
from mosaic.core.pipeline.job import Cancelled
from mosaic.core.pipeline.models import model_index_path, model_run_root
from mosaic.core.pipeline.ops import OPS, run_op
from mosaic.core.pipeline.run_log import read_runs, run_log_dir
from mosaic.core.pipeline.subprocess_util import ProcessCancelled
from mosaic.tracking import register_ops
from mosaic.tracking.ops._common import fingerprint_dataset
from mosaic.tracking.ops.train import TRAINING_PROVENANCE_FILENAME, train_run_id
from mosaic.tracking.ops.trained_models import trained_model_index
from mosaic.tracking.ops.train_sleap import TrainSleapParams, fingerprint_sleap_labels
from mosaic.tracking.sleap import probe as probe_module
from mosaic.tracking.sleap.probe import (
    SleapProbeResponse,
    require_identity_labels,
    require_sleap_nn,
)
from mosaic.tracking.sleap.run import SleapNotFoundError
from mosaic.tracking.sleap import training as training_module
from mosaic.tracking.sleap.training import sleap_device_overrides, sleap_train_config

from tests.helpers import dotted_values, make_dataset, write_sleap_model

register_ops()


def _fake_trainer(
    monkeypatch: pytest.MonkeyPatch,
    head: str = "centered_instance",
    writes_training_log: bool = False,
):
    """Stand in for ``sleap-nn train``, writing the directory it would have.

    *writes_training_log* says whether the run got far enough to log an epoch.
    A real run that did not leaves no ``training_log.csv``, which is the state
    the empty ``metrics_path`` cell records.
    """
    seen: list[list[str]] = []

    def run(argv: Sequence[str], **kw: object) -> tuple[str, str, int]:
        seen.append(list(argv))
        config_dir = Path(argv[argv.index("--config-dir") + 1])
        config = yaml.safe_load((config_dir / "config.yaml").read_text())
        produced = config_dir / config["trainer_config"]["run_name"]
        _ = write_sleap_model(produced, head=head)
        if writes_training_log:
            _ = (produced / "training_log.csv").write_text("epoch,train_loss\n0,1.0\n")
        return ("done", "", 0)

    monkeypatch.setattr(training_module, "run_supervised", run)
    return seen


def _point_at_sleap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "bin").mkdir(exist_ok=True)
    for name in ("sleap-nn", "sleap-convert", "python"):
        _ = (tmp_path / "bin" / name).write_text("")
    monkeypatch.setenv("MOSAIC_SLEAP_BIN", str(tmp_path / "bin" / "sleap-convert"))


def _fake_probe(
    monkeypatch: pytest.MonkeyPatch,
    *,
    has_sleap_nn: bool = True,
    n_tracks: int = 2,
    labels_load_error: str = "",
) -> list[list[str]]:
    """Stand in for the probe subprocess, writing the answer it would have."""
    seen: list[list[str]] = []
    found = {
        "has_sleap_nn": has_sleap_nn,
        "sleap_nn_version": "0.3.1" if has_sleap_nn else "",
        "sleap_nn_import_error": (
            "" if has_sleap_nn else "ModuleNotFoundError: No module named 'sleap_nn'"
        ),
        "sleap_io_version": "0.9.2",
        "n_labeled_frames": 12,
        "n_tracks": n_tracks,
        "labels_load_error": labels_load_error,
    }

    def run(argv: Sequence[str], **kw: object) -> tuple[str, str, int]:
        seen.append(list(argv))
        _ = Path(argv[argv.index("-c") + 3]).write_text(json.dumps(found))
        return ("probed", "", 0)

    monkeypatch.setattr(probe_module, "run_supervised", run)
    return seen


@pytest.fixture(autouse=True)
def _an_environment_that_can_train(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test here runs against a SLEAP install that holds sleap-nn.

    The probe is a real subprocess launch, so without a default every op test
    would try to run the empty stub ``_point_at_sleap`` writes. A test about
    what the probe found re-patches with ``_fake_probe``.
    """
    _ = _fake_probe(monkeypatch)


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
        validation_labels=None,
        base_weights=None,
        run_name="model",
    )
    assert config["data_config"]["train_labels_path"] == ["session.slp"]
    assert config["data_config"].get("validation_fraction") == 0.25
    assert "val_labels_path" not in config["data_config"]
    assert config["model_config"]["backbone_config"] == {"convnext": {}}
    assert config["model_config"]["head_configs"] == {"centroid": {"confmaps": {}}}
    assert "pretrained_backbone_weights" not in config["model_config"]
    assert "pretrained_head_weights" not in config["model_config"]
    assert config["trainer_config"]["max_epochs"] == 30
    assert config["trainer_config"]["seed"] == 7
    assert config["trainer_config"]["save_ckpt"] is True, "otherwise nothing is written"


def test_base_weights_initialize_the_backbone_and_the_head() -> None:
    """sleap-nn's fine-tune mapping, which SLEAP's GUI and --base_checkpoint set."""
    config = sleap_train_config(
        Path("labels.slp"),
        Path("/runs/r1"),
        head="centroid",
        backbone="unet",
        max_epochs=30,
        seed=7,
        validation_fraction=0.25,
        validation_labels=None,
        base_weights=Path("/models/base/best.ckpt"),
        run_name="model",
    )
    model_config = config["model_config"]
    assert model_config.get("pretrained_backbone_weights") == "/models/base/best.ckpt"
    assert model_config.get("pretrained_head_weights") == "/models/base/best.ckpt"
    assert "resume_ckpt_path" not in config["trainer_config"]


def test_a_validation_file_takes_the_place_of_the_validation_fraction() -> None:
    """sleap-nn validates on the file, so no fraction of the labels is held out."""
    config = sleap_train_config(
        Path("labels.slp"),
        Path("/runs/r1"),
        head="centroid",
        backbone="unet",
        max_epochs=30,
        seed=7,
        validation_fraction=0.25,
        validation_labels=Path("labels.val.slp"),
        base_weights=None,
        run_name="model",
    )
    assert config["data_config"]["train_labels_path"] == ["labels.slp"]
    assert config["data_config"].get("val_labels_path") == ["labels.val.slp"]
    assert "validation_fraction" not in config["data_config"]


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
        validation_labels=None,
        base_weights=None,
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
        validation_labels=None,
        base_weights=None,
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
    ds = make_dataset(tmp_path, save=False)
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
    ds = make_dataset(tmp_path, save=False)
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

    ds = make_dataset(tmp_path, save=False)
    _point_at_sleap(tmp_path, monkeypatch)
    _ = _fake_trainer(monkeypatch)
    labels = tmp_path / "session.slp"
    _ = labels.write_bytes(b"slp")

    run_id = run_op(ds, "train-sleap", {"labels": str(labels), "max_epochs": 1})
    resolved = resolve_model(ds, run_id, "train-sleap")

    assert resolved.model_id == run_id, "named by its run, not a digest"
    assert resolved.path.is_dir(), "a directory, which is what sleap-nn track wants"
    assert [p.name for p in resolved.significant_files] == ["best.ckpt"]


def test_different_labels_are_a_different_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = make_dataset(tmp_path, save=False)
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
    ds = make_dataset(tmp_path, save=False)
    _point_at_sleap(tmp_path, monkeypatch)
    _ = _fake_trainer(monkeypatch)

    with pytest.raises(FileNotFoundError, match="labels file does not exist"):
        _ = run_op(ds, "train-sleap", {"labels": str(tmp_path / "nope.slp")})


def test_a_finished_run_is_reused_unless_overwrite_says_otherwise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reuse gate reads the ``overwrite`` argument, where it read a field.

    Both directions. An op ignoring the argument reuses forever and an op whose
    gate lost its completion half retrains forever, and one direction alone
    cannot tell those apart.
    """
    ds = make_dataset(tmp_path / "ds", save=False)
    _point_at_sleap(tmp_path, monkeypatch)
    seen = _fake_trainer(monkeypatch)
    # At the dataset root, beside `models/` and the run-logs, where the first
    # run's output used to move the second run's identity.
    labels = ds.base_dir / "session.slp"
    _ = labels.write_bytes(b"slp")
    params = {"labels": str(labels), "max_epochs": 1}

    first = run_op(ds, "train-sleap", dict(params))
    assert len(seen) == 1

    assert run_op(ds, "train-sleap", dict(params)) == first
    assert len(seen) == 1, "a finished run must not train again"

    assert run_op(ds, "train-sleap", dict(params), overwrite=True) == first
    assert len(seen) == 2, "overwrite must reach the gate"


def test_a_killed_trainer_is_recorded_as_cancelled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A trainer killed on a cancel is a cancelled attempt, not a failed one."""
    ds = make_dataset(tmp_path / "ds", save=False)
    _point_at_sleap(tmp_path, monkeypatch)

    def killed(argv: Sequence[str], **kw: object) -> tuple[str, str, int]:
        raise ProcessCancelled(argv)

    monkeypatch.setattr(training_module, "run_supervised", killed)
    labels = ds.base_dir / "session.slp"
    _ = labels.write_bytes(b"slp")

    with pytest.raises(Cancelled):
        _ = run_op(ds, "train-sleap", {"labels": str(labels), "max_epochs": 1})

    (run,) = read_runs(run_log_dir(ds.base_dir), kind="train-sleap")
    assert run["status"] == "cancelled"
    assert run["error_json"] == ""


# --- overrides and the device ----------------------------------------------


def test_it_runs_the_train_subcommand_of_sleap_nn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """sleap-nn 0.3.3 removed ``sleap-nn-train``; ``sleap-nn train`` takes its flags.

    The subcommand is present in 0.3.1 as well, so running it does not drop the
    older environments.
    """
    argv = _run_and_capture(tmp_path, monkeypatch, {"head": "centroid"})
    assert Path(argv[0]).name == "sleap-nn"
    assert argv[1:3] == ["train", "--config-dir"]
    assert argv[argv.index("--config-name") + 1] == "config"


def _assignments(argv: Sequence[str]) -> list[str]:
    """The Hydra assignments in *argv*, after ``--config-dir`` and its name."""
    return list(argv[list(argv).index("--config-name") + 2 :])


def _labels_and_fake_trainer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    head: str = "centered_instance",
    writes_training_log: bool = False,
) -> tuple[Dataset, Path, list[list[str]]]:
    """Make a dataset, labels outside it and a faked trainer of *head*.

    Returns:
        The dataset, the labels, and the list that the trainer appends each
        argv to.
    """
    ds = make_dataset(tmp_path / "ds", save=False)
    _point_at_sleap(tmp_path, monkeypatch)
    seen = _fake_trainer(
        monkeypatch, head=head, writes_training_log=writes_training_log
    )
    labels = tmp_path / "labels" / "session.slp"
    labels.parent.mkdir()
    _ = labels.write_bytes(b"slp")
    return ds, labels, seen


def _run_and_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, params: dict[str, object]
) -> list[str]:
    """Run ``train-sleap`` once and return the argv the trainer was given."""
    ds, labels, seen = _labels_and_fake_trainer(tmp_path, monkeypatch)
    _ = run_op(ds, "train-sleap", {"labels": str(labels), "max_epochs": 1, **params})
    return seen[0]


def _written_config(argv: Sequence[str]) -> Mapping[str, object]:
    """The document the trainer was handed."""
    config_dir = Path(argv[argv.index("--config-dir") + 1])
    written: object = yaml.safe_load((config_dir / "config.yaml").read_text())
    assert is_str_mapping(written)
    return written


def _written_section(argv: Sequence[str], section: str) -> Mapping[str, object]:
    """One section of the document the trainer was handed."""
    written = _written_config(argv)[section]
    assert is_str_mapping(written)
    return written


def _train_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, head: str = "centered_instance"
) -> tuple[Dataset, Path, str, list[list[str]]]:
    """Train a *head* base, and return what a run fine-tuning from it needs.

    Returns:
        The dataset, the labels, the base's run id, and the trainer's argv list
        with the base's own call cleared from it.
    """
    ds, labels, seen = _labels_and_fake_trainer(tmp_path, monkeypatch, head=head)
    base = run_op(
        ds, "train-sleap", {"labels": str(labels), "max_epochs": 1, "head": head}
    )
    seen.clear()
    return ds, labels, base, seen


def test_an_override_the_config_carries_is_assigned_bare(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hydra's bare form is right for a key the written document holds."""
    argv = _run_and_capture(
        tmp_path, monkeypatch, {"sleap_overrides": {"trainer_config.seed": 7}}
    )
    assert "trainer_config.seed=7" in _assignments(argv)


def test_an_override_the_config_does_not_carry_is_appended(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Matching input resolution across models needs exactly this.

    ``data_config.preprocessing`` is written and ``...scale`` is not, so the
    append form is decided per key rather than per section.
    """
    argv = _run_and_capture(
        tmp_path,
        monkeypatch,
        {"sleap_overrides": {"data_config.preprocessing.scale": 0.5}},
    )
    assert "+data_config.preprocessing.scale=0.5" in _assignments(argv)


def test_an_explicit_hydra_form_is_passed_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A caller who spelled the form has already answered the question."""
    argv = _run_and_capture(
        tmp_path,
        monkeypatch,
        {"sleap_overrides": {"+trainer_config.profiler": "simple"}},
    )
    assert "+trainer_config.profiler=simple" in _assignments(argv)
    assert "++trainer_config.profiler=simple" not in _assignments(argv)


def test_every_injected_override_composes_against_the_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The guard on everything the op injects, whatever it grows next.

    Hydra refuses a bare assignment to a key the composed config does not
    already hold, so an injected override naming a key ``sleap_train_config``
    does not write kills the run before its first epoch. The written keys are
    read back off the document on disk rather than listed here, because a test
    that hard-codes them cannot notice the next key someone injects -- and
    flattened by a walk of its own rather than by ``_declared_keys``, so a walk
    that is wrong renders wrongly and reads wrongly, instead of agreeing with
    itself.

    The run under test names a base model, a device and an override. Its base
    adds declared keys to the document and injects no assignment. Its device
    injects keys that the document does not carry.
    """
    ds, labels, base, seen = _train_base(tmp_path, monkeypatch)
    _ = run_op(
        ds,
        "train-sleap",
        {
            "labels": str(labels),
            "max_epochs": 1,
            "base_model": base,
            "device": "0",
            "sleap_overrides": {"data_config.preprocessing.crop_size": 128},
        },
    )

    argv = seen[0]
    declared = dotted_values(_written_config(argv))
    assignments = _assignments(argv)
    assert assignments, "the run under test must inject something"
    for assignment in assignments:
        key = assignment.split("=", 1)[0]
        assert key.startswith(("+", "~")) or key in declared, assignment


def test_a_base_model_initializes_the_backbone_and_the_head(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fine-tune starts a new run from the base's checkpoint.

    Both pretrained keys name the checkpoint. The base's run is not resumed,
    which would continue its epoch count and optimizer state.
    """
    ds, labels, base, seen = _train_base(tmp_path, monkeypatch)
    _ = run_op(
        ds, "train-sleap", {"labels": str(labels), "max_epochs": 1, "base_model": base}
    )

    checkpoint = str(model_run_root(ds, "train-sleap", base) / "model" / "best.ckpt")
    model_config = _written_section(seen[0], "model_config")
    assert model_config["pretrained_backbone_weights"] == checkpoint
    assert model_config["pretrained_head_weights"] == checkpoint
    written = dotted_values(_written_config(seen[0]))
    assert not [key for key in written if key.endswith("resume_ckpt_path")]
    assert not [arg for arg in _assignments(seen[0]) if "resume_ckpt_path" in arg]


@pytest.mark.parametrize("weights_name", ["best.ckpt", "best_model.h5"])
def test_a_base_with_no_readable_head_is_not_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, weights_name: str
) -> None:
    """A base whose config does not read reports no head and is not refused.

    Its weights are the checkpoint its directory holds, ``best_model.h5`` for a
    classic SLEAP model.
    """
    base = tmp_path / "base"
    base.mkdir()
    weights = base / weights_name
    _ = weights.write_bytes(b"weights")

    argv = _run_and_capture(
        tmp_path, monkeypatch, {"head": "centroid", "base_model": str(base)}
    )

    model_config = _written_section(argv, "model_config")
    assert model_config["pretrained_backbone_weights"] == str(weights)
    assert model_config["pretrained_head_weights"] == str(weights)


def test_a_base_of_another_head_is_refused_before_the_run_root_is_made(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A centroid base cannot initialize a centered_instance network.

    The refusal leaves no run root, no index row and no trainer call.
    """
    ds, labels, base, seen = _train_base(tmp_path, monkeypatch, head="centroid")

    with pytest.raises(ValueError, match="centroid model.*centered_instance run"):
        _ = run_op(
            ds,
            "train-sleap",
            {
                "labels": str(labels),
                "max_epochs": 1,
                "head": "centered_instance",
                "base_model": base,
            },
        )

    assert seen == [], "the trainer must not run"
    index_path = model_index_path(ds, "train-sleap")
    run_roots = sorted(
        path.name for path in index_path.parent.iterdir() if path.is_dir()
    )
    assert run_roots == [base]
    assert list(trained_model_index(index_path).read()["run_id"]) == [base]


def test_an_override_of_the_pretrained_head_weights_beats_the_base_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The document declares the key, and Hydra applies the bare assignment over it."""
    ds, labels, base, seen = _train_base(tmp_path, monkeypatch)
    _ = run_op(
        ds,
        "train-sleap",
        {
            "labels": str(labels),
            "max_epochs": 1,
            "base_model": base,
            "sleap_overrides": {
                "model_config.pretrained_head_weights": "/models/other/best.ckpt"
            },
        },
    )

    assignments = _assignments(seen[0])
    assert "model_config.pretrained_head_weights=/models/other/best.ckpt" in assignments
    checkpoint = str(model_run_root(ds, "train-sleap", base) / "model" / "best.ckpt")
    model_config = _written_section(seen[0], "model_config")
    assert model_config["pretrained_backbone_weights"] == checkpoint


@pytest.mark.parametrize(
    ("device", "expected"),
    [
        ("auto", {}),
        ("", {}),
        ("cpu", {"trainer_config.trainer_accelerator": "cpu"}),
        ("mps", {"trainer_config.trainer_accelerator": "mps"}),
        ("gpu", {"trainer_config.trainer_accelerator": "gpu"}),
        (
            "0",
            {
                "trainer_config.trainer_accelerator": "gpu",
                "trainer_config.trainer_device_indices": [0],
            },
        ),
        (
            "0,1",
            {
                "trainer_config.trainer_accelerator": "gpu",
                "trainer_config.trainer_device_indices": [0, 1],
            },
        ),
    ],
)
def test_the_device_translation(device: str, expected: dict[str, object]) -> None:
    """sleap-nn names the family and the indices separately; mosaic names one.

    ``trainer_accelerator`` alone cannot select GPU 1, and handed ``"0"`` it
    would set the family to the string ``"0"``.
    """
    assert sleap_device_overrides(device) == expected


@pytest.mark.parametrize("device", ["cuda:0", "CPU", "gpu:1", "0,x"])
def test_an_unusable_device_is_refused_at_submit(device: str) -> None:
    """A 422 the caller can act on, rather than a Hydra error on a GPU node."""
    with pytest.raises(ValidationError, match="unusable device"):
        _ = TrainSleapParams(labels="session.slp", device=device)


def test_an_auto_device_injects_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The default leaves the choice to sleap-nn, which is what auto means."""
    argv = _run_and_capture(tmp_path, monkeypatch, {"device": "auto"})
    assert not [arg for arg in _assignments(argv) if "trainer_accelerator" in arg]


def test_a_device_index_names_the_family_and_the_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both keys reach the trainer, appended because neither is written."""
    argv = _run_and_capture(tmp_path, monkeypatch, {"device": "1"})
    assignments = _assignments(argv)
    assert "+trainer_config.trainer_accelerator=gpu" in assignments
    assert "+trainer_config.trainer_device_indices=[1]" in assignments


def test_an_override_beats_the_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What ``sleap_overrides``' declaration promises."""
    argv = _run_and_capture(
        tmp_path,
        monkeypatch,
        {
            "device": "cpu",
            "sleap_overrides": {"trainer_config.trainer_accelerator": "mps"},
        },
    )
    assignments = _assignments(argv)
    assert "+trainer_config.trainer_accelerator=mps" in assignments
    assert "+trainer_config.trainer_accelerator=cpu" not in assignments


# --- what the environment and the labels hold -------------------------------


def test_an_environment_without_the_nn_extra_is_refused() -> None:
    """A ``sleap`` install and a ``sleap[nn]`` one look identical on $PATH.

    SLEAP's console scripts belong to the ``sleap`` distribution itself, so
    the location ladder finds them either way and the run dies inside
    the subprocess. Decided from what the probe reported, so this needs no
    SLEAP at all.
    """
    probe = SleapProbeResponse(
        has_sleap_nn=False,
        sleap_nn_import_error="ModuleNotFoundError: No module named 'sleap_nn'",
    )
    with pytest.raises(SleapNotFoundError, match=r"sleap\[nn\]"):
        require_sleap_nn(probe)


def test_an_environment_that_can_train_is_not_refused() -> None:
    """The control: a refusal that fires on everything refuses nothing."""
    require_sleap_nn(SleapProbeResponse(has_sleap_nn=True, sleap_nn_version="0.3.1"))


@pytest.mark.parametrize("head", ["multi_class_bottomup", "multi_class_topdown"])
def test_an_identity_head_refuses_labels_with_no_tracks(head: str) -> None:
    """Training one of these against no classes succeeds and learns nothing."""
    probe = SleapProbeResponse(has_sleap_nn=True, n_labeled_frames=12, n_tracks=0)
    with pytest.raises(ValueError, match="carries a track"):
        require_identity_labels(probe, head, "session.slp")


@pytest.mark.parametrize(
    "head", ["single_instance", "centroid", "centered_instance", "bottomup"]
)
def test_a_head_that_classifies_nothing_needs_no_tracks(head: str) -> None:
    """Only the two identity heads have a requirement here to enforce."""
    probe = SleapProbeResponse(has_sleap_nn=True, n_labeled_frames=12, n_tracks=0)
    require_identity_labels(probe, head, "session.slp")


def test_unreadable_labels_are_refused_before_the_head_is_considered() -> None:
    """A file that did not load reports no tracks, which is not the same thing."""
    probe = SleapProbeResponse(
        has_sleap_nn=True, labels_load_error="OSError: unable to open file"
    )
    with pytest.raises(ValueError, match="would not load"):
        require_identity_labels(probe, "centered_instance", "session.slp")


def test_an_identity_head_with_no_tracks_never_reaches_the_trainer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole point is not paying for the run.

    Nothing may be claimed either: a refusal that left an inflight marker
    behind would hold the run root for the next half hour over a run that was
    never going to start.
    """
    ds, labels, seen = _labels_and_fake_trainer(tmp_path, monkeypatch)
    _ = _fake_probe(monkeypatch, n_tracks=0)

    with pytest.raises(ValueError, match="carries a track"):
        _ = run_op(
            ds,
            "train-sleap",
            {
                "labels": str(labels),
                "head": "multi_class_topdown",
                "max_epochs": 1,
            },
        )

    assert seen == [], "the trainer must not run"
    assert not list((ds.base_dir / "models").rglob(".mosaic-inflight.json"))


def test_an_identity_head_with_tracked_labels_trains(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The mirror, so the refusal is not simply refusing everything."""
    ds, labels, seen = _labels_and_fake_trainer(tmp_path, monkeypatch)
    _ = _fake_probe(monkeypatch, n_tracks=4)

    _ = run_op(
        ds,
        "train-sleap",
        {"labels": str(labels), "head": "multi_class_bottomup", "max_epochs": 1},
    )
    assert len(seen) == 1


def test_a_finished_run_pays_for_no_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Probed after the reuse gate, so a cache hit costs no cold import."""
    ds, labels, _ = _labels_and_fake_trainer(tmp_path, monkeypatch)
    probed = _fake_probe(monkeypatch)
    params = {"labels": str(labels), "max_epochs": 1}

    first = run_op(ds, "train-sleap", dict(params))
    assert len(probed) == 1

    assert run_op(ds, "train-sleap", dict(params)) == first
    assert len(probed) == 1, "a reused run must not probe again"


def test_the_probe_is_handed_the_labels_it_reports_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both questions are answered from one launch, so both need the file."""
    ds, labels, _ = _labels_and_fake_trainer(tmp_path, monkeypatch)
    probed = _fake_probe(monkeypatch)

    _ = run_op(ds, "train-sleap", {"labels": str(labels), "max_epochs": 1})
    assert str(labels) in probed[0]


def test_the_recorded_metrics_path_holds_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A row advertising a config as its metrics is worse than advertising none.

    A reader has to open the file to discover it holds no metric, where an
    empty cell says so at a glance.
    """
    ds, labels, _ = _labels_and_fake_trainer(
        tmp_path, monkeypatch, writes_training_log=True
    )

    _ = run_op(ds, "train-sleap", {"labels": str(labels), "max_epochs": 1})
    row = trained_model_index(model_index_path(ds, "train-sleap")).read().iloc[0]
    assert str(row["metrics_path"]).endswith("training_log.csv")


def test_a_run_that_logged_no_metrics_records_no_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The empty cell is a real state, and the only honest one here."""
    ds, labels, _ = _labels_and_fake_trainer(tmp_path, monkeypatch)

    _ = run_op(ds, "train-sleap", {"labels": str(labels), "max_epochs": 1})
    row = trained_model_index(model_index_path(ds, "train-sleap")).read().iloc[0]
    assert row["metrics_path"] == ""


# --- the validation file ---------------------------------------------------


def _labels_with_validation(tmp_path: Path) -> Path:
    """Write ``labels.slp`` and ``labels.val.slp`` beside it, as a preparation does."""
    labels = tmp_path / "labels" / "labels.slp"
    labels.parent.mkdir()
    _ = labels.write_bytes(b"slp")
    _ = (labels.parent / "labels.val.slp").write_bytes(b"validation slp")
    return labels


def test_a_validation_file_beside_the_labels_reaches_the_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = make_dataset(tmp_path / "ds", save=False)
    _point_at_sleap(tmp_path, monkeypatch)
    seen = _fake_trainer(monkeypatch)
    labels = _labels_with_validation(tmp_path)

    _ = run_op(ds, "train-sleap", {"labels": str(labels), "max_epochs": 1})

    data_config = _written_section(seen[0], "data_config")
    assert data_config["val_labels_path"] == [str(labels.parent / "labels.val.slp")]
    assert "validation_fraction" not in data_config


def test_labels_without_a_validation_file_hold_out_the_fraction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    argv = _run_and_capture(tmp_path, monkeypatch, {"validation_fraction": 0.2})

    data_config = _written_section(argv, "data_config")
    assert data_config["validation_fraction"] == 0.2
    assert "val_labels_path" not in data_config


def test_labels_without_a_validation_file_are_fingerprinted_by_their_bytes(
    tmp_path: Path,
) -> None:
    labels = tmp_path / "labels.slp"
    _ = labels.write_bytes(b"slp")

    assert fingerprint_sleap_labels(labels) == fingerprint_dataset(labels)


def test_the_validation_file_enters_the_fingerprint(tmp_path: Path) -> None:
    labels = _labels_with_validation(tmp_path)
    validation = labels.parent / "labels.val.slp"

    with_validation = fingerprint_sleap_labels(labels)

    assert with_validation == hash_params(
        {
            "labels": fingerprint_dataset(labels),
            "validation": fingerprint_dataset(validation),
        }
    )
    assert with_validation != fingerprint_dataset(labels)
    _ = validation.write_bytes(b"other validation slp")
    assert fingerprint_sleap_labels(labels) != with_validation


def test_the_recorded_data_fingerprint_is_the_one_in_the_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The row and ``training.json`` record the digest that named the run."""
    ds = make_dataset(tmp_path / "ds", save=False)
    _point_at_sleap(tmp_path, monkeypatch)
    _ = _fake_trainer(monkeypatch)
    labels = _labels_with_validation(tmp_path)

    run_id = run_op(ds, "train-sleap", {"labels": str(labels), "max_epochs": 1})

    row = trained_model_index(model_index_path(ds, "train-sleap")).read().iloc[0]
    recorded = str(row["data_fingerprint"])
    assert recorded == fingerprint_sleap_labels(labels)
    params = TrainSleapParams(labels=str(labels), max_epochs=1)
    assert run_id == train_run_id(
        "train-sleap", OPS["train-sleap"].version, params, recorded, ""
    )
    document: object = json.loads(
        (
            model_run_root(ds, "train-sleap", run_id) / TRAINING_PROVENANCE_FILENAME
        ).read_text()
    )
    assert is_str_mapping(document)
    data = document["data"]
    assert is_str_mapping(data) and data["fingerprint"] == recorded
