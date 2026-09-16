"""The SLEAP training op, with the trainer itself faked.

What matters here is everything around the subprocess: that the config mosaic
writes says what the parameters said, that the row records a *directory*, and
that the head is read back off the artifact rather than echoed from the request.
The last one is the point -- a row claiming a model type the directory does not
have is a row that lies about what was trained.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from mosaic.core.pipeline.models import model_index_path
from mosaic.core.pipeline.ops import run_op
from mosaic.tracking import register_ops
from mosaic.tracking.ops.train import trained_model_index
from mosaic.tracking.ops.train_sleap import TrainSleapParams
from mosaic.tracking.sleap import probe as probe_module
from mosaic.tracking.sleap.probe import (
    SleapProbeResponse,
    require_identity_labels,
    require_sleap_nn,
)
from mosaic.tracking.sleap.run import SleapNotFoundError
from mosaic.tracking.sleap import training as training_module
from mosaic.tracking.sleap.training import sleap_device_overrides, sleap_train_config

from tests.helpers import dotted_values, is_section, make_dataset

register_ops()


def _fake_trainer(
    monkeypatch: pytest.MonkeyPatch,
    head: str = "centered_instance",
    writes_training_log: bool = False,
):
    """Stand in for sleap-nn-train, writing the directory it would have.

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
        produced.mkdir(parents=True, exist_ok=True)
        _ = (produced / "best.ckpt").write_bytes(b"weights")
        _ = (produced / "training_config.yaml").write_text(
            f"head_configs:\n  {head}: {{}}\n"
        )
        if writes_training_log:
            _ = (produced / "training_log.csv").write_text("epoch,train_loss\n0,1.0\n")
        return ("done", "", 0)

    monkeypatch.setattr(training_module, "run_supervised", run)
    return seen


def _point_at_sleap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "bin").mkdir(exist_ok=True)
    for name in ("sleap-nn-train", "sleap-track", "python"):
        _ = (tmp_path / "bin" / name).write_text("")
    monkeypatch.setenv("MOSAIC_SLEAP_BIN", str(tmp_path / "bin" / "sleap-nn-train"))


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
    assert resolved.path.is_dir(), "a directory, which is what sleap-track wants"
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
    # In a directory of its own. The labels fingerprint walks the file's
    # parent, and a run writing beside it would move the identity between the
    # two calls.
    labels = tmp_path / "labels" / "session.slp"
    labels.parent.mkdir()
    _ = labels.write_bytes(b"slp")
    params = {"labels": str(labels), "max_epochs": 1}

    first = run_op(ds, "train-sleap", dict(params))
    assert len(seen) == 1

    assert run_op(ds, "train-sleap", dict(params)) == first
    assert len(seen) == 1, "a finished run must not train again"

    assert run_op(ds, "train-sleap", dict(params), overwrite=True) == first
    assert len(seen) == 2, "overwrite must reach the gate"


# --- overrides and the device ----------------------------------------------


def _assignments(argv: Sequence[str]) -> list[str]:
    """The Hydra assignments in *argv*, after ``--config-dir`` and its name."""
    return list(argv[list(argv).index("--config-name") + 2 :])


def _run_and_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, params: dict[str, object]
) -> list[str]:
    """Run ``train-sleap`` once and return the argv the trainer was given."""
    ds = make_dataset(tmp_path / "ds", save=False)
    _point_at_sleap(tmp_path, monkeypatch)
    seen = _fake_trainer(monkeypatch)
    labels = tmp_path / "labels" / "session.slp"
    labels.parent.mkdir()
    _ = labels.write_bytes(b"slp")
    _ = run_op(ds, "train-sleap", {"labels": str(labels), "max_epochs": 1, **params})
    return seen[0]


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

    A resume and a device are both exercised, since both inject keys the
    minimal document has never carried.
    """
    ds = make_dataset(tmp_path / "ds", save=False)
    _point_at_sleap(tmp_path, monkeypatch)
    seen = _fake_trainer(monkeypatch)
    labels = tmp_path / "labels" / "session.slp"
    labels.parent.mkdir()
    _ = labels.write_bytes(b"slp")

    base = run_op(ds, "train-sleap", {"labels": str(labels), "max_epochs": 1})
    seen.clear()
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
    config_dir = Path(argv[argv.index("--config-dir") + 1])
    written = yaml.safe_load((config_dir / "config.yaml").read_text())
    assert is_section(written)
    declared = dotted_values(written)
    assignments = _assignments(argv)
    assert assignments, "the run under test must inject something"
    for assignment in assignments:
        key = assignment.split("=", 1)[0]
        assert key.startswith(("+", "~")) or key in declared, assignment


def test_a_base_model_resume_names_the_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resuming injects a key the minimal document has never carried."""
    ds = make_dataset(tmp_path / "ds", save=False)
    _point_at_sleap(tmp_path, monkeypatch)
    seen = _fake_trainer(monkeypatch)
    labels = tmp_path / "labels" / "session.slp"
    labels.parent.mkdir()
    _ = labels.write_bytes(b"slp")

    base = run_op(ds, "train-sleap", {"labels": str(labels), "max_epochs": 1})
    seen.clear()
    _ = run_op(
        ds, "train-sleap", {"labels": str(labels), "max_epochs": 1, "base_model": base}
    )
    resumes = [
        arg
        for arg in _assignments(seen[0])
        if arg.startswith("+trainer_config.resume_ckpt_path=")
    ]
    assert len(resumes) == 1, _assignments(seen[0])
    assert resumes[0].endswith("best.ckpt")


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

    ``sleap-nn-train`` is a console script of the ``sleap`` distribution
    itself, so the location ladder finds it either way and the run dies inside
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
    ds = make_dataset(tmp_path / "ds", save=False)
    _point_at_sleap(tmp_path, monkeypatch)
    seen = _fake_trainer(monkeypatch)
    _ = _fake_probe(monkeypatch, n_tracks=0)
    labels = tmp_path / "labels" / "session.slp"
    labels.parent.mkdir()
    _ = labels.write_bytes(b"slp")

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
    ds = make_dataset(tmp_path / "ds", save=False)
    _point_at_sleap(tmp_path, monkeypatch)
    seen = _fake_trainer(monkeypatch)
    _ = _fake_probe(monkeypatch, n_tracks=4)
    labels = tmp_path / "labels" / "session.slp"
    labels.parent.mkdir()
    _ = labels.write_bytes(b"slp")

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
    ds = make_dataset(tmp_path / "ds", save=False)
    _point_at_sleap(tmp_path, monkeypatch)
    _ = _fake_trainer(monkeypatch)
    probed = _fake_probe(monkeypatch)
    labels = tmp_path / "labels" / "session.slp"
    labels.parent.mkdir()
    _ = labels.write_bytes(b"slp")
    params = {"labels": str(labels), "max_epochs": 1}

    first = run_op(ds, "train-sleap", dict(params))
    assert len(probed) == 1

    assert run_op(ds, "train-sleap", dict(params)) == first
    assert len(probed) == 1, "a reused run must not probe again"


def test_the_probe_is_handed_the_labels_it_reports_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both questions are answered from one launch, so both need the file."""
    ds = make_dataset(tmp_path / "ds", save=False)
    _point_at_sleap(tmp_path, monkeypatch)
    _ = _fake_trainer(monkeypatch)
    probed = _fake_probe(monkeypatch)
    labels = tmp_path / "labels" / "session.slp"
    labels.parent.mkdir()
    _ = labels.write_bytes(b"slp")

    _ = run_op(ds, "train-sleap", {"labels": str(labels), "max_epochs": 1})
    assert str(labels) in probed[0]


def test_the_recorded_metrics_path_holds_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A row advertising a config as its metrics is worse than advertising none.

    A reader has to open the file to discover it holds no metric, where an
    empty cell says so at a glance.
    """
    ds = make_dataset(tmp_path / "ds", save=False)
    _point_at_sleap(tmp_path, monkeypatch)
    _ = _fake_trainer(monkeypatch, writes_training_log=True)
    labels = tmp_path / "labels" / "session.slp"
    labels.parent.mkdir()
    _ = labels.write_bytes(b"slp")

    _ = run_op(ds, "train-sleap", {"labels": str(labels), "max_epochs": 1})
    row = trained_model_index(model_index_path(ds, "train-sleap")).read().iloc[0]
    assert str(row["metrics_path"]).endswith("training_log.csv")


def test_a_run_that_logged_no_metrics_records_no_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The empty cell is a real state, and the only honest one here."""
    ds = make_dataset(tmp_path / "ds", save=False)
    _point_at_sleap(tmp_path, monkeypatch)
    _ = _fake_trainer(monkeypatch)
    labels = tmp_path / "labels" / "session.slp"
    labels.parent.mkdir()
    _ = labels.write_bytes(b"slp")

    _ = run_op(ds, "train-sleap", {"labels": str(labels), "max_epochs": 1})
    row = trained_model_index(model_index_path(ds, "train-sleap")).read().iloc[0]
    assert row["metrics_path"] == ""
