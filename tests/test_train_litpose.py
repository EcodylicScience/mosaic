"""The Lightning Pose training op, with the trainer faked.

Lightning Pose is not installed on the machine this was written on, so unlike
the SLEAP side none of it has been run against the real thing. What is asserted
is the part mosaic owns: what it hands over, what it records, and that the model
type on the row comes from the artifact rather than the request.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import NamedTuple

import pytest
import yaml
from pydantic import ValidationError

from mosaic.core.pipeline.models import model_index_path
from mosaic.core.pipeline.ops import run_op
from mosaic.tracking import register_ops
from mosaic.tracking.litpose import training as training_module
from mosaic.tracking.litpose.templates import default_config_path
from mosaic.tracking.litpose.training import epoch_coupled_assignments
from mosaic.tracking.ops.train import trained_model_index
from mosaic.tracking.ops.train_litpose import TrainLitposeParams

from tests.helpers import dotted_values, is_section, make_dataset

register_ops()


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


class _Launches(NamedTuple):
    """What the fake trainer was handed, one entry per launch.

    The environment is recorded beside argv because Lightning Pose takes half
    of a device request through each: the count is a config key and the
    selection is ``CUDA_VISIBLE_DEVICES``.
    """

    argv: list[list[str]]
    env: list[dict[str, str]]


def _fake_trainer(
    monkeypatch: pytest.MonkeyPatch,
    model_type: str = "heatmap",
    writes_predictions: bool = False,
) -> _Launches:
    """Stand in for Lightning Pose, writing the model directory it would have.

    *writes_predictions* says whether the run got far enough to predict on its
    labelled images. A run that never validated leaves no prediction file,
    which is the state the empty ``metrics_path`` cell records.
    """
    launches = _Launches(argv=[], env=[])

    def run(
        argv: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        **kw: object,
    ) -> tuple[str, str, int]:
        launches.argv.append(list(argv))
        launches.env.append(dict(env or {}))
        out = Path(argv[argv.index("-c") + 4])
        checkpoints = out / "tb_logs" / "run" / "version_0" / "checkpoints"
        checkpoints.mkdir(parents=True, exist_ok=True)
        _ = (out / "config.yaml").write_text(f"model:\n  model_type: {model_type}\n")
        _ = (checkpoints / "best.ckpt").write_bytes(b"weights")
        if writes_predictions:
            _ = (out / "predictions_pixel_error.csv").write_text("head\n1.5\n")
        return ("done", "", 0)

    monkeypatch.setattr(training_module, "run_supervised", run)
    return launches


def _point_at_litpose(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "bin").mkdir(exist_ok=True)
    for name in ("litpose", "python"):
        _ = (tmp_path / "bin" / name).write_text("")
    monkeypatch.setenv("MOSAIC_LITPOSE_BIN", str(tmp_path / "bin" / "litpose"))


def test_it_registers_a_directory_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = make_dataset(tmp_path, save=False)
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
    ds = make_dataset(tmp_path, save=False)
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
    ds = make_dataset(tmp_path, save=False)
    _point_at_litpose(tmp_path, monkeypatch)
    launches = _fake_trainer(monkeypatch)

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
    argv = launches.argv[0]
    assert "model.model_type=heatmap_mhcrnn" in argv
    assert "model.backbone=vitb_sam" in argv
    assert "training.max_epochs=5" in argv


def test_the_device_reaches_the_trainer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both halves of a device request, because Lightning Pose splits it.

    Nothing in its config names a device: ``training.num_gpus`` is a count and
    ``lightning_pose.train`` fixes ``accelerator="gpu"``. So *which* has to
    reach the subprocess through its environment, and a test asserting only on
    argv would pass while the run still landed on GPU 0.
    """
    ds = make_dataset(tmp_path, save=False)
    _point_at_litpose(tmp_path, monkeypatch)
    launches = _fake_trainer(monkeypatch)

    _ = run_op(
        ds,
        "train-litpose",
        {
            "project": str(_project(tmp_path)),
            "base_config": str(_base_config(tmp_path)),
            "device": "1",
            "max_epochs": 5,
        },
    )

    assert "training.num_gpus=1" in launches.argv[0]
    assert launches.env[0]["CUDA_VISIBLE_DEVICES"] == "1"


def test_two_devices_are_counted_as_well_as_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The count is what makes ``0,1`` mean two GPUs rather than one."""
    ds = make_dataset(tmp_path, save=False)
    _point_at_litpose(tmp_path, monkeypatch)
    launches = _fake_trainer(monkeypatch)

    _ = run_op(
        ds,
        "train-litpose",
        {
            "project": str(_project(tmp_path)),
            "base_config": str(_base_config(tmp_path)),
            "device": "0,1",
            "max_epochs": 5,
        },
    )

    assert "training.num_gpus=2" in launches.argv[0]
    assert launches.env[0]["CUDA_VISIBLE_DEVICES"] == "0,1"


def test_an_auto_device_sets_neither_half(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``auto`` leaves the choice to Lightning Pose, so it must not narrow it.

    Laying ``CUDA_VISIBLE_DEVICES`` over an environment that already carries
    one would silently override a selection the caller made outside mosaic.
    """
    ds = make_dataset(tmp_path, save=False)
    _point_at_litpose(tmp_path, monkeypatch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    launches = _fake_trainer(monkeypatch)

    _ = run_op(
        ds,
        "train-litpose",
        {
            "project": str(_project(tmp_path)),
            "base_config": str(_base_config(tmp_path)),
            "max_epochs": 5,
        },
    )

    assert not [arg for arg in launches.argv[0] if arg.startswith("training.num_gpus")]
    assert launches.env[0]["CUDA_VISIBLE_DEVICES"] == "3", "inherited, not replaced"


@pytest.mark.parametrize("device", ["cpu", "mps", "cuda:0", "0,x"])
def test_an_unusable_device_is_refused_at_submit(device: str) -> None:
    """Lightning Pose trains on CUDA, so a family name has nothing to set.

    Accepting one and setting nothing is the state this field was in before it
    was wired, which is what its ``unwired`` record used to say.
    """
    with pytest.raises(ValidationError, match="unusable device"):
        _ = TrainLitposeParams(project="project", device=device)


def test_the_trained_model_resolves_back_as_a_litpose_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mosaic.tracking.model_refs import resolve_model

    ds = make_dataset(tmp_path, save=False)
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
    ds = make_dataset(tmp_path, save=False)
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


# --- what identifies a run's configuration ----------------------------------


def test_two_configs_at_one_path_are_two_runs(tmp_path: Path) -> None:
    """The config is identified by content, because a path is a location.

    Held as a path, two different Lightning Pose configurations sitting at one
    filename minted one identifier, and one configuration reachable by two paths
    minted two. `train_run_id` already says this about `base_model` -- "never the
    path itself" -- and a config carries training settings mosaic has no field
    for, so it is the same argument.
    """
    from mosaic.tracking.ops.train import train_run_id
    from mosaic.core.pipeline.file_digest import file_digest

    first = tmp_path / "cfg.yaml"
    _ = first.write_text("training:\n  num_gpus: 0\n")
    params = TrainLitposeParams(project="p", base_config=str(first))
    before = train_run_id(
        "train-litpose", "0.1", params, "data", "", extra={"config": file_digest(first)}
    )

    _ = first.write_text("training:\n  num_gpus: 1\n")  # same path, other config
    after = train_run_id(
        "train-litpose", "0.1", params, "data", "", extra={"config": file_digest(first)}
    )

    assert before != after


def test_the_same_config_at_two_paths_is_one_run(tmp_path: Path) -> None:
    """The mirror: moving a config, or vendoring it, does not fork the identity."""
    from mosaic.tracking.ops.train import train_run_id
    from mosaic.core.pipeline.file_digest import file_digest

    body = "training:\n  num_gpus: 0\n"
    here, there = tmp_path / "a.yaml", tmp_path / "b.yaml"
    _ = here.write_text(body)
    _ = there.write_text(body)

    ids = {
        train_run_id(
            "train-litpose",
            "0.1",
            TrainLitposeParams(project="p", base_config=str(path)),
            "data",
            "",
            extra={"config": file_digest(path)},
        )
        for path in (here, there)
    }

    assert len(ids) == 1


def test_an_unset_base_config_uses_the_vendored_template(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An API or app caller has no filesystem to point at, so there is a default.

    Lightning Pose ships no template, so without one carried here every caller
    would have to fetch `config_default.yaml` from its repository first.
    """
    from mosaic.tracking.litpose.templates import default_config_path

    assert default_config_path().is_file(), "the template must ship, not just exist"

    ds = make_dataset(tmp_path, save=False)
    _point_at_litpose(tmp_path, monkeypatch)
    launches = _fake_trainer(monkeypatch)
    _ = run_op(ds, "train-litpose", {"project": str(_project(tmp_path))})

    handed = Path(launches.argv[0][launches.argv[0].index("-c") + 2])
    assert handed == default_config_path()


def test_a_finished_run_is_reused_unless_overwrite_says_otherwise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reuse gate reads the ``overwrite`` argument, where it read a field.

    Both directions. An op ignoring the argument reuses forever and an op whose
    gate lost its completion half retrains forever, and one direction alone
    cannot tell those apart.
    """
    ds = make_dataset(tmp_path, save=False)
    _point_at_litpose(tmp_path, monkeypatch)
    launches = _fake_trainer(monkeypatch)
    params = {"project": str(_project(tmp_path))}

    first = run_op(ds, "train-litpose", dict(params))
    assert len(launches.argv) == 1

    assert run_op(ds, "train-litpose", dict(params)) == first
    assert len(launches.argv) == 1, "a finished run must not train again"

    assert run_op(ds, "train-litpose", dict(params), overwrite=True) == first
    assert len(launches.argv) == 2, "overwrite must reach the gate"


# --- the keys coupled to max_epochs -----------------------------------------


def _template() -> dict[str, object]:
    """The vendored config, flattened to the dotted keys an override names."""
    loaded = yaml.safe_load(default_config_path().read_text())
    assert is_section(loaded)
    return dotted_values(loaded)


def test_the_derivation_reproduces_the_template_at_its_own_length() -> None:
    """A run finished before this existed must keep training the way it did.

    Read off the template rather than restated, so the two cannot drift: if the
    vendored config's schedule changes, this says so instead of passing against
    numbers copied out of it once. Compared key for key, because a derivation
    that reproduced three of the four values would otherwise pass.
    """
    template = _template()
    max_epochs = template["training.max_epochs"]
    assert isinstance(max_epochs, int)

    derived = epoch_coupled_assignments(max_epochs)
    assert derived == {key: template[key] for key in derived}


@pytest.mark.parametrize("max_epochs", [1, 2, 3, 4, 10, 37, 300])
def test_every_milestone_stays_inside_the_training_length(max_epochs: int) -> None:
    """``ModelConfig.validate`` asserts exactly this, and dies before epoch one.

    The bound is what made every short run fail, which is every smoke test.
    """
    derived = epoch_coupled_assignments(max_epochs)
    milestones = derived["training.lr_scheduler_params.multisteplr.milestones"]
    assert isinstance(milestones, list)
    epochs = [value for value in milestones if isinstance(value, int)]
    assert len(epochs) == len(milestones), "a milestone is an epoch number"
    assert epochs, "a schedule with no milestone is not a schedule"
    assert all(1 <= epoch <= max_epochs for epoch in epochs)
    assert epochs == sorted(set(epochs))


@pytest.mark.parametrize("max_epochs", [1, 2, 3, 4, 10, 300])
def test_validation_runs_at_least_once(max_epochs: int) -> None:
    """Otherwise no checkpoint is written and the run fails having paid for it.

    Between three and four epochs the template's interval of 5 means validation
    never runs, so the checkpoint callback never fires and post-training
    evaluation dies with the model nowhere on disk.
    """
    interval = epoch_coupled_assignments(max_epochs)["training.check_val_every_n_epoch"]
    assert isinstance(interval, int)
    assert 1 <= interval <= max_epochs


@pytest.mark.parametrize("max_epochs", [1, 2, 3, 300])
def test_the_backbone_unfreezes_within_the_run(max_epochs: int) -> None:
    """Left at the template's 20, a two-epoch run trains only the head."""
    epoch = epoch_coupled_assignments(max_epochs)["training.unfreezing_epoch"]
    assert isinstance(epoch, int)
    assert 0 <= epoch < max_epochs


def test_the_coupled_keys_reach_the_trainer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Derived below ``Params``, so argv is where they become observable."""
    ds = make_dataset(tmp_path, save=False)
    _point_at_litpose(tmp_path, monkeypatch)
    launches = _fake_trainer(monkeypatch)

    _ = run_op(
        ds,
        "train-litpose",
        {
            "project": str(_project(tmp_path)),
            "base_config": str(_base_config(tmp_path)),
            "max_epochs": 2,
        },
    )

    argv = launches.argv[0]
    assert "training.max_epochs=2" in argv
    assert "training.min_epochs=2" in argv
    assert "training.check_val_every_n_epoch=1" in argv
    assert "training.unfreezing_epoch=1" in argv
    assert "training.lr_scheduler_params.multisteplr.milestones=[1, 2]" in argv


def test_an_override_still_beats_a_derived_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What ``litpose_overrides``' declaration promises, applied last."""
    ds = make_dataset(tmp_path, save=False)
    _point_at_litpose(tmp_path, monkeypatch)
    launches = _fake_trainer(monkeypatch)

    _ = run_op(
        ds,
        "train-litpose",
        {
            "project": str(_project(tmp_path)),
            "base_config": str(_base_config(tmp_path)),
            "max_epochs": 2,
            "litpose_overrides": {"training.unfreezing_epoch": 0},
        },
    )

    argv = launches.argv[0]
    assert "training.unfreezing_epoch=0" in argv
    assert "training.unfreezing_epoch=1" not in argv


def test_the_derivation_reads_nothing_but_the_training_length() -> None:
    """The reason these values may stay out of the ``run_id``.

    They are computed below ``Params``, so nothing about them reaches the
    identifier. That is sound only while they are a pure function of
    ``max_epochs``, which is hashed: a derivation reading a second input would
    let two runs share one identifier and produce different models.
    """
    assert epoch_coupled_assignments(17) == epoch_coupled_assignments(17)
    assert epoch_coupled_assignments(17) != epoch_coupled_assignments(18)


def test_the_recorded_metrics_path_holds_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A row advertising a config as its metrics is worse than advertising none.

    A reader has to open the file to discover it holds no metric, where an
    empty cell says so at a glance.
    """
    ds = make_dataset(tmp_path, save=False)
    _point_at_litpose(tmp_path, monkeypatch)
    _ = _fake_trainer(monkeypatch, writes_predictions=True)

    _ = run_op(
        ds,
        "train-litpose",
        {
            "project": str(_project(tmp_path)),
            "base_config": str(_base_config(tmp_path)),
            "max_epochs": 5,
        },
    )
    row = trained_model_index(model_index_path(ds, "train-litpose")).read().iloc[0]
    assert str(row["metrics_path"]).endswith("predictions_pixel_error.csv")


def test_a_run_that_predicted_nothing_records_no_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The empty cell is a real state, and the only honest one here."""
    ds = make_dataset(tmp_path, save=False)
    _point_at_litpose(tmp_path, monkeypatch)
    _ = _fake_trainer(monkeypatch)

    _ = run_op(
        ds,
        "train-litpose",
        {
            "project": str(_project(tmp_path)),
            "base_config": str(_base_config(tmp_path)),
            "max_epochs": 5,
        },
    )
    row = trained_model_index(model_index_path(ds, "train-litpose")).read().iloc[0]
    assert row["metrics_path"] == ""
