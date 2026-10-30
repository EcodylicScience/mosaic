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
from typing import Final, get_args

import pytest
import yaml
from pydantic import ValidationError

from mosaic.core.dataset import Dataset
from mosaic.core.json_value import JsonValue
from mosaic.core.narrowing import is_str_mapping
from mosaic.core.pipeline._utils import hash_params
from mosaic.core.pipeline.job import Cancelled
from mosaic.core.pipeline.models import model_index_path, model_run_root
from mosaic.core.pipeline.ops import OPS, IdentityDeferred, run_op
from mosaic.core.pipeline.run_log import read_runs, run_log_dir
from mosaic.core.pipeline.subprocess_util import ProcessCancelled
from mosaic.tracking import register_ops
from mosaic.tracking.ops._common import fingerprint_dataset
from mosaic.tracking.ops.train import TRAINING_PROVENANCE_FILENAME, train_run_id
from mosaic.tracking.ops.trained_models import trained_model_index
from mosaic.tracking.ops.train_sleap import (
    MODEL_DIRECTORY,
    TrainSleapInvalid,
    TrainSleapOp,
    TrainSleapParams,
    embedding_objective,
    fingerprint_sleap_labels,
)
from mosaic.tracking.sleap import probe as probe_module
from mosaic.tracking.sleap.probe import (
    SleapProbeResponse,
    require_embedding_labels,
    require_identity_labels,
    require_sleap_nn,
)
from mosaic.tracking.sleap.run import SleapError, SleapNotFoundError
from mosaic.tracking.sleap import training as training_module
from mosaic.tracking.sleap.training import (
    DEFAULT_EMBEDDING_OBJECTIVE,
    PREPARED_EMBEDDING_OBJECTIVE,
    EmbeddingObjective,
    EmbeddingPositiveScope,
    SleapBackbone,
    SleapBackboneSize,
    SleapHead,
    SleapTrainSettings,
    sleap_device_overrides,
    sleap_train_config,
)
from mosaic.tracking.sleap.version import PREPARE_EMBEDDING_DATA_KIND

from tests.helpers import (
    dotted_values,
    make_dataset,
    read_string_golden,
    regenerate_command,
    updating_golden,
    write_golden,
    write_sleap_model,
)

register_ops()


def _fake_trainer(
    monkeypatch: pytest.MonkeyPatch,
    head: str = "centered_instance",
    writes_training_log: bool = False,
):
    """Stand in for ``sleap-nn train``, writing the directory it would have.

    The directory holds the merged config that sleap-nn saves, which names every
    head it knows and configures *head*.

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
        if produced.exists():
            # sleap-nn never writes into a model directory that exists. It
            # numbers the new one beside it.
            produced = produced.with_name(f"{produced.name}-1")
        _ = write_sleap_model(produced, head=head, merged=True)
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
    sleap_nn_version: str = "0.3.5",
) -> list[list[str]]:
    """Stand in for the probe subprocess, writing the answer it would have."""
    seen: list[list[str]] = []
    found = {
        "has_sleap_nn": has_sleap_nn,
        "sleap_nn_version": sleap_nn_version if has_sleap_nn else "",
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
        ("embedding", {"embedding"}),
    ],
)
def test_every_head_names_its_output_sections(
    head: SleapHead, sections: set[str]
) -> None:
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
        head=head,
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
    if head in _POSE_HEADS:
        assert all(value == {} for value in written[head].values())


_POSE_CONFIGS_GOLDEN: Final = "sleap_pose_head_configs.json"
"""Every pose head's ``config.yaml``, stored because no run id hashes it."""

_POSE_HEADS: Final[tuple[SleapHead, ...]] = (
    "single_instance",
    "centroid",
    "centered_instance",
    "bottomup",
    "multi_class_bottomup",
    "multi_class_topdown",
)


def _pose_documents() -> dict[str, str]:
    """The ``config.yaml`` text of every pose head and backbone, by case id.

    Each pair is written twice: holding out a fraction, and with a validation
    file and base weights.
    """
    documents: dict[str, str] = {}
    for head in _POSE_HEADS:
        for backbone in get_args(SleapBackbone):
            for variant, validation, weights in (
                ("fraction", None, None),
                ("file", Path("labels.val.slp"), Path("/models/base/best.ckpt")),
            ):
                config = sleap_train_config(
                    Path("labels.slp"),
                    Path("/runs/r1"),
                    head=head,
                    backbone=backbone,
                    max_epochs=30,
                    seed=7,
                    validation_fraction=0.25,
                    validation_labels=validation,
                    base_weights=weights,
                    run_name="model",
                )
                documents[f"{head}/{backbone}/{variant}"] = yaml.safe_dump(
                    config, sort_keys=False
                )
    return documents


def _op_config_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """The ``config.yaml`` that a default train-sleap run writes, paths masked."""
    ds, labels, seen = _labels_and_fake_trainer(tmp_path, monkeypatch)
    run_id = run_op(ds, "train-sleap", {"labels": str(labels), "max_epochs": 1})
    config_dir = Path(seen[0][seen[0].index("--config-dir") + 1])
    text = (config_dir / "config.yaml").read_text()
    return text.replace(run_id, "<run_id>").replace(str(tmp_path), "<tmp>")


def test_every_pose_head_document_is_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pose head's document is byte for byte the one stored in the golden file.

    A train-sleap run id does not hash the document. A changed document
    would train a different model under an unchanged identifier.
    """
    documents = _pose_documents()
    documents["op/centered_instance/unet/fraction"] = _op_config_text(
        tmp_path, monkeypatch
    )
    if updating_golden():
        write_golden(_POSE_CONFIGS_GOLDEN, documents)
        return
    golden = read_string_golden(_POSE_CONFIGS_GOLDEN)
    assert documents == golden, regenerate_command(__name__)


# --- the embedding head -------------------------------------------------------

_EMBEDDING_LEAF: Final = "model_config.head_configs.embedding.embedding"

_DEFAULT_OBJECTIVE_TERMS: Final[dict[str, JsonValue]] = {
    "positives.scope": "global_id",
    "positives.aug_views": 2,
    "negatives.sources": ["same_frame", "in_batch"],
    "negatives.exclude_same_track": True,
    "negatives.restrict_same_video": False,
    "loss.name": "supcon",
    "loss.temperature": 0.1,
    "sampler.kind": "pk",
    "sampler.groups_per_batch": 8,
    "sampler.samples_per_group": 16,
    "use_projection": True,
    "projection_dim": 128,
}
"""sleap-nn 0.3.5's default objective, term by term."""

_PREPARED_OBJECTIVE_TERMS: Final[dict[str, JsonValue]] = {
    **_DEFAULT_OBJECTIVE_TERMS,
    "positives.scope": "tracklet",
    "negatives.restrict_same_video": True,
    "sampler.kind": "within_video",
    "sampler.samples_per_group": 8,
}
"""The objective for labels that a prepare-embedding-data run cut."""


def _embedding_document(
    backbone: SleapBackbone = "unet",
    objective: EmbeddingObjective = DEFAULT_EMBEDDING_OBJECTIVE,
) -> Mapping[str, object]:
    """The embedding head's document, read back from the YAML it is written as."""
    config = sleap_train_config(
        Path("labels.slp"),
        Path("/runs/r1"),
        head="embedding",
        backbone=backbone,
        max_epochs=30,
        seed=7,
        validation_fraction=0.25,
        validation_labels=None,
        base_weights=None,
        run_name="model",
        objective=objective,
    )
    written: object = yaml.safe_load(yaml.safe_dump(config, sort_keys=False))
    assert is_str_mapping(written)
    return written


@pytest.mark.parametrize(
    ("backbone", "stride"), [("unet", 16), ("convnext", 64), ("swint", 64)]
)
def test_the_embedding_leaf_takes_its_stride_from_the_backbone(
    backbone: SleapBackbone, stride: int
) -> None:
    """The leaf pools at the backbone's ``max_stride``.

    A convnext or swint stride is 16 times the stem stride of 4.
    """
    values = dotted_values(_embedding_document(backbone))
    backbone_block = f"model_config.backbone_config.{backbone}"
    assert values[f"{_EMBEDDING_LEAF}.embedding_dim"] == 128
    assert values[f"{_EMBEDDING_LEAF}.output_stride"] == stride
    assert values[f"{backbone_block}.max_stride"] == stride
    assert values[f"{backbone_block}.output_stride"] == stride


@pytest.mark.parametrize(
    ("objective", "terms"),
    [
        (DEFAULT_EMBEDDING_OBJECTIVE, _DEFAULT_OBJECTIVE_TERMS),
        (PREPARED_EMBEDDING_OBJECTIVE, _PREPARED_OBJECTIVE_TERMS),
    ],
    ids=["default", "prepare-embedding-data"],
)
def test_the_embedding_leaf_writes_every_objective_term(
    objective: EmbeddingObjective, terms: dict[str, JsonValue]
) -> None:
    """The run id names the objective that trained, because every term is written."""
    leaf = dotted_values(_embedding_document(objective=objective))
    prefix = f"{_EMBEDDING_LEAF}.objective."
    written = {
        key.removeprefix(prefix): value
        for key, value in leaf.items()
        if key.startswith(prefix) and not is_str_mapping(value)
    }
    assert written == terms


def test_the_embedding_document_declares_track_names_per_video() -> None:
    """mosaic does not assert that a track name means one animal across videos."""
    values = dotted_values(_embedding_document())
    assert values["data_config.identity.track_names_are_global"] is False


def test_the_embedding_document_augments_without_erasing_appearance() -> None:
    """Small rotations and shifts, and no flips, erasing or intensity changes."""
    values = dotted_values(_embedding_document())
    geometric = "data_config.augmentation_config.geometric"
    intensity = "data_config.augmentation_config.intensity"
    assert values["data_config.use_augmentations_train"] is True
    assert {
        key: values[f"{geometric}.{key}"]
        for key in (
            "rotation_min",
            "rotation_max",
            "scale_min",
            "scale_max",
            "translate_width",
            "translate_height",
            "affine_p",
            "erase_p",
            "flip_p",
        )
    } == {
        "rotation_min": -10.0,
        "rotation_max": 10.0,
        "scale_min": 1.0,
        "scale_max": 1.0,
        "translate_width": 0.05,
        "translate_height": 0.05,
        "affine_p": 1.0,
        "erase_p": 0.0,
        "flip_p": 0.0,
    }
    assert {
        key: values[f"{intensity}.{key}"]
        for key in ("contrast_p", "brightness_p", "gaussian_noise_p")
    } == {"contrast_p": 0.0, "brightness_p": 0.0, "gaussian_noise_p": 0.0}


def test_the_embedding_document_trains_on_one_device() -> None:
    values = dotted_values(_embedding_document())
    assert values["trainer_config.trainer_devices"] == 1


def test_the_embedding_document_leaves_the_crop_size_to_sleap_nn() -> None:
    """sleap-nn sizes the crop from the labels, and an override sets it."""
    values = dotted_values(_embedding_document())
    assert "data_config.preprocessing.crop_size" not in values


@pytest.mark.parametrize(
    ("labels", "objective"),
    [
        ("prepare-embedding-data.0.1-0123456789", PREPARED_EMBEDDING_OBJECTIVE),
        ("prepare-training-data.0.3-0123456789", DEFAULT_EMBEDDING_OBJECTIVE),
        ("/data/session.slp", DEFAULT_EMBEDDING_OBJECTIVE),
    ],
)
def test_the_objective_follows_the_kind_of_the_labels(
    labels: str, objective: EmbeddingObjective
) -> None:
    """Labels named by a prepare-embedding-data run id take its objective.

    A path, which names no run, takes the default.
    """
    assert embedding_objective(labels) == objective


def test_an_embedding_run_writes_the_default_objective_for_other_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Labels that no prepare-embedding-data run cut take sleap-nn's default."""
    ds, labels, seen = _embedding_setup(
        tmp_path, monkeypatch, _PREPARE_TRAINING_DATA, track_by="alias"
    )
    _ = run_op(ds, "train-sleap", _embedding_params(labels, {_DECLARATION: True}))
    values = dotted_values(_written_config(seen[0]))
    assert values[f"{_OBJECTIVE}.positives.scope"] == "global_id"
    assert values[f"{_EMBEDDING_LEAF}.output_stride"] == 16


def test_an_embedding_run_writes_the_prepared_objective_for_its_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds, labels, seen = _embedding_setup(tmp_path, monkeypatch)
    _ = run_op(ds, "train-sleap", _embedding_params(labels))
    values = dotted_values(_written_config(seen[0]))
    assert values[f"{_OBJECTIVE}.positives.scope"] == "tracklet"
    assert values[f"{_OBJECTIVE}.negatives.restrict_same_video"] is True


def test_an_override_of_an_objective_term_is_assigned_bare(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The document writes every term, and Hydra requires the bare form for it.

    A leaf field that mosaic leaves to sleap-nn is appended.
    """
    ds, labels, seen = _embedding_setup(tmp_path, monkeypatch)
    _ = run_op(
        ds,
        "train-sleap",
        _embedding_params(
            labels,
            {
                f"{_OBJECTIVE}.loss.temperature": 0.07,
                f"{_EMBEDDING_LEAF}.pool": "avg",
            },
        ),
    )
    assignments = _assignments(seen[0])
    assert f"{_OBJECTIVE}.loss.temperature=0.07" in assignments
    assert f"+{_EMBEDDING_LEAF}.pool=avg" in assignments


# --- typed training settings ------------------------------------------------


def _settings_document(
    head: SleapHead, backbone: SleapBackbone, settings: SleapTrainSettings
) -> dict[str, object]:
    """The dotted values of a document written with *settings*."""
    config = sleap_train_config(
        Path("labels.slp"),
        Path("/runs/r1"),
        head=head,
        backbone=backbone,
        max_epochs=30,
        seed=7,
        validation_fraction=0.25,
        validation_labels=None,
        base_weights=None,
        run_name="model",
        settings=settings,
    )
    written: object = yaml.safe_load(yaml.safe_dump(config, sort_keys=False))
    assert is_str_mapping(written)
    return dotted_values(written)


_EMBEDDING_TRIAL_VALUES: Final[dict[str, JsonValue]] = {
    "trainer_config.train_data_loader.batch_size": 64,
    "trainer_config.val_data_loader.batch_size": 64,
    "model_config.backbone_config.convnext.model_type": "tiny",
    "model_config.backbone_config.convnext.pre_trained_weights": (
        "ConvNeXt_Tiny_Weights"
    ),
    "model_config.backbone_config.convnext.stem_patch_stride": 4,
    "model_config.backbone_config.convnext.max_stride": 64,
    "model_config.backbone_config.convnext.output_stride": 64,
    f"{_EMBEDDING_LEAF}.output_stride": 64,
    "trainer_config.optimizer_name": "AdamW",
    "trainer_config.optimizer.lr": 1e-4,
    "data_config.preprocessing.ensure_grayscale": False,
    "data_config.data_pipeline_fw": "torch_dataset",
    "model_config.init_weights": "xavier",
    "trainer_config.eval.enabled": True,
    "trainer_config.eval.frequency": 1,
    "trainer_config.model_ckpt.monitor": "eval/val/rank1",
    "trainer_config.model_ckpt.mode": "max",
}
"""The settings of the measured embedding training, at their document keys."""


def test_an_embedding_document_writes_the_measured_settings() -> None:
    """Left unset, every typed setting takes the measured training's value.

    The checkpoint monitor names the metric that sleap-nn selects an embedding
    checkpoint on, so the written document states what selected it.
    """
    values = _settings_document("embedding", "convnext", SleapTrainSettings())
    assert {key: values.get(key) for key in _EMBEDDING_TRIAL_VALUES} == (
        _EMBEDDING_TRIAL_VALUES
    )


def test_a_swint_embedding_document_strides_its_patches_at_the_stem_stride() -> None:
    values = _settings_document("embedding", "swint", SleapTrainSettings())
    block = "model_config.backbone_config.swint"
    assert values[f"{block}.model_type"] == "tiny"
    assert values[f"{block}.pre_trained_weights"] == "Swin_T_Weights"
    assert values[f"{block}.stem_patch_stride"] == 4
    assert values[f"{block}.max_stride"] == 64


def test_a_unet_embedding_document_writes_no_convnext_setting() -> None:
    """A UNet has no size, ImageNet weights or patch stem."""
    values = _settings_document("embedding", "unet", SleapTrainSettings())
    assert values["model_config.backbone_config.unet"] == {
        "max_stride": 16,
        "output_stride": 16,
    }
    assert values["trainer_config.train_data_loader.batch_size"] == 64


def test_an_embedding_stem_stride_sets_the_stride_of_the_leaf() -> None:
    """sleap-nn sets a convnext max_stride to 16 times its stem stride.

    The stem keeps its ImageNet kernel, so the default weights still load.
    """
    values = _settings_document(
        "embedding", "convnext", SleapTrainSettings(stem_stride=2)
    )
    block = "model_config.backbone_config.convnext"
    assert values[f"{block}.pre_trained_weights"] == "ConvNeXt_Tiny_Weights"
    assert values[f"{block}.stem_patch_stride"] == 2
    assert values[f"{block}.max_stride"] == 32
    assert values[f"{_EMBEDDING_LEAF}.output_stride"] == 32


@pytest.mark.parametrize("head", ["centroid", "embedding"])
@pytest.mark.parametrize("backbone", ["convnext", "swint"])
@pytest.mark.parametrize("stem_stride", [None, 2, 4, 8])
def test_no_document_writes_a_stem_kernel(
    head: SleapHead, backbone: SleapBackbone, stem_stride: int | None
) -> None:
    """ImageNet weights load only into sleap-nn's default stem kernel of 4."""
    values = _settings_document(
        head,
        backbone,
        SleapTrainSettings(pretrained_backbone=True, stem_stride=stem_stride),
    )
    block = f"model_config.backbone_config.{backbone}"
    assert f"{block}.stem_patch_kernel" not in values
    assert f"{block}.patch_size" not in values


def test_an_unset_embedding_setting_can_be_turned_off() -> None:
    values = _settings_document(
        "embedding",
        "convnext",
        SleapTrainSettings(
            pretrained_backbone=False, ensure_grayscale=True, cache_images=True
        ),
    )
    assert "model_config.backbone_config.convnext.pre_trained_weights" not in values
    assert values["data_config.preprocessing.ensure_grayscale"] is True
    assert values["data_config.data_pipeline_fw"] == "torch_dataset_cache_img_memory"


@pytest.mark.parametrize(
    ("backbone", "settings", "expected"),
    [
        (
            "unet",
            SleapTrainSettings(batch_size=16),
            {
                "trainer_config.train_data_loader.batch_size": 16,
                "trainer_config.val_data_loader.batch_size": 16,
            },
        ),
        (
            "convnext",
            SleapTrainSettings(backbone_size="base"),
            {"model_config.backbone_config.convnext.model_type": "base"},
        ),
        (
            "convnext",
            SleapTrainSettings(pretrained_backbone=True),
            {
                "model_config.backbone_config.convnext.pre_trained_weights": (
                    "ConvNeXt_Tiny_Weights"
                )
            },
        ),
        (
            "convnext",
            SleapTrainSettings(stem_stride=4),
            {"model_config.backbone_config.convnext.stem_patch_stride": 4},
        ),
        (
            "swint",
            SleapTrainSettings(stem_stride=4),
            {"model_config.backbone_config.swint.stem_patch_stride": 4},
        ),
        (
            "unet",
            SleapTrainSettings(max_stride=32),
            {"model_config.backbone_config.unet.max_stride": 32},
        ),
        (
            "unet",
            SleapTrainSettings(optimizer="AdamW"),
            {"trainer_config.optimizer_name": "AdamW"},
        ),
        (
            "unet",
            SleapTrainSettings(learning_rate=3e-4),
            {"trainer_config.optimizer.lr": 3e-4},
        ),
        (
            "unet",
            SleapTrainSettings(ensure_grayscale=True),
            {"data_config.preprocessing.ensure_grayscale": True},
        ),
        (
            "unet",
            SleapTrainSettings(cache_images=True),
            {"data_config.data_pipeline_fw": "torch_dataset_cache_img_memory"},
        ),
    ],
    ids=[
        "batch_size",
        "backbone_size",
        "pretrained_backbone",
        "stem_stride-convnext",
        "stem_stride-swint",
        "max_stride",
        "optimizer",
        "learning_rate",
        "ensure_grayscale",
        "cache_images",
    ],
)
def test_a_pose_head_writes_each_setting_it_is_given(
    backbone: SleapBackbone,
    settings: SleapTrainSettings,
    expected: dict[str, JsonValue],
) -> None:
    """A pose head writes a setting only when it is set, and none of the
    embedding's evaluation or initialization keys."""
    values = _settings_document("centroid", backbone, settings)
    assert {key: values.get(key) for key in expected} == expected
    assert "model_config.init_weights" not in values
    assert "trainer_config.eval" not in values
    assert "trainer_config.model_ckpt" not in values


@pytest.mark.parametrize(
    ("backbone", "size", "weights"),
    [
        ("convnext", "tiny", "ConvNeXt_Tiny_Weights"),
        ("convnext", "small", "ConvNeXt_Small_Weights"),
        ("convnext", "base", "ConvNeXt_Base_Weights"),
        ("convnext", "large", "ConvNeXt_Large_Weights"),
        ("swint", "tiny", "Swin_T_Weights"),
        ("swint", "small", "Swin_S_Weights"),
        ("swint", "base", "Swin_B_Weights"),
    ],
)
def test_the_imagenet_weights_match_the_backbone_size(
    backbone: SleapBackbone, size: SleapBackboneSize, weights: str
) -> None:
    values = _settings_document(
        "centroid",
        backbone,
        SleapTrainSettings(backbone_size=size, pretrained_backbone=True),
    )
    block = f"model_config.backbone_config.{backbone}"
    assert values[f"{block}.model_type"] == size
    assert values[f"{block}.pre_trained_weights"] == weights


@pytest.mark.parametrize(
    ("head", "backbone", "fields", "named"),
    [
        ("centroid", "unet", {"backbone_size": "tiny"}, "backbone_size"),
        ("embedding", "unet", {"backbone_size": "tiny"}, "backbone_size"),
        ("centroid", "unet", {"pretrained_backbone": True}, "pretrained_backbone"),
        ("centroid", "unet", {"stem_stride": 2}, "stem_stride"),
        ("embedding", "unet", {"stem_stride": 2}, "stem_stride"),
        ("centroid", "swint", {"backbone_size": "large"}, "backbone_size"),
        ("centroid", "convnext", {"max_stride": 64}, "max_stride"),
        ("embedding", "convnext", {"max_stride": 32}, "max_stride"),
        ("embedding", "swint", {"stem_stride": 2, "max_stride": 64}, "max_stride"),
        (
            "centroid",
            "convnext",
            {"pretrained_backbone": True, "ensure_grayscale": True},
            "pretrained_backbone.*ensure_grayscale",
        ),
        (
            "centered_instance",
            "swint",
            {"pretrained_backbone": True, "ensure_grayscale": True},
            "pretrained_backbone.*ensure_grayscale",
        ),
    ],
)
def test_a_setting_sleap_nn_refuses_is_refused_at_submit(
    head: SleapHead, backbone: SleapBackbone, fields: dict[str, object], named: str
) -> None:
    """Each combination that sleap-nn refuses or replaces fails validation.

    sleap-nn sets a convnext or swint max_stride to 16 times the stem stride,
    so any other max_stride would be written and not trained. A pose head
    with ImageNet weights trains on RGB whatever ensure_grayscale says.
    """
    with pytest.raises(ValidationError, match=named):
        _ = TrainSleapParams.model_validate(
            {"labels": "labels.slp", "head": head, "backbone": backbone, **fields}
        )


@pytest.mark.parametrize(
    ("fields", "named"),
    [
        ({"batch_size": 0}, "batch_size"),
        ({"stem_stride": 0}, "stem_stride"),
        ({"max_stride": 0}, "max_stride"),
        ({"learning_rate": 0.0}, "learning_rate"),
        ({"optimizer": "SGD"}, "optimizer"),
    ],
)
def test_an_out_of_range_setting_is_refused_at_submit(
    fields: dict[str, object], named: str
) -> None:
    with pytest.raises(ValidationError, match=named):
        _ = TrainSleapParams.model_validate({"labels": "labels.slp", **fields})


def test_a_settings_combination_is_refused_by_the_document_builder() -> None:
    with pytest.raises(ValueError, match="stem_stride"):
        _ = _settings_document("centroid", "unet", SleapTrainSettings(stem_stride=2))


def test_an_embedding_run_writes_the_measured_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds, labels, seen = _embedding_setup(tmp_path, monkeypatch)
    params = {**_embedding_params(labels), "backbone": "convnext"}
    _ = run_op(ds, "train-sleap", params)
    values = dotted_values(_written_config(seen[0]))
    assert {key: values.get(key) for key in _EMBEDDING_TRIAL_VALUES} == (
        _EMBEDDING_TRIAL_VALUES
    )


def test_a_set_field_reaches_the_written_document(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds, labels, seen = _embedding_setup(tmp_path, monkeypatch)
    params = {**_embedding_params(labels), "batch_size": 32, "learning_rate": 5e-5}
    _ = run_op(ds, "train-sleap", params)
    values = dotted_values(_written_config(seen[0]))
    assert values["trainer_config.train_data_loader.batch_size"] == 32
    assert values["trainer_config.optimizer.lr"] == 5e-5


def test_an_override_of_a_written_setting_is_assigned_bare(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """sleap_overrides still apply last, over the written settings."""
    ds, labels, seen = _embedding_setup(tmp_path, monkeypatch)
    _ = run_op(
        ds,
        "train-sleap",
        _embedding_params(labels, {"trainer_config.train_data_loader.batch_size": 8}),
    )
    assert "trainer_config.train_data_loader.batch_size=8" in _assignments(seen[0])


def test_the_settings_enter_the_run_id(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds", save=False)
    labels = tmp_path / "labels.slp"
    _ = labels.write_bytes(b"slp")
    op = TrainSleapOp()

    def run_id(**fields: object) -> str:
        params = TrainSleapParams.model_validate({"labels": str(labels), **fields})
        return op.plan_identity(ds, params, ds.resolve_scope(None)).run_id

    assert run_id() != run_id(batch_size=8)


# --- embedding refusals -----------------------------------------------------

_OBJECTIVE: Final = f"{_EMBEDDING_LEAF}.objective"
_DECLARATION: Final = "data_config.identity.track_names_are_global"
_PREPARE_TRAINING_DATA: Final = "prepare-training-data"


def _preparation(
    ds: Dataset, kind: str, *, track_by: str = "", validation: bool = True
) -> str:
    """Record a *kind* run that wrote ``labels.slp``, and return its run id.

    The row holds the cells that train-sleap reads: where the labels are and,
    for a ``prepare-training-data`` run, how its tracks were named.
    """
    run_id = f"{kind}.0.1-0123456789"
    root = model_run_root(ds, kind, run_id)
    root.mkdir(parents=True)
    labels = root / "labels.slp"
    _ = labels.write_bytes(b"slp")
    if validation:
        _ = (root / "labels.val.slp").write_bytes(b"validation slp")
    _ = model_index_path(ds, kind).write_text(
        "run_id,artifact_path,track_by\n"
        f"{run_id},{ds.relative_to_root(labels)},{track_by}\n"
    )
    return run_id


def _embedding_setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str = PREPARE_EMBEDDING_DATA_KIND,
    *,
    track_by: str = "",
    validation: bool = True,
) -> tuple[Dataset, str, list[list[str]]]:
    """A dataset holding a *kind* preparation, and a faked embedding trainer.

    Returns:
        The dataset, the preparation's run id, and the trainer's argv list.
    """
    ds = make_dataset(tmp_path / "ds", save=False)
    _point_at_sleap(tmp_path, monkeypatch)
    seen = _fake_trainer(monkeypatch, head="embedding")
    run_id = _preparation(ds, kind, track_by=track_by, validation=validation)
    return ds, run_id, seen


def _embedding_params(
    labels: str, overrides: dict[str, JsonValue] | None = None, device: str = "auto"
) -> dict[str, object]:
    return {
        "labels": labels,
        "max_epochs": 1,
        "head": "embedding",
        "sleap_overrides": overrides,
        "device": device,
    }


def _assert_refused(
    ds: Dataset,
    params: dict[str, object],
    seen: list[list[str]],
    match: str,
    *,
    by_plan: bool = True,
) -> None:
    """Assert that the run refuses *params* before any claim, and the plan too.

    A refusal leaves no run root and never reaches the trainer. *by_plan* says
    whether the refusal needs no probe, which ``plan_identity`` raises as well.
    """
    with pytest.raises(ValueError, match=match):
        _ = run_op(ds, "train-sleap", dict(params))
    assert seen == [], "the trainer must not run"
    kind_root = model_index_path(ds, "train-sleap").parent
    assert not kind_root.exists() or not any(kind_root.iterdir())
    if by_plan:
        with pytest.raises(TrainSleapInvalid, match=match):
            _ = TrainSleapOp().plan_identity(
                ds, TrainSleapParams.model_validate(params), ds.resolve_scope(None)
            )


def test_an_older_sleap_nn_is_refused_for_an_embedding_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds, labels, seen = _embedding_setup(tmp_path, monkeypatch)
    _ = _fake_probe(monkeypatch, sleap_nn_version="0.3.4")
    _assert_refused(
        ds, _embedding_params(labels), seen, r"'0\.3\.4'.*0\.3\.5", by_plan=False
    )


def test_a_pose_run_takes_no_release_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds, labels, seen = _labels_and_fake_trainer(tmp_path, monkeypatch)
    _ = _fake_probe(monkeypatch, sleap_nn_version="0.3.1")
    _ = run_op(ds, "train-sleap", {"labels": str(labels), "max_epochs": 1})
    assert len(seen) == 1


def test_global_id_without_the_declaration_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds, labels, seen = _embedding_setup(
        tmp_path, monkeypatch, _PREPARE_TRAINING_DATA, track_by="alias"
    )
    _assert_refused(
        ds,
        _embedding_params(labels),
        seen,
        f"{_DECLARATION}.*one individual in every video",
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {f"{_OBJECTIVE}.positives.scope": "global_id"},
        {f"++{_OBJECTIVE}.positives.scope": "global_id"},
        {f"+{_OBJECTIVE}.positives": {"scope": "global_id"}},
        {f"{_OBJECTIVE}.positives": {"scope": "global_id"}},
        {_OBJECTIVE: {"positives": {"scope": "global_id"}}},
        {f"~{_OBJECTIVE}.positives.scope": None},
        {f"~{_OBJECTIVE}.positives": None},
    ],
    ids=["leaf", "++leaf", "+parent", "parent", "objective", "~leaf", "~parent"],
)
def test_every_spelling_of_global_id_needs_the_declaration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, overrides: dict[str, JsonValue]
) -> None:
    """The check reads the document after the overrides, however they reach it.

    A deleted scope takes sleap-nn's default, ``global_id``.
    """
    ds, labels, seen = _embedding_setup(tmp_path, monkeypatch)
    _assert_refused(ds, _embedding_params(labels, overrides), seen, _DECLARATION)


@pytest.mark.parametrize(
    "overrides",
    [
        {f"~{_DECLARATION}": None},
        {"~data_config.identity": None},
        {"data_config.identity": {"track_names_are_global": False}},
    ],
    ids=["~leaf", "~parent", "parent"],
)
def test_a_withdrawn_declaration_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, overrides: dict[str, JsonValue]
) -> None:
    ds, labels, seen = _embedding_setup(
        tmp_path, monkeypatch, _PREPARE_TRAINING_DATA, track_by="alias"
    )
    _assert_refused(ds, _embedding_params(labels, overrides), seen, _DECLARATION)


@pytest.mark.parametrize(
    "overrides",
    [
        {_DECLARATION: True},
        {f"++{_DECLARATION}": True},
        {"data_config.identity": {"track_names_are_global": True}},
        {"+data_config.identity": {"track_names_are_global": True}},
    ],
    ids=["leaf", "++leaf", "parent", "+parent"],
)
def test_global_id_trains_with_the_declaration_and_an_alias_preparation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, overrides: dict[str, JsonValue]
) -> None:
    ds, labels, seen = _embedding_setup(
        tmp_path, monkeypatch, _PREPARE_TRAINING_DATA, track_by="alias"
    )
    _ = run_op(ds, "train-sleap", _embedding_params(labels, overrides))
    assert len(seen) == 1


def test_global_id_on_a_preparation_not_named_by_alias_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The run reads a blank ``track_by`` cell as "not alias".

    The plan cannot: a recipe that runs an older preparation again fills the
    cell before training starts. It defers the step under *require_data*, and
    without it leaves the check to the run. The refusal names both readings of
    the blank cell and the run that fills it.
    """
    ds, labels, seen = _embedding_setup(tmp_path, monkeypatch, _PREPARE_TRAINING_DATA)
    params = _embedding_params(labels, {_DECLARATION: True})
    _assert_refused(
        ds,
        params,
        seen,
        r"records no track_by.*before.*recorded.*again with the same params",
        by_plan=False,
    )
    validated = TrainSleapParams.model_validate(params)
    with pytest.raises(IdentityDeferred, match="track_by"):
        _ = TrainSleapOp().plan_identity(ds, validated, ds.resolve_scope(None))
    planned = TrainSleapOp().plan_identity(
        ds, validated, ds.resolve_scope(None), require_data=False
    )
    assert planned.run_id


def test_global_id_on_a_labels_path_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """mosaic cannot read how the tracks of a ``.slp`` were named."""
    ds, _, seen = _embedding_setup(tmp_path, monkeypatch)
    labels = _labels_with_validation(tmp_path)
    _assert_refused(
        ds,
        _embedding_params(str(labels), {_DECLARATION: True}),
        seen,
        'track_by="alias"',
    )


def test_tracklet_on_labels_no_embedding_preparation_cut_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds, labels, seen = _embedding_setup(
        tmp_path, monkeypatch, _PREPARE_TRAINING_DATA, track_by="alias"
    )
    overrides: dict[str, JsonValue] = {
        f"{_OBJECTIVE}.positives.scope": "tracklet",
        f"{_OBJECTIVE}.negatives.restrict_same_video": True,
    }
    _assert_refused(
        ds,
        _embedding_params(labels, overrides),
        seen,
        "one SLEAP video per image size",
    )


def test_tracklet_across_videos_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds, labels, seen = _embedding_setup(tmp_path, monkeypatch)
    overrides: dict[str, JsonValue] = {
        f"{_OBJECTIVE}.negatives.restrict_same_video": False
    }
    _assert_refused(ds, _embedding_params(labels, overrides), seen, "another window")


def test_tracklet_trains_on_an_embedding_preparation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds, labels, seen = _embedding_setup(tmp_path, monkeypatch)
    _ = run_op(ds, "train-sleap", _embedding_params(labels))
    assert len(seen) == 1


def test_aug_view_trains_on_labels_without_tracks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two views of one crop need neither track names nor a second track."""
    ds, _, seen = _embedding_setup(tmp_path, monkeypatch)
    _ = _fake_probe(monkeypatch, n_tracks=0)
    labels = _labels_with_validation(tmp_path)
    overrides: dict[str, JsonValue] = {f"{_OBJECTIVE}.positives.scope": "aug_view"}
    _ = run_op(ds, "train-sleap", _embedding_params(str(labels), overrides))
    assert len(seen) == 1


def test_one_track_is_refused_for_a_contrastive_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds, labels, seen = _embedding_setup(tmp_path, monkeypatch)
    _ = _fake_probe(monkeypatch, n_tracks=1)
    _assert_refused(ds, _embedding_params(labels), seen, "1 track", by_plan=False)


@pytest.mark.parametrize(
    ("scope", "n_tracks"), [("global_id", 0), ("tracklet", 1), ("global_id", 1)]
)
def test_the_track_count_check(scope: EmbeddingPositiveScope, n_tracks: int) -> None:
    probe = SleapProbeResponse(has_sleap_nn=True, n_tracks=n_tracks)
    with pytest.raises(ValueError, match=f"{n_tracks} track.*two individuals"):
        require_embedding_labels(
            probe, scope=scope, labels_kind=PREPARE_EMBEDDING_DATA_KIND
        )


def test_aug_view_needs_no_tracks() -> None:
    probe = SleapProbeResponse(has_sleap_nn=True, n_tracks=0)
    require_embedding_labels(probe, scope="aug_view", labels_kind="")


def test_an_embedding_run_without_a_validation_file_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A random split puts frames of one track on both sides."""
    ds, labels, seen = _embedding_setup(tmp_path, monkeypatch, validation=False)
    _assert_refused(ds, _embedding_params(labels), seen, "labels.val.slp")


@pytest.mark.parametrize(
    ("overrides", "device"),
    [
        (None, "0,1"),
        ({"++trainer_config.trainer_device_indices": [0, 1]}, "auto"),
        ({"trainer_config.trainer_devices": 2}, "auto"),
        ({"trainer_config.trainer_devices": "auto"}, "auto"),
    ],
    ids=["device", "indices", "devices", "devices-auto"],
)
def test_an_embedding_run_on_more_than_one_device_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, JsonValue] | None,
    device: str,
) -> None:
    ds, labels, seen = _embedding_setup(tmp_path, monkeypatch)
    _assert_refused(
        ds, _embedding_params(labels, overrides, device), seen, "one device"
    )


_SELECT_METRIC: Final = "++trainer_config.eval.select_metric"
_MONITOR: Final = "trainer_config.model_ckpt.monitor"
_MODE: Final = "trainer_config.model_ckpt.mode"


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({_MONITOR: "eval/val/mAP"}, "'eval/val/rank1' with mode 'max'"),
        ({_SELECT_METRIC: "mAP", _MONITOR: "eval/val/auc"}, "'eval/val/mAP'"),
        ({_SELECT_METRIC: "eer", _MODE: "max"}, "'eval/val/eer' with mode 'min'"),
        ({_SELECT_METRIC: "top5"}, "one of rank1, mAP, auc, knn_acc, eer"),
    ],
    ids=["monitor", "monitor-off-select-metric", "mode", "unknown-metric"],
)
def test_a_checkpoint_kept_off_the_selection_metric_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, JsonValue],
    match: str,
) -> None:
    """sleap-nn early-stops on select_metric and keeps the monitor's best."""
    ds, labels, seen = _embedding_setup(tmp_path, monkeypatch)
    _assert_refused(ds, _embedding_params(labels, overrides), seen, match)


@pytest.mark.parametrize(
    "overrides",
    [
        {_SELECT_METRIC: "mAP", _MONITOR: "eval/val/mAP"},
        {_SELECT_METRIC: "eer", _MONITOR: "eval/val/eer", _MODE: "min"},
        {_SELECT_METRIC: "auc", _MONITOR: "val/loss"},
    ],
    ids=["max", "min", "monitor-follows-select-metric"],
)
def test_a_checkpoint_kept_on_the_selection_metric_trains(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, overrides: dict[str, JsonValue]
) -> None:
    ds, labels, seen = _embedding_setup(tmp_path, monkeypatch)
    _ = run_op(ds, "train-sleap", _embedding_params(labels, overrides))
    assert len(seen) == 1


@pytest.mark.parametrize(
    ("overrides", "monitor", "mode"),
    [
        (None, "eval/val/rank1", "max"),
        ({_SELECT_METRIC: "mAP"}, "eval/val/mAP", "max"),
        ({_SELECT_METRIC: "eer"}, "eval/val/eer", "min"),
    ],
    ids=["default", "max", "min"],
)
def test_the_written_checkpoint_monitor_follows_the_selection_metric(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, JsonValue] | None,
    monitor: str,
    mode: str,
) -> None:
    """A select_metric set through sleap_overrides also keeps the checkpoint."""
    ds, labels, seen = _embedding_setup(tmp_path, monkeypatch)
    _ = run_op(ds, "train-sleap", _embedding_params(labels, overrides))
    values = dotted_values(_written_config(seen[0]))
    assert values["trainer_config.model_ckpt.monitor"] == monitor
    assert values["trainer_config.model_ckpt.mode"] == mode


def test_an_embedding_run_on_one_cuda_index_trains(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds, labels, seen = _embedding_setup(tmp_path, monkeypatch)
    _ = run_op(ds, "train-sleap", _embedding_params(labels, device="1"))
    assert len(seen) == 1


@pytest.mark.parametrize(
    "overrides",
    [
        {"+trainer_config.seed": 1},
        {"~trainer_config.profiler": None},
        {"~trainer_config.seed": 3},
        {"~trainer_config.trainer_devices": None},
    ],
    ids=["+present", "~absent", "~other-value", "~devices"],
)
def test_an_override_that_hydra_rejects_is_refused_before_the_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, overrides: dict[str, JsonValue]
) -> None:
    ds, labels, seen = _embedding_setup(tmp_path, monkeypatch)
    _assert_refused(ds, _embedding_params(labels, overrides), seen, "Override")


def test_a_pose_run_refuses_an_override_that_hydra_rejects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds, labels, seen = _labels_and_fake_trainer(tmp_path, monkeypatch)
    params: dict[str, object] = {
        "labels": str(labels),
        "max_epochs": 1,
        "sleap_overrides": {"+trainer_config.seed": 1},
    }
    _assert_refused(ds, params, seen, r"\+trainer_config\.seed")


def test_a_finished_run_records_the_sleap_nn_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Provenance in ``training.json``, which no identifier reads."""
    ds, labels, _ = _labels_and_fake_trainer(tmp_path, monkeypatch)
    _ = _fake_probe(monkeypatch, sleap_nn_version="0.3.7")

    run_id = run_op(ds, "train-sleap", {"labels": str(labels), "max_epochs": 1})

    document: object = json.loads(
        (
            model_run_root(ds, "train-sleap", run_id) / TRAINING_PROVENANCE_FILENAME
        ).read_text()
    )
    assert is_str_mapping(document)
    assert document["tool_version"] == "0.3.7"


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


def _registered_model_directory(ds: Dataset, run_id: str) -> Path:
    """The directory that the run's index row registers as its artifact."""
    index = trained_model_index(model_index_path(ds, "train-sleap")).read()
    (artifact_path,) = index.loc[index["run_id"] == run_id, "artifact_path"]
    return ds.resolve_path(str(artifact_path))


def test_overwrite_trains_into_a_fresh_model_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The directory a run registers holds that run's model, not the last one's.

    sleap-nn writes ``model-1`` beside a ``model`` that exists, and the row
    would then register the old directory.
    """
    ds, labels, seen = _labels_and_fake_trainer(tmp_path, monkeypatch)
    params = {"labels": str(labels), "max_epochs": 1}
    run_id = run_op(ds, "train-sleap", dict(params))
    model = _registered_model_directory(ds, run_id)
    _ = (model / "from-the-first-run.txt").write_text("stale")

    assert run_op(ds, "train-sleap", dict(params), overwrite=True) == run_id

    assert len(seen) == 2
    assert _registered_model_directory(ds, run_id) == model
    assert not (model / "from-the-first-run.txt").exists()
    assert not (model.parent / "model-1").exists()
    assert (model / "best.ckpt").exists()


def test_a_retry_after_a_failed_run_trains_into_a_fresh_model_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run that died after writing part of its model leaves nothing behind."""
    ds, labels, _ = _labels_and_fake_trainer(tmp_path, monkeypatch)

    def dies_midway(argv: Sequence[str], **kw: object) -> tuple[str, str, int]:
        config_dir = Path(argv[argv.index("--config-dir") + 1])
        partial = config_dir / MODEL_DIRECTORY
        partial.mkdir()
        _ = (partial / "partial.ckpt").write_text("half")
        return ("", "out of memory", 1)

    monkeypatch.setattr(training_module, "run_supervised", dies_midway)
    params = {"labels": str(labels), "max_epochs": 1}
    with pytest.raises(SleapError):
        _ = run_op(ds, "train-sleap", dict(params))

    seen = _fake_trainer(monkeypatch)
    run_id = run_op(ds, "train-sleap", dict(params))

    assert len(seen) == 1
    model = _registered_model_directory(ds, run_id)
    assert (model / "best.ckpt").exists()
    assert not (model / "partial.ckpt").exists()
    assert not (model.parent / "model-1").exists()


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
    """``sleap-nn train`` replaces the ``sleap-nn-train`` script, and takes its flags.

    ``sleap-nn-train`` was a script of the ``sleap`` distribution, which sleap
    1.6.5 removed. sleap-nn's own ``sleap-nn`` script takes ``train
    --config-dir`` and ``--config-name`` from sleap-nn 0.3.1, so running it does
    not drop the older environments.
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


@pytest.mark.parametrize(
    ("value", "spelled"),
    [
        (None, "null"),
        (True, "true"),
        (0.5, "0.5"),
        ("simple", "simple"),
        ("/data/a.slp", "/data/a.slp"),
        ("two words", "'two words'"),
        ("0.5", "'0.5'"),
        ("null", "'null'"),
        ("it's", "'it\\'s'"),
        ("${run}", "'\\${run}'"),
        (["a b", 1, None], "['a b',1,null]"),
        ({"scope": "global_id", "max_gap": None}, "{scope:global_id,max_gap:null}"),
    ],
)
def test_an_override_value_is_spelled_in_hydra_grammar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: JsonValue, spelled: str
) -> None:
    """Hydra parses each spelling back to *value*; a Python repr it does not."""
    argv = _run_and_capture(
        tmp_path,
        monkeypatch,
        {"sleap_overrides": {"+trainer_config.profiler": value}},
    )
    assert f"+trainer_config.profiler={spelled}" in _assignments(argv)


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

    checkpoint = str(
        model_run_root(ds, "train-sleap", base) / MODEL_DIRECTORY / "best.ckpt"
    )
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


def test_an_embedding_run_fine_tunes_from_an_embedding_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An embedding base matches, because it is a training of the embedding head."""
    ds, labels, seen = _embedding_setup(tmp_path, monkeypatch)
    base = run_op(ds, "train-sleap", _embedding_params(labels))
    seen.clear()

    run_id = run_op(
        ds, "train-sleap", {**_embedding_params(labels), "base_model": base}
    )

    assert run_id != base
    assert len(seen) == 1
    checkpoint = str(
        model_run_root(ds, "train-sleap", base) / MODEL_DIRECTORY / "best.ckpt"
    )
    model_config = _written_section(seen[0], "model_config")
    assert model_config["pretrained_backbone_weights"] == checkpoint
    assert model_config["pretrained_head_weights"] == checkpoint


def test_an_embedding_run_refuses_a_centroid_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only an embedding model initializes the embedding head."""
    ds, labels, seen = _embedding_setup(tmp_path, monkeypatch)
    base = write_sleap_model(tmp_path / "centroid", head="centroid", merged=True)

    with pytest.raises(ValueError, match="is a centroid model.*embedding run"):
        _ = run_op(
            ds,
            "train-sleap",
            {**_embedding_params(labels), "base_model": str(base)},
        )

    assert seen == [], "the trainer must not run"
    kind_root = model_index_path(ds, "train-sleap").parent
    assert not kind_root.exists() or not any(kind_root.iterdir())


@pytest.mark.parametrize("head", ["centroid", "centered_instance", "bottomup"])
def test_an_embedding_base_is_refused_before_the_run_root_is_made(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, head: str
) -> None:
    """An embedding model's head is no head that train-sleap trains.

    Its directory holds the merged config that sleap-nn saves, which names every
    other head too. The refusal leaves no run root and no trainer call.
    """
    ds, labels, seen = _labels_and_fake_trainer(tmp_path, monkeypatch, head=head)
    base = write_sleap_model(tmp_path / "reid", head="embedding", merged=True)

    with pytest.raises(ValueError, match=f"is an embedding model.*{head} run"):
        _ = run_op(
            ds,
            "train-sleap",
            {
                "labels": str(labels),
                "max_epochs": 1,
                "head": head,
                "base_model": str(base),
            },
        )

    assert seen == [], "the trainer must not run"
    kind_root = model_index_path(ds, "train-sleap").parent
    assert not kind_root.exists() or not any(kind_root.iterdir())


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
    checkpoint = str(
        model_run_root(ds, "train-sleap", base) / MODEL_DIRECTORY / "best.ckpt"
    )
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


def test_the_recorded_epochs_are_those_the_training_log_holds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cap is a limit, and a run that stopped early trained fewer epochs."""
    ds, labels, _ = _labels_and_fake_trainer(
        tmp_path, monkeypatch, writes_training_log=True
    )

    _ = run_op(ds, "train-sleap", {"labels": str(labels), "max_epochs": 5})
    row = trained_model_index(model_index_path(ds, "train-sleap")).read().iloc[0]
    assert int(row["n_epochs"]) == 1, "one epoch logged under a cap of five"


def test_a_run_that_logged_no_epoch_records_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds, labels, _ = _labels_and_fake_trainer(tmp_path, monkeypatch)

    _ = run_op(ds, "train-sleap", {"labels": str(labels), "max_epochs": 5})
    row = trained_model_index(model_index_path(ds, "train-sleap")).read().iloc[0]
    assert int(row["n_epochs"]) == 0


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
