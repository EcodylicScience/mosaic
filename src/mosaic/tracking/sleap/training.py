"""Training a SLEAP model from a mosaic-owned labels file.

``sleap-nn-train`` is configuration-driven, not flag-driven: it takes a directory
and the name of a YAML inside it, plus Hydra-style ``key=value`` overrides. So
training from mosaic means *writing* that YAML, which is what this module does --
mosaic owns the config, and the config is what makes a run reproducible.

**A generated config, not a template the user maintains.** The alternative is
asking for a config path, and then a run identifier means nothing: two runs with
the same mosaic parameters and different YAML on disk would share a name. What is
declared here reaches the identifier; what is not, cannot be varied.

**Everything sleap-nn can express is still reachable.** The typed fields are the
few decisions mosaic has an opinion about -- which head, which backbone, how long.
The rest is ``overrides``, passed through as Hydra assignments and folded into
identity, because a run trained with a different learning rate is a different
model whether or not mosaic has a field for it.

Requires:
    ``sleap-nn-train``, from a SLEAP install. It is a sibling of ``sleap-track``,
    so the environment variables that already point mosaic at SLEAP for tracking
    serve training too -- see :data:`mosaic.tracking.sleap.run.SLEAP_ENV`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Final, Literal, TypedDict

import yaml

from mosaic.core.pipeline.subprocess_util import run_supervised
from mosaic.core.pipeline.types import JsonValue
from mosaic.tracking.common.toolenv import subprocess_env, tool_invocation
from mosaic.tracking.sleap.run import SLEAP_ENV, SleapError

logger = logging.getLogger(__name__)

__all__ = [
    "SleapBackbone",
    "SleapHead",
    "SleapTrainConfig",
    "sleap_train_config",
    "train_sleap",
]

_SLEAP_NN_TRAIN: str = "sleap-nn-train"

SleapHead = Literal[
    "single_instance",
    "centroid",
    "centered_instance",
    "bottomup",
    "multi_class_bottomup",
    "multi_class_topdown",
]
"""Which task the network is trained for.

``centroid`` and ``centered_instance`` are the two halves of a top-down model and
are trained separately; the pair is what inference is later handed, in that
order. The ``multi_class_`` heads add identity classification.
"""

SleapBackbone = Literal["unet", "convnext", "swint"]
"""The feature extractor. Orthogonal to the head, which is why it is its own
field rather than folded into a single model name."""

_HEAD_SECTIONS: Final[Mapping[SleapHead, tuple[str, ...]]] = {
    "single_instance": ("confmaps",),
    "centroid": ("confmaps",),
    "centered_instance": ("confmaps",),
    "bottomup": ("confmaps", "pafs"),
    "multi_class_bottomup": ("confmaps", "class_maps"),
    "multi_class_topdown": ("confmaps", "class_vectors"),
}
"""The output sections each head owns, which must be present for sleap-nn to fill.

An empty head block is **not** the same as a defaulted one. sleap-nn merges what
is written here over its own structured config, where every section defaults to
``None``; it then walks the head's sections looking for ``part_names`` and
``edges`` to fill in from the labels file, and a section left at ``None`` has no
keys to walk. Writing ``{head: {}}`` therefore reaches
``AttributeError: 'NoneType' object has no attribute 'keys'`` inside
``model_trainer._setup_head_config`` before training starts. Naming each section
as an empty mapping instantiates it at its own defaults, which is what leaves
``part_names`` and ``edges`` present-and-``None`` for sleap-nn to complete.

Listed here rather than derived from sleap-nn's classes because mosaic never
imports it: the tool lives in its own environment, and this file is the record of
what mosaic asks of it.
"""


class _PreprocessingConfig(TypedDict):
    ensure_rgb: bool
    ensure_grayscale: bool


class _DataConfig(TypedDict):
    train_labels_path: list[str]
    validation_fraction: float
    provider: str
    preprocessing: _PreprocessingConfig


class _ModelConfig(TypedDict):
    backbone_config: dict[str, dict[str, object]]
    head_configs: dict[str, dict[str, object]]


class _TrainerConfig(TypedDict):
    max_epochs: int
    seed: int
    save_ckpt: bool
    ckpt_dir: str
    run_name: str


class SleapTrainConfig(TypedDict):
    """The sleap-nn configuration document mosaic writes.

    Typed rather than a bare mapping so the sections can be read back and
    asserted on -- a config is the whole of what a run was, and a test that
    cannot index it can only check that a file exists.
    """

    data_config: _DataConfig
    model_config: _ModelConfig
    trainer_config: _TrainerConfig


def sleap_train_config(
    labels_path: Path,
    run_root: Path,
    *,
    head: SleapHead,
    backbone: SleapBackbone,
    max_epochs: int,
    seed: int,
    validation_fraction: float,
    run_name: str,
) -> SleapTrainConfig:
    """The training configuration, as the mapping that is written to YAML.

    Built rather than templated so it can be asserted on without running
    anything, which is what the tests do.
    """
    return {
        "data_config": {
            "train_labels_path": [str(labels_path)],
            "validation_fraction": validation_fraction,
            "provider": "LabelsReader",
            # Written at sleap-nn's own defaults, and only because sleap-nn reads
            # these two off the *unmerged* document. ``run_training`` completes the
            # config through ``verify_training_cfg`` and keeps the result on
            # ``trainer.config``, but its post-training evaluation reaches back to
            # the raw ``config`` for ``ensure_rgb`` / ``ensure_grayscale`` -- so a
            # config that omits them trains to completion, writes its checkpoint,
            # and then dies on the evaluation pass with
            # ``ConfigAttributeError: Key 'preprocessing' is not in struct``.
            # Stating them changes no behaviour and keeps the run from ending on an
            # error after the model is already on disk.
            "preprocessing": {"ensure_rgb": False, "ensure_grayscale": False},
        },
        "model_config": {
            # The backbone takes an empty block: its fields carry real defaults,
            # so merging one over sleap-nn's structured config yields those. A
            # head does not -- see ``_HEAD_SECTIONS``.
            "backbone_config": {backbone: {}},
            "head_configs": {head: {section: {} for section in _HEAD_SECTIONS[head]}},
        },
        "trainer_config": {
            "max_epochs": max_epochs,
            "seed": seed,
            "save_ckpt": True,
            "ckpt_dir": str(run_root),
            "run_name": run_name,
        },
    }


def train_sleap(
    labels_path: str | Path,
    run_root: str | Path,
    *,
    head: SleapHead = "centered_instance",
    backbone: SleapBackbone = "unet",
    max_epochs: int = 200,
    seed: int = 42,
    validation_fraction: float = 0.1,
    run_name: str = "model",
    overrides: Mapping[str, JsonValue] | None = None,
    sleap_conda_env: str | None = None,
    sleap_bin: str | Path | None = None,
    idle_timeout: float = 1800,
    max_runtime: float | None = None,
    cancel_check: Callable[[], bool] | None = None,
    on_output: Callable[[str], None] | None = None,
) -> Path:
    """Train a SLEAP model and return the directory it produced.

    Args:
        labels_path: A ``.slp`` to train on.
        run_root: Where the model directory is written. The configuration is
            written here too, so a finished run carries the exact inputs it ran
            with beside its weights.
        head: Which task to train. See :data:`SleapHead`.
        backbone: Which feature extractor. See :data:`SleapBackbone`.
        max_epochs: Training length.
        seed: Seeds sleap-nn's own initialisation.
        validation_fraction: Held out from *labels_path* when no separate
            validation file is given.
        run_name: The directory name under *run_root*.
        overrides: Hydra assignments applied on top, for anything sleap-nn
            exposes that this signature does not.
        sleap_conda_env: Run in this conda env, overriding the environment.
        sleap_bin: A SLEAP console script naming the install, overriding the
            environment.
        idle_timeout: Kill the subprocess after this long with no output.
            Generous by default -- an epoch on a large set is slow, and the
            watchdog must not mistake slow for dead.
        max_runtime: Optional absolute ceiling.

    Returns:
        The model directory, which is what
        :data:`mosaic.tracking.model_refs.MODEL_KINDS` describes as a ``sleap``
        artifact and what inference is later handed.

    Raises:
        FileNotFoundError: *labels_path* does not exist, or training exited
            zero without producing a model directory.
        SleapNotFoundError: No SLEAP install could be located.
        SleapError: Training exited non-zero.
    """
    labels_path = Path(labels_path)
    if not labels_path.exists():
        raise FileNotFoundError(f"SLEAP labels file does not exist: {labels_path}")
    run_root = Path(run_root)
    run_root.mkdir(parents=True, exist_ok=True)

    config = sleap_train_config(
        labels_path,
        run_root,
        head=head,
        backbone=backbone,
        max_epochs=max_epochs,
        seed=seed,
        validation_fraction=validation_fraction,
        run_name=run_name,
    )
    config_path = run_root / "config.yaml"
    _ = config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    args = [
        "--config-dir",
        str(run_root),
        "--config-name",
        config_path.stem,
        *(f"{key}={value}" for key, value in (overrides or {}).items()),
    ]
    invocation = tool_invocation(
        SLEAP_ENV,
        executable=_SLEAP_NN_TRAIN,
        conda_env=sleap_conda_env,
        bin_path=sleap_bin,
    )
    cmd = [*invocation, *args]
    logger.info("Running: %s", " ".join(cmd))

    stdout, stderr, returncode = run_supervised(
        cmd,
        env=subprocess_env(),
        cancel_check=cancel_check,
        timeout=max_runtime,
        idle_timeout=idle_timeout,
        on_output=on_output,
    )
    if returncode != 0:
        raise SleapError(cmd, returncode, stdout, stderr)

    produced = run_root / run_name
    if not produced.is_dir():
        raise FileNotFoundError(
            f"sleap-nn-train exited cleanly but wrote no model directory at {produced}"
        )
    return produced
