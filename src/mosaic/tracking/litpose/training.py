"""Training a Lightning Pose model from a mosaic-written project directory.

Lightning Pose's console verbs write into the model directory and expose no way
to say where the run should land -- the same limitation that made inference drive
the Python API through a static snippet. Training has the same shape, so it uses
the same arrangement: a program whose only inputs are argv, run by the Lightning
Pose interpreter, so mosaic never imports ``lightning_pose`` and the heavy stack
stays out of its environment and its type checker.

**The project directory is the input, and mosaic wrote it.** Lightning Pose reads
a ``config.yaml`` beside a ``CollectedData.csv``; both come from
:func:`mosaic.tracking.litpose.labels.write_litpose_dataset`, so what a run
consumed is a thing mosaic can point at rather than a state of somebody's disk.

Not verified against a real install. Unlike SLEAP, Lightning Pose is not present
on the machine this was written on, so the invocation follows the documented
Python API and the tests patch the subprocess. The seam is one snippet.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from mosaic.core.pipeline.subprocess_util import run_supervised
from mosaic.core.pipeline.types import JsonValue
from mosaic.tracking.common.toolenv import subprocess_env, tool_invocation
from mosaic.tracking.litpose.run import LITPOSE_ENV, LitposeError

logger = logging.getLogger(__name__)

__all__ = [
    "LitposeDevicePlacement",
    "epoch_coupled_assignments",
    "litpose_device_placement",
    "train_litpose",
]

# Run by the Lightning Pose interpreter. Only argv reaches it -- the project
# directory, where the model should land, and any Hydra overrides -- so no path
# is interpolated into this source.
_TRAIN_SNIPPET: str = """
import sys
from lightning_pose.train import train
from omegaconf import OmegaConf

_base, _project, _out = sys.argv[1], sys.argv[2], sys.argv[3]
_overrides = sys.argv[4:]

# The base config carries everything Lightning Pose needs and mosaic does not
# choose; the project's own config carries the data half mosaic wrote. The
# project wins, because it describes the labels actually being trained on.
_cfg = OmegaConf.merge(
    OmegaConf.load(_base),
    OmegaConf.load(f"{_project}/config.yaml"),
    OmegaConf.from_dotlist(_overrides),
)
OmegaConf.update(_cfg, "data.data_dir", _project, force_add=True)
# Where the model lands is an argument, not a config key. `train` does
# `model_dir = Path(model_dir or os.getcwd())` and never reads `hydra.run.dir`
# -- that key is applied by Hydra's `@hydra.main` decorator, which calling
# `train(cfg)` directly bypasses. Setting it and passing nothing trained
# successfully into the *caller's working directory*, left the run root empty,
# and surfaced as "exited cleanly but wrote no model".
_model = train(_cfg, model_dir=_out)
print(f"trained into {_out}")
"""


_MILESTONE_RATIOS: Final = (0.5, 2 / 3, 5 / 6)
"""Where the learning rate drops, as fractions of the training length.

The vendored config's own ``[150, 200, 250]`` over 300 epochs, read back as
ratios so a shorter run keeps the same schedule shape instead of keeping the
same absolute epochs.
"""

_DEFAULT_CHECK_VAL_EVERY_N_EPOCH: Final = 5
"""The template's validation interval, and the ceiling for a shorter run.

A run shorter than twice this never validates, so the checkpoint callback never
fires and post-training evaluation dies with "Checkpoint file not found" after
every epoch has already been paid for.
"""

_DEFAULT_UNFREEZING_EPOCH: Final = 20
"""The template's backbone-unfreezing epoch, and the ceiling for a shorter run.

Left at 20 on a two-epoch run the backbone never unfreezes at all, so the run
completes having trained only the head.
"""


def epoch_coupled_assignments(max_epochs: int) -> dict[str, JsonValue]:
    """The configuration keys that have to move when *max_epochs* does.

    The config Lightning Pose is given is written against 300 epochs, and four
    further keys are tied to that number: ``min_epochs``, the multi-step
    learning-rate ``milestones``, ``check_val_every_n_epoch`` and
    ``unfreezing_epoch``. Scaling ``max_epochs`` by itself leaves the other
    four, and ``ModelConfig.validate`` then refuses the config outright --
    every milestone must be at most ``max_epochs`` -- or, between three and
    four epochs, accepts it and never runs validation, so no checkpoint is
    written and the run fails after paying for every epoch. Either way every
    short run fails, which is every smoke test.

    **These values never reach the run_id.** They are derived here, below
    ``Params``, and that is sound only because they are a pure function of
    *max_epochs*, which is hashed. A derivation reading anything else would let
    two runs share one identifier and produce different models.

    At the template's own 300 they reproduce the template's own values, so a
    run finished before this existed keeps training the way it did.

    Args:
        max_epochs: The training length the caller asked for.

    Returns:
        Hydra assignments, to be applied before a caller's own overrides.
    """
    epochs = sorted(
        {
            min(max(1, round(ratio * max_epochs)), max_epochs)
            for ratio in _MILESTONE_RATIOS
        }
    )
    # Copied rather than returned directly: a ``list[int]`` is not a
    # ``list[JsonValue]``, because a list is invariant in its element type.
    milestones: list[JsonValue] = list(epochs)
    return {
        # Lightning Pose's config says both bounds must be set, or neither.
        "training.min_epochs": max_epochs,
        "training.lr_scheduler_params.multisteplr.milestones": milestones,
        "training.check_val_every_n_epoch": min(
            _DEFAULT_CHECK_VAL_EVERY_N_EPOCH, max(1, max_epochs // 2)
        ),
        "training.unfreezing_epoch": min(
            _DEFAULT_UNFREEZING_EPOCH, max(0, max_epochs - 1)
        ),
    }


_CUDA_VISIBLE_DEVICES: Final = "CUDA_VISIBLE_DEVICES"
"""What selects a GPU for Lightning Pose, because nothing in its config does.

``training.num_gpus`` is a *count*, and ``lightning_pose.train`` builds its
Trainer with ``accelerator="gpu"`` and ``devices=cfg.training.num_gpus``. So
there is no config key naming device 1, and the selection has to reach the
process from outside it.
"""


@dataclass(frozen=True, slots=True)
class LitposeDevicePlacement:
    """How a device request reaches Lightning Pose, split by where it lands.

    Two halves because the question is answered in two places: *which* devices
    is an environment variable the subprocess inherits, and *how many* is a
    config key. Returned together so one parse of the request decides both.

    Attributes:
        assignments: Hydra assignments, applied before a caller's overrides.
        env_overlay: Variables laid over ``subprocess_env()``.
    """

    assignments: Mapping[str, JsonValue]
    env_overlay: Mapping[str, str]


def litpose_device_placement(device: str) -> LitposeDevicePlacement:
    """Where *device* sends training.

    Lightning Pose trains on CUDA and only on CUDA: its Trainer fixes
    ``accelerator="gpu"``, and ``num_gpus: 0`` is clamped back up to 1 with a
    deprecation warning. So the accelerator *families* the other training ops
    accept have nothing to set here, and a request for one is refused rather
    than accepted and quietly ignored.

    Args:
        device: ``auto``, ``gpu`` or empty to take whatever Lightning Pose
            finds; or a comma-separated list of CUDA indices such as ``0`` or
            ``0,1``.

    Returns:
        The assignments and the environment overlay, both empty when *device*
        leaves the choice open.

    Raises:
        ValueError: *device* is neither. Raised from a field validator too, so
            an unusable value is refused when the run is submitted rather than
            on a GPU node once it is scheduled.
    """
    if device in ("", "auto", "gpu"):
        return LitposeDevicePlacement(assignments={}, env_overlay={})
    parts = [part.strip() for part in device.split(",")]
    if parts and all(part.isdigit() for part in parts):
        return LitposeDevicePlacement(
            assignments={"training.num_gpus": len(parts)},
            env_overlay={_CUDA_VISIBLE_DEVICES: ",".join(parts)},
        )
    msg = (
        f"unusable device {device!r}: Lightning Pose trains on CUDA, fixing "
        f"its trainer's accelerator to 'gpu', so give 'auto' or a "
        f"comma-separated list of CUDA indices such as '0' or '0,1'."
    )
    raise ValueError(msg)


def train_litpose(
    project_dir: str | Path,
    run_root: str | Path,
    *,
    base_config: str | Path,
    model_type: str = "heatmap",
    backbone: str = "resnet50_animal_ap10k",
    max_epochs: int = 300,
    device: str = "auto",
    overrides: Mapping[str, JsonValue] | None = None,
    litpose_conda_env: str | None = None,
    litpose_bin: str | Path | None = None,
    idle_timeout: float = 1800,
    max_runtime: float | None = None,
    cancel_check: Callable[[], bool] | None = None,
    on_output: Callable[[str], None] | None = None,
    on_activity: Callable[[str], None] | None = None,
) -> Path:
    """Train a Lightning Pose model and return the directory it produced.

    Args:
        project_dir: A Lightning Pose project, as written by
            :func:`~mosaic.tracking.litpose.labels.write_litpose_dataset`.
        run_root: Where the model directory is written.
        model_type: ``heatmap``, ``heatmap_mhcrnn``, ``regression`` or
            ``heatmap_multiview_transformer``. Passed as a Hydra override, so a
            value Lightning Pose does not know fails there rather than here.
        backbone: The feature extractor, likewise an override. The default is
            Lightning Pose's own, a ResNet-50 pretrained on animal pose rather
            than ImageNet.
        max_epochs: Training length.
        device: Which CUDA devices train the model. See
            :func:`litpose_device_placement`.
        overrides: Further Hydra assignments, applied last.
        litpose_conda_env: Run in this conda env, overriding the environment.
        base_config: A complete Lightning Pose config to train from. Lightning
            Pose composes its config with Hydra from the file it is given and
            merges no defaults of its own, so this has to be complete rather than
            partial. Callers who have no opinion pass
            :func:`mosaic.tracking.litpose.templates.default_config_path`, which
            is what the ``train-litpose`` op does when its ``base_config`` names
            nothing. The project written by ``write_litpose_dataset`` supplies the
            ``data`` half and is merged over this, and per-call overrides over
            both.
        litpose_bin: A Lightning Pose script naming the install, overriding the
            environment.
        idle_timeout: Kill the subprocess after this long with no output.
        max_runtime: Optional absolute ceiling.
        cancel_check: Polled while the subprocess runs.
        on_output: Called with each line Lightning Pose writes to standard
            output, for a caller parsing epochs out of them.
        on_activity: Called with each line on **either** stream, for a caller
            keeping the run root's claim alive. Both are passed straight to
            ``run_supervised``; see its two parameters of the same names.

    Returns:
        The model directory, which is the ``litpose`` artifact shape:
        a ``config.yaml`` beside a checkpoint under ``tb_logs``.

    Raises:
        FileNotFoundError: *base_config* or *project_dir*'s ``config.yaml`` is
            missing, or training exited zero without producing a model directory.
        LitposeNotFoundError: No Lightning Pose install could be located.
        LitposeError: Training exited non-zero.
    """
    project_dir = Path(project_dir)
    if not (project_dir / "config.yaml").exists():
        raise FileNotFoundError(
            f"not a Lightning Pose project -- no config.yaml in {project_dir}"
        )
    base_config = Path(base_config)
    if not base_config.is_file():
        raise FileNotFoundError(
            f"no Lightning Pose base config at {base_config}. Lightning Pose "
            f"merges no defaults of its own, so training needs a complete "
            f"config: leave 'base_config' unset to use the one mosaic carries, "
            f"or point it at a config of your own. The project's own config.yaml "
            f"supplies the data half and is merged over it."
        )
    run_root = Path(run_root)
    run_root.mkdir(parents=True, exist_ok=True)

    placement = litpose_device_placement(device)
    assignments: dict[str, JsonValue] = {
        "model.model_type": model_type,
        "model.backbone": backbone,
        "training.max_epochs": max_epochs,
        **epoch_coupled_assignments(max_epochs),
        **placement.assignments,
    }
    assignments.update(overrides or {})

    invocation = tool_invocation(
        LITPOSE_ENV.placed(conda_env=litpose_conda_env, bin_path=litpose_bin),
        executable="python",
    )
    cmd = [
        *invocation,
        "-c",
        _TRAIN_SNIPPET,
        str(base_config),
        str(project_dir),
        str(run_root),
        *(f"{key}={value}" for key, value in assignments.items()),
    ]
    logger.info("Running: %s", " ".join(cmd[:4]) + " ...")

    stdout, stderr, returncode = run_supervised(
        cmd,
        env=subprocess_env(placement.env_overlay),
        cancel_check=cancel_check,
        timeout=max_runtime,
        idle_timeout=idle_timeout,
        on_output=on_output,
        on_activity=on_activity,
    )
    if returncode != 0:
        raise LitposeError(cmd, returncode, stdout, stderr)

    if not (run_root / "config.yaml").exists():
        raise FileNotFoundError(
            f"Lightning Pose exited cleanly but wrote no model at {run_root}"
        )
    return run_root
