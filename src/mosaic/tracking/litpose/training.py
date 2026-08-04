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
from pathlib import Path

from mosaic.core.pipeline.subprocess_util import run_supervised
from mosaic.core.pipeline.types import JsonValue
from mosaic.tracking.common.toolenv import subprocess_env, tool_invocation
from mosaic.tracking.litpose.run import LITPOSE_ENV, LitposeError

logger = logging.getLogger(__name__)

__all__ = ["train_litpose"]

# Run by the Lightning Pose interpreter. Only argv reaches it -- the project
# directory, where the model should land, and any Hydra overrides -- so no path
# is interpolated into this source.
_TRAIN_SNIPPET: str = """
import sys
from lightning_pose.train import train
from omegaconf import OmegaConf

_project, _out = sys.argv[1], sys.argv[2]
_overrides = sys.argv[3:]

_cfg = OmegaConf.load(f"{_project}/config.yaml")
_cfg = OmegaConf.merge(_cfg, OmegaConf.from_dotlist(_overrides))
OmegaConf.update(_cfg, "data.data_dir", _project, force_add=True)
OmegaConf.update(_cfg, "hydra.run.dir", _out, force_add=True)
_model = train(_cfg)
print(f"trained into {_out}")
"""


def train_litpose(
    project_dir: str | Path,
    run_root: str | Path,
    *,
    model_type: str = "heatmap",
    backbone: str = "resnet50_animal_ap10k",
    max_epochs: int = 300,
    overrides: Mapping[str, JsonValue] | None = None,
    litpose_conda_env: str | None = None,
    litpose_bin: str | Path | None = None,
    idle_timeout: float = 1800,
    max_runtime: float | None = None,
    cancel_check: Callable[[], bool] | None = None,
    on_output: Callable[[str], None] | None = None,
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
        overrides: Further Hydra assignments, applied last.
        litpose_conda_env: Run in this conda env, overriding the environment.
        litpose_bin: A Lightning Pose script naming the install, overriding the
            environment.
        idle_timeout: Kill the subprocess after this long with no output.
        max_runtime: Optional absolute ceiling.

    Returns:
        The model directory, which is the ``litpose`` artifact shape:
        a ``config.yaml`` beside a checkpoint under ``tb_logs``.

    Raises:
        FileNotFoundError: *project_dir* holds no ``config.yaml``, or training
            exited zero without producing a model directory.
        LitposeNotFoundError: No Lightning Pose install could be located.
        LitposeError: Training exited non-zero.
    """
    project_dir = Path(project_dir)
    if not (project_dir / "config.yaml").exists():
        raise FileNotFoundError(
            f"not a Lightning Pose project -- no config.yaml in {project_dir}"
        )
    run_root = Path(run_root)
    run_root.mkdir(parents=True, exist_ok=True)

    assignments: dict[str, JsonValue] = {
        "model.model_type": model_type,
        "model.backbone": backbone,
        "training.max_epochs": max_epochs,
    }
    assignments.update(overrides or {})

    invocation = tool_invocation(
        LITPOSE_ENV,
        executable="python",
        conda_env=litpose_conda_env,
        bin_path=litpose_bin,
    )
    cmd = [
        *invocation,
        "-c",
        _TRAIN_SNIPPET,
        str(project_dir),
        str(run_root),
        *(f"{key}={value}" for key, value in assignments.items()),
    ]
    logger.info("Running: %s", " ".join(cmd[:4]) + " ...")

    stdout, stderr, returncode = run_supervised(
        cmd,
        env=subprocess_env(),
        cancel_check=cancel_check,
        timeout=max_runtime,
        idle_timeout=idle_timeout,
        on_output=on_output,
    )
    if returncode != 0:
        raise LitposeError(cmd, returncode, stdout, stderr)

    if not (run_root / "config.yaml").exists():
        raise FileNotFoundError(
            f"Lightning Pose exited cleanly but wrote no model at {run_root}"
        )
    return run_root
