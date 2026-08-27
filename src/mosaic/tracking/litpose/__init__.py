"""Lightning Pose integration for single-animal, per-frame pose inference.

This module drives the Lightning Pose Python API headlessly against a pre-trained
model and bridges the results into standardized tracks. Lightning Pose is
single-animal and produces no cross-frame identity, so each video becomes one
``id=0`` track; its DeepLabCut-style CSV is read by the existing ``deeplabcut``
converter.

Requires:
    A Lightning Pose install (https://lightning-pose.readthedocs.io). It is heavy
    (PyTorch + Lightning + NVIDIA DALI) and video inference needs a Linux CUDA
    GPU, so it usually lives in its **own** environment; point the wrapper at it
    with ``litpose_conda_env=`` / ``MOSAIC_LITPOSE_CONDA_ENV`` (or ``litpose_bin=``
    / ``MOSAIC_LITPOSE_BIN``), else ``litpose`` is found on ``$PATH``. Lightning
    Pose inference is headless and needs no ``DISPLAY``.

Usage
-----
>>> from mosaic.tracking.litpose import LitposeParams, run_litpose
>>> run_litpose(ds, LitposeParams(model_path="models/litpose_model"))

Equivalently set ``MOSAIC_LITPOSE_CONDA_ENV=litpose`` once and drive it from the
``mosaic run --kind litpose`` op.
"""

from mosaic.tracking.litpose.dataset_runs import (
    LitposeIndexRow,
    list_litpose_runs,
    run_litpose,
)
from mosaic.tracking.litpose.params import LitposeParams
from mosaic.tracking.litpose.run import (
    LitposeError,
    LitposeNotFoundError,
    LitposePredictResult,
    run_litpose_predict,
)

__all__ = [
    "LitposeError",
    "LitposeIndexRow",
    "LitposeNotFoundError",
    "LitposeParams",
    "LitposePredictResult",
    "list_litpose_runs",
    "run_litpose",
    "run_litpose_predict",
]
