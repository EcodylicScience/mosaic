"""Advice for a large ``xgboost`` fit left on the CPU of a host that has a GPU.

``xgboost`` fits on the CPU unless its ``device`` says ``cuda``, and the default
stays ``cpu`` because a GPU histogram build need not give the model a CPU one
does. On many entries that is slow, and the plan can tell the person submitting
it. The advice is a caution and nothing more: it is not part of the fingerprint,
it changes no param, and it never moves a step to another lane.

Whether the host has a GPU is read without importing ``torch``, from the
presence of the package and from the environment and device files that name a
GPU.

Imports the standard library and :mod:`mosaic.optional_dependency` only.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Final

from mosaic.optional_dependency import OptionalModule

from .wire import Problem

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from .plan import PlannedStep

__all__ = ["LARGE_CPU_FIT_ENTRIES", "cpu_fit_cautions", "host_has_gpu"]

LARGE_CPU_FIT_ENTRIES: Final = 50
"""The most entries an ``xgboost`` step may be asked for before the plan advises a GPU.

The plan cannot know how wide the inputs are, so the entries stand for their size.
"""

_FITS_ON_CPU_BY_DEFAULT: Final = frozenset({"xgboost"})
"""The feature slugs whose ``device`` param defaults to ``cpu``."""

_TORCH: Final = OptionalModule("torch", "deep-learning", "the GPU check")

_NVIDIA_DEVICE: Final = Path("/dev/nvidia0")


def _torch_installed() -> bool:
    """Whether ``torch`` can be imported here, found without importing it."""
    return _TORCH.missing() is None


def _nvidia_device_exists() -> bool:
    return _NVIDIA_DEVICE.exists()


def host_has_gpu() -> bool:
    """Whether this host has ``torch`` and a GPU it may use.

    ``CUDA_VISIBLE_DEVICES`` decides when it is set: empty or ``-1`` hides every
    GPU. Unset, the device file of the first NVIDIA card decides.
    """
    if not _torch_installed():
        return False
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        return visible.strip() not in ("", "-1")
    return _nvidia_device_exists()


def cpu_fit_cautions(
    steps: Sequence[PlannedStep], asked: Mapping[str, int]
) -> tuple[Problem, ...]:
    """The steps that fit a large model on the CPU of a host with a GPU.

    Args:
        steps: The planned steps.
        asked: The entries each step would be asked to compute, by step id.

    Returns:
        One caution per such step, none when the host has no GPU.
    """
    large = [
        planned
        for planned in steps
        if planned.kind == "feature"
        and planned.runs in _FITS_ON_CPU_BY_DEFAULT
        and planned.spec.params.get("device", "cpu") == "cpu"
        and asked.get(planned.step_id, 0) > LARGE_CPU_FIT_ENTRIES
    ]
    if not large or not host_has_gpu():
        return ()
    return tuple(
        Problem(
            planned.step_id,
            "params.device",
            f"fits on the CPU over {asked[planned.step_id]} entries, and this host "
            "has a GPU. Set device to 'cuda' to fit on it. The step stays on its "
            "lane either way",
        )
        for planned in large
    )
