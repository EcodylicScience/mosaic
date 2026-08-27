"""What a Lightning Pose run is, declared once for every consumer.

One statement per field serves validation, run identity, invocation and
discovery. ``Field`` states the constraint pydantic enforces and
:class:`~mosaic.core.params.Declared` states the prose a client draws a control
from. Lightning Pose infers in one gated phase. Its fields do not name a phase.

The model is declared beside the integration rather than beside the op, because
:func:`~mosaic.tracking.litpose.dataset_runs.run_litpose` and
:func:`~mosaic.tracking.litpose.dataset_runs.litpose_settings` take it: declared
in ``tracking/ops/litpose.py``, the integration would import its own adapter.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from mosaic.core.pipeline.types import JsonValue
from mosaic.core.params import (
    HASH_EXCLUDE,
    Declared,
)
from mosaic.tracking.common.params import TrackerOpParams

__all__ = ["LitposeParams"]

_MODEL_PATH_DESCRIPTION = (
    "A trained Lightning Pose model directory (config.yaml plus a checkpoint "
    "under tb_logs/)."
)

_LITPOSE_OVERRIDES_DESCRIPTION = "Hydra config overrides applied at inference time."

_PRECISION_DESCRIPTION = "The forward-pass precision: fp32, fp16, or bf16."


class LitposeParams(TrackerOpParams):
    """Parameters for the ``litpose`` tracking op and for ``run_litpose``."""

    # model: one external Lightning Pose model directory (config.yaml plus a
    # checkpoint under tb_logs/). Part of the run_id identity -- via a content
    # digest of the weights + config, never the path itself.
    model_path: Annotated[str, Declared(_MODEL_PATH_DESCRIPTION)]
    # Hydra config overrides applied at inference time. Identity, because they
    # change the produced keypoints. JsonValue rather than object, so an
    # unrepresentable value is rejected at params construction.
    litpose_overrides: Annotated[
        dict[str, JsonValue] | None, Declared(_LITPOSE_OVERRIDES_DESCRIPTION)
    ] = None
    # execution knobs -- throughput/environment only, excluded from the run_id.
    # fp16/bf16 change the forward pass numerically but are a "how it ran" choice,
    # like SLEAP's device, so precision is excluded from identity.
    precision: Annotated[
        str,
        HASH_EXCLUDE,
        Field(examples=["fp32", "fp16", "bf16"]),
        Declared(_PRECISION_DESCRIPTION),
    ] = "fp32"
