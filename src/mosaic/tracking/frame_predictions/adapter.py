"""Declares the call that runs one framework's tool over extracted images.

A predict-frames run hands a :class:`FrameAdapter` the images of a request, the
resolved model and the tool's :class:`ToolProbe`, and reads back the objects
found in each image with the pose that the tool reports.

Imports ``core``, ``tracking.common`` and ``tracking.model_refs`` and no op
module. The adapters and the op both import it without a cycle.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from mosaic.core.annotations.model import KeypointSchema
from mosaic.core.annotations.pose_predictions import PredictedObject
from mosaic.tracking.frame_predictions.params import PredictFramesParams
from mosaic.tracking.model_refs import ResolvedModel

__all__ = ["AdapterResult", "FrameAdapter", "FrameImage", "ToolProbe"]


@dataclass(frozen=True, slots=True)
class ToolProbe:
    """Records what the tool reported before any image was predicted.

    Attributes:
        tool_version: The installed tool's version, recorded with the
            predictions.
        n_keypoints: How many keypoints the model predicts, as the Ultralytics
            runner's probe counts them: the model's count for a yolo-pose
            model, and 1 for a POLO model. ``None`` for SLEAP and Lightning
            Pose, whose keypoints are read from the predictions.
    """

    tool_version: str
    n_keypoints: int | None


@dataclass(frozen=True, slots=True)
class FrameImage:
    """Names one image of a request.

    Attributes:
        position: The image's place in the request.
        key: The image's dataset-relative POSIX path, its image key.
        path: The absolute path of the image file, resolved from its image key.
        width: Image width in pixels.
        height: Image height in pixels.
    """

    position: int
    key: str
    path: Path
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class AdapterResult:
    """Returns the objects that a tool found in the images of a request.

    Attributes:
        objects: One tuple of objects per image, in request order. Each
            object's keypoints are in the tool's order.
        tool_pose: The pose that the tool reports. The run checks it against
            the pose that the model's provenance records, or adopts it.
    """

    objects: tuple[tuple[PredictedObject, ...], ...]
    tool_pose: KeypointSchema


class FrameAdapter(Protocol):
    """Runs one trained model over the images of a request with its framework's tool."""

    def __call__(
        self,
        images: Sequence[FrameImage],
        model: ResolvedModel,
        params: PredictFramesParams,
        probe: ToolProbe,
        work_dir: Path,
        *,
        on_output: Callable[[str], None],
        cancel_check: Callable[[], bool],
    ) -> AdapterResult:
        """Predict on *images* and return the objects found in each.

        Args:
            images: The images, in request order.
            model: The resolved model, or the SLEAP top-down pair.
            params: The run's parameters.
            probe: What the tool reported before the run took its claims.
            work_dir: The directory for the tool's exchange files, which the run
                removes after it writes the predictions.
            on_output: Called with each line that the tool prints.
            cancel_check: Returns ``True`` once the run is cancelled.

        Returns:
            One tuple of objects per image, in request order, and the tool's pose.
        """
        ...
