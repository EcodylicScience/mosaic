"""Calibrated cameras, and the projection of world-frame points into their images.

A world-frame tracks table (``mosaic_world_v1``) reaches a video only through the
recording's calibration. This module is that calibration and that projection,
and nothing else: no storage, no index, no importer for a vendor's file. Those
build on it.

**The conventions are OpenCV's**, which is also what Loopy's multicamcal writes and
what anipose reads, so a calibration from any of them is taken in without
re-deriving it:

- Intrinsics ``K`` (3x3) and distortion ``[k1, k2, p1, p2(, k3(, k4, k5, k6))]``
  -- 0, 4, 5 or 8 coefficients, the rational model at 8.
- Camera axes +X right, +Y down, +Z forward.
- Extrinsics world -> camera: ``x_cam = R @ X + t``, held as one 4x4 matrix.

**A camera that moves has one extrinsic per frame**: a head-mounted camera, say,
whose pose an inside-out tracker reports every frame. Its ``world_to_camera`` is a
``(T, 4, 4)`` array on the recording's frame axis, and projecting a point names
the frame it belongs to. A frame with no pose is a matrix of NaN, and projects to
NaN.

Every rotation is checked to be a proper rotation, determinant +1. A calibration
fitted without a reflection guard can come out mirrored, and a mirrored camera
projects every point to a plausible pixel on the wrong side of the image.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import numpy.typing as npt

__all__ = [
    "Camera",
    "CameraSystem",
    "WorldFrame",
    "project",
]

type WorldFrame = Literal["mosaic_world_v1", "camera0_optical", "unknown"]
"""What a camera system's world coordinates are.

- ``"mosaic_world_v1"`` -- the world-frame tracks convention: metres, right-handed,
  +Z up. A projection of a ``mosaic_world_v1`` table needs a system in this frame.
- ``"camera0_optical"`` -- the optical frame of the first camera, which is what a
  calibration fitted from point correspondences alone lands in before anyone
  aligns it: Y down, Z forward, often with an arbitrary scale.
- ``"unknown"`` -- nothing says.
"""

_DISTORTION_LENGTHS = (0, 4, 5, 8)
_ROTATION_TOLERANCE = 1e-3

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Camera:
    """One calibrated camera.

    Attributes:
        name: The camera's key in its system: a hardware serial where the rig
            records one, which is stable across recordings where an ordinal is
            not.
        image_size: ``(width, height)`` in pixels.
        K: The 3x3 intrinsic matrix.
        distortion: OpenCV distortion coefficients, 0, 4, 5 or 8 of them.
        world_to_camera: A 4x4 world -> camera matrix, or ``(T, 4, 4)`` for a
            camera that moves, one per frame of the recording. A frame without a
            pose is all NaN.

    Raises:
        ValueError: On construction, if a shape is wrong, a focal length is not
            positive, or a rotation is not a proper rotation.
    """

    name: str
    image_size: tuple[int, int]
    K: FloatArray
    distortion: FloatArray
    world_to_camera: FloatArray

    def __post_init__(self) -> None:
        intrinsics = np.asarray(self.K, dtype=np.float64)
        distortion = np.asarray(self.distortion, dtype=np.float64).reshape(-1)
        extrinsics = np.asarray(self.world_to_camera, dtype=np.float64)
        if intrinsics.shape != (3, 3):
            raise ValueError(
                f"Camera {self.name!r}: K must be 3x3, got {intrinsics.shape}"
            )
        if not (intrinsics[0, 0] > 0 and intrinsics[1, 1] > 0):
            raise ValueError(f"Camera {self.name!r}: focal lengths must be positive")
        if distortion.size not in _DISTORTION_LENGTHS:
            raise ValueError(
                f"Camera {self.name!r}: {distortion.size} distortion coefficients; "
                f"OpenCV's pinhole model takes one of {_DISTORTION_LENGTHS}"
            )
        if extrinsics.shape[-2:] != (4, 4) or extrinsics.ndim not in (2, 3):
            raise ValueError(
                f"Camera {self.name!r}: world_to_camera must be 4x4 or (T, 4, 4), "
                f"got {extrinsics.shape}"
            )
        _require_proper_rotations(self.name, extrinsics)
        # Normalised copies, so a caller's list or a float32 array is held as the
        # float64 arrays every method assumes. ``object.__setattr__`` because the
        # dataclass is frozen.
        object.__setattr__(self, "K", intrinsics)
        object.__setattr__(self, "distortion", distortion)
        object.__setattr__(self, "world_to_camera", extrinsics)

    @property
    def is_moving(self) -> bool:
        """Whether this camera has one extrinsic per frame."""
        return self.world_to_camera.ndim == 3

    @property
    def n_frames(self) -> int | None:
        """How many frames a moving camera has a pose for; ``None`` when static."""
        return int(self.world_to_camera.shape[0]) if self.is_moving else None

    def extrinsics_at(self, frames: npt.ArrayLike) -> FloatArray:
        """The world -> camera matrices at *frames*, shape ``(N, 4, 4)``.

        A static camera repeats its one matrix. A frame outside a moving camera's
        range is refused rather than clamped: a pose from another moment would
        project to a plausible pixel.
        """
        index = np.asarray(frames, dtype=np.int64).reshape(-1)
        if not self.is_moving:
            return np.broadcast_to(self.world_to_camera, (index.size, 4, 4))
        n = self.world_to_camera.shape[0]
        if index.size and (index.min() < 0 or index.max() >= n):
            raise IndexError(
                f"Camera {self.name!r} has a pose for frames 0..{n - 1}; asked for "
                f"{int(index.min())}..{int(index.max())}"
            )
        return self.world_to_camera[index]

    def center(self, frame: int | None = None) -> FloatArray:
        """The camera centre in world coordinates, ``C = -R^T t``.

        Args:
            frame: The frame, required for a moving camera and ignored otherwise.
        """
        if self.is_moving:
            if frame is None:
                raise ValueError(f"Camera {self.name!r} moves; name the frame")
            matrix = self.extrinsics_at([frame])[0]
        else:
            matrix = self.world_to_camera
        rotation, translation = matrix[:3, :3], matrix[:3, 3]
        return -rotation.T @ translation


@dataclass(frozen=True)
class CameraSystem:
    """The calibrated cameras of one recording.

    Calibration, like synchronisation, belongs to the recording rather than to
    any one of its cameras, which is why a system is what a projection is asked
    of.

    Attributes:
        cameras: The cameras, by name.
        world_frame: What the system's world coordinates are.
        units: The unit of the world coordinates.
        scale_known: Whether that unit is real. A calibration fitted from point
            correspondences alone has the right shape and an arbitrary scale
            until it is aligned to something measured.
    """

    cameras: Mapping[str, Camera]
    world_frame: WorldFrame = "unknown"
    units: str = "m"
    scale_known: bool = False
    _names: tuple[str, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        for key, camera in self.cameras.items():
            if key != camera.name:
                raise ValueError(f"Camera keyed {key!r} is named {camera.name!r}")
        object.__setattr__(self, "_names", tuple(self.cameras))

    def camera(self, name: str) -> Camera:
        """The camera called *name*, or a ``KeyError`` naming the ones there are."""
        try:
            return self.cameras[name]
        except KeyError:
            raise KeyError(
                f"No camera {name!r} in this system; it has {list(self._names)}"
            ) from None


def project(
    points: npt.ArrayLike,
    camera: Camera,
    frames: npt.ArrayLike | None = None,
) -> FloatArray:
    """Project world points into *camera*'s image, applying its distortion.

    Args:
        points: World coordinates, shape ``(..., 3)``. For a moving camera the
            first axis is time: ``(N, ..., 3)``, one row per entry of *frames*.
        camera: The camera to project into.
        frames: The frame of each row of *points*. Required for a moving camera;
            for a static one it is ignored.

    Returns:
        Pixel coordinates ``(..., 2)``. A point behind the camera, or with a NaN
        coordinate, or at a frame with no pose, is NaN -- never a pixel, because
        a point behind the lens projects to a perfectly plausible one.
    """
    world = np.asarray(points, dtype=np.float64)
    if world.shape[-1] != 3:
        raise ValueError(f"points must end in an axis of 3, got shape {world.shape}")
    if camera.is_moving:
        if frames is None:
            raise ValueError(
                f"Camera {camera.name!r} moves; pass the frame of each row"
            )
        index = np.asarray(frames, dtype=np.int64).reshape(-1)
        if world.ndim < 2 or world.shape[0] != index.size:
            raise ValueError(
                f"A moving camera projects one row of points per frame: got "
                f"{index.size} frames for points of shape {world.shape}"
            )
        matrices = camera.extrinsics_at(index)
        # Broadcast each frame's matrix over that row's trailing point axes.
        flat = world.reshape(index.size, -1, 3)
        rotation = matrices[:, None, :3, :3]
        translation = matrices[:, None, :3, 3]
        cam = (rotation @ flat[..., None])[..., 0] + translation
        cam = cam.reshape(world.shape)
    else:
        rotation = camera.world_to_camera[:3, :3]
        translation = camera.world_to_camera[:3, 3]
        cam = world @ rotation.T + translation
    return _camera_to_pixels(cam, camera.K, camera.distortion)


def _camera_to_pixels(
    cam: FloatArray, K: FloatArray, distortion: FloatArray
) -> FloatArray:
    """OpenCV's pinhole projection with its distortion model, vectorised."""
    z = cam[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        x = cam[..., 0] / z
        y = cam[..., 1] / z
    coefficients = np.zeros(8)
    coefficients[: distortion.size] = distortion
    k1, k2, p1, p2, k3, k4, k5, k6 = coefficients
    r2 = x * x + y * y
    radial = (1 + r2 * (k1 + r2 * (k2 + r2 * k3))) / (
        1 + r2 * (k4 + r2 * (k5 + r2 * k6))
    )
    xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    u = K[0, 0] * xd + K[0, 1] * yd + K[0, 2]
    v = K[1, 1] * yd + K[1, 2]
    pixels = np.stack([u, v], axis=-1)
    behind = ~(z > 0)
    pixels[behind] = np.nan
    return pixels


def _require_proper_rotations(name: str, extrinsics: FloatArray) -> None:
    """Refuse a rotation block that is not orthonormal with determinant +1."""
    matrices = extrinsics.reshape(-1, 4, 4)
    posed = ~np.isnan(matrices).any(axis=(1, 2))
    rotations = matrices[posed, :3, :3]
    if rotations.size == 0:
        return
    determinants = np.linalg.det(rotations)
    gram = rotations @ np.swapaxes(rotations, 1, 2)
    off = np.abs(gram - np.eye(3)).max(axis=(1, 2))
    bad = (np.abs(determinants - 1.0) > _ROTATION_TOLERANCE) | (
        off > _ROTATION_TOLERANCE
    )
    if bad.any():
        first = int(np.flatnonzero(posed)[np.flatnonzero(bad)[0]])
        raise ValueError(
            f"Camera {name!r}: the rotation of world_to_camera"
            f"{'' if extrinsics.ndim == 2 else f' at frame {first}'} is not a proper "
            f"rotation (determinant {determinants[bad][0]:+.4f}). A mirrored "
            "calibration projects every point to a plausible pixel on the wrong side."
        )
