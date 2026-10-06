"""A world point reaches a video only through a calibrated camera.

``project`` is the one place world-frame coordinates become pixels, so it is held
to OpenCV's own ``projectPoints`` -- the convention every calibration it takes in
was written in. A point behind the camera projects to NaN, never to a pixel, and
a mirrored rotation is refused when the camera is built rather than projecting
every point to the wrong side of the image.
"""

from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt
import pytest

from mosaic.core.cameras import Camera, CameraSystem, project

_K = np.array([[600.0, 0.0, 640.0], [0.0, 605.0, 360.0], [0.0, 0.0, 1.0]])
_RATIONAL = np.array([0.53, -3.1, 6e-5, -2.3e-4, 1.85, 0.4, -2.9, 1.75])


def _pose(rvec: tuple[float, float, float], tvec: tuple[float, float, float]):
    rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = tvec
    return matrix


def _camera(
    distortion: npt.ArrayLike = _RATIONAL, world_to_camera: npt.ArrayLike | None = None
) -> Camera:
    return Camera(
        name="cam0",
        image_size=(1280, 720),
        K=_K,
        distortion=np.asarray(distortion, dtype=np.float64),
        world_to_camera=(
            _pose((0.1, -0.2, 0.05), (0.1, -0.2, 2.0))
            if world_to_camera is None
            else np.asarray(world_to_camera, dtype=np.float64)
        ),
    )


def _points_in_front(n: int = 200) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.uniform([-0.6, -0.4, -0.5], [0.6, 0.4, 0.5], size=(n, 3))


@pytest.mark.parametrize("n_coefficients", [0, 4, 5, 8])
def test_projection_matches_opencv(n_coefficients: int) -> None:
    camera = _camera(_RATIONAL[:n_coefficients])
    points = _points_in_front()
    rvec, _ = cv2.Rodrigues(camera.world_to_camera[:3, :3])
    expected, _ = cv2.projectPoints(
        points, rvec, camera.world_to_camera[:3, 3], _K, camera.distortion
    )
    np.testing.assert_allclose(project(points, camera), expected[:, 0, :], atol=1e-6)


def test_leading_axes_are_kept() -> None:
    camera = _camera()
    points = _points_in_front(24).reshape(4, 6, 3)
    assert project(points, camera).shape == (4, 6, 2)


def test_a_point_behind_the_camera_is_nan_not_a_pixel() -> None:
    camera = _camera(world_to_camera=np.eye(4))
    pixels = project(
        np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [0.1, 0.0, 0.0]]), camera
    )
    assert np.isfinite(pixels[0]).all()
    assert np.isnan(pixels[1]).all()
    assert np.isnan(pixels[2]).all()


def test_a_nan_point_projects_to_nan() -> None:
    pixels = project(np.array([[np.nan, 0.0, 0.0]]), _camera())
    assert np.isnan(pixels).all()


def test_a_moving_camera_uses_each_frames_pose() -> None:
    poses = np.stack(
        [
            _pose((0.0, 0.0, 0.0), (0.0, 0.0, 2.0)),
            _pose((0.0, 0.3, 0.0), (0.2, 0.0, 2.5)),
        ]
    )
    moving = _camera(world_to_camera=poses)
    points = _points_in_front(10).reshape(2, 5, 3)
    pixels = project(points, moving, frames=[1, 0])
    for row, frame in enumerate([1, 0]):
        static = _camera(world_to_camera=poses[frame])
        np.testing.assert_allclose(pixels[row], project(points[row], static), atol=1e-9)


def test_a_moving_camera_requires_the_frames() -> None:
    poses = np.stack([np.eye(4), np.eye(4)])
    with pytest.raises(ValueError, match="moves"):
        _ = project(np.zeros((2, 3)), _camera(world_to_camera=poses))


def test_a_frame_without_a_pose_projects_to_nan() -> None:
    poses = np.stack([_pose((0, 0, 0), (0, 0, 2.0)), np.full((4, 4), np.nan)])
    pixels = project(np.zeros((2, 1, 3)), _camera(world_to_camera=poses), frames=[0, 1])
    assert np.isfinite(pixels[0]).all()
    assert np.isnan(pixels[1]).all()


def test_a_frame_outside_the_recorded_poses_is_refused() -> None:
    poses = np.stack([np.eye(4), np.eye(4)])
    with pytest.raises(IndexError, match="frames 0..1"):
        _ = project(np.zeros((1, 3)), _camera(world_to_camera=poses), frames=[5])


def test_a_mirrored_rotation_is_refused() -> None:
    mirror = np.diag([1.0, 1.0, -1.0, 1.0])
    with pytest.raises(ValueError, match="proper rotation"):
        _ = _camera(world_to_camera=mirror)


def test_a_mirrored_frame_of_a_moving_camera_is_named() -> None:
    poses = np.stack([np.eye(4), np.diag([1.0, -1.0, 1.0, 1.0])])
    with pytest.raises(ValueError, match="at frame 1"):
        _ = _camera(world_to_camera=poses)


@pytest.mark.parametrize("distortion", [np.zeros(3), np.zeros(14)])
def test_an_unsupported_distortion_length_is_refused(distortion: np.ndarray) -> None:
    with pytest.raises(ValueError, match="distortion"):
        _ = _camera(distortion)


def test_the_centre_is_minus_r_transpose_t() -> None:
    camera = _camera()
    rotation = camera.world_to_camera[:3, :3]
    centre = camera.center()
    np.testing.assert_allclose(
        rotation @ centre + camera.world_to_camera[:3, 3], 0, atol=1e-12
    )


def test_a_system_names_its_cameras_when_one_is_missing() -> None:
    system = CameraSystem(cameras={"cam0": _camera()}, world_frame="mosaic_world_v1")
    assert system.camera("cam0").name == "cam0"
    with pytest.raises(KeyError, match="cam0"):
        _ = system.camera("cam9")


def test_a_system_refuses_a_camera_under_another_name() -> None:
    with pytest.raises(ValueError, match="named"):
        _ = CameraSystem(cameras={"other": _camera()})
