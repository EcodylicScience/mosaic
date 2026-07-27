"""Deployment configuration for media probing.

`mosaic_media` reads no configuration: a caller passes `Thresholds` as an
argument, which is what keeps that package a pure library. Something has to
bridge the environment to that argument, and this toolkit is the lowest place
both consumers -- mosaic itself and the backend -- depend on.

Top level rather than under the core subpackage on purpose. Every import from
that subpackage pulls `Dataset` and the imgstore and video readers, hence
pandas, cv2, and numpy; a consumer that needs only a threshold should not pay
for those. `mosaic.runlog` is the same shape.
"""

from __future__ import annotations

import os

from mosaic_media import DEFAULT_THRESHOLDS, Thresholds


def _int_env(name: str, default: int) -> int:
    """The integer value of environment variable *name*, or *default*.

    An unset or blank value falls back; a value that is not an integer raises
    naming the variable, because a silently ignored typo would produce a verdict
    nobody could explain.
    """
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        message = f"{name} must be an integer, got {raw!r}"
        raise ValueError(message) from exc


def media_thresholds() -> Thresholds:
    """Media verdict thresholds, defaults overridden from the environment.

    Every `derive` call in this toolkit and in the backend resolves its
    thresholds here, so one file yields one verdict whatever the deployment
    sets. These values reach `derive`, not `probe_media`: that function's
    thresholds parameter feeds only `drift_frame_periods`, which nothing here
    overrides.

    The defaults are read from the `DEFAULT_THRESHOLDS` instance, never from the
    `Thresholds` class. `Thresholds` is a `slots=True` dataclass, so
    `Thresholds.max_gop_bytes` is a `member_descriptor` rather than the integer
    default, and a comparison against it raises `TypeError` at verdict time.
    """
    return Thresholds(
        max_gop_bytes=_int_env(
            "MEDIA_PROBE_MAX_GOP_BYTES", DEFAULT_THRESHOLDS.max_gop_bytes
        ),
        max_keyframe_interval_frames=_int_env(
            "MEDIA_PROBE_MAX_KEYFRAME_INTERVAL_FRAMES",
            DEFAULT_THRESHOLDS.max_keyframe_interval_frames,
        ),
    )
