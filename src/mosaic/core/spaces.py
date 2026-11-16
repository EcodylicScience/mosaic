"""The spaces a track table's spatial columns are measured in.

This leaf imports only the standard library. A reader of a declared space, such
as a definition's sidecar, avoids loading pandas.
"""

from __future__ import annotations

from typing import Final, Literal, TypeIs

__all__ = ["TRACK_SPACES", "TrackSpace", "is_track_space"]

type TrackSpace = Literal["image_px", "plane_cm", "world_m"]
"""What a table's spatial columns are measured in, and against what.

- ``"image_px"``: video pixels of the view the producer read. ``mosaic_v1`` and
  everything extending it use it. A coordinate is drawn as it stands only in this
  space: an overlay or a crop places it on the frame directly.
- ``"plane_cm"``: centimeters in the image plane, for archived tables whose
  pixel factor cannot be recovered (``mosaic_cm_v1``, ``trex_v1``). Nothing draws
  it, though a planar feature computes in it.
- ``"world_m"``: meters in a calibrated world frame, right-handed with +Z up,
  for a multi-view or fused producer (``mosaic_world_v1``). It has three axes and
  reaches a video only through a projection with the recording's calibration.

A schema family declares its space. The columns cannot identify it: a centimeter
table and a pixel table both have ``X`` and ``Y`` columns, and each reads as a
plausible number.
"""

TRACK_SPACES: Final[tuple[TrackSpace, ...]] = ("image_px", "plane_cm", "world_m")
"""Every :data:`TrackSpace`, in a stable order for messages and iteration."""


def is_track_space(value: object) -> TypeIs[TrackSpace]:
    """Whether *value* is one of :data:`TRACK_SPACES`."""
    return isinstance(value, str) and value in TRACK_SPACES
