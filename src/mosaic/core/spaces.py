"""The spaces a track table's spatial columns are measured in.

A leaf that imports only the standard library, so a reader of a declared space,
such as a definition's sidecar, loads no pandas.
"""

from __future__ import annotations

from typing import Final, Literal

__all__ = ["TRACK_SPACES", "TrackSpace"]

type TrackSpace = Literal["image_px", "plane_cm", "world_m"]
"""What a table's spatial columns are measured in, and against what.

- ``"image_px"`` -- video pixels of the view the producer read. ``mosaic_v1`` and
  everything extending it. The only space a coordinate may be *drawn* in as it
  stands: an overlay or a crop places it on the frame directly.
- ``"plane_cm"`` -- centimeters in the image plane, for archived tables whose
  pixel factor cannot be recovered (``mosaic_cm_v1``, ``trex_v1``). Nothing draws
  it, though a planar feature computes in it.
- ``"world_m"`` -- meters in a calibrated world frame, right-handed with +Z up,
  for a multi-view or fused producer (``mosaic_world_v1``). Three axes. It
  reaches a video only through a projection with the recording's calibration.

Declared on a schema family rather than inferred from columns, because the two
2D spaces are indistinguishable by their columns -- a centimeter table and a
pixel table both carry ``X`` and ``Y`` -- and the difference reads as a
plausible number either way.
"""

TRACK_SPACES: Final[tuple[TrackSpace, ...]] = ("image_px", "plane_cm", "world_m")
"""Every :data:`TrackSpace`, in a stable order for messages and iteration."""
