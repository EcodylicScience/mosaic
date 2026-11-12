"""Field prose shared by features whose ``sampling: SamplingConfig`` field
means the same thing.

``PairEgocentricFeatures`` (:mod:`mosaic.behavior.feature_library.pair_egocentric`)
and ``PairPositionFeatures`` (:mod:`mosaic.behavior.feature_library.pair_position`)
smooth their tracks with ``sampling.smooth_win`` before computing kinematics.
They take no frame rate: a velocity divides by the seconds between rows, read from
the ``time`` column.
"""

from __future__ import annotations

SAMPLING_WITH_SMOOTHING_DESCRIPTION = "Smoothing settings."
