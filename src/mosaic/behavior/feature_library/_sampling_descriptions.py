"""Field prose shared by features whose ``sampling: SamplingConfig`` field
means the same thing.

``PairEgocentricFeatures`` (:mod:`mosaic.behavior.feature_library.pair_egocentric`)
and ``PairPositionFeatures`` (:mod:`mosaic.behavior.feature_library.pair_position`)
smooth their tracks with ``sampling.smooth_win`` before computing kinematics.
``NearestNeighborDelta`` (:mod:`mosaic.behavior.feature_library.nn_delta_response`)
and ``PairWavelet`` (:mod:`mosaic.behavior.feature_library.pair_wavelet`) read
only ``sampling.fps_default`` and apply no smoothing. The two groups need two
descriptions: a field named ``sampling`` that smooths says something false
under the fps-only text, and the reverse promises a smoothing pass that never
runs.
"""

from __future__ import annotations

SAMPLING_WITH_SMOOTHING_DESCRIPTION = "Frame rate and smoothing settings."

SAMPLING_FPS_ONLY_DESCRIPTION = (
    "Frame rate settings. Only fps_default is read; smoothing is not applied."
)
