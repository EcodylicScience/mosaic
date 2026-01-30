"""
Feature library for behavior datasets.

This module provides a collection of features for behavioral analysis.
Features are automatically registered on import via the @register_feature decorator.

All features are automatically loaded when the feature_library is imported,
making them available in the global FEATURES registry.

Usage
-----
>>> from behavior import Dataset
>>> from behavior.dataset import FEATURES
>>>
>>> # Features are auto-registered and available by name
>>> dataset.run_feature(feature="speed-angvel", params={...})
>>>
>>> # Or by class
>>> from behavior import feature_library
>>> dataset.run_feature(
...     feature=feature_library.speed_angvel.SpeedAngvel,
...     params={...}
... )
>>>
>>> # List all registered features
>>> print(list(FEATURES.keys()))
"""

from behavior.dataset import register_feature

# Import shared helpers (used by multiple features)
from . import helpers

# Import all features to trigger auto-registration
# Per-sequence features
from . import speed_angvel
from . import body_scale
from . import orientation_relative
from . import nearestneighbor
from . import pair_egocentric
from . import pair_position
from . import pair_wavelet
from . import pairposedistancepca
from . import id_tag_columns
from . import nn_delta_response
from . import nn_delta_bins

# Transformation/context features
from . import temporal_stacking
from . import model_predict

# Group/social features
from . import ffgroups
from . import ffgroups_metrics

# Global fit-transform features
from . import global_tsne
from . import global_kmeans
from . import global_ward
from . import ward_assign

# Visualization features
from . import viz_global_colored

# Note: Templates are not imported (they're just examples)
# from . import feature_template__per_sequence
# from . import feature_template__global

__all__ = [
    "helpers",
    # Per-sequence features
    "speed_angvel",
    "body_scale",
    "orientation_relative",
    "nearestneighbor",
    "pair_egocentric",
    "pair_position",
    "pair_wavelet",
    "pairposedistancepca",
    "id_tag_columns",
    "nn_delta_response",
    "nn_delta_bins",
    # Transformation features
    "temporal_stacking",
    "model_predict",
    # Group features
    "ffgroups",
    "ffgroups_metrics",
    # Global features
    "global_tsne",
    "global_kmeans",
    "global_ward",
    "ward_assign",
    # Visualization
    "viz_global_colored",
]
