"""
Feature library for behavior datasets.

This module provides a collection of features for behavioral analysis.
Features are automatically registered on import via the @register_feature decorator.

All features are automatically loaded when the feature_library is imported,
making them available in the global FEATURES registry.

Feature Output Types
--------------------
Features have an `output_type` attribute indicating their output structure:
- "per_frame": One row per frame (or per frame×pair/id)
- "summary": Aggregated stats per sequence/chunk/id
- "global": Operates across all sequences (embeddings, clustering)
- "viz": Produces visualizations, not data
- None: Complex/custom output

Usage
-----
>>> from mosaic.core import Dataset
>>> from mosaic.core.dataset import FEATURES
>>>
>>> # Features are auto-registered and available by name
>>> dataset.run_feature(feature="speed-angvel", params={...})
>>>
>>> # Or by class
>>> from mosaic.behavior import feature_library
>>> dataset.run_feature(
...     feature=feature_library.speed_angvel.SpeedAngvel,
...     params={...}
... )
>>>
>>> # List all registered features
>>> print(list(FEATURES.keys()))
>>>
>>> # List features by output type
>>> from mosaic.behavior.feature_library import list_features_by_type
>>> print(list_features_by_type("per_frame"))
"""

from typing import Optional

from mosaic.core.dataset import register_feature, FEATURES


def list_features_by_type(output_type: Optional[str] = None) -> list[str]:
    """
    Return feature names filtered by output_type.

    Parameters
    ----------
    output_type : str or None
        Filter to features with this output_type. Valid values:
        - "per_frame": Per-frame features
        - "summary": Summary/aggregated features
        - "global": Global fit-transform features
        - "viz": Visualization features
        - None with filter=True: Features with output_type=None (custom)
        - None with filter=False (default): Return ALL features

    Returns
    -------
    list[str]
        List of feature names (the .name attribute, e.g., "speed-angvel")
    """
    result = []
    for cls in FEATURES.values():
        feat_output_type = getattr(cls, "output_type", None)
        feat_name = getattr(cls, "name", cls.__name__)
        if output_type is None:
            # No filter - return all
            result.append(feat_name)
        elif feat_output_type == output_type:
            result.append(feat_name)
    return sorted(result)


def get_feature_output_type(feature_name: str) -> Optional[str]:
    """
    Return the output_type for a registered feature.

    Parameters
    ----------
    feature_name : str
        The feature name (e.g., "speed-angvel") or class name (e.g., "SpeedAngvel")

    Returns
    -------
    str or None
        The output_type attribute, or None if not set or feature not found
    """
    # Try direct class name lookup
    if feature_name in FEATURES:
        return getattr(FEATURES[feature_name], "output_type", None)

    # Try matching by .name attribute
    for cls in FEATURES.values():
        if getattr(cls, "name", None) == feature_name:
            return getattr(cls, "output_type", None)

    return None

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
from . import approach_avoidance
from . import id_tag_columns
from . import nn_delta_response
from . import nn_delta_bins

# Transformation/context features
from . import temporal_stacking
from . import model_predict

# Group/social features
from . import ffgroups
from . import ffgroups_metrics

# External features (keypoint-moseq via subprocess)
from . import kpms_fit
from . import kpms_apply

# Global fit-transform features
from . import global_tsne
from . import global_kmeans
from . import global_ward
from . import ward_assign


# Note: Templates are not imported (they're just examples)
# from . import feature_template__per_sequence
# from . import feature_template__global

__all__ = [
    # Helper functions
    "list_features_by_type",
    "get_feature_output_type",
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
    "approach_avoidance",
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
    # External features (keypoint-moseq)
    "kpms_fit",
    "kpms_apply",
]
