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
>>> from mosaic.behavior.feature_library import Inputs, Result
>>> from mosaic.behavior.feature_library.speed_angvel import SpeedAngvel
>>>
>>> # Track-only feature (default inputs)
>>> feat = SpeedAngvel()
>>> dataset.run_feature(feat)
>>>
>>> # Feature consuming another feature's output
>>> feat = SpeedAngvel(inputs=Inputs((Result(feature="nn"),)))
>>> dataset.run_feature(feat)
>>>
>>> # List all registered features
>>> from mosaic.behavior.feature_library.spec import FEATURES
>>> print(list(FEATURES.keys()))
>>>
>>> # List features by output type
>>> from mosaic.behavior.feature_library import list_features_by_type
>>> print(list_features_by_type("per_frame"))
"""

# Import all submodules to trigger @register_feature auto-registration
from . import (
    approach_avoidance,
    body_scale,
    ffgroups,
    ffgroups_metrics,
    global_kmeans,
    global_tsne,
    global_ward,
    helpers,
    id_tag_columns,
    kpms_apply,
    kpms_fit,
    model_predict,
    nearestneighbor,
    nn_delta_bins,
    nn_delta_response,
    orientation_relative,
    pair_egocentric,
    pair_position,
    pair_wavelet,
    pairposedistancepca,
    speed_angvel,
    temporal_stacking,
    ward_assign,
)
from .approach_avoidance import ApproachAvoidance
from .body_scale import BodyScaleFeature
from .ffgroups import FFGroups
from .ffgroups_metrics import FFGroupsMetrics
from .global_kmeans import GlobalKMeansClustering
from .global_tsne import GlobalTSNE
from .global_ward import GlobalWardClustering
from .id_tag_columns import IdTagColumns
from .kpms_apply import KpmsApply
from .kpms_fit import KpmsFit
from .model_predict import ModelPredictFeature
from .nearestneighbor import NearestNeighbor
from .nn_delta_bins import NearestNeighborDeltaBins
from .nn_delta_response import NearestNeighborDelta
from .orientation_relative import OrientationRelativeFeature
from .pair_egocentric import PairEgocentricFeatures
from .pair_position import PairPositionFeatures
from .pair_wavelet import PairWavelet
from .pairposedistancepca import PairPoseDistancePCA
from .spec import (
    COLUMNS,
    FEATURES,
    ArtifactSpec,
    Feature,
    FeatureLabelsSource,
    GroundTruthLabelsSource,
    Inputs,
    InputsLike,
    OutputType,
    Result,
    ResultColumn,
    TrackInput,
    register_feature,
)
from .speed_angvel import SpeedAngvel
from .temporal_stacking import TemporalStackingFeature
from .ward_assign import WardAssignClustering

# Note: Templates are not imported (they're just examples)
# from . import feature_template__per_sequence
# from . import feature_template__global


def list_features_by_type(output_type: OutputType = None) -> list[str]:
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
        feat_output_type = cls.output_type
        feat_name = cls.name
        if output_type is None:
            result.append(feat_name)
        elif feat_output_type == output_type:
            result.append(feat_name)
    return sorted(result)


def get_feature_output_type(feature_name: str) -> OutputType:
    """
    Return the output_type for a registered feature.

    Parameters
    ----------
    feature_name : str
        The feature name (e.g., "speed-angvel") or class name (e.g., "SpeedAngvel")

    Returns
    -------
    OutputType
        The output_type attribute, or None if feature not found
    """
    # Try direct class name lookup
    if feature_name in FEATURES:
        return FEATURES[feature_name].output_type

    # Try matching by .name attribute
    for cls in FEATURES.values():
        if cls.name == feature_name:
            return cls.output_type

    return None


__all__ = [
    # Registry
    "Feature",
    "FEATURES",
    "register_feature",
    # Types and params
    "ArtifactSpec",
    "COLUMNS",
    "FeatureLabelsSource",
    "GroundTruthLabelsSource",
    "Inputs",
    "InputsLike",
    "OutputType",
    "Result",
    "ResultColumn",
    "TrackInput",
    # Helper functions
    "list_features_by_type",
    "get_feature_output_type",
    "helpers",
    # Feature classes
    "ApproachAvoidance",
    "BodyScaleFeature",
    "FFGroups",
    "FFGroupsMetrics",
    "GlobalKMeansClustering",
    "GlobalTSNE",
    "GlobalWardClustering",
    "IdTagColumns",
    "KpmsApply",
    "KpmsFit",
    "ModelPredictFeature",
    "NearestNeighbor",
    "NearestNeighborDelta",
    "NearestNeighborDeltaBins",
    "OrientationRelativeFeature",
    "PairEgocentricFeatures",
    "PairPositionFeatures",
    "PairPoseDistancePCA",
    "PairWavelet",
    "SpeedAngvel",
    "TemporalStackingFeature",
    "WardAssignClustering",
    # Submodules
    "approach_avoidance",
    "body_scale",
    "ffgroups",
    "ffgroups_metrics",
    "global_kmeans",
    "global_tsne",
    "global_ward",
    "id_tag_columns",
    "kpms_apply",
    "kpms_fit",
    "model_predict",
    "nearestneighbor",
    "nn_delta_bins",
    "nn_delta_response",
    "orientation_relative",
    "pair_egocentric",
    "pair_position",
    "pair_wavelet",
    "pairposedistancepca",
    "speed_angvel",
    "temporal_stacking",
    "ward_assign",
]
